"""
F1 2020 UDP Telemetry Capture (size-optimized)

Key changes vs. the original version:
  1. Packet-type filtering: Motion (and other packet types you don't need)
     are skipped by default. Motion is the single biggest contributor to
     file size because it carries full physics data for all 22 cars,
     at high frequency.
  2. Optional gzip compression on the output file (~5-10x smaller on top
     of the filtering, since JSON telemetry is highly repetitive).
  3. Packet-type byte/count summary printed at the end so you can see
     exactly where your file size is coming from.

Also recommended (no code needed): in F1 2020, go to
Game Options -> Settings -> Telemetry Settings -> UDP Send Rate,
and lower it (e.g. to 10Hz). This reduces the raw volume of every
packet type at the source, independent of anything below.
"""

import re
import socket
import json
import gzip
import time
import os
import ctypes
from collections import Counter
from datetime import datetime

from f1_2020_telemetry.packets import unpack_udp_packet

HOST = "0.0.0.0"
PORT = 20777

# Compress the output file. Strongly recommended - jsonl telemetry
# compresses very well (repetitive field names/structure).
USE_GZIP = True
OUTPUT_FILE = "f1_telemetry_capture.jsonl.gz" if USE_GZIP else "f1_telemetry_capture.jsonl"

# Packet types to KEEP. Comment/uncomment as needed.
# Names come from type(packet).__name__ in the f1_2020_telemetry lib.
# Matched after stripping any trailing _V<number> suffix, so this list
# stays correct even if the library's version suffix changes.
PACKET_TYPES_TO_CAPTURE = {
    "PacketMotionData",             # world position/velocity/orientation for all cars - needed to retrace track position
    "PacketLapData",                # lap times, sector times, track distance, positions, penalties
    "PacketCarTelemetryData",       # speed, throttle, brake, gear, DRS, tyre temps
    "PacketCarStatusData",          # tyre wear, fuel, ERS, damage flags
    "PacketEventData",              # collisions (with car indices), flags, penalties, fastest lap
    "PacketSessionData",            # track, weather, session type/time remaining
    "PacketParticipantsData",       # driver names, team, AI/human
    "PacketFinalClassificationData" # end-of-session results
    # Deliberately excluded by default (uncomment if you really need them):
    # "PacketCarSetupData",        # setups of all cars (privacy + size)
    # "PacketLobbyInfoData",       # lobby-only info
}

# Optional: further down-sample a specific packet type by only keeping
# 1 out of every N packets of that type. Leave empty dict to disable.
# Example: {"PacketCarTelemetryData": 2} keeps every 2nd telemetry packet.
DOWNSAMPLE = {}


def ctypes_to_dict(obj):
    """Recursively convert ctypes structures/arrays into JSON data."""

    if isinstance(obj, ctypes.Array):
        return [ctypes_to_dict(x) for x in obj]

    if isinstance(obj, ctypes.Structure):
        result = {}
        for field_name, field_type in obj._fields_:
            value = getattr(obj, field_name)
            if isinstance(value, bytes):
                try:
                    value = value.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
                except Exception:
                    value = repr(value)
            result[field_name] = ctypes_to_dict(value)
        return result

    if isinstance(obj, bytes):
        try:
            return obj.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
        except Exception:
            return repr(obj)

    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj

    try:
        return obj.value
    except AttributeError:
        pass

    return str(obj)


def normalize_type_name(name):
    """Strip a trailing _V<number> version suffix, e.g. 'PacketMotionData_V1' -> 'PacketMotionData'.
    Makes matching robust to library version-suffix differences."""
    return re.sub(r"_V\d+$", "", name)


def open_output(path):
    if USE_GZIP:
        return gzip.open(path, "at", encoding="utf-8")
    return open(path, "a", buffering=1)


print("=" * 60)
print("F1 2020 TELEMETRY CAPTURE (filtered)")
print("=" * 60)
print(f"Listening on UDP {HOST}:{PORT}")
print(f"Saving to: {os.path.abspath(OUTPUT_FILE)}")
print(f"Capturing packet types: {sorted(PACKET_TYPES_TO_CAPTURE)}")
print(f"Gzip: {USE_GZIP}")
print()
print("Start F1 2020 and drive.")
print("Press Ctrl+C to stop.")
print("=" * 60)

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
sock.bind((HOST, PORT))

packet_count = 0
written_count = 0
skipped_type_count = 0
type_counter = Counter()
downsample_counter = Counter()
start_time = time.time()

with open_output(OUTPUT_FILE) as f:
    try:
        while True:
            raw_data, addr = sock.recvfrom(4096)
            packet_count += 1

            try:
                packet = unpack_udp_packet(raw_data)
                raw_type_name = type(packet).__name__
                type_name = normalize_type_name(raw_type_name)
                type_counter[raw_type_name] += 1

                if type_name not in PACKET_TYPES_TO_CAPTURE:
                    skipped_type_count += 1
                    continue

                if type_name in DOWNSAMPLE:
                    downsample_counter[type_name] += 1
                    n = DOWNSAMPLE[type_name]
                    if downsample_counter[type_name] % n != 0:
                        continue

                record = {
                    "capture_time": datetime.now().isoformat(),
                    "source_ip": addr[0],
                    "source_port": addr[1],
                    "packet_size": len(raw_data),
                    "packet_type": raw_type_name,
                    "data": ctypes_to_dict(packet),
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                written_count += 1

                if written_count <= 10 or written_count % 100 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"[{written_count:6d}] {type_name:30s} "
                        f"{len(raw_data):4d} bytes from {addr[0]} ({elapsed:.1f}s)"
                    )

            except Exception as e:
                print(f"Decode error: {e}")

    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print("CAPTURE STOPPED")
        print(f"UDP packets received:  {packet_count}")
        print(f"Records written:       {written_count}")
        print(f"Skipped (filtered):    {skipped_type_count}")
        print(f"File: {os.path.abspath(OUTPUT_FILE)}")
        if os.path.exists(OUTPUT_FILE):
            size_mb = os.path.getsize(OUTPUT_FILE) / (1024 * 1024)
            print(f"File size: {size_mb:.2f} MB")
        print()
        print("Packet type breakdown (all received, incl. filtered):")
        for t, c in type_counter.most_common():
            kept = "kept" if normalize_type_name(t) in PACKET_TYPES_TO_CAPTURE else "SKIPPED"
            print(f"  {t:30s} {c:6d}  [{kept}]")
        print("=" * 60)

    finally:
        sock.close()
