"""Checks f1live.packets against the f1_2020_telemetry reference library.

For every packet type we encode sample records with f1live, decode the bytes with the
reference library, and check both decoders agree field by field. Samples come from
the old JSON capture when it is present, otherwise from synthetic packets.

    python -m pytest tests/test_packets.py      (or)      python tests/test_packets.py
"""

from __future__ import annotations

import ctypes
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from f1live import packets as P  # noqa: E402

try:
    from f1_2020_telemetry.packets import unpack_udp_packet
except ImportError:  # pragma: no cover
    unpack_udp_packet = None

CAPTURE = os.path.join(os.path.dirname(__file__), "..", "f1_telemetry_capture.jsonl")


def lib_to_dict(obj):
    if isinstance(obj, ctypes.Array):
        return [lib_to_dict(x) for x in obj]
    if isinstance(obj, (ctypes.Structure, ctypes.Union)):
        return {name: lib_to_dict(getattr(obj, name)) for name, _ in obj._fields_}
    if isinstance(obj, bytes):
        return obj.split(b"\x00", 1)[0].decode("utf-8", errors="replace")
    return obj


def same(a, b, path="") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        errs = []
        for k in a:
            if k in ("id",):
                continue
            if k not in b:
                errs.append(f"{path}.{k} missing")
                continue
            errs += same(a[k], b[k], f"{path}.{k}")
        return errs
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path} len {len(a)} != {len(b)}"]
        errs = []
        for i, (x, y) in enumerate(zip(a, b)):
            errs += same(x, y, f"{path}[{i}]")
        return errs
    if isinstance(a, float) or isinstance(b, float):
        if not math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-6):
            return [f"{path}: {a} != {b}"]
        return []
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def samples(limit_per_type: int = 40) -> list[dict]:
    got: dict[str, list] = {}
    if os.path.exists(CAPTURE):
        with open(CAPTURE, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i > 60000:
                    break
                r = json.loads(line)
                t = r["packet_type"].rsplit("_V", 1)[0]
                if len(got.setdefault(t, [])) < limit_per_type:
                    got[t].append(r["data"])
    # synthetic event packets so every union member is exercised
    base = {"packetFormat": 2020, "gameMajorVersion": 1, "gameMinorVersion": 18, "packetVersion": 1,
            "packetId": P.EVENT, "sessionUID": 1, "sessionTime": 12.5, "frameIdentifier": 9,
            "playerCarIndex": 3, "secondaryPlayerCarIndex": 255}
    got.setdefault("synthetic_events", [])
    for code, det in [
        ("PENA", {"penaltyType": 4, "infringementType": 3, "vehicleIdx": 17, "otherVehicleIdx": 3,
                  "time": 5, "lapNum": 12, "placesGained": 0}),
        ("FTLP", {"vehicleIdx": 1, "lapTime": 95.826}),
        ("SPTP", {"vehicleIdx": 2, "speed": 312.5}),
        ("RTMT", {"vehicleIdx": 14}),
        ("SSTA", {}),
    ]:
        got["synthetic_events"].append({"header": base, "eventStringCode": code, "eventDetails": det})
    return [p for v in got.values() for p in v]


def test_sizes():
    assert P.PACKET_SIZES == {0: 1464, 1: 251, 2: 1190, 3: 35, 4: 1213, 5: 1102, 6: 1307, 7: 1344, 8: 839, 9: 1169}


def test_roundtrip_against_reference():
    if unpack_udp_packet is None:
        print("f1_2020_telemetry not installed; skipping reference comparison")
        return
    n = 0
    for rec in samples():
        raw = P.encode(rec)
        pid = rec["header"]["packetId"]
        assert len(raw) == P.PACKET_SIZES[pid], (pid, len(raw))
        ours = P.decode(raw)
        ref = lib_to_dict(unpack_udp_packet(raw))
        if pid == P.EVENT:
            code = ref["eventStringCode"]
            code = code if isinstance(code, str) else bytes(code).decode()
            ref["eventStringCode"] = code
            member = {"FTLP": "fastestLap", "RTMT": "retirement", "TMPT": "teamMateInPits",
                      "RCWN": "raceWinner", "PENA": "penalty", "SPTP": "speedTrap"}.get(code)
            ref["eventDetails"] = ref["eventDetails"][member] if member else {}
        errs = same(ours, ref)
        assert not errs, (P.PACKET_NAMES[pid], errs[:5])
        # and our decode must reproduce the original record where it was well-formed
        if pid != P.EVENT or isinstance(rec.get("eventDetails"), dict):
            errs = same({k: v for k, v in rec.items() if k != "eventDetails" or pid == P.EVENT}, ours)
            assert not errs, (P.PACKET_NAMES[pid], errs[:5])
        n += 1
    print(f"{n} packets matched the reference decoder")


if __name__ == "__main__":
    test_sizes()
    test_roundtrip_against_reference()
    print("ok")
