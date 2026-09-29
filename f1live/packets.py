"""F1 2020 UDP packet decoder/encoder (packet format 2020, no external dependencies).

Decoded packets are plain dicts whose field names match the ones used by the
`f1_2020_telemetry` library, so captures made with the old `f1_capture.py`
(which wrote the library's structures as JSON) can be fed through the same code.

Every decoded packet has the shape::

    {"id": <packet id>, "header": {...}, <packet fields>...}
"""

from __future__ import annotations

import struct
from typing import Any

MOTION, SESSION, LAP, EVENT, PARTICIPANTS, SETUPS, TELEMETRY, STATUS, FINAL, LOBBY = range(10)

PACKET_NAMES = {
    MOTION: "PacketMotionData",
    SESSION: "PacketSessionData",
    LAP: "PacketLapData",
    EVENT: "PacketEventData",
    PARTICIPANTS: "PacketParticipantsData",
    SETUPS: "PacketCarSetupData",
    TELEMETRY: "PacketCarTelemetryData",
    STATUS: "PacketCarStatusData",
    FINAL: "PacketFinalClassificationData",
    LOBBY: "PacketLobbyInfoData",
}
PACKET_IDS = {v: k for k, v in PACKET_NAMES.items()}


class Layout:
    """A packed little-endian record described as (name, struct code, count) triples."""

    def __init__(self, spec: list[tuple[str, str, int]]):
        self.spec = spec
        fmt = "<"
        self.slots: list[tuple[str, int, int, bool]] = []
        i = 0
        for name, code, count in spec:
            if code.endswith("s"):  # fixed-size string: one struct item
                fmt += code
                self.slots.append((name, i, 1, True))
                i += 1
            else:
                fmt += code * count
                self.slots.append((name, i, count, False))
                i += count
        self.struct = struct.Struct(fmt)
        self.size = self.struct.size

    def decode(self, buf: bytes, offset: int = 0) -> dict[str, Any]:
        vals = self.struct.unpack_from(buf, offset)
        out: dict[str, Any] = {}
        for name, i, n, is_str in self.slots:
            if is_str:
                out[name] = vals[i].split(b"\x00", 1)[0].decode("utf-8", errors="replace")
            elif n == 1:
                out[name] = vals[i]
            else:
                out[name] = list(vals[i:i + n])
        return out

    def encode(self, d: dict[str, Any]) -> bytes:
        vals: list[Any] = []
        for name, code, count in self.spec:
            v = d.get(name, 0)
            if code.endswith("s"):
                vals.append((v or "").encode("utf-8")[: int(code[:-1])])
            elif count == 1:
                vals.append(_coerce(code, v))
            else:
                seq = list(v) if isinstance(v, (list, tuple)) else [0] * count
                vals.extend(_coerce(code, x) for x in (seq + [0] * count)[:count])
        return self.struct.pack(*vals)


def _coerce(code: str, v: Any) -> Any:
    if code in "fd":
        return float(v or 0.0)
    return int(v or 0)


HEADER = Layout([
    ("packetFormat", "H", 1), ("gameMajorVersion", "B", 1), ("gameMinorVersion", "B", 1),
    ("packetVersion", "B", 1), ("packetId", "B", 1), ("sessionUID", "Q", 1),
    ("sessionTime", "f", 1), ("frameIdentifier", "I", 1), ("playerCarIndex", "B", 1),
    ("secondaryPlayerCarIndex", "B", 1),
])

CAR_MOTION = Layout([
    ("worldPositionX", "f", 1), ("worldPositionY", "f", 1), ("worldPositionZ", "f", 1),
    ("worldVelocityX", "f", 1), ("worldVelocityY", "f", 1), ("worldVelocityZ", "f", 1),
    ("worldForwardDirX", "h", 1), ("worldForwardDirY", "h", 1), ("worldForwardDirZ", "h", 1),
    ("worldRightDirX", "h", 1), ("worldRightDirY", "h", 1), ("worldRightDirZ", "h", 1),
    ("gForceLateral", "f", 1), ("gForceLongitudinal", "f", 1), ("gForceVertical", "f", 1),
    ("yaw", "f", 1), ("pitch", "f", 1), ("roll", "f", 1),
])
MOTION_EXTRA = Layout([
    ("suspensionPosition", "f", 4), ("suspensionVelocity", "f", 4), ("suspensionAcceleration", "f", 4),
    ("wheelSpeed", "f", 4), ("wheelSlip", "f", 4),
    ("localVelocityX", "f", 1), ("localVelocityY", "f", 1), ("localVelocityZ", "f", 1),
    ("angularVelocityX", "f", 1), ("angularVelocityY", "f", 1), ("angularVelocityZ", "f", 1),
    ("angularAccelerationX", "f", 1), ("angularAccelerationY", "f", 1), ("angularAccelerationZ", "f", 1),
    ("frontWheelsAngle", "f", 1),
])

MARSHAL_ZONE = Layout([("zoneStart", "f", 1), ("zoneFlag", "b", 1)])
WEATHER_SAMPLE = Layout([
    ("sessionType", "B", 1), ("timeOffset", "B", 1), ("weather", "B", 1),
    ("trackTemperature", "b", 1), ("airTemperature", "b", 1),
])
SESSION_A = Layout([
    ("weather", "B", 1), ("trackTemperature", "b", 1), ("airTemperature", "b", 1),
    ("totalLaps", "B", 1), ("trackLength", "H", 1), ("sessionType", "B", 1), ("trackId", "b", 1),
    ("formula", "B", 1), ("sessionTimeLeft", "H", 1), ("sessionDuration", "H", 1),
    ("pitSpeedLimit", "B", 1), ("gamePaused", "B", 1), ("isSpectating", "B", 1),
    ("spectatorCarIndex", "B", 1), ("sliProNativeSupport", "B", 1), ("numMarshalZones", "B", 1),
])
SESSION_B = Layout([("safetyCarStatus", "B", 1), ("networkGame", "B", 1), ("numWeatherForecastSamples", "B", 1)])

CAR_LAP = Layout([
    ("lastLapTime", "f", 1), ("currentLapTime", "f", 1), ("sector1TimeInMS", "H", 1),
    ("sector2TimeInMS", "H", 1), ("bestLapTime", "f", 1), ("bestLapNum", "B", 1),
    ("bestLapSector1TimeInMS", "H", 1), ("bestLapSector2TimeInMS", "H", 1),
    ("bestLapSector3TimeInMS", "H", 1), ("bestOverallSector1TimeInMS", "H", 1),
    ("bestOverallSector1LapNum", "B", 1), ("bestOverallSector2TimeInMS", "H", 1),
    ("bestOverallSector2LapNum", "B", 1), ("bestOverallSector3TimeInMS", "H", 1),
    ("bestOverallSector3LapNum", "B", 1), ("lapDistance", "f", 1), ("totalDistance", "f", 1),
    ("safetyCarDelta", "f", 1), ("carPosition", "B", 1), ("currentLapNum", "B", 1),
    ("pitStatus", "B", 1), ("sector", "B", 1), ("currentLapInvalid", "B", 1), ("penalties", "B", 1),
    ("gridPosition", "B", 1), ("driverStatus", "B", 1), ("resultStatus", "B", 1),
])

EVENT_DETAILS = {
    "FTLP": Layout([("vehicleIdx", "B", 1), ("lapTime", "f", 1)]),
    "RTMT": Layout([("vehicleIdx", "B", 1)]),
    "TMPT": Layout([("vehicleIdx", "B", 1)]),
    "RCWN": Layout([("vehicleIdx", "B", 1)]),
    "PENA": Layout([
        ("penaltyType", "B", 1), ("infringementType", "B", 1), ("vehicleIdx", "B", 1),
        ("otherVehicleIdx", "B", 1), ("time", "B", 1), ("lapNum", "B", 1), ("placesGained", "B", 1),
    ]),
    "SPTP": Layout([("vehicleIdx", "B", 1), ("speed", "f", 1)]),
}
EVENT_UNION_SIZE = 7  # largest member (PENA)

PARTICIPANT = Layout([
    ("aiControlled", "B", 1), ("driverId", "B", 1), ("teamId", "B", 1), ("raceNumber", "B", 1),
    ("nationality", "B", 1), ("name", "48s", 1), ("yourTelemetry", "B", 1),
])

CAR_SETUP = Layout([
    ("frontWing", "B", 1), ("rearWing", "B", 1), ("onThrottle", "B", 1), ("offThrottle", "B", 1),
    ("frontCamber", "f", 1), ("rearCamber", "f", 1), ("frontToe", "f", 1), ("rearToe", "f", 1),
    ("frontSuspension", "B", 1), ("rearSuspension", "B", 1), ("frontAntiRollBar", "B", 1),
    ("rearAntiRollBar", "B", 1), ("frontSuspensionHeight", "B", 1), ("rearSuspensionHeight", "B", 1),
    ("brakePressure", "B", 1), ("brakeBias", "B", 1), ("rearLeftTyrePressure", "f", 1),
    ("rearRightTyrePressure", "f", 1), ("frontLeftTyrePressure", "f", 1),
    ("frontRightTyrePressure", "f", 1), ("ballast", "B", 1), ("fuelLoad", "f", 1),
])

CAR_TELEMETRY = Layout([
    ("speed", "H", 1), ("throttle", "f", 1), ("steer", "f", 1), ("brake", "f", 1), ("clutch", "B", 1),
    ("gear", "b", 1), ("engineRPM", "H", 1), ("drs", "B", 1), ("revLightsPercent", "B", 1),
    ("brakesTemperature", "H", 4), ("tyresSurfaceTemperature", "B", 4),
    ("tyresInnerTemperature", "B", 4), ("engineTemperature", "H", 1), ("tyresPressure", "f", 4),
    ("surfaceType", "B", 4),
])
TELEMETRY_TAIL = Layout([
    ("buttonStatus", "I", 1), ("mfdPanelIndex", "B", 1), ("mfdPanelIndexSecondaryPlayer", "B", 1),
    ("suggestedGear", "b", 1),
])

CAR_STATUS = Layout([
    ("tractionControl", "B", 1), ("antiLockBrakes", "B", 1), ("fuelMix", "B", 1),
    ("frontBrakeBias", "B", 1), ("pitLimiterStatus", "B", 1), ("fuelInTank", "f", 1),
    ("fuelCapacity", "f", 1), ("fuelRemainingLaps", "f", 1), ("maxRPM", "H", 1), ("idleRPM", "H", 1),
    ("maxGears", "B", 1), ("drsAllowed", "B", 1), ("drsActivationDistance", "H", 1),
    ("tyresWear", "B", 4), ("actualTyreCompound", "B", 1), ("visualTyreCompound", "B", 1),
    ("tyresAgeLaps", "B", 1), ("tyresDamage", "B", 4), ("frontLeftWingDamage", "B", 1),
    ("frontRightWingDamage", "B", 1), ("rearWingDamage", "B", 1), ("drsFault", "B", 1),
    ("engineDamage", "B", 1), ("gearBoxDamage", "B", 1), ("vehicleFiaFlags", "b", 1),
    ("ersStoreEnergy", "f", 1), ("ersDeployMode", "B", 1), ("ersHarvestedThisLapMGUK", "f", 1),
    ("ersHarvestedThisLapMGUH", "f", 1), ("ersDeployedThisLap", "f", 1),
])

CAR_FINAL = Layout([
    ("position", "B", 1), ("numLaps", "B", 1), ("gridPosition", "B", 1), ("points", "B", 1),
    ("numPitStops", "B", 1), ("resultStatus", "B", 1), ("bestLapTime", "f", 1),
    ("totalRaceTime", "d", 1), ("penaltiesTime", "B", 1), ("numPenalties", "B", 1),
    ("numTyreStints", "B", 1), ("tyreStintsActual", "B", 8), ("tyreStintsVisual", "B", 8),
])

LOBBY_PLAYER = Layout([
    ("aiControlled", "B", 1), ("teamId", "B", 1), ("nationality", "B", 1), ("name", "48s", 1),
    ("readyStatus", "B", 1),
])

U8 = struct.Struct("<B")

# Total sizes of each packet in the 2020 format, used to reject malformed/other-game packets.
PACKET_SIZES = {
    MOTION: HEADER.size + 22 * CAR_MOTION.size + MOTION_EXTRA.size,
    SESSION: HEADER.size + SESSION_A.size + 21 * MARSHAL_ZONE.size + SESSION_B.size + 20 * WEATHER_SAMPLE.size,
    LAP: HEADER.size + 22 * CAR_LAP.size,
    EVENT: HEADER.size + 4 + EVENT_UNION_SIZE,
    PARTICIPANTS: HEADER.size + 1 + 22 * PARTICIPANT.size,
    SETUPS: HEADER.size + 22 * CAR_SETUP.size,
    TELEMETRY: HEADER.size + 22 * CAR_TELEMETRY.size + TELEMETRY_TAIL.size,
    STATUS: HEADER.size + 22 * CAR_STATUS.size,
    FINAL: HEADER.size + 1 + 22 * CAR_FINAL.size,
    LOBBY: HEADER.size + 1 + 22 * LOBBY_PLAYER.size,
}


class DecodeError(ValueError):
    pass


def _cars(layout: Layout, buf: bytes, off: int, n: int = 22) -> tuple[list[dict], int]:
    out = []
    for _ in range(n):
        out.append(layout.decode(buf, off))
        off += layout.size
    return out, off


def peek_header(buf: bytes) -> dict[str, Any]:
    if len(buf) < HEADER.size:
        raise DecodeError("packet shorter than header")
    return HEADER.decode(buf, 0)


def decode(buf: bytes) -> dict[str, Any]:
    """Decode one UDP payload. Raises DecodeError for anything that isn't an F1 2020 packet."""
    h = peek_header(buf)
    if h["packetFormat"] != 2020:
        raise DecodeError(f"unsupported packet format {h['packetFormat']}")
    pid = h["packetId"]
    size = PACKET_SIZES.get(pid)
    if size is None or len(buf) < size:
        raise DecodeError(f"bad packet id {pid} / length {len(buf)}")
    p: dict[str, Any] = {"id": pid, "header": h}
    off = HEADER.size
    if pid == MOTION:
        p["carMotionData"], off = _cars(CAR_MOTION, buf, off)
        p.update(MOTION_EXTRA.decode(buf, off))
    elif pid == SESSION:
        p.update(SESSION_A.decode(buf, off))
        off += SESSION_A.size
        p["marshalZones"], off = _cars(MARSHAL_ZONE, buf, off, 21)
        p.update(SESSION_B.decode(buf, off))
        off += SESSION_B.size
        p["weatherForecastSamples"], off = _cars(WEATHER_SAMPLE, buf, off, 20)
    elif pid == LAP:
        p["lapData"], off = _cars(CAR_LAP, buf, off)
    elif pid == EVENT:
        code = buf[off:off + 4].decode("ascii", errors="replace")
        p["eventStringCode"] = code
        layout = EVENT_DETAILS.get(code)
        p["eventDetails"] = layout.decode(buf, off + 4) if layout else {}
    elif pid == PARTICIPANTS:
        p["numActiveCars"] = U8.unpack_from(buf, off)[0]
        p["participants"], off = _cars(PARTICIPANT, buf, off + 1)
    elif pid == SETUPS:
        p["carSetups"], off = _cars(CAR_SETUP, buf, off)
    elif pid == TELEMETRY:
        p["carTelemetryData"], off = _cars(CAR_TELEMETRY, buf, off)
        p.update(TELEMETRY_TAIL.decode(buf, off))
    elif pid == STATUS:
        p["carStatusData"], off = _cars(CAR_STATUS, buf, off)
    elif pid == FINAL:
        p["numCars"] = U8.unpack_from(buf, off)[0]
        p["classificationData"], off = _cars(CAR_FINAL, buf, off + 1)
    elif pid == LOBBY:
        p["numPlayers"] = U8.unpack_from(buf, off)[0]
        p["lobbyPlayers"], off = _cars(LOBBY_PLAYER, buf, off + 1)
    return p


def encode(p: dict[str, Any]) -> bytes:
    """Inverse of decode(). Accepts decoded dicts or dicts from the old JSON capture."""
    h = dict(p["header"])
    pid = h["packetId"]
    parts = [HEADER.encode(h)]

    def cars(layout: Layout, rows: list[dict] | None, n: int = 22) -> None:
        rows = list(rows or [])
        for i in range(n):
            parts.append(layout.encode(rows[i] if i < len(rows) else {}))

    if pid == MOTION:
        cars(CAR_MOTION, p.get("carMotionData"))
        parts.append(MOTION_EXTRA.encode(p))
    elif pid == SESSION:
        parts.append(SESSION_A.encode(p))
        cars(MARSHAL_ZONE, p.get("marshalZones"), 21)
        parts.append(SESSION_B.encode(p))
        cars(WEATHER_SAMPLE, p.get("weatherForecastSamples"), 20)
    elif pid == LAP:
        cars(CAR_LAP, p.get("lapData"))
    elif pid == EVENT:
        code = str(p.get("eventStringCode", "????"))[:4].ljust(4)
        details = p.get("eventDetails")
        layout = EVENT_DETAILS.get(code)
        body = layout.encode(details) if (layout and isinstance(details, dict)) else b""
        parts.append(code.encode("ascii") + body.ljust(EVENT_UNION_SIZE, b"\x00"))
    elif pid == PARTICIPANTS:
        parts.append(U8.pack(int(p.get("numActiveCars", 0))))
        cars(PARTICIPANT, p.get("participants"))
    elif pid == SETUPS:
        cars(CAR_SETUP, p.get("carSetups"))
    elif pid == TELEMETRY:
        cars(CAR_TELEMETRY, p.get("carTelemetryData"))
        parts.append(TELEMETRY_TAIL.encode(p))
    elif pid == STATUS:
        cars(CAR_STATUS, p.get("carStatusData"))
    elif pid == FINAL:
        parts.append(U8.pack(int(p.get("numCars", 0))))
        cars(CAR_FINAL, p.get("classificationData"))
    elif pid == LOBBY:
        parts.append(U8.pack(int(p.get("numPlayers", 0))))
        cars(LOBBY_PLAYER, p.get("lobbyPlayers"))
    else:
        raise DecodeError(f"cannot encode packet id {pid}")
    return b"".join(parts)
