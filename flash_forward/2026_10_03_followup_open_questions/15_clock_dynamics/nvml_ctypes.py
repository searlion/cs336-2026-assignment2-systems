# nvml_ctypes.py
# A minimal ctypes binding to NVML (libnvidia-ml.so.1), so the polling needs no extra package.
# Only the calls clock_dynamics.py uses: SM clock, instantaneous and 1-second-average power,
# temperature, P-state, the active clock-event ("throttle") reasons, and the per-policy
# violation counters (cumulative time, in ns, that each limiter held the clock down).
import ctypes

_lib = ctypes.CDLL("libnvidia-ml.so.1")

FI_POWER_AVERAGE = 185  # NVML_FI_DEV_POWER_AVERAGE: power averaged over 1 s (Ampere and newer)
FI_POWER_INSTANT = 186  # NVML_FI_DEV_POWER_INSTANT
REASONS = {0x1: "idle", 0x2: "app clocks", 0x4: "SW power cap", 0x8: "HW slowdown", 0x10: "sync boost",
           0x20: "SW thermal", 0x40: "HW thermal", 0x80: "HW power brake", 0x100: "display clock"}
POLICIES = {0: "power", 1: "thermal", 2: "sync boost", 3: "board limit", 4: "low utilization",
            5: "reliability", 10: "total app clocks", 11: "total base clocks"}


class _Value(ctypes.Union):
    _fields_ = [("dVal", ctypes.c_double), ("uiVal", ctypes.c_uint), ("ulVal", ctypes.c_ulong),
                ("ullVal", ctypes.c_ulonglong), ("sllVal", ctypes.c_longlong), ("usVal", ctypes.c_ushort)]


class _FieldValue(ctypes.Structure):
    _fields_ = [("fieldId", ctypes.c_uint), ("scopeId", ctypes.c_uint), ("timestamp", ctypes.c_longlong),
                ("latencyUsec", ctypes.c_longlong), ("valueType", ctypes.c_int), ("nvmlReturn", ctypes.c_int),
                ("value", _Value)]


class _Violation(ctypes.Structure):
    _fields_ = [("referenceTime", ctypes.c_ulonglong), ("violationTime", ctypes.c_ulonglong)]


def _check(ret, what):
    if ret != 0:
        raise RuntimeError(f"{what} failed with NVML error {ret}")


class Device:
    def __init__(self, index=0):
        _check(_lib.nvmlInit_v2(), "nvmlInit_v2")
        self.h = ctypes.c_void_p()
        _check(_lib.nvmlDeviceGetHandleByIndex_v2(ctypes.c_uint(index), ctypes.byref(self.h)), "handle")
        self._u = ctypes.c_uint()
        self._i = ctypes.c_int()
        self._ull = ctypes.c_ulonglong()
        self._viol = _Violation()
        self._fields = (_FieldValue * 2)()
        self._fields[0].fieldId, self._fields[1].fieldId = FI_POWER_INSTANT, FI_POWER_AVERAGE

    def sm_clock(self):  # MHz
        _check(_lib.nvmlDeviceGetClockInfo(self.h, 1, ctypes.byref(self._u)), "clock")
        return self._u.value

    def clock(self, clock_type, clock_id):  # MHz; type 0 graphics, 1 SM, 2 memory; id 0 current, 3 customer boost max
        ret = _lib.nvmlDeviceGetClock(self.h, clock_type, clock_id, ctypes.byref(self._u))
        return self._u.value if ret == 0 else f"NVML error {ret}"

    def power(self):  # (instantaneous W, 1-s average W)
        _check(_lib.nvmlDeviceGetFieldValues(self.h, 2, self._fields), "field values")
        return tuple(f.value.uiVal / 1000 if f.nvmlReturn == 0 else float("nan") for f in self._fields)

    def temperature(self):
        _check(_lib.nvmlDeviceGetTemperature(self.h, 0, ctypes.byref(self._u)), "temperature")
        return self._u.value

    def pstate(self):
        _check(_lib.nvmlDeviceGetPerformanceState(self.h, ctypes.byref(self._i)), "pstate")
        return self._i.value

    def reasons(self):  # bitmask, see REASONS
        _check(_lib.nvmlDeviceGetCurrentClocksEventReasons(self.h, ctypes.byref(self._ull)), "reasons")
        return self._ull.value

    def violation_ns(self, policy):  # cumulative ns this policy held the clock down, or None
        ret = _lib.nvmlDeviceGetViolationStatus(self.h, policy, ctypes.byref(self._viol))
        return self._viol.violationTime if ret == 0 else None
