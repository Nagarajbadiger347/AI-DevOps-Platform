"""Linux node health checker.

Reads system metrics directly from /proc (no external dependencies) with an
optional psutil upgrade when available.  Falls back gracefully on non-Linux
systems (macOS, Windows) so the import never crashes.

Return schema:
    {
        "status":  "healthy" | "degraded" | "error" | "unavailable",
        "success": bool,
        "details": {
            "cpu":    {"load_avg_1m", "load_avg_5m", "load_avg_15m", "cpu_count", "load_pct"},
            "memory": {"total_mb", "used_mb", "free_mb", "used_pct"},
            "disk":   {"total_gb", "used_gb", "free_gb", "used_pct"},
            "uptime_hours": float,
            "hostname": str,
        }
    }

Thresholds for "degraded":
    CPU load > 90% of core count
    Memory used > 90%
    Disk (/) used > 85%
"""
from __future__ import annotations

import os
import platform
import socket


# ── Optional psutil (faster / more accurate) ──────────────────────────────────

try:
    import psutil as _psutil
    _PSUTIL = True
except ImportError:
    _psutil = None  # type: ignore
    _PSUTIL = False


# ── /proc readers (Linux only) ─────────────────────────────────────────────────

def _read_meminfo() -> dict[str, int]:
    """Parse /proc/meminfo into a {key: kB_value} dict."""
    mem: dict[str, int] = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split(":", 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip().split()[0]
                    mem[key] = int(val)
    except Exception:
        pass
    return mem


def _read_uptime_secs() -> float:
    """Return system uptime in seconds from /proc/uptime."""
    try:
        with open("/proc/uptime") as f:
            return float(f.read().split()[0])
    except Exception:
        return 0.0


def _disk_usage(path: str = "/") -> dict:
    """Return disk usage for *path* using os.statvfs (no external deps)."""
    try:
        st = os.statvfs(path)
        total = st.f_frsize * st.f_blocks
        free  = st.f_frsize * st.f_bavail
        used  = total - free
        used_pct = round(used / total * 100, 1) if total else 0.0
        return {
            "total_gb": round(total / 1024**3, 1),
            "used_gb":  round(used  / 1024**3, 1),
            "free_gb":  round(free  / 1024**3, 1),
            "used_pct": used_pct,
        }
    except Exception:
        return {"total_gb": 0, "used_gb": 0, "free_gb": 0, "used_pct": 0}


# ── Public API ─────────────────────────────────────────────────────────────────

def check_linux_node() -> dict:
    """Check Linux node health — CPU, memory, disk, uptime.

    Works on any Linux host without psutil.  If psutil is installed the
    numbers are more accurate (uses instantaneous CPU % instead of load avg
    approximation).

    On non-Linux platforms returns status="unavailable" rather than raising.
    """
    system = platform.system()
    if system != "Linux":
        return {
            "status":  "unavailable",
            "success": False,
            "details": f"Linux checker only runs on Linux (current: {system})",
        }

    try:
        hostname  = socket.gethostname()
        cpu_count = os.cpu_count() or 1

        # ── CPU ──────────────────────────────────────────────────────────
        if _PSUTIL:
            cpu_pct   = _psutil.cpu_percent(interval=0.2)
            load1, load5, load15 = os.getloadavg()
        else:
            load1, load5, load15 = os.getloadavg()
            cpu_pct = round(load1 / cpu_count * 100, 1)

        cpu_info = {
            "load_avg_1m":  round(load1,  2),
            "load_avg_5m":  round(load5,  2),
            "load_avg_15m": round(load15, 2),
            "cpu_count":    cpu_count,
            "load_pct":     round(cpu_pct, 1),
        }

        # ── Memory ───────────────────────────────────────────────────────
        if _PSUTIL:
            vm = _psutil.virtual_memory()
            mem_info = {
                "total_mb": round(vm.total   / 1024**2, 1),
                "used_mb":  round(vm.used    / 1024**2, 1),
                "free_mb":  round(vm.available / 1024**2, 1),
                "used_pct": vm.percent,
            }
        else:
            mem = _read_meminfo()
            total_kb = mem.get("MemTotal",     0)
            avail_kb = mem.get("MemAvailable", 0)
            used_kb  = total_kb - avail_kb
            mem_info = {
                "total_mb": round(total_kb / 1024, 1),
                "used_mb":  round(used_kb  / 1024, 1),
                "free_mb":  round(avail_kb / 1024, 1),
                "used_pct": round(used_kb / total_kb * 100, 1) if total_kb else 0.0,
            }

        # ── Disk ─────────────────────────────────────────────────────────
        disk_info = _disk_usage("/")

        # ── Swap ─────────────────────────────────────────────────────────
        if _PSUTIL:
            sw = _psutil.swap_memory()
            swap_info = {
                "total_mb": round(sw.total / 1024**2, 1),
                "used_mb":  round(sw.used  / 1024**2, 1),
                "used_pct": sw.percent,
            }
        else:
            mem = _read_meminfo()
            sw_total = mem.get("SwapTotal", 0)
            sw_free  = mem.get("SwapFree",  0)
            sw_used  = sw_total - sw_free
            swap_info = {
                "total_mb": round(sw_total / 1024, 1),
                "used_mb":  round(sw_used  / 1024, 1),
                "used_pct": round(sw_used / sw_total * 100, 1) if sw_total else 0.0,
            }

        # ── Uptime ───────────────────────────────────────────────────────
        uptime_hours = round(_read_uptime_secs() / 3600, 1)

        # ── Health determination ─────────────────────────────────────────
        degraded_reasons: list[str] = []
        if cpu_info["load_pct"]   > 90:
            degraded_reasons.append(f"high CPU load ({cpu_info['load_pct']}%)")
        if mem_info["used_pct"]   > 90:
            degraded_reasons.append(f"high memory usage ({mem_info['used_pct']}%)")
        if disk_info["used_pct"]  > 85:
            degraded_reasons.append(f"high disk usage ({disk_info['used_pct']}%)")
        if swap_info["total_mb"] > 0 and swap_info["used_pct"] > 80:
            degraded_reasons.append(f"high swap usage ({swap_info['used_pct']}%)")

        overall = "degraded" if degraded_reasons else "healthy"

        return {
            "status":       overall,
            "success":      True,
            "degraded_reasons": degraded_reasons,
            "details": {
                "hostname":     hostname,
                "cpu":          cpu_info,
                "memory":       mem_info,
                "swap":         swap_info,
                "disk":         disk_info,
                "uptime_hours": uptime_hours,
            },
        }

    except Exception as exc:
        return {"status": "error", "success": False, "details": str(exc)}
