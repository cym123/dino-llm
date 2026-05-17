import argparse
import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path


def read_proc_status(pid: int) -> dict:
    """Read key fields from /proc/<pid>/status (Linux only)."""
    fields = {}
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                key, _, value = line.partition(":\t")
                fields[key.strip()] = value.strip()
    except FileNotFoundError:
        return {}
    return fields


def read_proc_stat(pid: int) -> tuple[int, int]:
    """Read utime and stime (CPU ticks) from /proc/<pid>/stat (Linux only)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            parts = f.read().split()
        # Fields 14 and 15 (0-indexed: 13, 14) are utime and stime in clock ticks
        utime = int(parts[13])
        stime = int(parts[14])
        return utime, stime
    except (FileNotFoundError, IndexError):
        return 0, 0


def read_psutil(pid: int) -> dict:
    """Fallback for Windows using psutil."""
    try:
        import psutil
        proc = psutil.Process(pid)
        mem = proc.memory_info()
        cpu = proc.cpu_times()
        ctx = proc.num_ctx_switches()
        return {
            "rss_kb": mem.rss // 1024,
            "vms_kb": mem.vms // 1024,
            "cpu_utime_ticks": int(cpu.user * 100),
            "cpu_stime_ticks": int(cpu.system * 100),
            "vol_ctx_switches": ctx.voluntary,
            "nonvol_ctx_switches": ctx.involuntary,
            "threads": proc.num_threads(),
        }
    except Exception as e:
        return {"error": str(e)}


def sample(pid: int, use_proc: bool) -> dict:
    ts = datetime.now().isoformat(timespec="milliseconds")
    if use_proc:
        status = read_proc_status(pid)
        utime, stime = read_proc_stat(pid)
        return {
            "timestamp": ts,
            "rss_kb": status.get("VmRSS", "0 kB").split()[0],
            "vms_kb": status.get("VmSize", "0 kB").split()[0],
            "vmpeak_kb": status.get("VmPeak", "0 kB").split()[0],
            "cpu_utime_ticks": utime,
            "cpu_stime_ticks": stime,
            "vol_ctx_switches": status.get("voluntary_ctxt_switches", "0"),
            "nonvol_ctx_switches": status.get("nonvoluntary_ctxt_switches", "0"),
            "threads": status.get("Threads", "0"),
        }
    else:
        d = read_psutil(pid)
        d["timestamp"] = ts
        return d


def main():
    parser = argparse.ArgumentParser(description="Live /proc monitor for the inference server")
    parser.add_argument("--pid", type=int, required=True, help="Server process PID")
    parser.add_argument("--duration", type=int, default=60, help="Monitoring duration in seconds")
    parser.add_argument("--interval", type=float, default=0.5, help="Polling interval in seconds")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path (auto-generated if not set)")
    args = parser.parse_args()

    use_proc = sys.platform == "linux" and os.path.exists(f"/proc/{args.pid}/status")
    if not use_proc:
        print(f"[INFO] /proc not available (platform: {sys.platform}). Using psutil fallback.")
        try:
            import psutil  # noqa
        except ImportError:
            print("[ERROR] psutil not installed. Run: pip install psutil")
            sys.exit(1)
    else:
        print(f"[INFO] Reading /proc/{args.pid}/status on Linux.")

    out_dir = Path("profiling_reports")
    out_dir.mkdir(exist_ok=True)
    out_path = args.output or str(out_dir / f"proc_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")

    print(f"[INFO] Monitoring PID {args.pid} for {args.duration}s → {out_path}")
    print(f"       Interval: {args.interval}s | Mode: {'proc' if use_proc else 'psutil'}")
    print()

    fieldnames = [
        "timestamp", "rss_kb", "vms_kb", "vmpeak_kb",
        "cpu_utime_ticks", "cpu_stime_ticks",
        "vol_ctx_switches", "nonvol_ctx_switches", "threads"
    ]

    start = time.time()
    samples = []

    try:
        with open(out_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()

            while time.time() - start < args.duration:
                row = sample(args.pid, use_proc)
                writer.writerow(row)
                csvfile.flush()

                # Print live summary
                print(
                    f"\r  [{row['timestamp']}]  "
                    f"RSS={row.get('rss_kb', '?'):>8} kB  "
                    f"Threads={row.get('threads', '?'):>3}  "
                    f"CtxSwitch(vol)={row.get('vol_ctx_switches', '?'):>6}",
                    end="", flush=True
                )
                samples.append(row)
                time.sleep(args.interval)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")

    print(f"\n\n[DONE] Captured {len(samples)} samples → {out_path}")

    # Summary stats
    try:
        rss_vals = [int(s["rss_kb"]) for s in samples if s.get("rss_kb", "0").isdigit()]
        if rss_vals:
            print(f"  RSS memory — min: {min(rss_vals):,} kB  max: {max(rss_vals):,} kB  "
                  f"delta: {max(rss_vals) - min(rss_vals):,} kB")
    except Exception:
        pass


if __name__ == "__main__":
    main()
