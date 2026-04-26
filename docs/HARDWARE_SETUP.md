# ArqueoGal Hardware Setup Guide

## Hardware

- Lenovo Legion laptop, Windows 10 + WSL2 Ubuntu
- RTX 3060 6 GB VRAM (CUDA via WSL2 passthrough)
- 32 GB host RAM
- NVMe storage (1 TB SSD)

## Current Environment (audit 2026-04-24)

As of the audit snapshot:

```
MemTotal:        10183888 kB (~9.7 GB)
MemAvailable:     5057348 kB (~4.8 GB)
SwapTotal:       3145728 kB (~3.0 GB)
Filesystem:      /dev/sdd (native ext4, 1007 GB, 159 GB used, 798 GB available)
/etc/wsl.conf:   present (systemd=true, generateResolvConf=false)
```

The data/ and models/ directories are on `/dev/sdd` (native WSL2 ext4), not on `/mnt/c/` (Windows-side 9P filesystem). This is correct for I/O performance.

## WSL2 Memory Ceiling

WSL2 runs as a Hyper-V VM with a dynamic memory ceiling. On Windows 10 with older WSL2 versions, the default is 50% of host RAM or 8 GB, whichever is smaller. The current audit shows MemTotal at 9.7 GB, which suggests WSL2 is allocated somewhat above the baseline; however, under Stream 3 inference workload (3–4 GB feature-matrix joins, 500 MB ensemble inference state, plus Python and OS overhead), the current ceiling is inadequate and risks VM-wide shutdown (harsher than native Linux OOM-kill).

### Recommended `.wslconfig`

The user must create this file manually on the Windows side at `%UserProfile%\.wslconfig`:

```ini
[wsl2]
memory=24GB
swap=8GB
processors=8
```

Rationale:

- `memory=24GB`: leaves 6–8 GB for Windows OS and concurrent processes.
- `swap=8GB`: provides relief if a transient spike approaches the ceiling.
- `processors=8`: matches the Lenovo Legion's hyperthreaded core count.

After creating the file, run `wsl --shutdown` from Windows PowerShell, then reopen the WSL2 terminal. Verify with `free -h` inside WSL2 (MemTotal should be ~24 GB).

## Linux Kernel Tuning

Inside WSL2, apply the sysctl settings in `etc/sysctl.d/99-arqueogal.conf` via:

```bash
sudo cp etc/sysctl.d/99-arqueogal.conf /etc/sysctl.d/
sudo sysctl --system
```

This reduces swappiness (prefer freeing page cache), lowers vfs_cache_pressure (retain directory/inode caches), and disables panic-on-OOM (kill the offending process instead).

## Filesystem Placement

ArqueoGal's hot data (data/, models/) MUST live on the WSL2 ext4 filesystem (under /home/...), not on `/mnt/c/` (Windows-side 9P filesystem). The 9P mount has 10–100x I/O overhead and undermines training and inference throughput.

The current setup is correct: data/ and models/ are on `/dev/sdd` (native ext4).

## Memory Monitoring During Stream 3

Before launching a Stream 3 inference run, start the memory monitor in a separate terminal:

```bash
nohup bash scripts/monitoring/wsl_memory_monitor.sh >> logs/inference_memory.log 2>&1 &
```

The monitor logs `/proc/meminfo` and the inference process's RSS/VMS every 10 seconds. After inference completes, inspect the log for memory spikes or pre-OOM signatures.

## OOM Recovery

If WSL2 shuts down mid-Stream-3 (the entire distribution disappears), the inference state on disk is salvageable but must be resumed from the last checkpoint. Steps:

1. From Windows PowerShell: `wsl --shutdown` then `wsl` to restart.
2. Inside WSL2: check `data/processed/stream3_output/` for the last partial Parquet emit.
3. Resume inference from the last batch index recorded in the memory monitor log.

## Outstanding Actions for the User

1. **Create `.wslconfig` on Windows:** Follow the template above and place at `%UserProfile%\.wslconfig`. Restart WSL2 with `wsl --shutdown` and verify MemTotal reaches ~24 GB.
2. **Apply sysctl settings:** Run the commands above in a WSL2 terminal (requires `sudo`).
3. **Monitor first Stream 3 run:** Start the memory monitor before inference to establish baseline memory behavior under your data and model workload.
