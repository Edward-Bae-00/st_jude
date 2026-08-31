# ===========================================================================
# Step 5b - build the model on the biggest local disk (e.g. local-scratch)
# Run this INSTEAD of Step 5 when Step 5 dies with "no space left on device".
# Requires Steps 1-4 to have run (it reuses MODEL / DRIVE_DIR).
# ===========================================================================
import os, pathlib, shutil, subprocess, time, urllib.request

HOST      = "http://localhost:11434"
MODEL_TAG = globals().get("MODEL", "medgemma-27b-bf16").split(":")[0]
MODEL_GB  = float(globals().get("MODEL_GB", 54.0))
NEED_GB   = MODEL_GB * 3.2          # staged copy + blob + compatibility rewrite

# --- 1. every real local filesystem, most free first -----------------------
SKIP = {"tmpfs", "devtmpfs", "squashfs", "proc", "sysfs", "cgroup", "cgroup2",
        "devpts", "fuse", "fuse.drive", "fuseblk"}
disks = {}
for line in open("/proc/mounts"):
    parts = line.split()
    if len(parts) < 3:
        continue
    target, fstype = parts[1], parts[2]
    if fstype in SKIP or "/drive" in target:
        continue
    try:
        u = shutil.disk_usage(target)
    except OSError:
        continue
    if os.access(target, os.W_OK) and u.total > 20e9:
        disks[target] = (u.free, u.total, fstype)

print("writable local filesystems, most free first:")
ranked = sorted(disks.items(), key=lambda kv: -kv[1][0])
for target, (free, total, fstype) in ranked:
    print(f"   {target:30s} {fstype:12s} {free/1e9:7.0f} GB free / {total/1e9:.0f} GB")
if not ranked:
    raise RuntimeError("Found no writable local filesystem. Run `!df -h` and set "
                       "SCRATCH_ROOT by hand below.")

SCRATCH_ROOT = next((t for t, _ in ranked if "scratch" in t.lower()), ranked[0][0])
# SCRATCH_ROOT = "/mnt/local-scratch"     # <- uncomment to force a specific disk
free_gb = disks[SCRATCH_ROOT][0] / 1e9
print(f"\nusing {SCRATCH_ROOT}  ({free_gb:.0f} GB free, need ~{NEED_GB:.0f} GB)")
if free_gb < NEED_GB:
    raise RuntimeError(
        f"{SCRATCH_ROOT} has {free_gb:.0f} GB free but staging + `ollama create` needs "
        f"about {NEED_GB:.0f} GB. Pick another disk above, or set MODEL_OVERRIDE = None "
        f"in Step 4 to fall back to Q8_0 (28.7 GB).")

STORE = pathlib.Path(SCRATCH_ROOT) / "ollama_models"
STORE.mkdir(parents=True, exist_ok=True)

# --- 2. find the merged .gguf ---------------------------------------------
cands = []
if globals().get("MERGED_IN_DRIVE"):
    cands.append(MERGED_IN_DRIVE)
if globals().get("DRIVE_DIR"):
    cands += [q for q in pathlib.Path(DRIVE_DIR).glob("**/*.gguf") if "-of-" not in q.name]
cands.append(pathlib.Path("/content/merged.gguf"))

SRC = None
for q in cands:
    try:
        if q.exists() and q.stat().st_size >= MODEL_GB * 1e9 * 0.99:
            with open(q, "rb") as fh:
                if fh.read(4) == b"GGUF":
                    SRC = q
                    break
    except OSError:
        continue
if SRC is None:
    raise RuntimeError("No complete merged .gguf found in Drive or at /content/merged.gguf.")
print(f"source: {SRC}  ({SRC.stat().st_size / 1e9:.1f} GB)")

# --- 3. stage it onto the fast disk, with progress -------------------------
# `ollama create` reads and hashes the whole file before writing anything. Doing
# that over Drive FUSE is 15-30 silent minutes. One bulk sequential copy first is
# faster and, more to the point, visible.
STAGED = pathlib.Path(SCRATCH_ROOT) / SRC.name
total = SRC.stat().st_size
if STAGED.exists() and STAGED.stat().st_size == total:
    print(f"already staged at {STAGED}")
else:
    print(f"staging -> {STAGED}")
    t0, done, last = time.time(), 0, 0.0
    with open(SRC, "rb") as fi, open(STAGED, "wb") as fo:
        while chunk := fi.read(64 << 20):          # 64 MB at a time
            fo.write(chunk)
            done += len(chunk)
            if time.time() - last > 5:
                el = time.time() - t0
                rate = done / el / 1e6
                eta = (total - done) / (done / el) if done else 0
                print(f"   {done/1e9:5.1f} / {total/1e9:.1f} GB   "
                      f"{rate:5.0f} MB/s   eta {eta/60:4.1f} min", flush=True)
                last = time.time()
    print(f"   staged {total/1e9:.1f} GB in {(time.time()-t0)/60:.1f} min")

# --- 4. restart the daemon with its store on the big disk ------------------
# OLLAMA_MODELS is read ONCE at startup, so the daemon has to be restarted.
print("\nrestarting the daemon with the store on the big disk ...")
subprocess.run(["pkill", "-f", "ollama serve"], capture_output=True)
time.sleep(4)
os.environ["OLLAMA_MODELS"] = str(STORE)
os.environ.setdefault("OLLAMA_KEEP_ALIVE", "-1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", str(globals().get("NUM_PARALLEL", 4)))
os.environ.setdefault("OLLAMA_CONTEXT_LENGTH", str(globals().get("NUM_CTX", 4096)))
os.environ.setdefault("OLLAMA_FLASH_ATTENTION", "1")
SERVE_DIR = STORE            # keep later cells pointing at the right place

subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(45):
    time.sleep(2)
    try:
        urllib.request.urlopen(HOST, timeout=2); break
    except Exception:
        pass
else:
    raise RuntimeError("Ollama daemon never came up on :11434")
print(f"    daemon up, OLLAMA_MODELS={STORE}")

# --- 5. create, with the output actually visible ---------------------------
mf = pathlib.Path("/content/Modelfile")
mf.write_text(f"FROM {STAGED}\n")
print(f"\nollama create {MODEL_TAG} - progress below, do not interrupt:")

# The `!` form streams to the cell; subprocess.run(capture_output=True) hides it
# all until the command ends, which is what made this look frozen.
get_ipython().system(f'ollama create {MODEL_TAG} -f {mf}')

# --- 6. verify -------------------------------------------------------------
listed = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
print("\n" + listed)
if MODEL_TAG not in listed:
    raise RuntimeError(f"{MODEL_TAG} is not in the store - read the create output above.")
print(f"store: {STORE}  ({shutil.disk_usage(STORE).free / 1e9:.0f} GB still free)")
print(f"You can reclaim {total/1e9:.0f} GB now:   !rm {STAGED}")
print("This disk is wiped when the runtime ends; Steps 6 and 7 work for this session.")
