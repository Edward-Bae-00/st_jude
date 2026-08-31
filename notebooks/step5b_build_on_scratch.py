# ===========================================================================
# Step 5b - build the model on the biggest local disk (e.g. local-scratch)
# Run this INSTEAD of Step 5 when Step 5 dies with "no space left on device".
# Requires Steps 1-4 to have run (it reuses MODEL / DRIVE_DIR).
# ===========================================================================
import os, pathlib, shutil, subprocess, time, urllib.request

HOST      = "http://localhost:11434"
MODEL_TAG = globals().get("MODEL", "medgemma-27b-bf16").split(":")[0]
MODEL_GB  = float(globals().get("MODEL_GB", 54.0))
NEED_GB   = MODEL_GB * 2.2          # blob + llama-quantize compatibility rewrite

# --- 1. every real local filesystem, most free first -----------------------
SKIP = {"tmpfs", "devtmpfs", "squashfs", "proc", "sysfs", "cgroup", "cgroup2",
        "devpts", "fuse", "fuse.drive", "fuseblk", "overlayfs?"}
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

# Prefer anything that looks like a scratch disk; otherwise just take the biggest.
SCRATCH_ROOT = next((t for t, _ in ranked if "scratch" in t.lower()), ranked[0][0])
# SCRATCH_ROOT = "/mnt/local-scratch"     # <- uncomment to force a specific disk
free_gb = disks[SCRATCH_ROOT][0] / 1e9
print(f"\nusing {SCRATCH_ROOT}  ({free_gb:.0f} GB free, need ~{NEED_GB:.0f} GB)")
if free_gb < NEED_GB:
    raise RuntimeError(
        f"{SCRATCH_ROOT} has {free_gb:.0f} GB free but `ollama create` needs about "
        f"{NEED_GB:.0f} GB. Pick another disk above, or set MODEL_OVERRIDE = None in "
        f"Step 4 to fall back to Q8_0 (28.7 GB).")

STORE = pathlib.Path(SCRATCH_ROOT) / "ollama_models"
STORE.mkdir(parents=True, exist_ok=True)

# --- 2. find the merged .gguf to build from --------------------------------
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
    raise RuntimeError("No complete merged .gguf found in Drive or at /content/merged.gguf. "
                       "Re-run Step 5 to produce one - it will stop before `ollama create`.")
print(f"building from {SRC}  ({SRC.stat().st_size / 1e9:.1f} GB)")

# --- 3. restart the daemon with its store on the big disk ------------------
# OLLAMA_MODELS is read ONCE at startup, so the daemon has to be restarted.
print("\nrestarting the daemon with the store on the big disk ...")
subprocess.run(["pkill", "-f", "ollama serve"], capture_output=True)
time.sleep(4)

os.environ["OLLAMA_MODELS"]     = str(STORE)
os.environ.setdefault("OLLAMA_KEEP_ALIVE", "-1")
os.environ.setdefault("OLLAMA_NUM_PARALLEL", str(globals().get("NUM_PARALLEL", 4)))
os.environ.setdefault("OLLAMA_CONTEXT_LENGTH", str(globals().get("NUM_CTX", 4096)))
os.environ.setdefault("OLLAMA_FLASH_ATTENTION", "1")
SERVE_DIR = STORE            # keep later cells pointing at the right place

subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(45):
    time.sleep(2)
    try:
        urllib.request.urlopen(HOST, timeout=2)
        break
    except Exception:
        pass
else:
    raise RuntimeError("Ollama daemon never came up on :11434")
print(f"    daemon up, OLLAMA_MODELS={STORE}")

# --- 4. create ------------------------------------------------------------
mf = pathlib.Path("/content/Modelfile")
mf.write_text(f"FROM {SRC}\n")
t0 = time.time()
print(f"\nollama create {MODEL_TAG} ... (several minutes; it writes ~{NEED_GB:.0f} GB)")
r = subprocess.run(["ollama", "create", MODEL_TAG, "-f", str(mf)],
                   capture_output=True, text=True)
if r.returncode != 0:
    raise RuntimeError(f"ollama create failed:\n{(r.stderr or r.stdout)[-1500:]}")
print(f"    created in {time.time() - t0:.0f}s")

print(f"\nstore now at {STORE}  ({shutil.disk_usage(STORE).free / 1e9:.0f} GB still free)")
subprocess.run(["ollama", "list"])
print("\nNote: this disk is wiped when the runtime ends. Step 6 and Step 7 will work "
      "for the rest of this session.")
