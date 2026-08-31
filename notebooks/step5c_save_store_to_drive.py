# ===========================================================================
# Step 5c - save the built Ollama store to Drive, so future sessions skip
#           the merge AND the create. Run after Step 5b succeeds.
# ===========================================================================
import json, os, pathlib, shutil, time

PRUNE_ORPHANS  = True     # delete blobs no manifest references (failed-attempt leftovers)
DELETE_SHARDS  = False    # delete the split shards from Drive after the store verifies
DELETE_MERGED  = False    # delete the merged .gguf from Drive after the store verifies

STORE = pathlib.Path(globals().get("SERVE_DIR", "/mnt/local-scratch/ollama_models"))
DRIVE = pathlib.Path(globals().get("DRIVE_DIR", "/content/drive/MyDrive/scogs_ollama_models"))
TAG   = globals().get("MODEL", "medgemma-27b-bf16").split(":")[0]

if not (STORE / "manifests").exists():
    raise RuntimeError(f"No Ollama store at {STORE}. Run Step 5b first.")

# --- 1. what is actually in the store? ------------------------------------
def referenced_digests(store):
    """Every blob named by any manifest, so we never delete one still in use."""
    keep = set()
    for mf in (store / "manifests").rglob("*"):
        if not mf.is_file():
            continue
        try:
            data = json.loads(mf.read_text())
        except Exception:
            continue
        for layer in (data.get("layers") or []) + [data.get("config") or {}]:
            d = layer.get("digest", "")
            if d.startswith("sha256:"):
                keep.add("sha256-" + d.split(":", 1)[1])
    return keep

blobs = {q.name: q.stat().st_size for q in (STORE / "blobs").glob("*") if q.is_file()}
keep  = referenced_digests(STORE)
orphans = {n: s for n, s in blobs.items() if n not in keep}

print(f"store {STORE}")
print(f"  blobs        {len(blobs):3d}  {sum(blobs.values())/1e9:7.1f} GB")
print(f"  referenced   {len(keep & set(blobs)):3d}  "
      f"{sum(s for n, s in blobs.items() if n in keep)/1e9:7.1f} GB")
print(f"  orphaned     {len(orphans):3d}  {sum(orphans.values())/1e9:7.1f} GB"
      + ("   <- leftovers from the failed attempts" if orphans else ""))

if orphans and PRUNE_ORPHANS:
    for n in orphans:
        (STORE / "blobs" / n).unlink()
    print(f"  pruned {len(orphans)} orphan blob(s), "
          f"{sum(orphans.values())/1e9:.1f} GB reclaimed")

need = sum(q.stat().st_size for q in STORE.rglob("*") if q.is_file())
print(f"\nto copy: {need/1e9:.1f} GB")

# --- 2. room in Drive? -----------------------------------------------------
drive_used = sum(q.stat().st_size for q in DRIVE.rglob("*") if q.is_file())
try:
    drive_free = shutil.disk_usage(DRIVE).free
except OSError:
    drive_free = None
print(f"Drive holds {drive_used/1e9:.1f} GB"
      + (f", {drive_free/1e9:.0f} GB free" if drive_free is not None else ""))

shards = sorted(DRIVE.glob("**/*-of-*.gguf"))
mergedf = [q for q in DRIVE.glob("**/*.gguf") if "-of-" not in q.name]
if drive_free is not None and drive_free < need * 1.05:
    msg = [f"Drive has {drive_free/1e9:.0f} GB free but the store needs {need/1e9:.0f} GB."]
    if shards:
        msg.append(f"The {len(shards)} shards (~{sum(q.stat().st_size for q in shards)/1e9:.0f} GB) "
                   f"are redundant - the merged .gguf is built and verified. Set "
                   f"DELETE_SHARDS = True, or delete them by hand, then re-run.")
    raise RuntimeError(" ".join(msg))

# --- 3. copy, with progress ------------------------------------------------
def copy_with_progress(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    total, done, t0, last = src.stat().st_size, 0, time.time(), 0.0
    with open(src, "rb") as fi, open(dst, "wb") as fo:
        while chunk := fi.read(64 << 20):
            fo.write(chunk); done += len(chunk)
            if total > 2e9 and time.time() - last > 5:
                el = time.time() - t0
                print(f"      {done/1e9:5.1f} / {total/1e9:.1f} GB   "
                      f"{done/el/1e6:5.0f} MB/s   eta {(total-done)/(done/el)/60:4.1f} min",
                      flush=True)
                last = time.time()

print(f"\ncopying {STORE} -> {DRIVE}")
t0 = time.time()
for src in sorted(STORE.rglob("*")):
    if not src.is_file():
        continue
    dst = DRIVE / src.relative_to(STORE)
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        continue
    print(f"   {src.relative_to(STORE)}  ({src.stat().st_size/1e9:.1f} GB)")
    copy_with_progress(src, dst)
print(f"copied in {(time.time()-t0)/60:.1f} min")

# --- 4. verify before trusting it ------------------------------------------
ok = True
for src in STORE.rglob("*"):
    if not src.is_file():
        continue
    dst = DRIVE / src.relative_to(STORE)
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        print(f"!!  MISMATCH {src.relative_to(STORE)}")
        ok = False
manifest_ok = (DRIVE / "manifests" / "registry.ollama.ai" / "library" / TAG).exists()
print(f"\nverify: files {'ok' if ok else 'FAILED'}, manifest for {TAG} "
      f"{'ok' if manifest_ok else 'MISSING'}")
if not (ok and manifest_ok):
    raise RuntimeError("The Drive copy is incomplete. Do NOT delete anything; re-run.")

print(f"\nDone. Next session, Step 5 finds this store and just copies it back - "
      f"no merge, no create.")

# --- 5. optional cleanup, only now that the store is verified --------------
for flag, files, label in ((DELETE_SHARDS, shards, "shards"),
                           (DELETE_MERGED, mergedf, "merged .gguf")):
    if flag and files:
        freed = sum(q.stat().st_size for q in files)
        for q in files:
            q.unlink()
        print(f"deleted {label} from Drive, {freed/1e9:.0f} GB reclaimed")
    elif files:
        print(f"{label} still in Drive (~{sum(q.stat().st_size for q in files)/1e9:.0f} GB) "
              f"- redundant now; delete by hand or set the flag above.")
