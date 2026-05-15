#!/usr/bin/env python
"""Shard + upload SurroundOcc occupancy GT to HuggingFace.

One-shot uploader. Splits ~34k .npy voxel-GT files into N zstd-compressed tar
shards, then pushes them to a private HF dataset repo via hf_transfer.

Resumable: existing shards are skipped; upload_large_folder tracks per-file
upload state in ~/.cache/huggingface/.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

SRC = Path("/media/skr/storage/self_driving/S2GO/data/nuscenes_occ/nuscenes_occ/samples")
OUT = Path("/media/skr/storage/self_driving/S2GO/data/nuscenes_occ_shards")
REPO = "sangramrout/surroundOCC"
N_SHARDS = 4
ZSTD_LEVEL = 3

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def shard_files(files: list[str]) -> list[list[str]]:
    chunk = (len(files) + N_SHARDS - 1) // N_SHARDS
    return [files[i * chunk:(i + 1) * chunk] for i in range(N_SHARDS)]


def make_shard(idx: int, members: list[str]) -> Path:
    out = OUT / f"surroundocc-{idx:03d}.tar.zst"
    if out.exists():
        log(f"shard {idx}: already exists ({out.stat().st_size / 1e9:.2f} GB) — skipping")
        return out
    listfile = OUT / f"shard-{idx:03d}.list"
    listfile.write_text("\n".join(members) + "\n")
    log(f"shard {idx}: tarring + zstd-compressing {len(members)} files")
    t0 = time.time()
    with open(out, "wb") as fh:
        tar = subprocess.Popen(
            ["tar", "-cf", "-", "-C", str(SRC), "-T", str(listfile)],
            stdout=subprocess.PIPE,
        )
        zstd = subprocess.Popen(
            ["zstd", f"-{ZSTD_LEVEL}", "-T0", "-q"],
            stdin=tar.stdout,
            stdout=fh,
        )
        tar.stdout.close()
        zstd_rc = zstd.wait()
        tar_rc = tar.wait()
    listfile.unlink()
    if tar_rc != 0 or zstd_rc != 0:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"shard {idx}: tar={tar_rc} zstd={zstd_rc}")
    dt = time.time() - t0
    sz = out.stat().st_size / 1e9
    log(f"shard {idx}: done, {sz:.2f} GB in {dt:.1f}s ({sz * 1024 / dt:.0f} MB/s wall)")
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(os.listdir(SRC))
    log(f"source: {SRC} ({len(files)} files)")
    log(f"target: {REPO} (private dataset)")
    log(f"shards: {N_SHARDS} × zstd-{ZSTD_LEVEL}")

    chunks = shard_files(files)
    manifest = {
        "source": str(SRC),
        "repo": REPO,
        "n_shards": N_SHARDS,
        "zstd_level": ZSTD_LEVEL,
        "total_files": len(files),
        "shards": [],
    }
    for i, members in enumerate(chunks):
        path = make_shard(i, members)
        manifest["shards"].append({
            "name": path.name,
            "files": len(members),
            "bytes": path.stat().st_size,
        })

    manifest_path = OUT / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log(f"manifest: {manifest_path}")

    log("=" * 60)
    log("uploading to HuggingFace …")
    from huggingface_hub import HfApi
    api = HfApi()
    api.upload_large_folder(
        folder_path=str(OUT),
        repo_id=REPO,
        repo_type="dataset",
        ignore_patterns=["*.list", "*.log", "upload.log"],
    )
    log("upload complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
