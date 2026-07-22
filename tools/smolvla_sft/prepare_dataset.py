#!/usr/bin/env python
"""Build a SmolVLA-ready **2-cam, standard-keyed** copy of a 5-cam datagen LeRobot
v2.1 dataset -- by KEY-REMAPPING an already-valid dataset (no re-encode, no per-frame
rebuild, no lerobot version-internal API). Runs in the lerobot env.

    python tools/smolvla_sft/prepare_dataset.py \
        --src <5cam_lerobot_dir> --out <2cam_dir> --repo-id <id> [--external-cam left]

Why a physical copy (not lerobot-train's ``--rename_map``): ``lerobot-train``
classifies dataset features purely by standard key prefix (``observation.images.*`` /
``observation.state`` / ``action``) and its load-time ``--rename_map`` renames only
OBSERVATIONS, **not the action key** (verified in lerobot 0.5.1
``processor/rename_processor.py``: ``transform_features`` touches only
``PipelineFeatureType.OBSERVATION``). So the datagen ``actions`` column must be
physically renamed to ``action`` on disk; while at it we rename the 2 kept cameras +
state and drop the 3 unused overviews + ``actions_commanded``.

We do it as a pure **v2.1 -> v2.1 key remap** of the source (which stays READ-ONLY):
  - ``meta/info.json``           : ``features`` re-keyed (2 cams renamed, 3 overviews
                                   + ``actions_commanded`` dropped); ``total_videos`` fixed.
  - ``meta/episodes_stats.jsonl`` (+ ``stats.json`` if present): per-feature stats re-keyed.
  - ``data/**.parquet``          : columns  ``state -> observation.state``,
                                   ``actions -> action``; drop ``actions_commanded``.
                                   (state/action VALUES are unchanged, so their MEAN/STD
                                   stats are unchanged -- only the key names move. SmolVLA
                                   normalizes state/action with MEAN_STD and images with
                                   IDENTITY, so no stats need recomputing.)
  - ``videos/**/<cam>/``         : the 2 kept camera dirs copied to their new key names
                                   (H.264 mp4s copied as-is -- no transcode).

Idempotent at dataset granularity: a complete ``out`` is a no-op; a partial ``out``
must be removed and retried.

NOTE (verify on the SFT box): this reads the source's own ``data_path`` / ``video_path``
templates from ``info.json`` (data-driven, so robust to path layout), but the exact v2.1
meta field names should be confirmed against your datagen datasets + a
``LeRobotDataset(...)`` load before the first real run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pyarrow.parquet as pq

# Make ``maniguard_sft`` importable when this file is run as a script from the fork root.
_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from maniguard_sft import embodiment as emb  # noqa: E402


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text())


def _rekey_stats(stats: dict, ren: dict, drop: set) -> dict:
    """Rename top-level feature keys; drop unwanted features. Values untouched."""
    out = {}
    for k, v in stats.items():
        if k in drop:
            continue
        out[ren.get(k, k)] = v
    return out


def prepare(src: Path, out: Path, external_cam: str, repo_id: str) -> dict:
    info_p = src / "meta" / "info.json"
    if not info_p.is_file():
        raise FileNotFoundError(f"{info_p} not found -- not a LeRobot v2.1 dataset")
    info = _read_json(info_p)

    ren = emb.rename_map(external_cam)              # {src_key: tgt_key} for KEPT features
    drop = set(emb.dropped_streams(external_cam))   # 3 overviews + actions_commanded
    n_ep = int(info["total_episodes"])

    # idempotency
    out_info = out / "meta" / "info.json"
    if out_info.is_file():
        done = int(_read_json(out_info)["total_episodes"])
        if done == n_ep:
            print(f"[prepare] already complete ({done} episodes), skip: {out}")
            return {"repo_id": repo_id, "episodes": done, "root": str(out), "skipped": True}
        raise FileExistsError(f"{out} is partial ({done}/{n_ep}); rm it and retry.")

    print(f"[prepare] {src.name}: {n_ep} episodes, external_cam={external_cam}")
    print(f"[prepare]   rename {ren}")
    print(f"[prepare]   drop   {sorted(drop)}")

    chunks_size = int(info.get("chunks_size", 1000))
    data_tmpl = info["data_path"]     # data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet
    video_tmpl = info["video_path"]   # videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4

    kept_vid = {s: t for s, t in ren.items() if t.startswith("observation.images.")}  # cam renames
    col_ren = {s: t for s, t in ren.items() if not t.startswith("observation.images.")}  # state/actions

    (out / "meta").mkdir(parents=True, exist_ok=True)

    # 1) meta/info.json -- re-key features, fix video count, stamp repo_id.
    new_features = {ren.get(k, k): v for k, v in info["features"].items() if k not in drop}
    new_info = dict(info)
    new_info["features"] = new_features
    new_info["total_videos"] = n_ep * len(kept_vid)
    new_info["repo_id"] = repo_id
    out_info.write_text(json.dumps(new_info, indent=4) + "\n")

    # 2) tasks.jsonl / episodes.jsonl -- verbatim (no feature keys inside).
    for name in ("tasks.jsonl", "episodes.jsonl"):
        s = src / "meta" / name
        if s.is_file():
            shutil.copy2(s, out / "meta" / name)

    # 3) stats -- re-key per-episode (and aggregated, if present).
    es = src / "meta" / "episodes_stats.jsonl"
    if es.is_file():
        lines = []
        for line in es.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            r["stats"] = _rekey_stats(r["stats"], ren, drop)
            lines.append(json.dumps(r))
        (out / "meta" / "episodes_stats.jsonl").write_text("\n".join(lines) + "\n")
    agg = src / "meta" / "stats.json"
    if agg.is_file():
        (out / "meta" / "stats.json").write_text(
            json.dumps(_rekey_stats(_read_json(agg), ren, drop), indent=4) + "\n")

    # 4) per-episode: parquet column-rename + video passthrough-copy.
    for ep in range(n_ep):
        chunk = ep // chunks_size
        src_pq = src / data_tmpl.format(episode_chunk=chunk, episode_index=ep)
        dst_pq = out / data_tmpl.format(episode_chunk=chunk, episode_index=ep)
        dst_pq.parent.mkdir(parents=True, exist_ok=True)
        t = pq.read_table(src_pq)
        t = t.select([c for c in t.column_names if c not in drop])           # drop actions_commanded
        t = t.rename_columns([col_ren.get(c, c) for c in t.column_names])    # state/actions -> standard
        pq.write_table(t, dst_pq)
        for s_cam, t_key in kept_vid.items():
            src_v = src / video_tmpl.format(episode_chunk=chunk, video_key=s_cam, episode_index=ep)
            dst_v = out / video_tmpl.format(episode_chunk=chunk, video_key=t_key, episode_index=ep)
            dst_v.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_v, dst_v)
        if (ep + 1) % 100 == 0 or ep + 1 == n_ep:
            print(f"[prepare]   {ep + 1}/{n_ep} episodes")

    summary = {"repo_id": repo_id, "episodes": n_ep, "root": str(out),
               "external_cam": external_cam, "skipped": False}
    print(f"[prepare] DONE {summary}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="source 5-cam datagen LeRobot dir (untouched)")
    ap.add_argument("--out", required=True, help="output 2-cam SmolVLA LeRobot dir")
    ap.add_argument("--repo-id", required=True, help="repo_id stamped into the new dataset meta")
    ap.add_argument("--external-cam", default=emb.DEFAULT_EXTERNAL_CAM, choices=emb.EXTERNAL_CAM_CHOICES,
                    help=f"which overview -> observation.images.top (default {emb.DEFAULT_EXTERNAL_CAM})")
    args = ap.parse_args()
    prepare(Path(args.src), Path(args.out), args.external_cam, args.repo_id)


if __name__ == "__main__":
    main()
