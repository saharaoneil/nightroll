#!/usr/bin/env python3
"""Nightroll prototype: turn a folder of phone clips into one vertical edit.

The experiment this exists to run
--------------------------------
The whole Nightroll concept rests on one assumption: that an automatically
framed edit of amateur night-out footage is good enough that someone would
post it unprompted. Everything else in the concept is downstream of that.

So this tool builds the SAME edit twice:

  * ``naive``  -- centre-crop every clip to 9:16. The control.
  * ``piped``  -- run each clip through the smart-cropping pipeline, which
                  finds the subject and follows them. The treatment.

Identical clip selection, identical ordering, identical pacing, identical
music. The only variable is the framing. Watch them back to back; if the
difference is not obvious, the concept does not work and you have saved
yourself building an app.

What is deliberately faked in v1
--------------------------------
Selection and pacing are stubbed so the framing question is isolated:

  * pacing    -- a fixed slot length, optionally derived from a BPM.
  * window    -- the middle N seconds of each clip.
  * coverage  -- spread over TIME (buckets across the night), not over
                 PEOPLE. Per-person coverage needs cross-clip face re-id,
                 which is real work and not what this test is about.
  * quality   -- duration and brightness only. No shake detection.

These are the right things to fake: they are all refinements if the
framing lands, and all irrelevant if it does not.

Usage
-----
    # Fast sanity check, no pipeline, no GPU:
    python nightroll.py ~/footage/saturday --dry-run

    # The real thing:
    python nightroll.py ~/footage/saturday --out ./out --music track.mp3
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}

OUT_W = 1080
OUT_H = 1920
OUT_FPS = 30


# --------------------------------------------------------------------------
# probing
# --------------------------------------------------------------------------


@dataclass
class Clip:
    """One source clip plus everything we measured about it."""

    path: Path
    duration: float
    width: int
    height: int
    shot_at: datetime
    luma: float | None = None
    # filled in later
    rejected_for: str | None = None
    cropped_path: Path | None = None
    slot_index: int | None = None
    window_start: float = 0.0

    @property
    def is_portrait(self) -> bool:
        return self.height >= self.width


def _run(cmd: list[str], *, capture: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        capture_output=capture,
        text=True,
        check=False,
    )


def probe_clip(path: Path) -> Clip | None:
    """Read duration, dimensions and capture time out of one file."""
    proc = _run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,duration:format=duration:format_tags=creation_time",
            "-of", "json",
            str(path),
        ]
    )
    if proc.returncode != 0:
        return None
    try:
        meta = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None

    streams = meta.get("streams") or []
    if not streams:
        return None
    stream = streams[0]

    duration = _first_float(
        stream.get("duration"),
        (meta.get("format") or {}).get("duration"),
    )
    if duration is None or duration <= 0:
        return None

    width = int(stream.get("width") or 0)
    height = int(stream.get("height") or 0)
    if not width or not height:
        return None

    shot_at = _creation_time(meta) or datetime.fromtimestamp(
        path.stat().st_mtime, tz=timezone.utc
    )

    return Clip(
        path=path,
        duration=duration,
        width=width,
        height=height,
        shot_at=shot_at,
    )


def _first_float(*values: object) -> float | None:
    for value in values:
        if value in (None, "", "N/A"):
            continue
        try:
            return float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
    return None


def _creation_time(meta: dict) -> datetime | None:
    raw = ((meta.get("format") or {}).get("tags") or {}).get("creation_time")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def measure_luma(path: Path, *, samples_per_second: float = 1.0) -> float | None:
    """Average frame brightness, 0-255. Used only to drop pitch-black clips."""
    proc = _run(
        [
            "ffmpeg", "-v", "info", "-nostdin", "-i", str(path),
            "-vf", f"fps={samples_per_second},signalstats,"
                   "metadata=print:key=lavfi.signalstats.YAVG:file=-",
            "-f", "null", "-",
        ]
    )
    values = [float(m) for m in re.findall(r"YAVG=([0-9.]+)", proc.stdout or "")]
    if not values:
        values = [float(m) for m in re.findall(r"YAVG=([0-9.]+)", proc.stderr or "")]
    if not values:
        return None
    return sum(values) / len(values)


# --------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------


def prefilter(
    clips: list[Clip],
    *,
    min_duration: float,
    min_luma: float,
    measure_brightness: bool,
) -> list[Clip]:
    """Cheap quality gate. Marks rejects in place, returns the survivors."""
    kept: list[Clip] = []
    for clip in clips:
        if clip.duration < min_duration:
            clip.rejected_for = f"too short ({clip.duration:.1f}s < {min_duration}s)"
            continue
        if measure_brightness:
            clip.luma = measure_luma(clip.path)
            if clip.luma is not None and clip.luma < min_luma:
                clip.rejected_for = f"too dark (luma {clip.luma:.0f} < {min_luma:.0f})"
                continue
        kept.append(clip)
    return kept


def select_for_edit(clips: list[Clip], *, slots: int) -> list[Clip]:
    """Pick ``slots`` clips spread evenly across the night.

    Buckets the night into ``slots`` equal time windows and takes the
    longest clip from each. Longest is a crude stand-in for "most
    substantial" -- it is a placeholder, not a claim. Empty buckets are
    backfilled from whatever is left, longest first, so a lull in the
    footage does not shorten the edit.
    """
    if not clips:
        return []
    ordered = sorted(clips, key=lambda c: c.shot_at)
    if len(ordered) <= slots:
        for index, clip in enumerate(ordered):
            clip.slot_index = index
        return ordered

    start = ordered[0].shot_at.timestamp()
    end = ordered[-1].shot_at.timestamp()
    span = max(end - start, 1e-6)

    buckets: dict[int, list[Clip]] = {}
    for clip in ordered:
        position = (clip.shot_at.timestamp() - start) / span
        bucket = min(int(position * slots), slots - 1)
        buckets.setdefault(bucket, []).append(clip)

    chosen: list[Clip] = []
    leftovers: list[Clip] = []
    for bucket in range(slots):
        candidates = sorted(
            buckets.get(bucket, []), key=lambda c: c.duration, reverse=True
        )
        if candidates:
            chosen.append(candidates[0])
            leftovers.extend(candidates[1:])

    backfill = sorted(leftovers, key=lambda c: c.duration, reverse=True)
    while len(chosen) < slots and backfill:
        chosen.append(backfill.pop(0))

    chosen.sort(key=lambda c: c.shot_at)
    for index, clip in enumerate(chosen):
        clip.slot_index = index
    return chosen


def set_windows(clips: list[Clip], *, slot_seconds: float) -> None:
    """Take the middle ``slot_seconds`` of each clip.

    The middle is a guess: people tend to start recording before anything
    happens and stop after it stops. It is not a content decision, and it
    is the first thing worth replacing once framing is proven.
    """
    for clip in clips:
        usable = max(clip.duration - slot_seconds, 0.0)
        clip.window_start = usable / 2.0


# --------------------------------------------------------------------------
# smart-cropping pipeline
# --------------------------------------------------------------------------


def run_pipeline(
    clip: Clip,
    *,
    repo: Path,
    env: str,
    genre: str,
    experiment: str,
    workspace: Path,
    verbose: bool,
) -> Path | None:
    """Crop one clip via run_pipeline_local.py, return the cropped mp4."""
    cmd = [
        "conda", "run", "-n", env, "python",
        "scripts/pipeline_runners/run_pipeline_local.py",
        str(clip.path.resolve()),
        "--genre", genre,
        "--experiment", experiment,
        "--no-annotated",
        "--no-json",
        "--no-state",
        "--no-progress",
    ]
    if verbose:
        print(f"      $ {' '.join(cmd)}", flush=True)

    proc = subprocess.run(cmd, cwd=repo, capture_output=not verbose, text=True)
    if proc.returncode != 0:
        tail = ""
        if not verbose and proc.stderr:
            tail = proc.stderr.strip().splitlines()[-1:] or [""]
            tail = f" -- {tail[0][:160]}"
        print(f"      pipeline failed (exit {proc.returncode}){tail}", flush=True)
        return None

    return newest_cropped_output(
        workspace=workspace, experiment=experiment, stem=clip.path.stem
    )


def newest_cropped_output(
    *, workspace: Path, experiment: str, stem: str
) -> Path | None:
    """Find the cropped mp4 the pipeline just wrote.

    The runner owns its own output layout
    (``{workspace}/v{version}/{experiment}/{stem}/...``) and exposes no
    flag to override it, so we glob for the artifact rather than
    reconstruct the path and risk drifting from it.
    """
    matches = list(workspace.glob(f"v*/{experiment}/**/{stem}*.cropped.mp4"))
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

NAIVE_VF = (
    f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,"
    f"crop={OUT_W}:{OUT_H},setsar=1,fps={OUT_FPS}"
)

PIPED_VF = (
    f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
    f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={OUT_FPS}"
)


def render_segment(
    source: Path,
    destination: Path,
    *,
    start: float,
    duration: float,
    video_filter: str,
) -> bool:
    proc = _run(
        [
            "ffmpeg", "-v", "error", "-nostdin", "-y",
            "-ss", f"{start:.3f}",
            "-i", str(source),
            "-t", f"{duration:.3f}",
            "-vf", video_filter,
            "-an",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p",
            str(destination),
        ]
    )
    if proc.returncode != 0:
        print(f"      segment failed: {proc.stderr.strip()[:160]}", flush=True)
    return proc.returncode == 0


def concat_segments(
    segments: list[Path], destination: Path, *, music: Path | None
) -> bool:
    if not segments:
        return False
    with tempfile.NamedTemporaryFile(
        "w", suffix=".txt", delete=False, encoding="utf-8"
    ) as handle:
        for segment in segments:
            handle.write(f"file '{segment.resolve()}'\n")
        listing = Path(handle.name)

    cmd = [
        "ffmpeg", "-v", "error", "-nostdin", "-y",
        "-f", "concat", "-safe", "0", "-i", str(listing),
    ]
    if music:
        cmd += ["-i", str(music), "-map", "0:v", "-map", "1:a", "-shortest",
                "-c:a", "aac", "-b:a", "192k"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-pix_fmt", "yuv420p", str(destination)]

    proc = _run(cmd)
    listing.unlink(missing_ok=True)
    if proc.returncode != 0:
        print(f"  concat failed: {proc.stderr.strip()[:300]}", flush=True)
    return proc.returncode == 0


def build_edit(
    clips: list[Clip],
    *,
    label: str,
    source_of: dict[Path, Path],
    video_filter: str,
    slot_seconds: float,
    out_dir: Path,
    music: Path | None,
) -> Path | None:
    """Render one full edit from the already-selected clips."""
    work = out_dir / f"_segments_{label}"
    work.mkdir(parents=True, exist_ok=True)

    segments: list[Path] = []
    for clip in clips:
        source = source_of.get(clip.path)
        if source is None:
            continue
        segment = work / f"{clip.slot_index:03d}_{clip.path.stem}.mp4"
        # A cropped clip can be marginally shorter than its source, so
        # clamp the window rather than trusting the source duration.
        start = clip.window_start if label == "naive" else 0.0
        if label != "naive":
            cropped_duration = _first_float(
                _run(["ffprobe", "-v", "error", "-show_entries",
                      "format=duration", "-of", "csv=p=0", str(source)]).stdout.strip()
            ) or slot_seconds
            start = max((cropped_duration - slot_seconds) / 2.0, 0.0)
        if render_segment(
            source, segment,
            start=start, duration=slot_seconds, video_filter=video_filter,
        ):
            segments.append(segment)

    if not segments:
        print(f"  {label}: no segments rendered", flush=True)
        return None

    destination = out_dir / f"nightroll_{label}.mp4"
    if not concat_segments(segments, destination, music=music):
        return None
    return destination


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a vertical night-out edit, twice: naive vs pipeline-framed.",
    )
    parser.add_argument("footage", type=Path, help="Folder of source clips.")
    parser.add_argument("--out", type=Path, default=Path("./out"),
                        help="Output folder (default: ./out).")
    parser.add_argument("--seconds", type=float, default=60.0,
                        help="Target edit length (default: 60).")
    parser.add_argument("--slot", type=float, default=2.5,
                        help="Seconds per clip (default: 2.5). Ignored if --bpm given.")
    parser.add_argument("--bpm", type=float, default=None,
                        help="Derive the slot from a tempo: 4 beats per clip.")
    parser.add_argument("--music", type=Path, default=None,
                        help="Audio track laid over both edits.")
    parser.add_argument("--min-duration", type=float, default=1.5,
                        help="Drop clips shorter than this (default: 1.5s).")
    parser.add_argument("--min-luma", type=float, default=18.0,
                        help="Drop clips darker than this 0-255 average (default: 18).")
    parser.add_argument("--no-brightness", action="store_true",
                        help="Skip the brightness pass (faster probing).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Probe and plan only. No pipeline, no rendering.")
    parser.add_argument("--only", choices=["naive", "piped"], default=None,
                        help="Build just one of the two edits.")
    parser.add_argument("--repo", type=Path,
                        default=Path.home() / "smart-cropping",
                        help="smart-cropping checkout.")
    parser.add_argument("--env", default="smartcropping-env",
                        help="Conda env for the pipeline.")
    parser.add_argument("--genre", default="entertainment",
                        help="Pipeline --genre (default: entertainment).")
    parser.add_argument("--workspace", type=Path,
                        default=Path.home() / "workspace" / "smart-cropping-workspace",
                        help="Pipeline workspace root, where cropped output lands.")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Stream pipeline output instead of capturing it.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if not shutil.which(tool):
            print(f"error: {tool} not found on PATH", file=sys.stderr)
            return 2

    if not args.footage.is_dir():
        print(f"error: {args.footage} is not a folder", file=sys.stderr)
        return 2

    slot_seconds = (4.0 * 60.0 / args.bpm) if args.bpm else args.slot
    slots = max(int(args.seconds // slot_seconds), 1)

    # ---- probe -----------------------------------------------------------
    sources = sorted(
        p for p in args.footage.iterdir()
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    )
    if not sources:
        print(f"error: no video files in {args.footage}", file=sys.stderr)
        return 2

    print(f"probing {len(sources)} clips ...", flush=True)
    clips = [c for c in (probe_clip(p) for p in sources) if c is not None]
    unreadable = len(sources) - len(clips)
    if unreadable:
        print(f"  {unreadable} unreadable, skipped")

    kept = prefilter(
        clips,
        min_duration=args.min_duration,
        min_luma=args.min_luma,
        measure_brightness=not args.no_brightness,
    )
    for clip in clips:
        if clip.rejected_for:
            print(f"  drop  {clip.path.name}: {clip.rejected_for}")

    if not kept:
        print("error: every clip was filtered out", file=sys.stderr)
        return 1

    selected = select_for_edit(kept, slots=slots)
    set_windows(selected, slot_seconds=slot_seconds)

    print()
    print(f"plan: {len(selected)} clips x {slot_seconds:.2f}s "
          f"= {len(selected) * slot_seconds:.1f}s "
          f"(from {len(kept)} usable of {len(sources)} found)")
    for clip in selected:
        orientation = "portrait" if clip.is_portrait else "landscape"
        print(f"  {clip.slot_index:>3}  {clip.shot_at.astimezone():%H:%M:%S}  "
              f"{clip.duration:>5.1f}s  {orientation:>9}  {clip.path.name}")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "footage": str(args.footage.resolve()),
        "found": len(sources),
        "usable": len(kept),
        "slot_seconds": slot_seconds,
        "slots": len(selected),
        "selected": [
            {
                "slot": c.slot_index,
                "file": c.path.name,
                "shot_at": c.shot_at.isoformat(),
                "duration": round(c.duration, 3),
                "window_start": round(c.window_start, 3),
                "orientation": "portrait" if c.is_portrait else "landscape",
                "luma": round(c.luma, 1) if c.luma is not None else None,
            }
            for c in selected
        ],
        "rejected": [
            {"file": c.path.name, "reason": c.rejected_for}
            for c in clips if c.rejected_for
        ],
    }

    if args.dry_run:
        (args.out / "plan.json").write_text(json.dumps(manifest, indent=2))
        print(f"\ndry run -- plan written to {args.out / 'plan.json'}")
        return 0

    # ---- naive edit (control) -------------------------------------------
    outputs: dict[str, str] = {}

    if args.only in (None, "naive"):
        print("\nbuilding naive edit (centre-crop control) ...", flush=True)
        naive = build_edit(
            selected,
            label="naive",
            source_of={c.path: c.path for c in selected},
            video_filter=NAIVE_VF,
            slot_seconds=slot_seconds,
            out_dir=args.out,
            music=args.music,
        )
        if naive:
            outputs["naive"] = str(naive)
            print(f"  -> {naive}")

    # ---- piped edit (treatment) -----------------------------------------
    if args.only in (None, "piped"):
        experiment = f"nightroll_{datetime.now():%Y%m%dT%H%M%S}"
        print(f"\ncropping {len(selected)} clips through the pipeline "
              f"(experiment {experiment}) ...", flush=True)

        cropped: dict[Path, Path] = {}
        for clip in selected:
            print(f"  [{clip.slot_index + 1}/{len(selected)}] {clip.path.name}",
                  flush=True)
            result = run_pipeline(
                clip,
                repo=args.repo, env=args.env, genre=args.genre,
                experiment=experiment, workspace=args.workspace,
                verbose=args.verbose,
            )
            if result:
                cropped[clip.path] = result
                clip.cropped_path = result

        if not cropped:
            print("  no clips survived the pipeline -- skipping piped edit")
        else:
            missing = len(selected) - len(cropped)
            if missing:
                print(f"  {missing} clip(s) failed; building from {len(cropped)}")
            print("\nbuilding piped edit (pipeline-framed) ...", flush=True)
            piped = build_edit(
                [c for c in selected if c.path in cropped],
                label="piped",
                source_of=cropped,
                video_filter=PIPED_VF,
                slot_seconds=slot_seconds,
                out_dir=args.out,
                music=args.music,
            )
            if piped:
                outputs["piped"] = str(piped)
                print(f"  -> {piped}")

        manifest["experiment"] = experiment
        manifest["cropped"] = {
            c.path.name: str(c.cropped_path)
            for c in selected if c.cropped_path
        }

    manifest["outputs"] = outputs
    (args.out / "plan.json").write_text(json.dumps(manifest, indent=2))

    print(f"\nmanifest: {args.out / 'plan.json'}")
    if len(outputs) == 2:
        print("\nWatch them back to back. If the difference is not obvious,")
        print("the concept does not work -- and that is the useful answer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
