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
  * coverage  -- spread over TIME (buckets across one event), not over
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
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".heic", ".heif", ".png", ".webp"}
MEDIA_SUFFIXES = VIDEO_SUFFIXES | IMAGE_SUFFIXES

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
    is_still: bool = False
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


def probe_media(path: Path) -> Clip | None:
    """Probe one file, video or still."""
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return probe_still(path)
    return probe_clip(path)


def probe_still(path: Path) -> Clip | None:
    """Read dimensions and capture time out of one photo.

    A still has no duration of its own; it is given the edit's slot length
    later, so ``duration`` is left at zero here and the duration filter
    skips stills entirely.
    """
    proc = _run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height",
            "-of", "json",
            str(path),
        ]
    )
    if proc.returncode != 0:
        return None
    try:
        streams = (json.loads(proc.stdout).get("streams") or [])
    except json.JSONDecodeError:
        return None
    if not streams:
        return None

    width = int(streams[0].get("width") or 0)
    height = int(streams[0].get("height") or 0)

    if path.suffix.lower() in {".heic", ".heif"}:
        # An iPhone HEIC is tiled HEVC and ffprobe reports the first tile
        # (typically 512x512), not the image. Spotlight has the real size.
        true_size = spotlight_pixel_size(path)
        if true_size:
            width, height = true_size

    if not width or not height:
        return None

    shot_at = exif_datetime(path) or datetime.fromtimestamp(
        path.stat().st_mtime, tz=timezone.utc
    )
    return Clip(
        path=path, duration=0.0, width=width, height=height,
        shot_at=shot_at, is_still=True,
    )


def exif_datetime(path: Path) -> datetime | None:
    """Capture time from Spotlight metadata (macOS).

    ffprobe does not surface EXIF dates for JPEG/HEIC, and mdls does, with
    no extra dependency. Off macOS this returns None and the caller falls
    back to mtime.
    """
    if not shutil.which("mdls"):
        return None
    proc = _run(
        ["mdls", "-raw", "-name", "kMDItemContentCreationDate", str(path)]
    )
    raw = (proc.stdout or "").strip()
    if not raw or raw == "(null)":
        return None
    try:
        return datetime.strptime(raw, "%Y-%m-%d %H:%M:%S %z")
    except ValueError:
        return None


def spotlight_pixel_size(path: Path) -> tuple[int, int] | None:
    """True pixel dimensions from Spotlight metadata (macOS)."""
    if not shutil.which("mdls"):
        return None

    # Queried one at a time on purpose: `mdls -raw` with several -name
    # flags prints the values sorted by attribute NAME, not in the order
    # the flags were given, so asking for Width then Height returns
    # height first and silently transposes the image.
    values: list[int] = []
    for attribute in ("kMDItemPixelWidth", "kMDItemPixelHeight"):
        raw = (_run(["mdls", "-raw", "-name", attribute, str(path)]).stdout
               or "").strip()
        if not raw.isdigit():
            return None
        values.append(int(raw))
    return values[0], values[1]


def ensure_renderable(path: Path, staging: Path) -> Path | None:
    """Return a path ffmpeg can apply a -vf filtergraph to.

    HEIC needs a decode pass of its own first. ffmpeg stitches its tiles
    using an internal complex filtergraph, and a simple ``-vf`` on the
    same stream is rejected outright ("Simple and complex filtering
    cannot be used together"). Decoding to JPEG first sidesteps that;
    every other format passes through untouched.
    """
    if path.suffix.lower() not in {".heic", ".heif"}:
        return path

    staging.mkdir(parents=True, exist_ok=True)
    destination = staging / f"{path.stem}.jpg"
    if destination.exists():
        return destination

    proc = _run([
        "ffmpeg", "-v", "error", "-nostdin", "-y",
        "-i", str(path), "-frames:v", "1", "-q:v", "2", str(destination),
    ])
    if proc.returncode != 0:
        print(f"      HEIC decode failed: {proc.stderr.strip()[:160]}", flush=True)
        return None
    return destination


def probe_clip(path: Path) -> Clip | None:
    """Read duration, dimensions and capture time out of one video."""
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
        if not clip.is_still and clip.duration < min_duration:
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

    Buckets the event into ``slots`` equal time windows and takes the
    longest clip from each. Longest is a crude stand-in for "most
    substantial" -- it is a placeholder, not a claim. Empty buckets are
    backfilled from whatever is left, longest first, so a lull in the
    footage does not shorten the edit.

    Stills have no duration, so within a bucket a video outranks a photo.
    That is deliberate for now: video is the harder framing case and the
    one the experiment is really asking about.
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


def cluster_events(clips: list[Clip], *, gap_hours: float) -> list[list[Clip]]:
    """Split clips into events wherever capture time jumps by more than a gap.

    A folder of gathered footage is rarely one night. Clustering on time
    gaps recovers the events without needing anything smarter: people film
    in bursts, and the quiet hours between bursts are unmistakable.
    """
    if not clips:
        return []
    ordered = sorted(clips, key=lambda c: c.shot_at)
    gap = gap_hours * 3600.0

    events: list[list[Clip]] = [[ordered[0]]]
    for previous, clip in zip(ordered, ordered[1:]):
        if clip.shot_at.timestamp() - previous.shot_at.timestamp() > gap:
            events.append([clip])
        else:
            events[-1].append(clip)
    return events


def timestamps_look_collapsed(clips: list[Clip]) -> bool:
    """True when every clip claims nearly the same capture time.

    Copying or AirDropping footage rewrites mtime, and many phone exports
    carry no ``creation_time`` tag at all. The result is a folder whose
    clips all landed within a few seconds of each other, which silently
    destroys both event clustering and ordering. Better to say so than to
    produce a confidently wrong running order.
    """
    if len(clips) < 3:
        return False
    stamps = [c.shot_at.timestamp() for c in clips]
    return (max(stamps) - min(stamps)) < 60.0


# --------------------------------------------------------------------------
# smart-cropping pipeline
# --------------------------------------------------------------------------


def still_to_video(
    clip: Clip, destination: Path, *, seconds: float, long_edge: int = 1920
) -> bool:
    """Hold a still as a short video so the real pipeline can consume it.

    The pipeline takes video, so a photo is turned into one rather than
    reimplementing what the pipeline does to a single frame. It is
    downscaled to roughly video resolution first: a 4032px phone photo is
    far larger than any frame the detector expects, and the extra pixels
    only cost time.
    """
    scale = (
        f"scale='if(gt(iw,ih),{long_edge},-2)':'if(gt(iw,ih),-2,{long_edge})'"
        f":force_original_aspect_ratio=decrease,"
        f"pad=ceil(iw/2)*2:ceil(ih/2)*2,fps={OUT_FPS}"
    )
    renderable = ensure_renderable(clip.path, destination.parent)
    if renderable is None:
        return False
    proc = _run([
        "ffmpeg", "-v", "error", "-nostdin", "-y",
        "-loop", "1", "-t", f"{seconds:.3f}", "-i", str(renderable),
        "-vf", scale,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-pix_fmt", "yuv420p", str(destination),
    ])
    if proc.returncode != 0:
        print(f"      still->video failed: {proc.stderr.strip()[:160]}", flush=True)
    return proc.returncode == 0


def run_pipeline(
    clip: Clip,
    *,
    repo: Path,
    env: str,
    genre: str,
    experiment: str,
    workspace: Path,
    verbose: bool,
    staging: Path,
    slot_seconds: float,
) -> Path | None:
    """Crop one clip via run_pipeline_local.py, return the cropped mp4."""
    source = clip.path
    if clip.is_still:
        # Keep the original stem so the output glob still finds it.
        staged = staging / f"{clip.path.stem}.mp4"
        if not still_to_video(clip, staged, seconds=slot_seconds):
            return None
        source = staged

    cmd = [
        "conda", "run", "-n", env, "python",
        "scripts/pipeline_runners/run_pipeline_local.py",
        str(source.resolve()),
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
        workspace=workspace, experiment=experiment, stem=source.stem
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

# Stills are cropped to the output aspect but left at full resolution, so
# the zoom afterwards has real pixels to push into rather than upscaling.
NAIVE_STILL_VF = f"crop='min(iw,ih*{OUT_W}/{OUT_H})':'min(ih,iw*{OUT_H}/{OUT_W})'"
PIPED_STILL_VF = "null"

PIPED_VF = (
    f"scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=decrease,"
    f"pad={OUT_W}:{OUT_H}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={OUT_FPS}"
)


def ken_burns(duration: float, *, zoom_to: float = 1.12) -> str:
    """A slow push across a held still.

    A photo sitting dead still for two and a half seconds inside a
    music-cut edit reads as a broken video. A gentle zoom reads as
    intentional. This is the standard fix and it is worth having in the
    prototype, because otherwise the stills look worse than they are and
    that would bias the comparison.
    """
    frames = still_frames(duration)
    step = (zoom_to - 1.0) / frames
    # zoompan's `d` is output frames PER INPUT FRAME, so this filter is only
    # correct on a single-frame input. Feeding it a looped stream multiplies
    # the output length by the number of looped frames -- which is exactly
    # the bug this replaced (63 input frames x 75 = 157s from a 2.5s slot).
    # render_segment therefore passes the image once and caps with -frames:v.
    #
    # Pre-scaled to a little above the output so the zoom has real pixels to
    # push into, and no higher: a full-resolution phone photo makes zoompan
    # crawl for no visible gain.
    return (
        f"scale={int(OUT_W * 1.4)}:-2,"
        f"zoompan=z='min(zoom+{step:.6f},{zoom_to})':d={frames}"
        f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":s={OUT_W}x{OUT_H}:fps={OUT_FPS},setsar=1"
    )


def still_frames(duration: float) -> int:
    """Output frame count for a held still."""
    return max(int(duration * OUT_FPS), 1)


def render_segment(
    source: Path,
    destination: Path,
    *,
    start: float,
    duration: float,
    video_filter: str,
    is_still: bool = False,
) -> bool:
    if is_still:
        # One input frame in, `still_frames` out -- see ken_burns().
        cmd = [
            "ffmpeg", "-v", "error", "-nostdin", "-y",
            "-i", str(source),
            "-vf", f"{video_filter},{ken_burns(duration)}",
            "-frames:v", str(still_frames(duration)),
        ]
    else:
        cmd = [
            "ffmpeg", "-v", "error", "-nostdin", "-y",
            "-ss", f"{start:.3f}",
            "-i", str(source),
            "-t", f"{duration:.3f}",
            "-vf", video_filter,
        ]
    proc = _run(
        cmd + [
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

        # A still that has been through the pipeline comes back as a video,
        # so it is rendered like one; only an un-piped still is held.
        held_still = clip.is_still and label == "naive"
        if held_still:
            renderable = ensure_renderable(source, out_dir / "_normalized")
            if renderable is None:
                continue
            ok = render_segment(
                renderable, segment, start=0.0, duration=slot_seconds,
                video_filter=NAIVE_STILL_VF, is_still=True,
            )
        else:
            # A cropped clip can be marginally shorter than its source, so
            # clamp the window rather than trusting the source duration.
            start = clip.window_start if label == "naive" else 0.0
            if label != "naive":
                cropped_duration = _first_float(
                    _run(["ffprobe", "-v", "error", "-show_entries",
                          "format=duration", "-of", "csv=p=0",
                          str(source)]).stdout.strip()
                ) or slot_seconds
                start = max((cropped_duration - slot_seconds) / 2.0, 0.0)
            ok = render_segment(
                source, segment, start=start, duration=slot_seconds,
                video_filter=video_filter,
            )
        if ok:
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
    parser.add_argument("--gap-hours", type=float, default=6.0,
                        help="A time gap this large starts a new event (default: 6).")
    parser.add_argument("--event", type=int, default=None,
                        help="Build from event N (1-indexed). Default: the biggest.")
    parser.add_argument("--pool", action="store_true",
                        help="Ignore events; treat the whole folder as one pool. "
                             "Right for a pure framing test, wrong for a real recap.")
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
        if p.is_file() and p.suffix.lower() in MEDIA_SUFFIXES
    )
    if not sources:
        print(f"error: no photos or videos in {args.footage}", file=sys.stderr)
        return 2

    stills_found = sum(1 for p in sources if p.suffix.lower() in IMAGE_SUFFIXES)
    print(f"probing {len(sources)} files "
          f"({len(sources) - stills_found} video, {stills_found} photo) ...",
          flush=True)
    clips = [c for c in (probe_media(p) for p in sources) if c is not None]
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

    if timestamps_look_collapsed(kept):
        print()
        print("  warning: every clip reports nearly the same capture time.")
        print("  Copying or AirDropping footage rewrites mtime, and many phone")
        print("  exports carry no creation_time tag. Event grouping and running")
        print("  order are meaningless here -- clips will be used in name order.")
        print("  To fix: copy with `cp -p`, or re-export preserving metadata.")
        events = [kept]
        pooled = True
    else:
        events = cluster_events(kept, gap_hours=args.gap_hours)
        pooled = args.pool

    if pooled or len(events) == 1:
        pool = kept if pooled else events[0]
        if len(events) > 1:
            print(f"\n{len(events)} events found; pooling all "
                  f"{len(pool)} clips (--pool).")
    else:
        print(f"\n{len(events)} events found (gap > {args.gap_hours:g}h):")
        for index, event in enumerate(events, start=1):
            first = event[0].shot_at.astimezone()
            last = event[-1].shot_at.astimezone()
            span = (last - first).total_seconds() / 3600.0
            print(f"  {index:>3}  {first:%a %d %b %H:%M}  "
                  f"{len(event):>3} clips  {span:>4.1f}h")

        if args.event is not None:
            if not 1 <= args.event <= len(events):
                print(f"error: --event {args.event} out of range "
                      f"(1-{len(events)})", file=sys.stderr)
                return 2
            pool = events[args.event - 1]
            print(f"\nusing event {args.event} ({len(pool)} clips)")
        else:
            pool = max(events, key=len)
            chosen_index = events.index(pool) + 1
            print(f"\nusing event {chosen_index}, the biggest "
                  f"({len(pool)} clips). --event N picks another, "
                  f"--pool uses everything.")

    # The folder-wide check above catches a wholly-flattened folder. This
    # catches the commoner case: one event's clips share a timestamp because
    # that batch was copied, while other events kept theirs.
    if not timestamps_look_collapsed(kept) and timestamps_look_collapsed(pool):
        print()
        print("  warning: the chosen clips all report the same capture time,")
        print("  so their running order is arbitrary (falling back to name")
        print("  order). Copying rewrites mtime -- use `cp -p` to preserve it.")

    selected = select_for_edit(pool, slots=slots)
    set_windows(selected, slot_seconds=slot_seconds)

    print()
    n_stills = sum(1 for c in selected if c.is_still)
    print(f"plan: {len(selected)} items ({len(selected) - n_stills} video, "
          f"{n_stills} photo) x {slot_seconds:.2f}s "
          f"= {len(selected) * slot_seconds:.1f}s "
          f"(from {len(kept)} usable of {len(sources)} found)")
    for clip in selected:
        orientation = "portrait" if clip.is_portrait else "landscape"
        kind = "photo" if clip.is_still else f"{clip.duration:.1f}s"
        print(f"  {clip.slot_index:>3}  {clip.shot_at.astimezone():%H:%M:%S}  "
              f"{kind:>6}  {orientation:>9}  {clip.path.name}")

    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "footage": str(args.footage.resolve()),
        "found": len(sources),
        "usable": len(kept),
        "events_found": len(events),
        "event_pool_size": len(pool),
        "pooled": pooled,
        "slot_seconds": slot_seconds,
        "slots": len(selected),
        "selected": [
            {
                "slot": c.slot_index,
                "file": c.path.name,
                "shot_at": c.shot_at.isoformat(),
                "kind": "photo" if c.is_still else "video",
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

        staging = args.out / "_staged_stills"
        staging.mkdir(parents=True, exist_ok=True)

        cropped: dict[Path, Path] = {}
        for clip in selected:
            print(f"  [{clip.slot_index + 1}/{len(selected)}] {clip.path.name}",
                  flush=True)
            result = run_pipeline(
                clip,
                repo=args.repo, env=args.env, genre=args.genre,
                experiment=experiment, workspace=args.workspace,
                verbose=args.verbose, staging=staging,
                slot_seconds=slot_seconds,
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
