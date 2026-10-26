"""Host-side DexJoCo demonstrations from the official human teleoperation dataset.

scripts/setup_dexjoco_demos.py installs a few official raw demonstrations per task (the
rand_obj set for ``<task>``, the rand_full set for ``<task>_random``) under
runtime/reference-demos/dexjoco/<task>/, with a sha256 manifest, base-camera keyframes and a
numpy export of each trajectory. It replays every candidate in the official environment and
keeps as experts the first three that succeed and whose layout is none of the official
evaluation layouts (formal), so no formal layout is ever shown.

``demonstration_context`` copies keyframe images of the experts into runtime/demonstrations;
with ``expert_demonstrations`` the expert episodes themselves (official files unchanged plus the
export) are shown read-only at runtime/experts (exploration and agent formal
copies; never in isolated rehearsal or formal runs).
"""
from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from services.demonstration_io import (
    DEMONSTRATION_CONTEXTS, DemonstrationError, _replace_directory, _validate_jpeg,
)

from .tasks import task_info

PROJECT = Path(__file__).resolve().parents[2]
SCHEMA = "dexjoco_demos_v1"
EXPERTS = 3


def demos_root():
    return PROJECT / "runtime" / "reference-demos" / "dexjoco"


def _load_manifest(directory, task):
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
    except (OSError, ValueError) as error:
        raise DemonstrationError(f"DexJoCo demonstrations for {task} are not installed; "
                                 f"run scripts/setup_dexjoco_demos.py --gpu <gpu> --task {task}") from error
    if manifest.get("schema") != SCHEMA or manifest.get("task") != task:
        raise DemonstrationError(f"Invalid DexJoCo demonstration manifest for {task}")
    if len(manifest.get("experts", ())) != EXPERTS:
        raise DemonstrationError(f"DexJoCo expert set for {task} must hold {EXPERTS} verified episodes; "
                                 f"rerun scripts/setup_dexjoco_demos.py --gpu <gpu> --task {task} with more candidates")
    return manifest


def cached_experts(cache_root, task):
    """The installed expert episode directories of one task, in order (light check; the setup
    script verified every file hash, the replay and the formal-layout exclusion)."""
    task_info(task)
    directory = (Path(cache_root).expanduser() / task).resolve()
    manifest = _load_manifest(directory, task)
    experts_dir = directory / "experts"
    demos = []
    for name in manifest["experts"]:
        path = (experts_dir / name).resolve()
        if not path.is_relative_to(experts_dir) or not (path / "trajectory.npz").is_file() \
                or not (path / "replay.zarr").is_dir():
            raise DemonstrationError(f"Incomplete DexJoCo expert episode {name} for {task}")
        demos.append(path)
    return demos


HUMAN_SCHEMA = "dexjoco_human_demos_v1"
# What each video-only view keeps of an expert episode directory.
ROBOT_VIDEO_PARTS = ("videos", "frames", "episode.json")
HUMAN_VIDEO_PARTS = ("video.mp4", "frames", "episode.json")
HUMAN_MOTION_FILES = ("hand_trajectory.npz", "phases.json")


def human_demos_root():
    return PROJECT / "runtime" / "reference-demos" / "dexjoco-human"


def cached_human_demos(cache_root, task, *, motion):
    """Installed human demonstrations of one task (scripts/build_human_demos.py), in order."""
    task_info(task)
    directory = (Path(cache_root).expanduser() / task).resolve()
    try:
        manifest = json.loads((directory / "manifest.json").read_text())
    except (OSError, ValueError) as error:
        raise DemonstrationError(f"Human demonstrations for {task} are not installed; "
                                 f"run scripts/build_human_demos.py for {task}") from error
    if manifest.get("schema") != HUMAN_SCHEMA or manifest.get("task") != task:
        raise DemonstrationError(f"Invalid human demonstration manifest for {task}")
    if len(manifest.get("experts", ())) != EXPERTS:
        raise DemonstrationError(f"Human demonstration set for {task} must hold {EXPERTS} clips")
    experts_dir = directory / "experts"
    demos = []
    for name in manifest["experts"]:
        path = (experts_dir / name).resolve()
        required = HUMAN_VIDEO_PARTS + (HUMAN_MOTION_FILES if motion else ())
        if not path.is_relative_to(experts_dir) or not all((path / part).exists() for part in required):
            raise DemonstrationError(f"Incomplete human demonstration {name} for {task}")
        demos.append(path)
    return demos


def _video_view(demos, view, parts):
    """A copy of the expert directories holding only `parts` (rebuilt when absent)."""
    view = Path(view)
    if view.is_dir() and all((view / demo.name).is_dir() for demo in demos):
        return view.resolve()
    staging = view.parent / f".{view.name}.{uuid.uuid4().hex}.tmp"
    staging.mkdir(mode=0o755)
    try:
        for demo in demos:
            (staging / demo.name).mkdir(mode=0o755)
            for part in parts:
                source = demo / part
                if source.is_dir():
                    shutil.copytree(source, staging / demo.name / part)
                elif source.is_file():
                    shutil.copyfile(source, staging / demo.name / part)
        _replace_directory(staging, view)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return view.resolve()


def expert_root(task, content="full"):
    """Host directory the agent reads at runtime/experts, for the session's expert content."""
    if content == "full":
        cached_experts(demos_root(), task)
        return (demos_root() / task / "experts").resolve()
    if content == "video":
        demos = cached_experts(demos_root(), task)
        return _video_view(demos, demos_root() / task / "experts-video", ROBOT_VIDEO_PARTS)
    if content in ("human-video", "human-full"):
        demos = cached_human_demos(human_demos_root(), task, motion=content == "human-full")
        if content == "human-full":
            return (human_demos_root() / task / "experts").resolve()
        return _video_view(demos, human_demos_root() / task / "experts-video", HUMAN_VIDEO_PARTS)
    raise DemonstrationError(f"DexJoCo has no {content!r} expert content")


def keyframes(demo):
    """The episode's base-camera keyframes, in time order (the last is the final frame)."""
    index = json.loads((demo / "frames" / "index.json").read_text())
    images = []
    for frame in sorted(index["frames"], key=lambda item: item["step"]):
        path = (demo / frame["file"]).resolve()
        if not path.is_relative_to(demo):
            raise DemonstrationError("DexJoCo keyframe escaped its episode directory")
        _validate_jpeg(path)
        images.append(path)
    if not images:
        raise DemonstrationError(f"No keyframes in {demo.name}")
    return images, index.get("camera")


def provision_trial_demonstration(*, cache_root, target, task, context_kind):
    """Expose the selected keyframes in the trial runtime."""
    if context_kind not in DEMONSTRATION_CONTEXTS:
        raise DemonstrationError("demonstration_context must be one of: " + ", ".join(DEMONSTRATION_CONTEXTS))
    demos = [keyframes(demo) for demo in cached_experts(cache_root, task)] if context_kind != "none" else []
    target = Path(target).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    target = target.parent.resolve() / target.name
    # Stage in the trusted trial root, outside the agent-writable runtime.
    staging = target.parent.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    staging.mkdir(mode=0o700)
    try:
        if context_kind == "none":
            _replace_directory(staging, target)
            return {"kind": "none", "image_count": 0, "root": "runtime/demonstrations", "manifest_path": None}
        sequences, camera = [], None
        for number, (images, camera) in enumerate(demos, 1):
            chosen = images if context_kind == "completion_sequence" else images[-1:]
            (staging / f"demo-{number}").mkdir(mode=0o700)
            names = []
            for order, source in enumerate(chosen):
                relative = (f"demo-{number}/frame_{order:03d}.jpg" if context_kind == "completion_sequence"
                            else f"demo-{number}/terminal.jpg")
                shutil.copyfile(source, staging / relative)
                (staging / relative).chmod(0o600)
                names.append(relative)
            sequences.append(names)
        ordered_images = [name for names in sequences for name in names]
        manifest = {
            "schema_version": 1, "task": task, "context_kind": context_kind, "demonstrator": "human_teleoperation",
            "source": "dexjoco_official_teleoperation_demos", "camera": camera,
            "image_count": len(ordered_images), "ordered_images": ordered_images,
            # Independent demonstrations, each in a different layout; order applies within one.
            "demonstrations": [{"ordered_images": names} for names in sequences],
            "contains": ["rgb_images"],
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (staging / "manifest.json").chmod(0o600)
        _replace_directory(staging, target)
        return {"kind": context_kind, "image_count": len(ordered_images), "root": "runtime/demonstrations",
                "manifest_path": "runtime/demonstrations/manifest.json"}
    finally:
        if staging.exists():
            shutil.rmtree(staging)
