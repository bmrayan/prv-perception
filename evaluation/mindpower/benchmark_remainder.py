from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, REPO_ROOT, BENCHMARK_ROOT, resolve_path


import csv
import json
import math
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import imageio_ffmpeg
from PIL import Image

from mindpower.common import (
    ALL_FIELDS,
    ROOT,
    build_contact_sheet,
    compute_phase1_frame_indices,
    find_reasoning_file,
    normalize_space,
    now_iso,
    parse_tagged_fields,
    parse_source_reasoning_file,
    relative_to_root,
    sample_dir_sort_key,
    slugify,
    write_json,
)


THREEDWORLD_ROOT = BENCHMARK_ROOT / "MindPower" / "ThreeDWorld"
VIRTUALHOME_ROOT = BENCHMARK_ROOT / "MindPower" / "VirtualHome"
DEFAULT_REMAINDER_OUTPUT_ROOT = ROOT / "results" / "MindPower" / "BenchmarkRemainder"
DEFAULT_REMAINDER_PREP_DIR = DEFAULT_REMAINDER_OUTPUT_ROOT / "prep"
DEFAULT_REMAINDER_RUN_DIR = DEFAULT_REMAINDER_OUTPUT_ROOT / "runs"
DEFAULT_REMAINDER_SCORE_DIR = DEFAULT_REMAINDER_OUTPUT_ROOT / "scores"


def read_text(path: Path) -> str:
    return normalize_space(path.read_text(encoding="utf-8", errors="ignore"))


def parse_reasoning_fields_relaxed(reasoning_path: Path, fallback_description: Optional[str] = None) -> Dict[str, object]:
    text = reasoning_path.read_text(encoding="utf-8", errors="ignore")
    fields = parse_tagged_fields(text)
    missing_non_description = [field for field in ALL_FIELDS if field != "description" and not fields[field]]
    if missing_non_description:
        raise ValueError(f"Missing required tagged fields in {reasoning_path}: {missing_non_description}")
    description_is_fallback = False
    if not fields["description"]:
        fields["description"] = normalize_space(fallback_description or "")
        description_is_fallback = True
    return {
        "fields": fields,
        "description_is_fallback": description_is_fallback,
    }


def find_non_reasoning_text_files(sample_dir: Path, reasoning_path: Optional[Path]) -> List[Path]:
    files: List[Path] = []
    for path in sorted(sample_dir.glob("*.txt")):
        if reasoning_path and path.resolve() == reasoning_path.resolve():
            continue
        if path.name == ".DS_Store":
            continue
        files.append(path)
    return files


def parse_virtualhome_script_metadata(script_path: Path) -> Dict[str, object]:
    text = script_path.read_text(encoding="utf-8", errors="ignore")
    scene_match = re.search(r"(?m)^\s*scene_id\s*=\s*(\d+)", text)
    camera_modes = [
        normalize_space(match)
        for match in re.findall(r"camera_mode\s*=\s*\[([^\]]+)\]", text)
        if normalize_space(match)
    ]
    file_name_prefixes = re.findall(r"file_name_prefix\s*=\s*['\"]([^'\"]+)['\"]", text)
    room_names = re.findall(r'add_four_cams_around_room\("([^"]+)"\)', text)
    return {
        "scene_id": int(scene_match.group(1)) if scene_match else None,
        "recording_stage_count": len(re.findall(r"recording\s*=\s*True", text)),
        "camera_modes": camera_modes,
        "file_name_prefixes": sorted(dict.fromkeys(file_name_prefixes)),
        "camera_room_names": sorted(dict.fromkeys(room_names)),
    }


def build_script_description_fallback(script_text: str, metadata: Dict[str, object]) -> str:
    verbs = []
    for verb in re.findall(r"\[([A-Za-z_]+)\]", script_text):
        clean = verb.strip().lower()
        if clean and clean not in verbs:
            verbs.append(clean)
        if len(verbs) >= 4:
            break
    objects = []
    for obj in re.findall(r"<([A-Za-z_]+)>", script_text):
        clean = obj.strip().lower()
        if clean.startswith("char"):
            continue
        if clean and clean not in objects:
            objects.append(clean)
        if len(objects) >= 5:
            break
    rooms = [str(room) for room in metadata.get("camera_room_names", []) if str(room)]
    pieces = ["A VirtualHome simulation script depicts household agents interacting in a domestic scene."]
    if verbs:
        pieces.append(f"Observed scripted actions include {', '.join(verbs)}.")
    if objects:
        pieces.append(f"Referenced objects include {', '.join(objects)}.")
    if rooms:
        pieces.append(f"The scene focuses on rooms such as {', '.join(rooms)}.")
    return " ".join(pieces)


def looks_like_virtualhome_script(text: str) -> bool:
    markers = [
        "UnityCommunication",
        "comm.render_script",
        "comm.reset(",
        "add_character(",
        "environment_graph(",
        "camera_mode",
        "file_name_prefix",
        "<char0>",
        "<char1>",
        "[Walk]",
        "[walk]",
    ]
    if any(marker in text for marker in markers):
        return True
    if re.search(r"(?m)^\s*(from|import)\s+virtualhome", text):
        return True
    if re.search(r"<char\d+>\s*\[[A-Za-z_]+\]", text):
        return True
    return False


def build_talk_rows(data_root: Path = THREEDWORLD_ROOT) -> List[dict]:
    rows: List[dict] = []
    for sample_dir in sorted((path for path in data_root.iterdir() if path.is_dir() and path.name.startswith("talk")), key=sample_dir_sort_key):
        reasoning_path = find_reasoning_file(sample_dir)
        if reasoning_path is None:
            continue
        context_files = find_non_reasoning_text_files(sample_dir, reasoning_path)
        if len(context_files) != 1:
            raise ValueError(
                f"Expected exactly one dialogue file for {sample_dir.name}, found {[path.name for path in context_files]}"
            )
        context_text = read_text(context_files[0])
        parsed = parse_reasoning_fields_relaxed(reasoning_path, fallback_description=context_text)
        fields = parsed["fields"]
        sample_id = f"threedworld_talk__{sample_dir.name}"
        rows.append(
            {
                "sample_id": sample_id,
                "source_sample_id": sample_dir.name,
                "source_group": "threedworld_talk",
                "task_family": "dialogue_only",
                "scenario_family": "threedworld_talk",
                "input_mode": "text_context",
                "context_label": "Dialogue transcript",
                "context_path": relative_to_root(context_files[0]),
                "context_text": context_text,
                "reasoning_path": relative_to_root(reasoning_path),
                "description_ref": fields["description"],
                "belief_ref": fields["belief"],
                "desire_ref": fields["desire"],
                "intention_ref": fields["intention"],
                "decision_ref": fields["decision"],
                "action_ref": fields["action"],
                "description_ref_is_fallback": bool(parsed["description_is_fallback"]),
                "render_required": False,
            }
        )
    return rows


def build_virtualhome_rows(data_root: Path = VIRTUALHOME_ROOT) -> Tuple[List[dict], List[str]]:
    rows: List[dict] = []
    skipped: List[str] = []
    for sample_dir in sorted((path for path in data_root.iterdir() if path.is_dir()), key=sample_dir_sort_key):
        reasoning_path = find_reasoning_file(sample_dir)
        if reasoning_path is None:
            skipped.append(sample_dir.name)
            continue
        context_files = find_non_reasoning_text_files(sample_dir, reasoning_path)
        script_files = sorted(path for path in sample_dir.glob("*.py") if path.name != ".DS_Store")
        sample_id = f"virtualhome__{sample_dir.name}"
        if len(script_files) == 1 and not context_files:
            script_path = script_files[0]
            context_text = script_path.read_text(encoding="utf-8", errors="ignore")
            if not looks_like_virtualhome_script(context_text):
                parsed = parse_reasoning_fields_relaxed(reasoning_path, fallback_description=context_text)
                fields = parsed["fields"]
                rows.append(
                    {
                        "sample_id": sample_id,
                        "source_sample_id": sample_dir.name,
                        "source_group": "virtualhome_dialogue",
                        "task_family": "dialogue_only",
                        "scenario_family": "virtualhome_dialogue",
                        "input_mode": "text_context",
                        "context_label": "Narrative transcript",
                        "context_path": relative_to_root(script_path),
                        "context_text": normalize_space(context_text),
                        "reasoning_path": relative_to_root(reasoning_path),
                        "description_ref": fields["description"],
                        "belief_ref": fields["belief"],
                        "desire_ref": fields["desire"],
                        "intention_ref": fields["intention"],
                        "decision_ref": fields["decision"],
                        "action_ref": fields["action"],
                        "description_ref_is_fallback": bool(parsed["description_is_fallback"]),
                        "render_required": False,
                    }
                )
                continue
            metadata = parse_virtualhome_script_metadata(script_path)
            parsed = parse_reasoning_fields_relaxed(
                reasoning_path,
                fallback_description=build_script_description_fallback(context_text, metadata),
            )
            fields = parsed["fields"]
            rows.append(
                {
                    "sample_id": sample_id,
                    "source_sample_id": sample_dir.name,
                    "source_group": "virtualhome_script",
                    "task_family": "virtualhome_script",
                    "scenario_family": "virtualhome_script",
                    "input_mode": "virtualhome_script",
                    "context_label": "VirtualHome simulation script",
                    "context_path": relative_to_root(script_path),
                    "context_text": context_text,
                    "script_path": relative_to_root(script_path),
                    "reasoning_path": relative_to_root(reasoning_path),
                    "description_ref": fields["description"],
                    "belief_ref": fields["belief"],
                    "desire_ref": fields["desire"],
                    "intention_ref": fields["intention"],
                    "decision_ref": fields["decision"],
                    "action_ref": fields["action"],
                    "description_ref_is_fallback": bool(parsed["description_is_fallback"]),
                    "render_required": True,
                    "script_scene_id": metadata["scene_id"],
                    "script_recording_stage_count": metadata["recording_stage_count"],
                    "script_camera_modes": metadata["camera_modes"],
                    "script_file_name_prefixes": metadata["file_name_prefixes"],
                    "script_camera_room_names": metadata["camera_room_names"],
                }
            )
            continue
        if len(context_files) == 1 and not script_files:
            context_path = context_files[0]
            context_text = read_text(context_path)
            parsed = parse_reasoning_fields_relaxed(reasoning_path, fallback_description=context_text)
            fields = parsed["fields"]
            rows.append(
                {
                    "sample_id": sample_id,
                    "source_sample_id": sample_dir.name,
                    "source_group": "virtualhome_dialogue",
                    "task_family": "dialogue_only",
                    "scenario_family": "virtualhome_dialogue",
                    "input_mode": "text_context",
                    "context_label": "Dialogue transcript",
                    "context_path": relative_to_root(context_path),
                    "context_text": context_text,
                    "reasoning_path": relative_to_root(reasoning_path),
                    "description_ref": fields["description"],
                    "belief_ref": fields["belief"],
                    "desire_ref": fields["desire"],
                    "intention_ref": fields["intention"],
                    "decision_ref": fields["decision"],
                    "action_ref": fields["action"],
                    "description_ref_is_fallback": bool(parsed["description_is_fallback"]),
                    "render_required": False,
                }
            )
            continue
        raise ValueError(
            f"Unsupported VirtualHome sample layout for {sample_dir.name}: "
            f"txt={[path.name for path in context_files]}, py={[path.name for path in script_files]}"
        )
    return rows, skipped


def build_remainder_rows(
    *,
    threedworld_root: Path = THREEDWORLD_ROOT,
    virtualhome_root: Path = VIRTUALHOME_ROOT,
) -> Tuple[List[dict], List[str]]:
    talk_rows = build_talk_rows(threedworld_root)
    virtualhome_rows, skipped = build_virtualhome_rows(virtualhome_root)
    rows = talk_rows + virtualhome_rows
    rows.sort(key=lambda row: row["sample_id"])
    return rows, skipped


def filter_text_eval_rows(rows: Sequence[dict], include_script_context: bool = False) -> List[dict]:
    allowed_modes = {"text_context"}
    if include_script_context:
        allowed_modes.add("virtualhome_script")
    return [row for row in rows if str(row.get("input_mode")) in allowed_modes]


def filter_virtualhome_script_rows(rows: Sequence[dict]) -> List[dict]:
    return [row for row in rows if str(row.get("input_mode")) == "virtualhome_script"]


def build_virtualhome_render_jobs(rows: Sequence[dict], render_root: Path) -> List[dict]:
    jobs: List[dict] = []
    for row in filter_virtualhome_script_rows(rows):
        bundle_id = slugify(str(row["sample_id"]))
        jobs.append(
            {
                "sample_id": row["sample_id"],
                "source_sample_id": row["source_sample_id"],
                "source_group": row["source_group"],
                "script_path": row["script_path"],
                "reasoning_path": row["reasoning_path"],
                "scene_id": row.get("script_scene_id"),
                "recording_stage_count": row.get("script_recording_stage_count"),
                "camera_modes": row.get("script_camera_modes", []),
                "camera_room_names": row.get("script_camera_room_names", []),
                "bundle_id": bundle_id,
                "render_work_dir": relative_to_root(render_root / "workdirs" / bundle_id),
            }
        )
    return jobs


def write_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_union_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def manifest_index(rows: Sequence[dict]) -> Dict[str, dict]:
    return {str(row["sample_id"]): dict(row) for row in rows}


def extract_storyboard_bundle_generic(video_path: Path, bundle_id: str, prep_dir: Path) -> dict:
    storyboard_dir = prep_dir / "storyboards" / bundle_id
    manifest_path = storyboard_dir / "storyboard_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        contact_sheet_path = resolve_path(manifest["contact_sheet_path"])
        frame_paths = [resolve_path(rel_path) for rel_path in manifest["frame_paths"]]
        if contact_sheet_path.exists() and all(path.exists() for path in frame_paths):
            return manifest

    storyboard_dir.mkdir(parents=True, exist_ok=True)
    frame_total, duration_sec = imageio_ffmpeg.count_frames_and_secs(str(video_path))
    frame_total = int(frame_total)
    duration_sec = float(duration_sec)
    indices = compute_phase1_frame_indices(frame_total)
    reader = imageio.get_reader(str(video_path), format="ffmpeg")
    frame_paths: List[Path] = []
    actual_indices: List[int] = []
    try:
        for order, requested_idx in enumerate(indices):
            actual_idx = requested_idx
            while actual_idx >= 0:
                try:
                    frame = reader.get_data(actual_idx)
                    break
                except IndexError:
                    actual_idx -= 1
            if actual_idx < 0:
                raise IndexError(f"Could not read any frame near index {requested_idx}")
            frame_path = storyboard_dir / f"frame_{order + 1:02d}_src{actual_idx:06d}.png"
            Image.fromarray(frame).save(frame_path)
            frame_paths.append(frame_path)
            actual_indices.append(actual_idx)
    finally:
        reader.close()

    if frame_total <= 1:
        timestamps_sec = [0.0 for _ in actual_indices]
    else:
        timestamps_sec = [duration_sec * idx / (frame_total - 1) for idx in actual_indices]

    contact_sheet_path = storyboard_dir / "storyboard_contact_sheet.png"
    build_contact_sheet(frame_paths, timestamps_sec, contact_sheet_path)
    manifest = {
        "generated_at_utc": now_iso(),
        "bundle_id": bundle_id,
        "video_path": relative_to_root(video_path),
        "frame_total": frame_total,
        "duration_sec": duration_sec,
        "requested_indices": indices,
        "source_indices": actual_indices,
        "timestamps_sec": timestamps_sec,
        "frame_paths": [relative_to_root(path) for path in frame_paths],
        "contact_sheet_path": relative_to_root(contact_sheet_path),
    }
    write_json(manifest_path, manifest)
    return manifest


def build_storyboard_from_images(
    image_paths: Sequence[Path],
    bundle_id: str,
    prep_dir: Path,
) -> Optional[dict]:
    if not image_paths:
        return None
    storyboard_dir = prep_dir / "storyboards" / bundle_id
    manifest_path = storyboard_dir / "storyboard_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        contact_sheet_path = resolve_path(manifest["contact_sheet_path"])
        frame_paths = [resolve_path(rel_path) for rel_path in manifest["frame_paths"]]
        if contact_sheet_path.exists() and all(path.exists() for path in frame_paths):
            return manifest

    storyboard_dir.mkdir(parents=True, exist_ok=True)
    total_images = len(image_paths)
    selected_indices = compute_phase1_frame_indices(total_images)
    selected = [image_paths[idx] for idx in selected_indices]
    timestamps_sec = [float(idx) for idx in selected_indices]
    saved_paths: List[Path] = []
    for idx, src in enumerate(selected, start=1):
        target = storyboard_dir / f"frame_{idx:02d}_{src.name}"
        if src.resolve() != target.resolve():
            target.write_bytes(src.read_bytes())
        saved_paths.append(target)
    contact_sheet_path = storyboard_dir / "storyboard_contact_sheet.png"
    build_contact_sheet(saved_paths, timestamps_sec, contact_sheet_path)
    manifest = {
        "generated_at_utc": now_iso(),
        "bundle_id": bundle_id,
        "image_sequence_source": [relative_to_root(path) for path in selected],
        "frame_total": total_images,
        "duration_sec": float(max(total_images - 1, 0)),
        "requested_indices": selected_indices,
        "source_indices": selected_indices,
        "timestamps_sec": timestamps_sec,
        "frame_paths": [relative_to_root(path) for path in saved_paths],
        "contact_sheet_path": relative_to_root(contact_sheet_path),
    }
    write_json(manifest_path, manifest)
    return manifest


def _description_token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", str(text).lower()))


def select_remainder_mismatch_row(row: dict, rows: Sequence[dict]) -> dict:
    current_tokens = _description_token_set(str(row["description_ref"]))
    ranked: List[Tuple[Tuple[int, int, float, str], dict]] = []
    for candidate in rows:
        if candidate["sample_id"] == row["sample_id"]:
            continue
        candidate_tokens = _description_token_set(str(candidate["description_ref"]))
        union = current_tokens | candidate_tokens
        overlap = len(current_tokens & candidate_tokens) / len(union) if union else 0.0
        lexical_distance = 1.0 - overlap
        cross_family = int(candidate.get("scenario_family") != row.get("scenario_family"))
        cross_group = int(candidate.get("source_group") != row.get("source_group"))
        ranked.append(
            (
                (
                    cross_family,
                    cross_group,
                    lexical_distance,
                    str(candidate["sample_id"]),
                ),
                candidate,
            )
        )
    if not ranked:
        raise ValueError(f"Could not find mismatch candidate for sample {row['sample_id']}")
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked[0][1]


def classify_render_record(record: dict) -> str:
    text = f"{record.get('stderr_tail') or ''}\n{record.get('stdout_tail') or ''}"
    if record.get("status") == "ok":
        return "ok"
    if record.get("storyboard_path"):
        if "Read timed out" in text:
            return "usable_timeout"
        return "usable_error"
    if "already being used" in text or "Address already in use" in text:
        return "port_in_use"
    if "Missing object:" in text:
        return "missing_object"
    if "SyntaxError" in text:
        return "syntax_error"
    return "error"


def choose_best_render_record(records: Sequence[dict]) -> Optional[dict]:
    if not records:
        return None

    priority = {
        "ok": 4,
        "usable_timeout": 3,
        "usable_error": 2,
        "port_in_use": 1,
        "missing_object": 1,
        "syntax_error": 1,
        "error": 0,
    }

    def key(record: dict) -> Tuple[int, int, str]:
        status = classify_render_record(record)
        return (
            int(bool(record.get("storyboard_path"))),
            priority.get(status, -1),
            str(record.get("timestamp_utc") or ""),
        )

    return max(records, key=key)


def build_virtualhome_visual_eval_rows(
    script_rows: Sequence[dict],
    render_rows: Sequence[dict],
) -> Tuple[List[dict], List[dict]]:
    render_rows_by_sample: Dict[str, List[dict]] = {}
    for record in render_rows:
        render_rows_by_sample.setdefault(str(record["sample_id"]), []).append(record)

    usable_rows: List[dict] = []
    dropped_rows: List[dict] = []
    for row in script_rows:
        sample_id = str(row["sample_id"])
        best_record = choose_best_render_record(render_rows_by_sample.get(sample_id, []))
        if best_record is None or not best_record.get("storyboard_path"):
            if best_record is None:
                reason = "no_render_record"
            elif int(row.get("script_recording_stage_count") or 0) <= 0:
                reason = "no_recording_stage"
            else:
                reason = classify_render_record(best_record)
            dropped_rows.append(
                {
                    "sample_id": sample_id,
                    "source_sample_id": str(row.get("source_sample_id") or sample_id),
                    "reason": reason,
                }
            )
            continue
        merged = dict(row)
        merged.update(
            {
                "storyboard_path": str(best_record["storyboard_path"]),
                "storyboard_frame_paths": list(best_record.get("storyboard_frame_paths") or []),
                "render_status": classify_render_record(best_record),
                "render_status_raw": str(best_record.get("status") or "unknown"),
                "render_returncode": best_record.get("returncode"),
                "render_work_dir": str(best_record.get("work_dir") or ""),
                "render_image_path_count": len(best_record.get("image_paths") or []),
                "render_image_paths_preview": list(best_record.get("image_paths") or [])[:24],
            }
        )
        usable_rows.append(merged)

    usable_rows.sort(key=lambda item: str(item["sample_id"]))
    for row in usable_rows:
        mismatch_row = select_remainder_mismatch_row(row, usable_rows)
        row["mismatch_sample_id"] = mismatch_row["sample_id"]
        row["mismatch_description_ref"] = mismatch_row["description_ref"]
    return usable_rows, dropped_rows
