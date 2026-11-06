from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, REPO_ROOT, BENCHMARK_ROOT, resolve_path


import base64
import json
import math
import os
import re
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import imageio.v2 as imageio
import imageio_ffmpeg
import requests
from PIL import Image, ImageDraw, ImageOps


DATA_ROOT = BENCHMARK_ROOT / "MindPower" / "ThreeDWorld"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "MindPower" / "ThreeDWorldPhase1"
DEFAULT_PREP_DIR = DEFAULT_OUTPUT_ROOT / "prep"
DEFAULT_RUN_DIR = DEFAULT_OUTPUT_ROOT / "runs"
DEFAULT_SCORE_DIR = DEFAULT_OUTPUT_ROOT / "scores"
DEFAULT_KEY_LIST = REPO_ROOT / "evaluation" / "mindpower" / "default_openrouter_key_files.txt"

API_URL = "https://openrouter.ai/api/v1/chat/completions"

EVAL_MODELS = [
    "openai/gpt-4o-2024-11-20",
    "google/gemini-2.5-pro",
    "anthropic/claude-4-sonnet-20250522",
    "qwen/qwen2.5-vl-72b-instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct",
    "openai/gpt-4.1-mini",
    "mistralai/ministral-8b-2512",
]
PARSER_MODELS = {
    "B_auto_gpt4o": "openai/gpt-4o-2024-11-20",
    "B_auto_claude": "anthropic/claude-4-sonnet-20250522",
}
MODEL_FRIENDLY_NAMES = {
    "openai/gpt-4o-2024-11-20": "GPT-4o",
    "google/gemini-2.5-pro": "Gemini 2.5 Pro",
    "anthropic/claude-4-sonnet-20250522": "Claude Sonnet 4",
    "qwen/qwen2.5-vl-72b-instruct": "Qwen2.5-VL 72B Instruct",
    "meta-llama/llama-4-maverick-17b-128e-instruct": "Llama 4 Maverick 17B",
    "openai/gpt-4.1-mini": "GPT-4.1 Mini",
    "mistralai/ministral-8b-2512": "Ministral 8B",
}
OPENROUTER_RUNTIME_MODEL_ALIASES = {
    "anthropic/claude-4-sonnet-20250522": "anthropic/claude-sonnet-4",
}

CONDITIONS = [
    "A_visual",
    "B_auto_gpt4o",
    "B_auto_claude",
    "B_percept",
    "B_full",
    "C_text",
    "D_mismatch",
]
TEXT_FIELDS = ["description", "belief", "desire", "intention", "decision"]
ALL_FIELDS = TEXT_FIELDS + ["action"]
SCENARIO_FAMILIES = {
    "reachability",
    "missing_tool",
    "lighting_visibility",
    "obstacle_removal",
}
SCENARIO_FAMILY_BY_SAMPLE_ID = {
    "1": "reachability",
    "2": "reachability",
    "3": "reachability",
    "4": "reachability",
    "5": "reachability",
    "6": "reachability",
    "7": "reachability",
    "8": "reachability",
    "9": "reachability",
    "10": "reachability",
    "11": "obstacle_removal",
    "12": "missing_tool",
    "13": "missing_tool",
    "14": "missing_tool",
    "15": "missing_tool",
    "16": "missing_tool",
    "17": "lighting_visibility",
    "18": "missing_tool",
    "19": "obstacle_removal",
    "20": "obstacle_removal",
    "21": "obstacle_removal",
    "22": "missing_tool",
    "23": "obstacle_removal",
    "24": "obstacle_removal",
    "25": "obstacle_removal",
    "26": "lighting_visibility",
    "27": "missing_tool",
    "28": "missing_tool",
    "29": "missing_tool",
    "30": "missing_tool",
    "31": "missing_tool",
    "32": "obstacle_removal",
    "33": "missing_tool",
    "34": "missing_tool",
    "35": "missing_tool",
    "36": "missing_tool",
}

_TAG_TO_FIELD = {
    "<description>": "description",
    "<perception>": "description",
    "<robot belief>": "belief",
    "<belief>": "belief",
    "<robot desire>": "desire",
    "<desire>": "desire",
    "<robot intention>": "intention",
    "<intention>": "intention",
    "<decision>": "decision",
    "<action>": "action",
}
_TAG_REGEX = re.compile(
    r"(?im)^\s*(<description>|<perception>|<robot belief>|<belief>|<robot desire>|<desire>|"
    r"<robot intention>|<intention>|<decision>|<action>)\s*:?\s*"
)
ACTION_VERB_ALIASES = {
    "go": "walk",
    "goto": "walk",
    "walk": "walk",
    "walkto": "walk",
    "moveto": "move",
    "move": "move",
    "moveaside": "move",
    "reach": "reach",
    "reachfor": "reach",
    "grab": "pick",
    "pickup": "pick",
    "pick": "pick",
    "pick_up": "pick",
    "take": "pick",
    "place": "place",
    "put": "place",
    "putdown": "place",
    "putin": "putin",
    "give": "give",
    "hand": "give",
    "switchon": "light",
    "turnon": "light",
    "light": "light",
    "open": "open",
    "close": "close",
}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def relative_to_root(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except Exception:
        return str(path.resolve())


def read_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def read_api_key_from_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return text.split()[-1].strip() or None


def load_api_keys(key_list_path: Optional[Path], extra_key_files: Sequence[str]) -> List[str]:
    keys: List[str] = []
    env_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if env_key:
        keys.append(env_key)

    candidate_items: List[Tuple[Path, str]] = []
    if key_list_path and key_list_path.exists():
        candidate_items.append((key_list_path.parent, key_list_path.read_text(encoding="utf-8")))
    for raw_path in extra_key_files:
        path = Path(raw_path)
        if path.exists():
            candidate_items.append((path.parent, path.read_text(encoding="utf-8")))
            continue
        repo_relative = ROOT / raw_path
        if repo_relative.exists():
            candidate_items.append((repo_relative.parent, repo_relative.read_text(encoding="utf-8")))

    for base_dir, content in candidate_items:
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("sk-or-"):
                keys.append(line)
                continue
            key_path = (base_dir / line).resolve()
            key = read_api_key_from_file(key_path)
            if key:
                keys.append(key)

    deduped: List[str] = []
    seen = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        deduped.append(key)
    if not deduped:
        raise RuntimeError("No OpenRouter keys found.")
    return deduped


def resolve_runtime_model(model: str) -> str:
    return OPENROUTER_RUNTIME_MODEL_ALIASES.get(model, model)


def parse_tagged_fields(text: str) -> Dict[str, str]:
    matches = list(_TAG_REGEX.finditer(text or ""))
    output = {field: "" for field in ALL_FIELDS}
    if not matches:
        return output
    for idx, match in enumerate(matches):
        field = _TAG_TO_FIELD[match.group(1).lower()]
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        output[field] = normalize_space(text[start:end])
    return output


def parse_source_reasoning_file(path: Path) -> Dict[str, str]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    fields = parse_tagged_fields(text)
    missing = [field for field in ALL_FIELDS if not fields[field]]
    if missing:
        raise ValueError(f"Missing tagged fields in {path}: {missing}")
    return fields


def find_reasoning_file(sample_dir: Path) -> Optional[Path]:
    preferred = sample_dir / "reasoning.txt"
    if preferred.exists():
        return preferred

    tagged_files: List[Path] = []
    for path in sorted(sample_dir.glob("*.txt")):
        try:
            parse_source_reasoning_file(path)
        except Exception:
            continue
        tagged_files.append(path)

    if len(tagged_files) == 1:
        return tagged_files[0]
    if len(tagged_files) > 1:
        raise ValueError(
            f"Expected exactly one tagged annotation file in {sample_dir}, found {[path.name for path in tagged_files]}"
        )
    return None


def sample_dir_sort_key(path: Path) -> Tuple[int, str]:
    text = path.name
    if text.isdigit():
        return (0, f"{int(text):09d}")
    return (1, text.lower())


def _description_token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def select_mismatch_row(row: dict, rows: Sequence[dict]) -> dict:
    current_tokens = _description_token_set(str(row["description_ref"]))
    candidates: List[Tuple[Tuple[int, float, int], dict]] = []
    for candidate in rows:
        if candidate["sample_id"] == row["sample_id"]:
            continue
        candidate_tokens = _description_token_set(str(candidate["description_ref"]))
        union = current_tokens | candidate_tokens
        overlap = len(current_tokens & candidate_tokens) / len(union) if union else 0.0
        lexical_distance = 1.0 - overlap
        cross_family = int(candidate["scenario_family"] != row["scenario_family"])
        sample_distance = abs(int(str(candidate["sample_id"])) - int(str(row["sample_id"])))
        candidates.append(((cross_family, lexical_distance, sample_distance), candidate))

    if not candidates:
        raise ValueError(f"Could not find mismatch candidate for sample {row['sample_id']}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def canonicalize_arg_token(text: str) -> str:
    token = text.strip().lower()
    token = token.replace("'", "")
    token = re.sub(r"\b(the|a|an)\b", " ", token)
    token = re.sub(r"[^a-z0-9]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token


def canonicalize_action_verb(verb: str) -> str:
    compact = re.sub(r"[^a-z]+", "", verb.lower())
    if compact in ACTION_VERB_ALIASES:
        return ACTION_VERB_ALIASES[compact]
    return compact


def parse_action_sequence(text: str) -> List[Tuple[str, Tuple[str, ...]]]:
    action_text = normalize_space(text).replace("; ", ", ")
    tuples: List[Tuple[str, Tuple[str, ...]]] = []
    for verb, raw_args in re.findall(r"([A-Za-z_ ]+?)\s*\(([^()]*)\)", action_text):
        clean_verb = canonicalize_action_verb(verb)
        args = tuple(
            canonicalize_arg_token(arg)
            for arg in raw_args.split(",")
            if canonicalize_arg_token(arg)
        )
        if clean_verb:
            tuples.append((clean_verb, args))
    return tuples


def canonical_action_text(text: str) -> str:
    pieces = []
    for verb, args in parse_action_sequence(text):
        if args:
            pieces.append(f"{verb}({', '.join(args)})")
        else:
            pieces.append(f"{verb}()")
    return ", ".join(pieces)


def build_manifest_rows(data_root: Path = DATA_ROOT) -> List[dict]:
    rows: List[dict] = []
    sample_dirs = sorted((path for path in data_root.iterdir() if path.is_dir()), key=sample_dir_sort_key)
    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        video_paths = sorted(sample_dir.glob("*.mp4"))
        if not video_paths:
            continue
        if len(video_paths) != 1:
            raise ValueError(f"Expected exactly one video for {sample_id}, found {len(video_paths)}")
        reasoning_path = find_reasoning_file(sample_dir)
        if reasoning_path is None:
            raise FileNotFoundError(f"Missing tagged annotation file for {sample_id}")
        fields = parse_source_reasoning_file(reasoning_path)
        scenario_family = SCENARIO_FAMILY_BY_SAMPLE_ID.get(sample_id)
        if scenario_family not in SCENARIO_FAMILIES:
            raise ValueError(f"Unsupported scenario family for {sample_id}: {scenario_family}")
        rows.append(
            {
                "sample_id": sample_id,
                "task_family": "implicit_goal",
                "scenario_family": scenario_family,
                "video_path": relative_to_root(video_paths[0]),
                "reasoning_path": relative_to_root(reasoning_path),
                "description_ref": fields["description"],
                "belief_ref": fields["belief"],
                "desire_ref": fields["desire"],
                "intention_ref": fields["intention"],
                "decision_ref": fields["decision"],
                "action_ref": fields["action"],
            }
        )

    if not rows:
        raise ValueError(f"No complete video samples found under {data_root}")

    rows.sort(key=lambda row: int(row["sample_id"]))
    for row in rows:
        mismatch_row = select_mismatch_row(row, rows)
        row["mismatch_sample_id"] = mismatch_row["sample_id"]
        row["mismatch_description_ref"] = mismatch_row["description_ref"]
    return rows


def write_manifest_jsonl(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_manifest_csv(path: Path, rows: Sequence[dict]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _uniform_indices(start: int, end: int, count: int) -> List[int]:
    if count <= 0:
        return []
    if start == end:
        return [start] * count
    step = (end - start) / max(count - 1, 1)
    indices = [int(round(start + step * idx)) for idx in range(count)]
    return [max(start, min(end, value)) for value in indices]


def compute_phase1_frame_indices(frame_total: int) -> List[int]:
    if frame_total <= 12:
        return sorted(set(_uniform_indices(0, max(frame_total - 1, 0), min(frame_total, 12))))

    split = int(math.floor(frame_total * 0.8))
    split = max(8, min(frame_total - 4, split))
    early = _uniform_indices(0, max(split - 1, 0), 8)
    late = _uniform_indices(split, frame_total - 1, 4)
    merged = early + late
    if len(set(merged)) == 12:
        return merged
    fallback = _uniform_indices(0, frame_total - 1, 12)
    deduped: List[int] = []
    for value in fallback:
        if value not in deduped:
            deduped.append(value)
    while len(deduped) < 12 and deduped:
        candidate = min(frame_total - 1, deduped[-1] + 1)
        if candidate not in deduped:
            deduped.append(candidate)
        else:
            break
    return deduped[:12]


def _draw_frame_label(image: Image.Image, label: str) -> None:
    drawer = ImageDraw.Draw(image)
    drawer.rectangle((0, 0, 108, 28), fill=(0, 0, 0))
    drawer.text((8, 7), label, fill=(255, 255, 255))


def build_contact_sheet(
    frame_paths: Sequence[Path],
    timestamps_sec: Sequence[float],
    output_path: Path,
    tile_size: Tuple[int, int] = (384, 216),
    columns: int = 4,
) -> None:
    if len(frame_paths) != len(timestamps_sec):
        raise ValueError("frame_paths and timestamps_sec must have the same length")
    rows = int(math.ceil(len(frame_paths) / columns))
    pad = 8
    width = columns * tile_size[0] + (columns + 1) * pad
    height = rows * tile_size[1] + (rows + 1) * pad
    canvas = Image.new("RGB", (width, height), color=(18, 18, 18))

    for idx, frame_path in enumerate(frame_paths):
        image = Image.open(frame_path).convert("RGB")
        tile = ImageOps.fit(image, tile_size, method=Image.Resampling.LANCZOS)
        timestamp = timestamps_sec[idx]
        _draw_frame_label(tile, f"{idx + 1:02d}  t={timestamp:05.1f}s")
        row_idx = idx // columns
        col_idx = idx % columns
        x = pad + col_idx * (tile_size[0] + pad)
        y = pad + row_idx * (tile_size[1] + pad)
        canvas.paste(tile, (x, y))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def extract_storyboard_bundle(
    video_path: Path,
    sample_id: str,
    prep_dir: Path = DEFAULT_PREP_DIR,
) -> dict:
    storyboard_dir = prep_dir / "storyboards" / f"sample_{int(sample_id):02d}"
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
        "sample_id": sample_id,
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


def encode_image(image_path: Path) -> str:
    with image_path.open("rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


def build_openrouter_content(prompt: str, image_paths: Sequence[Path]) -> List[dict]:
    content: List[dict] = [{"type": "text", "text": prompt}]
    for image_path in image_paths:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{encode_image(image_path)}",
                    "detail": "high",
                },
            }
        )
    return content


def build_openrouter_reasoning_config(model: str) -> Optional[dict]:
    model_name = str(model)
    if model_name.startswith("google/gemini-2.5-"):


        return {
            "max_tokens": 32,
            "exclude": True,
        }
    return None


def call_openrouter(
    model: str,
    prompt: str,
    api_key: str,
    image_paths: Optional[Sequence[Path]] = None,
    session: Optional[requests.Session] = None,
    max_tokens: int = 500,
    max_retries: int = 3,
) -> Tuple[str, Optional[dict]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": build_openrouter_content(prompt, image_paths or []),
            }
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    reasoning = build_openrouter_reasoning_config(model)
    if reasoning is not None:
        payload["reasoning"] = reasoning
    requester = session or requests
    for attempt in range(max_retries):
        try:
            response = requester.post(API_URL, headers=headers, json=payload, timeout=240)
            if response.status_code == 429:
                time.sleep(12 * (attempt + 1))
                continue
            if response.status_code >= 500:
                time.sleep(8 * (attempt + 1))
                continue
            if response.status_code >= 400:
                error_text = response.text[:1200]
                try:
                    data = response.json()
                    error_block = data.get("error") or {}
                    message = error_block.get("message") or error_text
                    return f"ERROR_HTTP_{response.status_code}: {message}", data
                except Exception:
                    return f"ERROR_HTTP_{response.status_code}: {error_text}", None
            data = response.json()
            if "error" in data:
                return f"ERROR_API: {data['error'].get('message', 'unknown error')}", data
            choices = data.get("choices") or []
            if not choices:
                return f"ERROR_BAD_RESPONSE: {json.dumps(data)[:500]}", data
            return str(choices[0]["message"]["content"]).strip(), data
        except Exception as exc:  # noqa: BLE001
            if attempt == max_retries - 1:
                return f"ERROR: {str(exc)[:240]}", None
            time.sleep(5 * (attempt + 1))
    return "ERROR: max retries", None


def build_parser_prompt() -> str:
    return (
        "You are describing a 12-frame storyboard of a household interaction in temporal order. "
        "Write only a short neutral scene description covering visible actions, object states, and "
        "temporal order. Do not mention beliefs, desires, intentions, decisions, robot actions, or advice. "
        "Do not answer with bullet points. Output only 3 to 6 plain sentences."
    )


def build_eval_prompt(
    *,
    include_storyboard: bool,
    scaffold_text: Optional[str] = None,
    scaffold_label: Optional[str] = None,
) -> str:
    lines = [
        "You are a robot helper observing a household situation and deciding how to assist the human.",
        "Infer the human's likely goal, then produce a robot-centric reasoning chain and an action plan.",
        "Return exactly these six tagged lines and nothing else:",
        "<Description> ...",
        "<Robot Belief> ...",
        "<Robot Desire> ...",
        "<Robot Intention> ...",
        "<Decision> ...",
        "<Action> ...",
        "",
        "Rules:",
        "- <Description> must contain only observable or explicitly provided facts.",
        "- The belief, desire, intention, decision, and action must be from the robot's perspective.",
        "- Never omit a tag, even if uncertain.",
        "- Keep each field concise and specific to the scene.",
        "- <Action> must be a comma-separated sequence of lowercase atomic actions like walk(object), pick(object), place(object, location), give(object, owner), move(object, location), light(object), reach(object), open(object), or close(object).",
    ]
    if include_storyboard:
        lines.append("- Use the storyboard as primary evidence.")
    else:
        lines.append("- No storyboard is provided; rely only on the text context below.")
    if scaffold_text:
        label = scaffold_label or "Scene scaffold"
        lines.extend(
            [
                "",
                f"{label}:",
                scaffold_text.strip(),
                "",
                "Treat the text above as neutral context rather than a final answer.",
            ]
        )
    return "\n".join(lines)


def build_bpc_prompt(candidate_text: str) -> str:
    return (
        "You are grading a robot-centric Theory-of-Mind reasoning chain.\n"
        "Score the candidate on four 0-10 scales:\n"
        "1. logical_consistency: do later layers follow from earlier layers without contradictions?\n"
        "2. completeness_specificity: is the chain concrete, useful, and sufficiently specific?\n"
        "3. robot_centric_perspective: does the reasoning stay in the helper robot's perspective instead of generic scene narration?\n"
        "4. overall_bpc: overall BDI and perspective consistency.\n"
        "Return strict JSON with keys logical_consistency, completeness_specificity, robot_centric_perspective, overall_bpc, rationale.\n"
        "Use numbers, not strings, for the scores.\n\n"
        "Candidate:\n"
        f"{candidate_text.strip()}\n"
    )


def parse_bpc_response(text: str) -> Optional[dict]:
    payload = None
    try:
        payload = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                payload = json.loads(match.group(0))
            except Exception:
                payload = None
    if not isinstance(payload, dict):
        return None
    output = {}
    for key in [
        "logical_consistency",
        "completeness_specificity",
        "robot_centric_perspective",
        "overall_bpc",
    ]:
        value = payload.get(key)
        try:
            output[key] = float(value)
        except Exception:
            return None
    output["rationale"] = normalize_space(str(payload.get("rationale", "")))
    return output


def render_reference_response(row: dict) -> str:
    return "\n".join(
        [
            f"<Description> {row['description_ref']}",
            f"<Robot Belief> {row['belief_ref']}",
            f"<Robot Desire> {row['desire_ref']}",
            f"<Robot Intention> {row['intention_ref']}",
            f"<Decision> {row['decision_ref']}",
            f"<Action> {row['action_ref']}",
        ]
    )


def build_full_scaffold_text(row: dict) -> str:
    return "\n".join(
        [
            f"Description: {row['description_ref']}",
            f"Robot belief: {row['belief_ref']}",
            f"Robot desire: {row['desire_ref']}",
            f"Robot intention: {row['intention_ref']}",
            f"Decision: {row['decision_ref']}",
            f"Action: {row['action_ref']}",
        ]
    )
