from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, REPO_ROOT, BENCHMARK_ROOT, resolve_path


import argparse
import base64
import json
import math
import os
import queue
import random
import re
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import requests
from PIL import Image, ImageDraw


API_URL = "https://openrouter.ai/api/v1/chat/completions"
PER_CALL_COST = 0.004
JSON_SHARD_MAX_BYTES = 28 * 1024 * 1024
JSON_SHARD_MANIFEST = "sharded_json_array_v1"
BOOTSTRAP_RESAMPLES = 10_000

STIMULI_DIR = ROOT
RESULTS_DIR = ROOT / "results" / "generated" / "abstract"
PROBES_PATH = STIMULI_DIR / "probes.json"
DOOR_RGB = (141, 110, 99)

MODELS = [
    "openai/gpt-4o-2024-11-20",
    "google/gemini-2.5-pro-preview",
    "anthropic/claude-sonnet-4",
    "qwen/qwen-2.5-vl-72b-instruct",
    "meta-llama/llama-4-maverick",
    "openai/gpt-4.1-mini",
    "mistralai/ministral-8b-2512",
]

MODEL_TIERS = {
    "gpt-4o-2024-11-20": "frontier",
    "gemini-2.5-pro-preview": "frontier",
    "claude-sonnet-4": "frontier",
    "qwen-2.5-vl-72b-instruct": "mid",
    "llama-4-maverick": "mid",
    "gpt-4.1-mini": "small",
    "ministral-8b-2512": "small",
}

AGENT_COLORS = [
    ("#1565C0", "blue"),
    ("#C62828", "red"),
    ("#2E7D32", "green"),
    ("#6A1B9A", "purple"),
    ("#E65100", "orange"),
    ("#00838F", "teal"),
]
OBJECT_COLORS = [
    ("#FFD600", "yellow"),
    ("#FF6F00", "amber"),
    ("#76FF03", "lime"),
    ("#E040FB", "pink"),
    ("#00E5FF", "cyan"),
]
CONTAINER_COLORS = [
    ("#42A5F5", "light-blue"),
    ("#EF5350", "coral"),
    ("#5D4037", "brown"),
    ("#66BB6A", "mint"),
    ("#AB47BC", "violet"),
    ("#FF7043", "peach"),
]

STAGE1_TEMPLATES = [
    "occlusion_consistent",
    "occlusion_violation",
    "count_two",
    "count_three",
    "permanence",
    "color_change",
]
STAGE2_TEMPLATES = [
    "goal_upper",
    "detour_obstacle",
    "no_goal",
    "wall_removed",
]
STAGE3_TEMPLATES = [
    "sally_anne_search",
    "sally_anne_belief",
    "sally_anne_reality",
    "reality_vs_belief",
    "knowledge_who_knows",
    "knowledge_belief",
    "unexpected_contents",
    "two_rooms",
    "perspectives_differ",
    "second_order",
]
ALL_TEMPLATES = STAGE1_TEMPLATES + STAGE2_TEMPLATES + STAGE3_TEMPLATES

PHASE_CONDITIONS = {
    "phase1": ["A", "B_auto", "B_percept", "B_full", "C"],
    "phase2": ["D_mismatch", "D_shuffled", "D_noise"],
}

STAGE_NAMES = {
    1: "Object Tracking",
    2: "Goal Attribution",
    3: "False Belief / ToM",
}

BONFERRONI_THRESHOLD = 0.05 / 12.0


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def slug_model(model: str) -> str:
    return model.split("/")[-1].replace("/", "-")


def hex_to_rgb(value: str) -> Tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def rgb_dist(a: Sequence[int], b: Sequence[int]) -> float:
    return float(math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))))


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def percent(value: Optional[float]) -> str:
    if value is None or math.isnan(value):
        return "nan"
    return f"{100.0 * value:.1f}%"


def percent_ci(point: Optional[float], low: Optional[float], high: Optional[float]) -> str:
    if any(v is None or math.isnan(v) for v in (point, low, high)):
        return "nan"
    return f"{100.0 * point:.1f}% [{100.0 * low:.1f}, {100.0 * high:.1f}]"


def select_models(count: int, explicit_models: Optional[Sequence[str]] = None) -> List[str]:
    if explicit_models:
        selected: List[str] = []
        seen = set()
        for raw_model in explicit_models:
            model = str(raw_model).strip()
            if not model or model in seen:
                continue
            seen.add(model)
            selected.append(model)
        if not selected:
            raise ValueError("Explicit model list is empty after normalization.")
        return selected
    count = max(1, min(count, len(MODELS)))
    return MODELS[:count]


def choose_distinct(palette: Sequence[Tuple[str, str]], n: int, rng: random.Random) -> List[Tuple[str, str]]:
    return rng.sample(list(palette), n)


def replace_vocabulary(text: str) -> str:
    replacements = [
        (r"\bcharacter\b", "figure"),
        (r"\bcharacters\b", "figures"),
        (r"\bcontainer\b", "box"),
        (r"\bcontainers\b", "boxes"),
        (r"\bobject\b", "item"),
        (r"\bobjects\b", "items"),
        (r"\bsearch\b", "look in"),
        (r"\bsearch first\b", "look in first"),
    ]
    out = text
    for pattern, replacement in replacements:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return out


def parse_answer_suffix(answer_instruction: str) -> str:
    return answer_instruction.strip()


def read_api_key_from_file(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    token = text.split()[-1].strip()
    return token or None


def load_api_keys(extra_key_files: Optional[Sequence[str]] = None, require: bool = True) -> List[str]:
    keys: List[str] = []
    env_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if env_key:
        keys.append(env_key)

    default_key = read_api_key_from_file(Path("OPENROUTER_Key.txt"))
    if default_key:
        keys.append(default_key)

    for key_file in extra_key_files or ():
        extra_key = read_api_key_from_file(Path(key_file))
        if extra_key:
            keys.append(extra_key)

    unique_keys: List[str] = []
    seen = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        unique_keys.append(key)

    if require and not unique_keys:
        raise RuntimeError("No usable OpenRouter API keys found")
    return unique_keys


def load_api_key(require: bool) -> Optional[str]:
    keys = load_api_keys(require=require)
    return keys[0] if keys else None


def build_headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}


def encode_image(path: str) -> str:
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


def draw_circle(draw: ImageDraw.ImageDraw, cx: float, cy: float, radius: float, color: str) -> None:
    draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], fill=color)


def draw_rect(draw: ImageDraw.ImageDraw, x: float, y: float, width: float, height: float, color: str) -> None:
    draw.rectangle([x, y, x + width, y + height], fill=color)


def draw_round_symbol(draw: ImageDraw.ImageDraw, x: float, y: float, color: str) -> None:
    for offset in (0, 10, 20):
        draw_circle(draw, x + offset, y, 3, color)


def draw_yellow_rectangles(draw: ImageDraw.ImageDraw, x: float, y: float, color: str) -> None:
    for offset in (0, 10, 20):
        draw_rect(draw, x + offset, y, 6, 18, color)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    color: str,
    width: int = 3,
) -> None:
    arrow_color = "#222222"
    draw.line([(x1, y1), (x2, y2)], fill=arrow_color, width=width)
    angle = math.atan2(y2 - y1, x2 - x1)
    arm = 10
    for delta in (2.5, -2.5):
        ax = x2 - arm * math.cos(angle + delta)
        ay = y2 - arm * math.sin(angle + delta)
        draw.line([(x2, y2), (ax, ay)], fill=arrow_color, width=width)


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Tuple[float, float]],
    color: str,
    width: int = 3,
    dash: int = 10,
    gap: int = 6,
) -> None:
    for start, end in zip(points, points[1:]):
        x1, y1 = start
        x2, y2 = end
        length = math.hypot(x2 - x1, y2 - y1)
        if length == 0:
            continue
        ux = (x2 - x1) / length
        uy = (y2 - y1) / length
        dist = 0.0
        while dist < length:
            seg_start = dist
            seg_end = min(dist + dash, length)
            sx = x1 + ux * seg_start
            sy = y1 + uy * seg_start
            ex = x1 + ux * seg_end
            ey = y1 + uy * seg_end
            draw.line([(sx, sy), (ex, ey)], fill=color, width=width)
            dist += dash + gap


def sentence_shuffle(lines: Sequence[str], seed: int) -> List[str]:
    rng = random.Random(seed)
    items = list(lines)
    rng.shuffle(items)
    return items


def inject_noise_into_scaffold(probe: Dict[str, object], seed: int) -> str:
    rng = random.Random(seed)
    lines = list(probe["panel_percept_lines"])
    text = " ".join(lines)
    replacements: Dict[str, str] = {}
    for category, palette in (
        ("agent", AGENT_COLORS),
        ("object", OBJECT_COLORS),
        ("container", CONTAINER_COLORS),
    ):
        names = list(dict.fromkeys(probe["entity_catalog"][category]))
        palette_names = [name for _, name in palette]
        for name in names:
            if rng.random() < 0.30:
                candidates = [candidate for candidate in palette_names if candidate != name]
                replacements[name] = rng.choice(candidates)
    for src, dst in replacements.items():
        text = re.sub(rf"\b{re.escape(src)}\b", dst, text)
    return text


def format_prompt(prefix: Optional[str], scaffold: Optional[str], task_prompt: str, answer_instruction: str) -> str:
    parts = []
    if prefix:
        parts.append(prefix.strip())
    if scaffold:
        parts.append(scaffold.strip())
    parts.append(task_prompt.strip())
    parts.append(answer_instruction.strip())
    return " ".join(part for part in parts if part)


def make_paraphrases(probe: Dict[str, object]) -> Dict[str, str]:
    task_prompt = str(probe["task_prompt"])
    answer_instruction = str(probe["answer_instruction"])
    panel_count = int(probe["panel_count"])
    original = str(probe["question_A"])
    restructured = (
        f"This image has {panel_count} panels showing colored figures, containers, and smaller objects. "
        f"Follow the sequence carefully before answering: {task_prompt} {answer_instruction}"
    )
    vocabulary = replace_vocabulary(original)
    distractor = f"The background is white. {original}"
    return {
        "original": original,
        "restructured": restructured,
        "vocabulary": vocabulary,
        "distractor": distractor,
    }


def response_contains_any(text: str, aliases: Sequence[str]) -> bool:
    norm = normalize_text(text)
    return any(alias in norm for alias in aliases)


def safe_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.nanmean(arr))


class VisionParser:
    """Pixel-only parser used for B_auto."""

    def __init__(self) -> None:
        self.palette = {}
        for hex_code, name in AGENT_COLORS + OBJECT_COLORS + CONTAINER_COLORS:
            self.palette[name] = hex_to_rgb(hex_code)
        self.agent_min = 400
        self.agent_max = 2000
        self.object_min = 100
        self.object_max = 550
        self.container_min = 1500

    def _foreground_mask(self, panel: np.ndarray) -> np.ndarray:
        arr = panel.astype(np.int32)
        r = arr[:, :, 0]
        g = arr[:, :, 1]
        b = arr[:, :, 2]
        white = np.all(arr > 210, axis=2)
        black = np.all(arr < 50, axis=2)
        mean = np.mean(arr, axis=2)
        spread = np.max(arr, axis=2) - np.min(arr, axis=2)
        gray = (spread < 30) & (mean > 80) & (mean < 210)
        brown = (
            (np.abs(r - DOOR_RGB[0]) < 25)
            & (np.abs(g - DOOR_RGB[1]) < 25)
            & (np.abs(b - DOOR_RGB[2]) < 25)
        )
        return ~(white | black | gray | brown)

    def _connected_components(self, panel: np.ndarray) -> List[Dict[str, object]]:
        h, w = panel.shape[:2]
        arr = panel.astype(np.int32)
        fg = self._foreground_mask(panel)
        visited = np.zeros((h, w), dtype=bool)
        blobs: List[Dict[str, object]] = []
        ys, xs = np.where(fg)
        for sy, sx in zip(ys.tolist(), xs.tolist()):
            if visited[sy, sx]:
                continue
            queue: deque[Tuple[int, int]] = deque([(sy, sx)])
            visited[sy, sx] = True
            pixels: List[Tuple[int, int]] = []
            seed_rgb = arr[sy, sx]
            while queue:
                cy, cx = queue.popleft()
                pixels.append((cy, cx))
                if len(pixels) > 6000:
                    break
                for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    ny = cy + dy
                    nx = cx + dx
                    if ny < 0 or ny >= h or nx < 0 or nx >= w:
                        continue
                    if visited[ny, nx] or not fg[ny, nx]:
                        continue
                    if int(np.max(np.abs(arr[ny, nx] - seed_rgb))) >= 60:
                        continue
                    visited[ny, nx] = True
                    queue.append((ny, nx))
            if len(pixels) < 40:
                continue
            ys_blob = [p[0] for p in pixels]
            xs_blob = [p[1] for p in pixels]
            area = len(pixels)
            min_x = min(xs_blob)
            max_x = max(xs_blob)
            min_y = min(ys_blob)
            max_y = max(ys_blob)
            bbox_w = max_x - min_x + 1
            bbox_h = max_y - min_y + 1
            aspect = bbox_w / max(bbox_h, 1)
            sample = np.array([arr[y, x] for y, x in pixels[: min(200, len(pixels))]])
            avg = tuple(int(v) for v in sample.mean(axis=0))
            best_name = "unknown"
            best_dist = 999.0
            for name, rgb in self.palette.items():
                dist = rgb_dist(avg, rgb)
                if dist < best_dist:
                    best_name = name
                    best_dist = dist
            circularity = area / max(bbox_w * bbox_h, 1)
            is_circle = 0.55 <= circularity <= 0.90 and 0.65 <= aspect <= 1.35
            role = "other"
            if is_circle and self.agent_min <= area <= self.agent_max:
                role = "agent"
            elif is_circle and self.object_min <= area <= self.object_max:
                role = "object"
            elif area >= self.container_min and not is_circle:
                role = "container"
            elif is_circle and area < self.object_min:
                role = "tiny"
            blobs.append(
                {
                    "name": best_name if best_dist < 80 else "unknown",
                    "cx": float(np.mean(xs_blob)),
                    "cy": float(np.mean(ys_blob)),
                    "area": area,
                    "role": role,
                }
            )
        return blobs

    def parse(self, image_path: str) -> Tuple[List[Dict[str, object]], str]:
        image = Image.open(image_path).convert("RGB")
        arr = np.array(image)
        height, width, _ = arr.shape
        panel_width = height
        panel_count = max(1, round(width / panel_width))
        panel_width = width // panel_count
        parsed_panels: List[Dict[str, object]] = []
        for panel_index in range(panel_count):
            start = panel_index * panel_width
            end = min((panel_index + 1) * panel_width, width)
            panel = arr[:, start:end, :]
            blobs = self._connected_components(panel)
            info = {"panel": panel_index + 1, "agents": [], "objects": [], "containers": []}
            for blob in blobs:
                side = "left" if blob["cx"] < panel.shape[1] * 0.5 else "right"
                vertical = "upper" if blob["cy"] < panel.shape[0] * 0.45 else "lower"
                entry = {"name": blob["name"], "side": side, "vertical": vertical}
                if blob["role"] == "agent":
                    info["agents"].append(entry)
                elif blob["role"] == "object":
                    info["objects"].append(entry)
                elif blob["role"] == "container":
                    info["containers"].append(entry)
            parsed_panels.append(info)
        lines = []
        for info in parsed_panels:
            parts = [f"Panel {info['panel']}:"]
            if info["agents"]:
                agent_text = ", ".join(
                    f"{entry['name']} agent ({entry['side']}-{entry['vertical']})"
                    for entry in sorted(info["agents"], key=lambda x: (x["name"], x["side"], x["vertical"]))
                )
                parts.append(f"agents: {agent_text}.")
            if info["objects"]:
                object_text = ", ".join(
                    f"{entry['name']} object ({entry['side']}-{entry['vertical']})"
                    for entry in sorted(info["objects"], key=lambda x: (x["name"], x["side"], x["vertical"]))
                )
                parts.append(f"objects: {object_text}.")
            if info["containers"]:
                container_text = ", ".join(
                    f"{entry['name']} container ({entry['side']}-{entry['vertical']})"
                    for entry in sorted(info["containers"], key=lambda x: (x["name"], x["side"], x["vertical"]))
                )
                parts.append(f"containers: {container_text}.")
            lines.append(" ".join(parts))
        return parsed_panels, " ".join(lines)


class ProbeGenerator:
    def __init__(self, variants: int) -> None:
        self.variants = variants
        self.panel_w = 200
        self.panel_h = 200
        self.agent_r = 18
        self.object_r = 10
        self.stage1_r = 12
        self.container_w = 58
        self.container_h = 38

    def _rng(self, template: str, variant_index: int) -> random.Random:
        template_seed = 10_000 * (ALL_TEMPLATES.index(template) + 1)
        return random.Random(template_seed + variant_index)

    def _make_image(self, panel_count: int) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
        width = self.panel_w * panel_count
        image = Image.new("RGB", (width, self.panel_h), "white")
        draw = ImageDraw.Draw(image)
        for panel_idx in range(1, panel_count):
            x = panel_idx * self.panel_w
            draw.line([(x, 0), (x, self.panel_h)], fill="#DDDDDD", width=2)
        return image, draw

    def _px(self, panel_index: int, rel_x: float) -> float:
        return panel_index * self.panel_w + rel_x * self.panel_w

    def _py(self, rel_y: float) -> float:
        return rel_y * self.panel_h

    def _draw_single_room_base(
        self,
        draw: ImageDraw.ImageDraw,
        panel_index: int,
        left_color: str,
        right_color: str,
    ) -> None:
        x0 = self._px(panel_index, 0.03)
        y0 = 3
        x1 = self._px(panel_index, 0.97)
        y1 = self.panel_h - 3
        draw.rectangle([x0, y0, x1, y1], outline="black", width=2)
        draw_rect(draw, self._px(panel_index, 0.10), self._py(0.69), self.container_w, self.container_h, left_color)
        draw_rect(draw, self._px(panel_index, 0.61), self._py(0.69), self.container_w, self.container_h, right_color)
        draw_rect(draw, self._px(panel_index, 0.90), self._py(0.36), 16, 48, "#8D6E63")

    def _draw_two_room_base(
        self,
        draw: ImageDraw.ImageDraw,
        panel_index: int,
        left_color: str,
        right_color: str,
    ) -> None:
        x0 = self._px(panel_index, 0.03)
        x1 = self._px(panel_index, 0.97)
        y0 = 3
        y1 = self.panel_h - 3
        draw.rectangle([x0, y0, x1, y1], outline="black", width=2)
        divider = self._px(panel_index, 0.50)
        draw.line([(divider, y0), (divider, y1)], fill="black", width=2)
        draw_rect(draw, self._px(panel_index, 0.12), self._py(0.69), 50, 36, left_color)
        draw_rect(draw, self._px(panel_index, 0.63), self._py(0.69), 50, 36, right_color)
        draw_rect(draw, self._px(panel_index, 0.90), self._py(0.36), 16, 48, "#8D6E63")

    def _draw_panel_agent(self, draw: ImageDraw.ImageDraw, panel_index: int, rel_x: float, rel_y: float, color: str) -> None:
        draw_circle(draw, self._px(panel_index, rel_x), self._py(rel_y), self.agent_r, color)

    def _draw_panel_object(self, draw: ImageDraw.ImageDraw, panel_index: int, rel_x: float, rel_y: float, color: str, radius: Optional[int] = None) -> None:
        draw_circle(draw, self._px(panel_index, rel_x), self._py(rel_y), radius or self.object_r, color)

    def _panel_agent_truth(self, agents: Sequence[str]) -> List[str]:
        return sorted(set(agents))

    def _stage1_meta(
        self,
        probe_id: str,
        template: str,
        variant_index: int,
        seed: int,
        panel_count: int,
        task_prompt: str,
        answer_instruction: str,
        panel_percept_lines: Sequence[str],
        text_narration: str,
        expected: str,
        answer_kind: str,
        expected_aliases: Sequence[str],
        actual_aliases: Optional[Sequence[str]] = None,
        entity_catalog: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, object]:
        question_a = f"Study this {panel_count}-panel visual sequence. {task_prompt} {answer_instruction}"
        return {
            "id": probe_id,
            "template": template,
            "stage": 1,
            "variant_index": variant_index,
            "seed": seed,
            "panel_count": panel_count,
            "task_prompt": task_prompt,
            "answer_instruction": answer_instruction,
            "question_A": question_a,
            "panel_percept_lines": list(panel_percept_lines),
            "scaffold_percept": " ".join(panel_percept_lines),
            "scaffold_full": " ".join(panel_percept_lines),
            "text_narration": text_narration,
            "question_C": f"{text_narration} {task_prompt} {answer_instruction}",
            "answer_kind": answer_kind,
            "expected": expected,
            "expected_aliases": list(expected_aliases),
            "actual_aliases": list(actual_aliases or []),
            "entity_catalog": entity_catalog or {"agent": [], "object": [], "container": []},
            "ground_truth_panels": [],
            "parser_agent_all_correct": None,
        }

    def _stage2_meta(self, *args, **kwargs) -> Dict[str, object]:
        meta = self._stage1_meta(*args, **kwargs)
        meta["stage"] = 2
        return meta

    def _stage3_meta(
        self,
        probe_id: str,
        template: str,
        variant_index: int,
        seed: int,
        panel_count: int,
        task_prompt: str,
        answer_instruction: str,
        panel_percept_lines: Sequence[str],
        full_extra_lines: Sequence[str],
        text_narration: str,
        expected: str,
        answer_kind: str,
        expected_aliases: Sequence[str],
        actual_aliases: Optional[Sequence[str]],
        entity_catalog: Dict[str, List[str]],
        ground_truth_panels: Sequence[Sequence[str]],
        probe_specific: Dict[str, object],
    ) -> Dict[str, object]:
        question_a = f"Study this {panel_count}-panel visual sequence. {task_prompt} {answer_instruction}"
        scaffold_full_lines = list(panel_percept_lines) + list(full_extra_lines)
        return {
            "id": probe_id,
            "template": template,
            "stage": 3,
            "variant_index": variant_index,
            "seed": seed,
            "panel_count": panel_count,
            "task_prompt": task_prompt,
            "answer_instruction": answer_instruction,
            "question_A": question_a,
            "panel_percept_lines": list(panel_percept_lines),
            "scaffold_percept": " ".join(panel_percept_lines),
            "scaffold_full": " ".join(scaffold_full_lines),
            "full_extra_lines": list(full_extra_lines),
            "text_narration": text_narration,
            "question_C": f"{text_narration} {task_prompt} {answer_instruction}",
            "answer_kind": answer_kind,
            "expected": expected,
            "expected_aliases": list(expected_aliases),
            "actual_aliases": list(actual_aliases or []),
            "entity_catalog": entity_catalog,
            "ground_truth_panels": [sorted(set(panel)) for panel in ground_truth_panels],
            "parser_agent_all_correct": False,
            **probe_specific,
        }

    def _build_sally_anne(self, template: str, variant_index: int) -> Tuple[Image.Image, Dict[str, object]]:
        rng = self._rng(template, variant_index)
        agents = choose_distinct(AGENT_COLORS, 2, rng)
        obj = rng.choice(OBJECT_COLORS)
        containers = choose_distinct(CONTAINER_COLORS, 2, rng)
        agent1_hex, agent1_name = agents[0]
        agent2_hex, agent2_name = agents[1]
        object_hex, object_name = obj
        left_hex, left_name = containers[0]
        right_hex, right_name = containers[1]
        panel_count = 6
        image, draw = self._make_image(panel_count)
        for panel in range(panel_count):
            self._draw_single_room_base(draw, panel, left_hex, right_hex)
        
        self._draw_panel_agent(draw, 0, 0.25, 0.27, agent1_hex)
        self._draw_panel_agent(draw, 0, 0.72, 0.27, agent2_hex)
        self._draw_panel_object(draw, 0, 0.23, 0.64, object_hex)
        draw_arrow(draw, self._px(0, 0.23), self._py(0.58), self._px(0, 0.23), self._py(0.69), object_hex)
        
        self._draw_panel_agent(draw, 1, 0.90, 0.48, agent1_hex)
        self._draw_panel_agent(draw, 1, 0.72, 0.27, agent2_hex)
        self._draw_panel_object(draw, 1, 0.24, 0.77, object_hex)
        draw_arrow(draw, self._px(1, 0.90), self._py(0.48), self._px(1, 0.98), self._py(0.48), agent1_hex)
        
        self._draw_panel_agent(draw, 2, 0.48, 0.27, agent2_hex)
        self._draw_panel_object(draw, 2, 0.47, 0.64, object_hex)
        draw_arrow(draw, self._px(2, 0.24), self._py(0.77), self._px(2, 0.69), self._py(0.77), object_hex)
        
        self._draw_panel_agent(draw, 3, 0.72, 0.27, agent2_hex)
        self._draw_panel_object(draw, 3, 0.69, 0.77, object_hex)
        
        self._draw_panel_agent(draw, 4, 0.90, 0.48, agent1_hex)
        self._draw_panel_agent(draw, 4, 0.72, 0.27, agent2_hex)
        self._draw_panel_object(draw, 4, 0.69, 0.77, object_hex)
        draw_arrow(draw, self._px(4, 0.98), self._py(0.48), self._px(4, 0.90), self._py(0.48), agent1_hex)
        
        self._draw_panel_agent(draw, 5, 0.25, 0.27, agent1_hex)
        self._draw_panel_agent(draw, 5, 0.72, 0.27, agent2_hex)
        self._draw_panel_object(draw, 5, 0.69, 0.77, object_hex)

        panel_lines = [
            f"Panel 1: {agent1_name} and {agent2_name} characters are in the room. {object_name} object is near the left {left_name} container. {agent1_name} character places the object there.",
            f"Panel 2: {agent1_name} character moves to the door and exits. {agent2_name} character remains in the room. {object_name} object is in the left {left_name} container.",
            f"Panel 3: Only {agent2_name} character is in the room. {object_name} object moves from the left {left_name} container toward the right {right_name} container.",
            f"Panel 4: {agent2_name} character is in the room. {object_name} object is in the right {right_name} container.",
            f"Panel 5: {agent1_name} character returns through the door. {agent2_name} character is in the room. {object_name} object is in the right {right_name} container.",
            f"Panel 6: {agent1_name} and {agent2_name} characters are both in the room. {object_name} object is in the right {right_name} container.",
        ]
        full_extra = [
            f"{agent1_name} character was outside when the object moved from left to right and did not see that move."
        ]
        text_narration = (
            f"{agent1_name} and {agent2_name} are in a room with a left {left_name} container and a right {right_name} container. "
            f"{agent1_name} places the {object_name} object in the left container and then leaves. "
            f"While {agent1_name} is away, {agent2_name} moves the object to the right container. "
            f"{agent1_name} returns after the move."
        )
        expected = "left"
        answer_kind = "left_right"
        expected_aliases = ["left", "left box", "left container", left_name]
        actual_aliases = ["right", "right box", "right container", right_name]
        probe_specific = {
            "agent1_name": agent1_name,
            "agent2_name": agent2_name,
            "object_name": object_name,
            "left_container_name": left_name,
            "right_container_name": right_name,
        }
        if template == "sally_anne_search":
            task_prompt = (
                f"Where will the {agent1_name} character search first for the {object_name} object, left or right?"
            )
        elif template == "sally_anne_belief":
            task_prompt = (
                f"Where does the {agent1_name} character believe the {object_name} object is, left or right?"
            )
        elif template == "sally_anne_reality":
            task_prompt = f"Where is the {object_name} object actually located in the final panel, left or right?"
            expected = "right"
            expected_aliases = ["right", "right box", "right container", right_name]
            actual_aliases = ["right", right_name]
        elif template == "reality_vs_belief":
            task_prompt = (
                f"Answer two questions about the final panel. Actually: where is the {object_name} object? "
                f"Belief: where does the {agent1_name} character believe it is?"
            )
            expected = "actually right belief left"
            answer_kind = "belief_split"
            expected_aliases = ["actually", "belief", "left", "right", left_name, right_name]
        else:
            raise ValueError(template)
        answer_instruction = (
            "Answer ONLY 'left' or 'right'."
            if answer_kind != "belief_split"
            else "Format exactly as 'Actually: [left/right]. Belief: [left/right].'"
        )
        metadata = self._stage3_meta(
            probe_id=f"s3_{template}_{variant_index:03d}",
            template=template,
            variant_index=variant_index,
            seed=10_000 * (ALL_TEMPLATES.index(template) + 1) + variant_index,
            panel_count=panel_count,
            task_prompt=task_prompt,
            answer_instruction=answer_instruction,
            panel_percept_lines=panel_lines,
            full_extra_lines=full_extra,
            text_narration=text_narration,
            expected=expected,
            answer_kind=answer_kind,
            expected_aliases=expected_aliases,
            actual_aliases=actual_aliases,
            entity_catalog={
                "agent": [agent1_name, agent2_name],
                "object": [object_name],
                "container": [left_name, right_name],
            },
            ground_truth_panels=[
                [agent1_name, agent2_name],
                [agent1_name, agent2_name],
                [agent2_name],
                [agent2_name],
                [agent1_name, agent2_name],
                [agent1_name, agent2_name],
            ],
            probe_specific=probe_specific,
        )
        return image, metadata

    def _build_knowledge(self, template: str, variant_index: int) -> Tuple[Image.Image, Dict[str, object]]:
        rng = self._rng(template, variant_index)
        agents = choose_distinct(AGENT_COLORS, 2, rng)
        obj = rng.choice(OBJECT_COLORS)
        containers = choose_distinct(CONTAINER_COLORS, 2, rng)
        agent1_hex, agent1_name = agents[0]
        agent2_hex, agent2_name = agents[1]
        object_hex, object_name = obj
        left_hex, left_name = containers[0]
        right_hex, right_name = containers[1]
        panel_count = 5
        image, draw = self._make_image(panel_count)
        for panel in range(panel_count):
            self._draw_single_room_base(draw, panel, left_hex, right_hex)
        self._draw_panel_agent(draw, 0, 0.25, 0.27, agent1_hex)
        self._draw_panel_agent(draw, 0, 0.72, 0.27, agent2_hex)
        self._draw_panel_object(draw, 0, 0.24, 0.77, object_hex)
        self._draw_panel_agent(draw, 1, 0.25, 0.27, agent1_hex)
        self._draw_panel_agent(draw, 1, 0.90, 0.48, agent2_hex)
        self._draw_panel_object(draw, 1, 0.24, 0.77, object_hex)
        draw_arrow(draw, self._px(1, 0.90), self._py(0.48), self._px(1, 0.98), self._py(0.48), agent2_hex)
        self._draw_panel_agent(draw, 2, 0.48, 0.27, agent1_hex)
        self._draw_panel_object(draw, 2, 0.47, 0.64, object_hex)
        draw_arrow(draw, self._px(2, 0.24), self._py(0.77), self._px(2, 0.69), self._py(0.77), object_hex)
        self._draw_panel_agent(draw, 3, 0.25, 0.27, agent1_hex)
        self._draw_panel_object(draw, 3, 0.69, 0.77, object_hex)
        self._draw_panel_agent(draw, 4, 0.25, 0.27, agent1_hex)
        self._draw_panel_agent(draw, 4, 0.72, 0.27, agent2_hex)
        self._draw_panel_object(draw, 4, 0.69, 0.77, object_hex)

        panel_lines = [
            f"Panel 1: {agent1_name} and {agent2_name} characters are in the room. {object_name} object is in the left {left_name} container.",
            f"Panel 2: {agent2_name} character exits through the door. {agent1_name} character remains. {object_name} object is in the left {left_name} container.",
            f"Panel 3: Only {agent1_name} character is in the room. {object_name} object moves from the left {left_name} container toward the right {right_name} container.",
            f"Panel 4: {agent1_name} character is in the room. {object_name} object is in the right {right_name} container.",
            f"Panel 5: {agent2_name} character returns. {agent1_name} character is in the room. {object_name} object is in the right {right_name} container.",
        ]
        full_extra = [
            f"{agent2_name} character did not see the move, so {agent2_name} still has the old belief. "
            f"{agent1_name} knows the move happened and knows that {agent2_name} missed it."
        ]
        text_narration = (
            f"{agent1_name} and {agent2_name} both see the {object_name} object in the left {left_name} container. "
            f"{agent2_name} leaves. While {agent2_name} is away, {agent1_name} moves the object to the right {right_name} container. "
            f"{agent2_name} returns after the move."
        )
        answer_instruction = "Answer ONLY 'left' or 'right'."
        expected = "left"
        answer_kind = "left_right"
        expected_aliases = ["left", "left box", "left container", left_name]
        actual_aliases = ["right", "right box", "right container", right_name]
        task_prompt = f"Where does the {agent2_name} character think the {object_name} object is, left or right?"
        if template == "knowledge_who_knows":
            task_prompt = "Which character knows the object's current location, the one who stayed or the one who left?"
            answer_instruction = "Answer ONLY 'stayed' or 'left'."
            expected = "stayed"
            answer_kind = "stayed_left"
            expected_aliases = ["stayed", agent1_name]
            actual_aliases = ["stayed", agent1_name]
        elif template == "knowledge_belief":
            pass
        elif template == "perspectives_differ":
            task_prompt = "Do the two characters have the same or different beliefs about the object's location?"
            answer_instruction = "Answer ONLY 'same' or 'different'."
            expected = "different"
            answer_kind = "same_different"
            expected_aliases = ["different"]
            actual_aliases = ["same"]
        elif template == "second_order":
            task_prompt = (
                f"Does the {agent1_name} character know that the {agent2_name} character has a wrong belief about the object's location, yes or no?"
            )
            answer_instruction = "Answer ONLY 'yes' or 'no'."
            expected = "yes"
            answer_kind = "yes_no"
            expected_aliases = ["yes"]
            actual_aliases = ["no"]
        else:
            raise ValueError(template)
        metadata = self._stage3_meta(
            probe_id=f"s3_{template}_{variant_index:03d}",
            template=template,
            variant_index=variant_index,
            seed=10_000 * (ALL_TEMPLATES.index(template) + 1) + variant_index,
            panel_count=panel_count,
            task_prompt=task_prompt,
            answer_instruction=answer_instruction,
            panel_percept_lines=panel_lines,
            full_extra_lines=full_extra,
            text_narration=text_narration,
            expected=expected,
            answer_kind=answer_kind,
            expected_aliases=expected_aliases,
            actual_aliases=actual_aliases,
            entity_catalog={
                "agent": [agent1_name, agent2_name],
                "object": [object_name],
                "container": [left_name, right_name],
            },
            ground_truth_panels=[
                [agent1_name, agent2_name],
                [agent1_name, agent2_name],
                [agent1_name],
                [agent1_name],
                [agent1_name, agent2_name],
            ],
            probe_specific={
                "agent1_name": agent1_name,
                "agent2_name": agent2_name,
                "object_name": object_name,
                "left_container_name": left_name,
                "right_container_name": right_name,
            },
        )
        return image, metadata

    def _build_unexpected_contents(self, variant_index: int) -> Tuple[Image.Image, Dict[str, object]]:
        template = "unexpected_contents"
        rng = self._rng(template, variant_index)
        agents = choose_distinct(AGENT_COLORS, 2, rng)
        object_hex, object_name = rng.choice(OBJECT_COLORS)
        box_hex, box_name = rng.choice(CONTAINER_COLORS)
        agent1_hex, agent1_name = agents[0]
        agent2_hex, agent2_name = agents[1]
        panel_count = 5
        image, draw = self._make_image(panel_count)
        for panel in range(panel_count):
            self._draw_single_room_base(draw, panel, box_hex, box_hex)
        
        for panel in range(panel_count):
            draw.rectangle(
                [self._px(panel, 0.58), self._py(0.66), self._px(panel, 0.95), self._py(0.93)],
                fill="white",
                outline="white",
            )
            draw_rect(draw, self._px(panel, 0.31), self._py(0.63), 70, 52, box_hex)
        
        self._draw_panel_agent(draw, 0, 0.18, 0.27, agent1_hex)
        draw_round_symbol(draw, self._px(0, 0.38), self._py(0.70), "#222222")
        
        self._draw_panel_agent(draw, 1, 0.18, 0.27, agent1_hex)
        draw.rectangle(
            [self._px(1, 0.31), self._py(0.63), self._px(1, 0.66), self._py(0.69)],
            fill=box_hex,
            outline=box_hex,
        )
        draw.rectangle(
            [self._px(1, 0.31), self._py(0.69), self._px(1, 0.66), self._py(0.86)],
            fill="#FFF8E1",
            outline="black",
        )
        draw_yellow_rectangles(draw, self._px(1, 0.38), self._py(0.72), object_hex)
        
        self._draw_panel_agent(draw, 2, 0.18, 0.27, agent1_hex)
        draw_round_symbol(draw, self._px(2, 0.38), self._py(0.70), "#222222")
        
        self._draw_panel_agent(draw, 3, 0.74, 0.27, agent2_hex)
        draw_round_symbol(draw, self._px(3, 0.38), self._py(0.70), "#222222")
        
        self._draw_panel_agent(draw, 4, 0.52, 0.27, agent2_hex)
        draw_round_symbol(draw, self._px(4, 0.38), self._py(0.70), "#222222")

        panel_lines = [
            f"Panel 1: {agent1_name} character sees a closed {box_name} box marked with round symbols.",
            f"Panel 2: {agent1_name} character opens the {box_name} box. Inside are {object_name} rectangular contents, not round objects.",
            f"Panel 3: The {box_name} box is closed again and still shows the round symbols.",
            f"Panel 4: {agent2_name} character arrives and sees only the closed {box_name} box with round symbols.",
            f"Panel 5: {agent2_name} character is still looking at the closed {box_name} box.",
        ]
        full_extra = [
            f"{agent2_name} character never saw the box opened and only saw the closed box with the round symbols."
        ]
        text_narration = (
            f"A {box_name} box has round symbols on it. {agent1_name} opens it and finds {object_name} rectangular contents inside. "
            f"The box is closed again. Later, {agent2_name} arrives and sees only the closed box with the round symbols."
        )
        metadata = self._stage3_meta(
            probe_id=f"s3_{template}_{variant_index:03d}",
            template=template,
            variant_index=variant_index,
            seed=10_000 * (ALL_TEMPLATES.index(template) + 1) + variant_index,
            panel_count=panel_count,
            task_prompt=f"What does the {agent2_name} character think is inside the box, round objects or yellow rectangles?",
            answer_instruction="Answer ONLY 'round objects' or 'yellow rectangles'.",
            panel_percept_lines=panel_lines,
            full_extra_lines=full_extra,
            text_narration=text_narration,
            expected="round objects",
            answer_kind="round_rectangles",
            expected_aliases=["round", "round objects", "symbols"],
            actual_aliases=["yellow rectangles", "rectangles", object_name],
            entity_catalog={
                "agent": [agent1_name, agent2_name],
                "object": [object_name],
                "container": [box_name],
            },
            ground_truth_panels=[
                [agent1_name],
                [agent1_name],
                [agent1_name],
                [agent2_name],
                [agent2_name],
            ],
            probe_specific={
                "agent1_name": agent1_name,
                "agent2_name": agent2_name,
                "object_name": object_name,
                "box_name": box_name,
            },
        )
        return image, metadata

    def _build_two_rooms(self, variant_index: int) -> Tuple[Image.Image, Dict[str, object]]:
        template = "two_rooms"
        rng = self._rng(template, variant_index)
        agents = choose_distinct(AGENT_COLORS, 2, rng)
        object_hex, object_name = rng.choice(OBJECT_COLORS)
        containers = choose_distinct(CONTAINER_COLORS, 2, rng)
        agent1_hex, agent1_name = agents[0]
        agent2_hex, agent2_name = agents[1]
        left_hex, left_name = containers[0]
        right_hex, right_name = containers[1]
        panel_count = 5
        image, draw = self._make_image(panel_count)
        for panel in range(panel_count):
            self._draw_two_room_base(draw, panel, left_hex, right_hex)
        self._draw_panel_agent(draw, 0, 0.20, 0.27, agent1_hex)
        self._draw_panel_agent(draw, 0, 0.76, 0.27, agent2_hex)
        self._draw_panel_object(draw, 0, 0.18, 0.77, object_hex)
        self._draw_panel_agent(draw, 1, 0.46, 0.34, agent1_hex)
        self._draw_panel_agent(draw, 1, 0.76, 0.27, agent2_hex)
        self._draw_panel_object(draw, 1, 0.43, 0.63, object_hex)
        draw_arrow(draw, self._px(1, 0.26), self._py(0.63), self._px(1, 0.64), self._py(0.63), object_hex)
        self._draw_panel_agent(draw, 2, 0.72, 0.27, agent1_hex)
        self._draw_panel_agent(draw, 2, 0.90, 0.48, agent2_hex)
        self._draw_panel_object(draw, 2, 0.68, 0.63, object_hex)
        draw_arrow(draw, self._px(2, 0.90), self._py(0.48), self._px(2, 0.98), self._py(0.48), agent2_hex)
        self._draw_panel_agent(draw, 3, 0.72, 0.27, agent1_hex)
        self._draw_panel_object(draw, 3, 0.69, 0.77, object_hex)
        self._draw_panel_agent(draw, 4, 0.72, 0.27, agent1_hex)
        self._draw_panel_agent(draw, 4, 0.82, 0.27, agent2_hex)
        self._draw_panel_object(draw, 4, 0.69, 0.77, object_hex)

        panel_lines = [
            f"Panel 1: {agent1_name} character is in the left room with the {object_name} object near the left {left_name} container. {agent2_name} character is in the right room.",
            f"Panel 2: {agent1_name} character carries the {object_name} object from the left room toward the right room. {agent2_name} character can see this transfer.",
            f"Panel 3: {agent2_name} character exits from the right room. {agent1_name} character and the {object_name} object are in the right room.",
            f"Panel 4: Only {agent1_name} character is present. {object_name} object is placed in the right {right_name} container.",
            f"Panel 5: {agent2_name} character returns to the right room. {object_name} object is in the right {right_name} container.",
        ]
        full_extra = [
            f"{agent2_name} character left before the object was placed in the right container, but had already seen the object taken into the right room."
        ]
        text_narration = (
            f"{agent1_name} starts in the left room with the {object_name} object. {agent2_name} is in the right room. "
            f"{agent1_name} carries the object into the right room while {agent2_name} is watching. "
            f"{agent2_name} then leaves. While {agent2_name} is gone, {agent1_name} places the object in the right {right_name} container. "
            f"{agent2_name} returns later."
        )
        metadata = self._stage3_meta(
            probe_id=f"s3_{template}_{variant_index:03d}",
            template=template,
            variant_index=variant_index,
            seed=10_000 * (ALL_TEMPLATES.index(template) + 1) + variant_index,
            panel_count=panel_count,
            task_prompt=f"Where does the {agent2_name} character think the {object_name} object is, left or right?",
            answer_instruction="Answer ONLY 'left' or 'right'.",
            panel_percept_lines=panel_lines,
            full_extra_lines=full_extra,
            text_narration=text_narration,
            expected="right",
            answer_kind="left_right",
            expected_aliases=["right", "right room", "right container", right_name],
            actual_aliases=["right", "right room", "right container", right_name],
            entity_catalog={
                "agent": [agent1_name, agent2_name],
                "object": [object_name],
                "container": [left_name, right_name],
            },
            ground_truth_panels=[
                [agent1_name, agent2_name],
                [agent1_name, agent2_name],
                [agent1_name, agent2_name],
                [agent1_name],
                [agent1_name, agent2_name],
            ],
            probe_specific={
                "agent1_name": agent1_name,
                "agent2_name": agent2_name,
                "object_name": object_name,
                "left_container_name": left_name,
                "right_container_name": right_name,
            },
        )
        return image, metadata

    def _build_occlusion(self, template: str, variant_index: int) -> Tuple[Image.Image, Dict[str, object]]:
        rng = self._rng(template, variant_index)
        object_hex, object_name = rng.choice(OBJECT_COLORS)
        consistent = template == "occlusion_consistent"
        panel_count = 5
        image, draw = self._make_image(panel_count)
        positions = [
            (0.16, 0.50),
            (0.31, 0.50),
            None,
            None,
            (0.80, 0.50) if consistent else (0.18, 0.50),
        ]
        for panel in range(panel_count):
            x0 = self._px(panel, 0.03)
            x1 = self._px(panel, 0.97)
            draw.rectangle([x0, 3, x1, self.panel_h - 3], outline="black", width=2)
            draw_rect(draw, self._px(panel, 0.45), self._py(0.12), 24, self.panel_h - 50, "#888888")
            if positions[panel] is not None:
                self._draw_panel_object(draw, panel, positions[panel][0], positions[panel][1], object_hex, radius=self.stage1_r)
        panel_lines = [
            f"Panel 1: {object_name} circle is on the left side of the scene.",
            f"Panel 2: {object_name} circle moves right toward the gray barrier.",
            "Panel 3: The circle is hidden behind the barrier.",
            "Panel 4: The circle is still hidden behind the barrier.",
            f"Panel 5: The circle appears on the {'right' if consistent else 'left'} side of the barrier.",
        ]
        text_narration = (
            f"A {object_name} circle moves rightward toward a gray barrier. It goes behind the barrier and then "
            f"reappears on the {'right' if consistent else 'left'} side."
        )
        metadata = self._stage1_meta(
            probe_id=f"s1_{template}_{variant_index:03d}",
            template=template,
            variant_index=variant_index,
            seed=10_000 * (ALL_TEMPLATES.index(template) + 1) + variant_index,
            panel_count=panel_count,
            task_prompt="Is the final reappearance position consistent with the earlier direction of motion, yes or no?",
            answer_instruction="Answer ONLY 'yes' or 'no'.",
            panel_percept_lines=panel_lines,
            text_narration=text_narration,
            expected="yes" if consistent else "no",
            answer_kind="yes_no",
            expected_aliases=["yes"] if consistent else ["no"],
            entity_catalog={"agent": [], "object": [object_name], "container": []},
        )
        return image, metadata

    def _build_count_two(self, variant_index: int) -> Tuple[Image.Image, Dict[str, object]]:
        template = "count_two"
        rng = self._rng(template, variant_index)
        object_hex, object_name = rng.choice(OBJECT_COLORS)
        panel_count = 4
        image, draw = self._make_image(panel_count)
        positions = [
            ((0.18, 0.25), (0.18, 0.75)),
            ((0.34, 0.40), (0.34, 0.60)),
            ((0.50, 0.56), (0.50, 0.44)),
            ((0.66, 0.72), (0.66, 0.28)),
        ]
        for panel in range(panel_count):
            draw.rectangle([self._px(panel, 0.03), 3, self._px(panel, 0.97), self.panel_h - 3], outline="black", width=2)
            for rel_x, rel_y in positions[panel]:
                self._draw_panel_object(draw, panel, rel_x, rel_y, object_hex, radius=self.stage1_r)
        panel_lines = [
            f"Each panel shows two identical {object_name} circles following crossing paths.",
            "One circle moves from upper left toward lower right.",
            "The other circle moves from lower left toward upper right.",
            "Both circles remain visible as separate objects.",
        ]
        metadata = self._stage1_meta(
            probe_id=f"s1_{template}_{variant_index:03d}",
            template=template,
            variant_index=variant_index,
            seed=10_000 * (ALL_TEMPLATES.index(template) + 1) + variant_index,
            panel_count=panel_count,
            task_prompt="How many distinct circular objects are present across the sequence?",
            answer_instruction="Answer ONLY '2'.",
            panel_percept_lines=panel_lines,
            text_narration="Two identical circles move on crossing trajectories while remaining separate objects.",
            expected="2",
            answer_kind="number",
            expected_aliases=["2"],
            entity_catalog={"agent": [], "object": [object_name], "container": []},
        )
        return image, metadata

    def _build_count_three(self, variant_index: int) -> Tuple[Image.Image, Dict[str, object]]:
        template = "count_three"
        rng = self._rng(template, variant_index)
        colors = choose_distinct(OBJECT_COLORS, 3, rng)
        panel_count = 4
        image, draw = self._make_image(panel_count)
        trajectories = [
            [(0.18, 0.25), (0.28, 0.35), (0.38, 0.45), (0.48, 0.55)],
            [(0.18, 0.75), (0.33, 0.65), (0.48, 0.55), (0.63, 0.45)],
            [(0.72, 0.25), (0.62, 0.35), (0.52, 0.45), (0.42, 0.55)],
        ]
        for panel in range(panel_count):
            draw.rectangle([self._px(panel, 0.03), 3, self._px(panel, 0.97), self.panel_h - 3], outline="black", width=2)
            for traj, (hex_code, _) in zip(trajectories, colors):
                self._draw_panel_object(draw, panel, traj[panel][0], traj[panel][1], hex_code, radius=self.stage1_r)
        object_names = [name for _, name in colors]
        panel_lines = [
            f"Each panel shows three circles in different colors: {object_names[0]}, {object_names[1]}, and {object_names[2]}.",
            "All three circles move on distinct trajectories.",
        ]
        metadata = self._stage1_meta(
            probe_id=f"s1_{template}_{variant_index:03d}",
            template=template,
            variant_index=variant_index,
            seed=10_000 * (ALL_TEMPLATES.index(template) + 1) + variant_index,
            panel_count=panel_count,
            task_prompt="How many separate circular objects are present across the sequence?",
            answer_instruction="Answer ONLY '3'.",
            panel_percept_lines=panel_lines,
            text_narration=f"Three distinct circles ({', '.join(object_names)}) move through the scene.",
            expected="3",
            answer_kind="number",
            expected_aliases=["3"],
            entity_catalog={"agent": [], "object": object_names, "container": []},
        )
        return image, metadata

    def _build_permanence(self, variant_index: int) -> Tuple[Image.Image, Dict[str, object]]:
        template = "permanence"
        rng = self._rng(template, variant_index)
        object_hex, object_name = rng.choice(OBJECT_COLORS)
        panel_count = 4
        image, draw = self._make_image(panel_count)
        positions = [(0.18, 0.55), (0.35, 0.55), None, None]
        for panel in range(panel_count):
            draw.rectangle([self._px(panel, 0.03), 3, self._px(panel, 0.97), self.panel_h - 3], outline="black", width=2)
            draw_rect(draw, self._px(panel, 0.45), self._py(0.18), 70, self.panel_h - 72, "#888888")
            if positions[panel] is not None:
                self._draw_panel_object(draw, panel, positions[panel][0], positions[panel][1], object_hex, radius=self.stage1_r)
        panel_lines = [
            f"Panel 1: {object_name} circle is left of a large gray occluder.",
            f"Panel 2: {object_name} circle moves toward the occluder.",
            "Panels 3-4: the circle is not visible because the occluder can hide it completely.",
        ]
        metadata = self._stage1_meta(
            probe_id=f"s1_{template}_{variant_index:03d}",
            template=template,
            variant_index=variant_index,
            seed=10_000 * (ALL_TEMPLATES.index(template) + 1) + variant_index,
            panel_count=panel_count,
            task_prompt="In the final panel, is the circle hidden or gone?",
            answer_instruction="Answer ONLY 'hidden'.",
            panel_percept_lines=panel_lines,
            text_narration=f"A {object_name} circle moves behind a large occluder and is no longer visible.",
            expected="hidden",
            answer_kind="hidden",
            expected_aliases=["hidden", "behind", "occluded"],
            entity_catalog={"agent": [], "object": [object_name], "container": []},
        )
        return image, metadata

    def _build_color_change(self, variant_index: int) -> Tuple[Image.Image, Dict[str, object]]:
        template = "color_change"
        rng = self._rng(template, variant_index)
        first, second = choose_distinct(OBJECT_COLORS, 2, rng)
        first_hex, first_name = first
        second_hex, second_name = second
        panel_count = 5
        image, draw = self._make_image(panel_count)
        positions = [(0.16, 0.50), (0.31, 0.50), None, None, (0.80, 0.50)]
        for panel in range(panel_count):
            draw.rectangle([self._px(panel, 0.03), 3, self._px(panel, 0.97), self.panel_h - 3], outline="black", width=2)
            draw_rect(draw, self._px(panel, 0.45), self._py(0.12), 24, self.panel_h - 50, "#888888")
            if positions[panel] is None:
                continue
            color = first_hex if panel < 2 else second_hex
            self._draw_panel_object(draw, panel, positions[panel][0], positions[panel][1], color, radius=self.stage1_r)
        panel_lines = [
            f"Panels 1-2: a {first_name} circle moves right toward a gray barrier.",
            "Panels 3-4: the circle is hidden behind the barrier.",
            f"Panel 5: a {second_name} circle emerges on the right side of the barrier.",
        ]
        metadata = self._stage1_meta(
            probe_id=f"s1_{template}_{variant_index:03d}",
            template=template,
            variant_index=variant_index,
            seed=10_000 * (ALL_TEMPLATES.index(template) + 1) + variant_index,
            panel_count=panel_count,
            task_prompt="Are the circle colors before and after the barrier the same or different?",
            answer_instruction="Answer ONLY 'different'.",
            panel_percept_lines=panel_lines,
            text_narration=f"A {first_name} circle goes behind a barrier and a {second_name} circle emerges on the other side.",
            expected="different",
            answer_kind="different",
            expected_aliases=["different"],
            entity_catalog={"agent": [], "object": [first_name, second_name], "container": []},
        )
        return image, metadata

    def _build_goal_upper(self, variant_index: int) -> Tuple[Image.Image, Dict[str, object]]:
        template = "goal_upper"
        rng = self._rng(template, variant_index)
        agent_hex, agent_name = rng.choice(AGENT_COLORS)
        targets = choose_distinct(CONTAINER_COLORS, 2, rng)
        upper_hex, upper_name = targets[0]
        lower_hex, lower_name = targets[1]
        panel_count = 4
        image, draw = self._make_image(panel_count)
        positions = [(0.16, 0.62), (0.33, 0.52), (0.50, 0.39), (0.68, 0.28)]
        for panel in range(panel_count):
            draw.rectangle([self._px(panel, 0.03), 3, self._px(panel, 0.97), self.panel_h - 3], outline="black", width=2)
            draw_rect(draw, self._px(panel, 0.77), self._py(0.14), 42, 42, upper_hex)
            draw_rect(draw, self._px(panel, 0.77), self._py(0.70), 42, 42, lower_hex)
            self._draw_panel_agent(draw, panel, positions[panel][0], positions[panel][1], agent_hex)
        panel_lines = [
            f"Panels 1-4: a {agent_name} character moves rightward and curves upward.",
            f"The upper target is {upper_name}. The lower target is {lower_name}.",
            "The trajectory heads toward the upper target.",
        ]
        metadata = self._stage2_meta(
            probe_id=f"s2_{template}_{variant_index:03d}",
            template=template,
            variant_index=variant_index,
            seed=10_000 * (ALL_TEMPLATES.index(template) + 1) + variant_index,
            panel_count=panel_count,
            task_prompt="Which target is the character heading toward, upper or lower?",
            answer_instruction="Answer ONLY 'upper' or 'lower'.",
            panel_percept_lines=panel_lines,
            text_narration=f"A {agent_name} character curves toward the upper of two targets.",
            expected="upper",
            answer_kind="upper_lower",
            expected_aliases=["upper", "top", upper_name],
            entity_catalog={"agent": [agent_name], "object": [], "container": [upper_name, lower_name]},
        )
        return image, metadata

    def _build_detour_obstacle(self, variant_index: int) -> Tuple[Image.Image, Dict[str, object]]:
        template = "detour_obstacle"
        rng = self._rng(template, variant_index)
        agent_hex, agent_name = rng.choice(AGENT_COLORS)
        target_hex, target_name = rng.choice(CONTAINER_COLORS)
        panel_count = 5
        image, draw = self._make_image(panel_count)
        positions = [(0.16, 0.60), (0.28, 0.43), (0.43, 0.28), (0.62, 0.36), (0.78, 0.60)]
        for panel in range(panel_count):
            draw.rectangle([self._px(panel, 0.03), 3, self._px(panel, 0.97), self.panel_h - 3], outline="black", width=2)
            draw_rect(draw, self._px(panel, 0.44), self._py(0.28), 34, 90, "#888888")
            draw_rect(draw, self._px(panel, 0.78), self._py(0.55), 42, 42, target_hex)
            self._draw_panel_agent(draw, panel, positions[panel][0], positions[panel][1], agent_hex)
        panel_lines = [
            f"A {agent_name} character moves toward the {target_name} target.",
            "A gray obstacle blocks the direct path.",
            "The trajectory curves around the obstacle before reaching the target.",
        ]
        metadata = self._stage2_meta(
            probe_id=f"s2_{template}_{variant_index:03d}",
            template=template,
            variant_index=variant_index,
            seed=10_000 * (ALL_TEMPLATES.index(template) + 1) + variant_index,
            panel_count=panel_count,
            task_prompt="Why did the character take the curved path instead of going straight: A) prefers curves, B) obstacle blocking, or C) avoiding the target?",
            answer_instruction="Answer ONLY 'A', 'B', or 'C'.",
            panel_percept_lines=panel_lines,
            text_narration=f"A {agent_name} character detours around a gray obstacle to reach the {target_name} target.",
            expected="B",
            answer_kind="choice_abc",
            expected_aliases=["b", "obstacle", "blocking", "blocked"],
            entity_catalog={"agent": [agent_name], "object": [], "container": [target_name]},
        )
        return image, metadata

    def _build_no_goal(self, variant_index: int) -> Tuple[Image.Image, Dict[str, object]]:
        template = "no_goal"
        rng = self._rng(template, variant_index)
        agent_hex, agent_name = rng.choice(AGENT_COLORS)
        targets = choose_distinct(CONTAINER_COLORS, 2, rng)
        left_hex, left_name = targets[0]
        right_hex, right_name = targets[1]
        panel_count = 4
        image, draw = self._make_image(panel_count)
        positions = [(0.16, 0.22), (0.34, 0.22), (0.52, 0.22), (0.70, 0.22)]
        for panel in range(panel_count):
            draw.rectangle([self._px(panel, 0.03), 3, self._px(panel, 0.97), self.panel_h - 3], outline="black", width=2)
            draw_rect(draw, self._px(panel, 0.18), self._py(0.74), 42, 42, left_hex)
            draw_rect(draw, self._px(panel, 0.68), self._py(0.74), 42, 42, right_hex)
            self._draw_panel_agent(draw, panel, positions[panel][0], positions[panel][1], agent_hex)
        panel_lines = [
            f"A {agent_name} character drifts horizontally across the top of the scene.",
            f"The lower targets are {left_name} and {right_name}.",
            "The trajectory never turns toward either target.",
        ]
        metadata = self._stage2_meta(
            probe_id=f"s2_{template}_{variant_index:03d}",
            template=template,
            variant_index=variant_index,
            seed=10_000 * (ALL_TEMPLATES.index(template) + 1) + variant_index,
            panel_count=panel_count,
            task_prompt="Is the character heading toward either bottom target, yes or no?",
            answer_instruction="Answer ONLY 'yes' or 'no'.",
            panel_percept_lines=panel_lines,
            text_narration=f"A {agent_name} character moves horizontally across the top while two targets stay far below.",
            expected="no",
            answer_kind="yes_no",
            expected_aliases=["no"],
            entity_catalog={"agent": [agent_name], "object": [], "container": [left_name, right_name]},
        )
        return image, metadata

    def _build_wall_removed(self, variant_index: int) -> Tuple[Image.Image, Dict[str, object]]:
        template = "wall_removed"
        rng = self._rng(template, variant_index)
        agent_hex, agent_name = rng.choice(AGENT_COLORS)
        target_hex, target_name = rng.choice(CONTAINER_COLORS)
        panel_count = 2
        image, draw = self._make_image(panel_count)
        for panel in range(panel_count):
            draw.rectangle([self._px(panel, 0.03), 3, self._px(panel, 0.97), self.panel_h - 3], outline="black", width=2)
            self._draw_panel_agent(draw, panel, 0.18, 0.55, agent_hex)
            draw_rect(draw, self._px(panel, 0.76), self._py(0.48), 42, 42, target_hex)
        draw_dashed_line(
            draw,
            [(self._px(0, 0.24), self._py(0.55)), (self._px(0, 0.76), self._py(0.55))],
            "#2E7D32",
            width=4,
        )
        draw_dashed_line(
            draw,
            [
                (self._px(1, 0.24), self._py(0.55)),
                (self._px(1, 0.38), self._py(0.24)),
                (self._px(1, 0.62), self._py(0.24)),
                (self._px(1, 0.76), self._py(0.55)),
            ],
            "#EF6C00",
            width=4,
        )
        panel_lines = [
            f"Both panels show a {agent_name} character on the left and a {target_name} target on the right with no obstacle between them.",
            "The left panel shows a straight path. The right panel shows a curved detour path.",
        ]
        metadata = self._stage2_meta(
            probe_id=f"s2_{template}_{variant_index:03d}",
            template=template,
            variant_index=variant_index,
            seed=10_000 * (ALL_TEMPLATES.index(template) + 1) + variant_index,
            panel_count=panel_count,
            task_prompt="With no obstacle present, which path would a goal-directed character choose, straight or curved?",
            answer_instruction="Answer ONLY 'straight' or 'curved'.",
            panel_percept_lines=panel_lines,
            text_narration=f"A {agent_name} character can reach a {target_name} target either by a straight path or an unnecessary curved detour with no obstacle present.",
            expected="straight",
            answer_kind="straight_curved",
            expected_aliases=["straight", "direct", "left panel"],
            entity_catalog={"agent": [agent_name], "object": [], "container": [target_name]},
        )
        return image, metadata

    def build_probe(self, template: str, variant_index: int) -> Tuple[Image.Image, Dict[str, object]]:
        if template in {"sally_anne_search", "sally_anne_belief", "sally_anne_reality", "reality_vs_belief"}:
            return self._build_sally_anne(template, variant_index)
        if template in {"knowledge_who_knows", "knowledge_belief", "perspectives_differ", "second_order"}:
            return self._build_knowledge(template, variant_index)
        if template == "unexpected_contents":
            return self._build_unexpected_contents(variant_index)
        if template == "two_rooms":
            return self._build_two_rooms(variant_index)
        if template in {"occlusion_consistent", "occlusion_violation"}:
            return self._build_occlusion(template, variant_index)
        if template == "count_two":
            return self._build_count_two(variant_index)
        if template == "count_three":
            return self._build_count_three(variant_index)
        if template == "permanence":
            return self._build_permanence(variant_index)
        if template == "color_change":
            return self._build_color_change(variant_index)
        if template == "goal_upper":
            return self._build_goal_upper(variant_index)
        if template == "detour_obstacle":
            return self._build_detour_obstacle(variant_index)
        if template == "no_goal":
            return self._build_no_goal(variant_index)
        if template == "wall_removed":
            return self._build_wall_removed(variant_index)
        raise ValueError(template)


def parser_agents_correct(parsed_panels: Sequence[Dict[str, object]], truth_panels: Sequence[Sequence[str]]) -> bool:
    if not truth_panels:
        return False
    if len(parsed_panels) != len(truth_panels):
        return False
    for parsed, truth in zip(parsed_panels, truth_panels):
        parsed_agents = sorted(entry["name"] for entry in parsed["agents"])
        if parsed_agents != sorted(truth):
            return False
    return True


def augment_prompts_and_controls(probes: List[Dict[str, object]]) -> None:
    by_template: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for probe in probes:
        by_template[str(probe["template"])].append(probe)
    for template, items in by_template.items():
        items.sort(key=lambda p: int(p["variant_index"]))
        total = len(items)
        for idx, probe in enumerate(items):
            partner = items[(idx + 1) % total]
            shuffled = " ".join(sentence_shuffle(probe["panel_percept_lines"], int(probe["seed"]) + 17))
            noisy = inject_noise_into_scaffold(probe, int(probe["seed"]) + 41)
            probe["mismatch_partner_id"] = partner["id"]
            probe["question_B_auto"] = format_prompt(
                "Study this image. Here is an automatic pixel-level scene description with no visibility inference:",
                str(probe["scaffold_auto"]),
                str(probe["task_prompt"]),
                str(probe["answer_instruction"]),
            )
            probe["question_B_percept"] = format_prompt(
                "Study this image. Here is a structured perceptual description:",
                str(probe["scaffold_percept"]),
                str(probe["task_prompt"]),
                str(probe["answer_instruction"]),
            )
            probe["question_B_full"] = format_prompt(
                "Study this image. Here is a full structured description:",
                str(probe["scaffold_full"]),
                str(probe["task_prompt"]),
                str(probe["answer_instruction"]),
            )
            probe["question_D_mismatch"] = format_prompt(
                "Study this image. Here is a structured scene description:",
                str(partner["scaffold_percept"]),
                str(probe["task_prompt"]),
                str(probe["answer_instruction"]),
            )
            probe["question_D_shuffled"] = format_prompt(
                "Study this image. Here is a structured scene description with the panel sentences reordered:",
                shuffled,
                str(probe["task_prompt"]),
                str(probe["answer_instruction"]),
            )
            probe["question_D_noise"] = format_prompt(
                "Study this image. Here is a structured scene description with some entity names corrupted:",
                noisy,
                str(probe["task_prompt"]),
                str(probe["answer_instruction"]),
            )
            probe["robustness_prompts"] = make_paraphrases(probe)


def ensure_probe_set(variants: int) -> List[Dict[str, object]]:
    STIMULI_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    should_regenerate = True
    if PROBES_PATH.exists():
        try:
            existing = json.loads(PROBES_PATH.read_text(encoding="utf-8"))
            if existing:
                config = existing[0].get("_generation_config", {})
                should_regenerate = int(config.get("variants", -1)) != variants
            else:
                should_regenerate = True
        except Exception:
            should_regenerate = True
    if not should_regenerate:
        return json.loads(PROBES_PATH.read_text(encoding="utf-8"))

    parser = VisionParser()
    generator = ProbeGenerator(variants)
    probes: List[Dict[str, object]] = []
    for template in ALL_TEMPLATES:
        for variant_index in range(variants):
            image, metadata = generator.build_probe(template, variant_index)
            image_path = STIMULI_DIR / f"{metadata['id']}.png"
            image.save(image_path)
            parsed_panels, scaffold_auto = parser.parse(str(image_path))
            metadata["image_path"] = str(image_path)
            metadata["scaffold_auto"] = scaffold_auto
            metadata["parser_panels"] = parsed_panels
            metadata["parser_agent_all_correct"] = bool(
                metadata["stage"] == 3 and parser_agents_correct(parsed_panels, metadata["ground_truth_panels"])
            )
            metadata["_generation_config"] = {"variants": variants}
            probes.append(metadata)

    augment_prompts_and_controls(probes)
    PROBES_PATH.write_text(json.dumps(probes, indent=2), encoding="utf-8")
    return probes


def call_model(
    model: str,
    prompt: str,
    image_path: Optional[str],
    api_key: str,
    session: Optional[requests.Session] = None,
    max_retries: int = 3,
) -> Tuple[str, Optional[Dict[str, object]]]:
    headers = build_headers(api_key)
    content: List[Dict[str, object]] = []
    if image_path:
        b64 = encode_image(image_path)
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    content.append({"type": "text", "text": prompt})
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 200,
        "temperature": 0.0,
    }
    requester = session or requests
    for attempt in range(max_retries):
        try:
            response = requester.post(API_URL, headers=headers, json=payload, timeout=120)
            if response.status_code == 429:
                time.sleep(30 * (2**attempt))
                continue
            if response.status_code >= 500:
                time.sleep(15)
                continue
            response.raise_for_status()
            data = response.json()
            if "error" in data:
                error = data["error"]
                message = error.get("message", "unknown provider error")
                code = error.get("code", "unknown")
                return f"ERROR_API[{code}]: {message}", data
            choices = data.get("choices") or []
            if not choices or "message" not in choices[0]:
                return f"ERROR_BAD_RESPONSE: {json.dumps(data)[:300]}", data
            return str(choices[0]["message"]["content"]).strip(), data
        except Exception as exc:  # noqa: BLE001
            if attempt < max_retries - 1:
                time.sleep(10 * (attempt + 1))
                continue
            return f"ERROR: {str(exc)[:120]}", None
    return "ERROR: max retries", None


def answer_correct(probe: Dict[str, object], response: str) -> bool:
    norm = normalize_text(response)
    kind = str(probe["answer_kind"])
    expected = str(probe["expected"])
    if kind == "yes_no":
        return norm.startswith(expected)
    if kind == "number":
        return expected in norm[:4]
    if kind in {"hidden", "different", "upper_lower", "same_different", "straight_curved", "round_rectangles"}:
        return response_contains_any(response, probe["expected_aliases"])
    if kind == "choice_abc":
        return norm.startswith(expected.lower()) or response_contains_any(response, probe["expected_aliases"])
    if kind == "stayed_left":
        return response_contains_any(response, probe["expected_aliases"])
    if kind == "left_right":
        return response_contains_any(response, probe["expected_aliases"])
    if kind == "belief_split":
        if "belief" in norm:
            before, after = norm.split("belief", 1)
        else:
            midpoint = len(norm) // 2
            before, after = norm[:midpoint], norm[midpoint:]
        right_ok = any(alias in before for alias in ["right", "right box", "right container", str(probe.get("right_container_name", ""))])
        left_ok = any(alias in after for alias in ["left", "left box", "left container", str(probe.get("left_container_name", ""))])
        return right_ok and left_ok
    return response_contains_any(response, probe["expected_aliases"])


def build_result_record(
    phase: str,
    model: str,
    probe: Dict[str, object],
    condition: str,
    prompt: str,
    image_path: Optional[str],
    response: str,
    raw_data: Optional[Dict[str, object]],
    correct: bool,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    model_slug = slug_model(model)
    record = {
        "phase": phase,
        "model_full": model,
        "model": model_slug,
        "model_tier": MODEL_TIERS.get(model_slug, "unknown"),
        "probe_id": probe["id"],
        "stage": probe["stage"],
        "template": probe["template"],
        "variant_index": probe["variant_index"],
        "condition": condition,
        "prompt": prompt,
        "image_path": image_path,
        "used_image": bool(image_path),
        "response": response,
        "correct": correct,
        "expected": probe["expected"],
        "answer_kind": probe["answer_kind"],
        "timestamp": now_iso(),
        "raw_api": raw_data,
        "parser_agent_all_correct": probe.get("parser_agent_all_correct"),
        "mismatch_partner_id": probe.get("mismatch_partner_id"),
    }
    if extra:
        record.update(extra)
    return record


def result_key(record: Dict[str, object]) -> Tuple[str, str, str, str, str]:
    return (
        str(record["phase"]),
        str(record["model"]),
        str(record["probe_id"]),
        str(record["condition"]),
        str(record.get("paraphrase", "")),
    )


def compact_json_dumps(data: object) -> str:
    return json.dumps(data, separators=(",", ":"), ensure_ascii=True)


def shard_filename(path: Path, index: int) -> str:
    return f"{path.stem}.part{index:03d}{path.suffix}"


def shard_array(items: Sequence[Dict[str, object]], max_bytes: int = JSON_SHARD_MAX_BYTES) -> List[List[Dict[str, object]]]:
    shards: List[List[Dict[str, object]]] = []
    current: List[Dict[str, object]] = []
    current_size = 2  
    for item in items:
        item_json = compact_json_dumps(item)
        item_size = len(item_json.encode("utf-8"))
        separator_size = 1 if current else 0
        if current and current_size + separator_size + item_size > max_bytes:
            shards.append(current)
            current = [item]
            current_size = 2 + item_size
        else:
            current.append(item)
            current_size += separator_size + item_size
    if current:
        shards.append(current)
    return shards


def remove_stale_shards(path: Path) -> None:
    pattern = f"{path.stem}.part*{path.suffix}"
    for shard_path in path.parent.glob(pattern):
        shard_path.unlink()


def load_results(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and payload.get("_format") == JSON_SHARD_MANIFEST:
        results: List[Dict[str, object]] = []
        for shard_name in payload.get("shards", []):
            shard_path = path.parent / str(shard_name)
            if not shard_path.exists():
                raise FileNotFoundError(f"Missing shard file for {path.name}: {shard_name}")
            shard_payload = json.loads(shard_path.read_text(encoding="utf-8"))
            if not isinstance(shard_payload, list):
                raise ValueError(f"Shard {shard_name} for {path.name} is not a JSON array")
            results.extend(shard_payload)
        return results
    if not isinstance(payload, list):
        raise ValueError(f"Expected {path.name} to contain a JSON array")
    return payload


def save_results(path: Path, results: List[Dict[str, object]]) -> None:
    compact_bytes = compact_json_dumps(results).encode("utf-8")
    if len(compact_bytes) <= JSON_SHARD_MAX_BYTES:
        remove_stale_shards(path)
        path.write_bytes(compact_bytes)
        return

    remove_stale_shards(path)
    shards = shard_array(results)
    shard_names: List[str] = []
    for index, shard in enumerate(shards, start=1):
        shard_name = shard_filename(path, index)
        shard_bytes = compact_json_dumps(shard).encode("utf-8")
        if len(shard_bytes) > JSON_SHARD_MAX_BYTES:
            raise ValueError(f"Shard {shard_name} still exceeds max size")
        (path.parent / shard_name).write_bytes(shard_bytes)
        shard_names.append(shard_name)

    manifest = {
        "_format": JSON_SHARD_MANIFEST,
        "source": path.name,
        "total_records": len(results),
        "shards": shard_names,
    }
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def save_model_intermediate(phase: str, model_slug: str, model_results: List[Dict[str, object]]) -> None:
    path = RESULTS_DIR / f"v5_intermediate_{model_slug}.json"
    payload = {"model": model_slug, "updated_at": now_iso(), "phases": {}}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("phases", {})
    payload["phases"][phase] = model_results
    payload["updated_at"] = now_iso()
    path.write_text(compact_json_dumps(payload), encoding="utf-8")


@dataclass
class Task:
    index: int
    total: int
    phase: str
    model: str
    probe: Dict[str, object]
    condition: str
    prompt: str
    image_path: Optional[str]
    extra: Dict[str, object]


@dataclass
class TaskOutcome:
    task: Task
    record: Dict[str, object]
    worker_name: str
    error: Optional[str] = None


def build_phase_tasks(phase: str, probes: Sequence[Dict[str, object]], models: Sequence[str], variants: int) -> List[Task]:
    tasks: List[Task] = []
    if phase in PHASE_CONDITIONS:
        total = len(probes) * len(PHASE_CONDITIONS[phase]) * len(models)
        counter = 0
        for model in models:
            for probe in probes:
                for condition in PHASE_CONDITIONS[phase]:
                    counter += 1
                    prompt = str(probe[f"question_{condition}"])
                    image_path = None if condition == "C" else str(probe["image_path"])
                    tasks.append(
                        Task(
                            index=counter,
                            total=total,
                            phase=phase,
                            model=model,
                            probe=probe,
                            condition=condition,
                            prompt=prompt,
                            image_path=image_path,
                            extra={},
                        )
                    )
        return tasks

    if phase == "robustness":
        stage3_templates = set(STAGE3_TEMPLATES)
        robustness_probes = [
            probe
            for probe in probes
            if int(probe["stage"]) == 3
            and str(probe["template"]) in stage3_templates
            and int(probe["variant_index"]) < min(20, variants)
        ]
        robustness_probes.sort(key=lambda p: (str(p["template"]), int(p["variant_index"])))
        total = len(robustness_probes) * 3 * len(models)
        counter = 0
        for model in models:
            for probe in robustness_probes:
                for paraphrase in ("restructured", "vocabulary", "distractor"):
                    counter += 1
                    tasks.append(
                        Task(
                            index=counter,
                            total=total,
                            phase=phase,
                            model=model,
                            probe=probe,
                            condition="A",
                            prompt=str(probe["robustness_prompts"][paraphrase]),
                            image_path=str(probe["image_path"]),
                            extra={"paraphrase": paraphrase},
                        )
                    )
        return tasks

    raise ValueError(f"Unsupported phase: {phase}")


def execute_task(task: Task, api_key: str, session: Optional[requests.Session] = None) -> Dict[str, object]:
    response, raw = call_model(task.model, task.prompt, task.image_path, api_key, session=session)
    correct = answer_correct(task.probe, response)
    return build_result_record(
        phase=task.phase,
        model=task.model,
        probe=task.probe,
        condition=task.condition,
        prompt=task.prompt,
        image_path=task.image_path,
        response=response,
        raw_data=raw,
        correct=correct,
        extra=task.extra,
    )


def worker_loop(
    worker_name: str,
    api_key: str,
    task_queue: "queue.Queue[Optional[Task]]",
    result_queue: "queue.Queue[TaskOutcome]",
) -> None:
    session = requests.Session()
    try:
        while True:
            task = task_queue.get()
            if task is None:
                task_queue.task_done()
                return
            try:
                record = execute_task(task, api_key, session=session)
                result_queue.put(TaskOutcome(task=task, record=record, worker_name=worker_name))
            except Exception as exc:  # noqa: BLE001
                fallback = build_result_record(
                    phase=task.phase,
                    model=task.model,
                    probe=task.probe,
                    condition=task.condition,
                    prompt=task.prompt,
                    image_path=task.image_path,
                    response=f"ERROR_INTERNAL: {str(exc)[:120]}",
                    raw_data=None,
                    correct=False,
                    extra=task.extra,
                )
                result_queue.put(
                    TaskOutcome(task=task, record=fallback, worker_name=worker_name, error=str(exc)[:120])
                )
            finally:
                task_queue.task_done()
                time.sleep(1.0)
    finally:
        session.close()


def run_phase(
    phase: str,
    probes: List[Dict[str, object]],
    models: List[str],
    variants: int,
    dry_run: bool,
    api_keys: Sequence[str],
    parallel_workers: int,
) -> None:
    tasks = build_phase_tasks(phase, probes, models, variants)
    estimated_cost = len(tasks) * PER_CALL_COST
    result_path = RESULTS_DIR / (
        "v5_robustness_results.json" if phase == "robustness" else f"v5_{phase}_results.json"
    )
    existing_results = load_results(result_path)
    existing_keys = {result_key(record) for record in existing_results}
    if dry_run:
        print(f"{phase}: {len(tasks)} calls, estimated cost ${estimated_cost:.2f}")
        return

    phase_results = list(existing_results)
    phase_by_model: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for record in phase_results:
        phase_by_model[str(record["model"])].append(record)
    remaining_tasks = [
        task
        for task in tasks
        if (phase, slug_model(task.model), str(task.probe["id"]), task.condition, str(task.extra.get("paraphrase", "")))
        not in existing_keys
    ]
    target_by_model: Dict[str, int] = defaultdict(int)
    for task in tasks:
        target_by_model[slug_model(task.model)] += 1
    completed_by_model: Dict[str, int] = defaultdict(int)
    for record in phase_results:
        completed_by_model[str(record["model"])] += 1
    saved_completed_models = {
        model_slug for model_slug, count in completed_by_model.items() if count >= target_by_model.get(model_slug, 0)
    }
    worker_count = max(1, min(parallel_workers, len(api_keys)))
    worker_keys = list(api_keys[:worker_count])

    print("=" * 72)
    print(f"Phase {phase}: {len(tasks)} tasks, estimated cost ${estimated_cost:.2f}")
    print(
        f"Resuming with {len(phase_results)} saved rows, {len(remaining_tasks)} remaining tasks, "
        f"{worker_count} parallel workers across {len(worker_keys)} API keys"
    )
    print("=" * 72)

    if not remaining_tasks:
        save_results(result_path, phase_results)
        for model_slug, model_results in phase_by_model.items():
            save_model_intermediate(phase, model_slug, model_results)
        return

    checkpoint_every = 100
    task_queue: "queue.Queue[Optional[Task]]" = queue.Queue()
    result_queue: "queue.Queue[TaskOutcome]" = queue.Queue()
    for task in remaining_tasks:
        task_queue.put(task)
    for _ in range(worker_count):
        task_queue.put(None)

    workers = []
    for worker_idx, api_key in enumerate(worker_keys, start=1):
        worker = threading.Thread(
            target=worker_loop,
            args=(f"worker-{worker_idx}", api_key, task_queue, result_queue),
            daemon=True,
        )
        worker.start()
        workers.append(worker)

    completed_new = 0
    while completed_new < len(remaining_tasks):
        outcome = result_queue.get()
        task = outcome.task
        record = outcome.record
        model_slug = slug_model(task.model)
        phase_results.append(record)
        phase_by_model[model_slug].append(record)
        existing_keys.add(result_key(record))
        completed_by_model[model_slug] += 1
        completed_new += 1

        expected = task.probe["expected"]
        response = str(record["response"])
        status = "✅" if record["correct"] else "❌"
        prefix = f"[{task.index}/{task.total}] {model_slug} | {task.probe['id']}/{task.condition}"
        if outcome.error:
            print(f"{prefix} {status} worker={outcome.worker_name} internal:{outcome.error}")
        elif record["correct"]:
            print(f"{prefix} {status} worker={outcome.worker_name} \"{response[:80]}\"")
        else:
            print(f"{prefix} {status} worker={outcome.worker_name} expected:{expected} got:\"{response[:80]}\"")

        if completed_new % checkpoint_every == 0:
            save_results(result_path, phase_results)
            for save_slug, model_results in phase_by_model.items():
                save_model_intermediate(phase, save_slug, model_results)

        if completed_by_model[model_slug] >= target_by_model.get(model_slug, 0) and model_slug not in saved_completed_models:
            save_results(result_path, phase_results)
            save_model_intermediate(phase, model_slug, phase_by_model[model_slug])
            saved_completed_models.add(model_slug)

    task_queue.join()
    for worker in workers:
        worker.join(timeout=1.0)
    save_results(result_path, phase_results)


def build_probe_index(probes: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    return {str(probe["id"]): probe for probe in probes}


def stage_probe_ids(probes: Sequence[Dict[str, object]], stage: int) -> List[str]:
    ids = [str(probe["id"]) for probe in probes if int(probe["stage"]) == stage]
    ids.sort()
    return ids


def matrix_for_condition(
    results: Sequence[Dict[str, object]],
    probe_ids: Sequence[str],
    models: Sequence[str],
    condition: str,
    stage: Optional[int] = None,
    paraphrase: Optional[str] = None,
) -> np.ndarray:
    probe_index = {pid: idx for idx, pid in enumerate(probe_ids)}
    model_index = {model: idx for idx, model in enumerate(models)}
    matrix = np.full((len(models), len(probe_ids)), np.nan, dtype=float)
    for record in results:
        if str(record["condition"]) != condition:
            continue
        if stage is not None and int(record["stage"]) != stage:
            continue
        if paraphrase is not None and str(record.get("paraphrase", "")) != paraphrase:
            continue
        mid = model_index.get(str(record["model"]))
        pid = probe_index.get(str(record["probe_id"]))
        if mid is None or pid is None:
            continue
        matrix[mid, pid] = 1.0 if record["correct"] else 0.0
    return matrix


def bootstrap_metric(matrix: np.ndarray, seed: int) -> Tuple[float, float, float]:
    n_models, n_probes = matrix.shape
    if n_probes == 0:
        return (float("nan"),) * 3
    point = float(np.nanmean(matrix))
    rng = np.random.default_rng(seed)
    chunk = 500
    values = []
    for start in range(0, BOOTSTRAP_RESAMPLES, chunk):
        count = min(chunk, BOOTSTRAP_RESAMPLES - start)
        sample_idx = rng.integers(0, n_probes, size=(count, n_probes))
        sampled = matrix[:, sample_idx]
        means = np.nanmean(sampled, axis=(0, 2))
        values.append(means)
    samples = np.concatenate(values)
    low, high = np.nanpercentile(samples, [2.5, 97.5])
    return point, float(low), float(high)


def bootstrap_difference(matrix_a: np.ndarray, matrix_b: np.ndarray, seed: int) -> Tuple[float, float, float]:
    n_models, n_probes = matrix_a.shape
    if n_probes == 0:
        return (float("nan"),) * 3
    point = float(np.nanmean(matrix_b) - np.nanmean(matrix_a))
    rng = np.random.default_rng(seed)
    chunk = 500
    values = []
    for start in range(0, BOOTSTRAP_RESAMPLES, chunk):
        count = min(chunk, BOOTSTRAP_RESAMPLES - start)
        sample_idx = rng.integers(0, n_probes, size=(count, n_probes))
        sampled_a = matrix_a[:, sample_idx]
        sampled_b = matrix_b[:, sample_idx]
        means = np.nanmean(sampled_b, axis=(0, 2)) - np.nanmean(sampled_a, axis=(0, 2))
        values.append(means)
    samples = np.concatenate(values)
    low, high = np.nanpercentile(samples, [2.5, 97.5])
    return point, float(low), float(high)


def bootstrap_pcr(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    matrix_c: np.ndarray,
    seed: int,
) -> Dict[str, float]:
    n_models, n_probes = matrix_a.shape
    if n_probes == 0:
        return {"point": float("nan"), "median": float("nan"), "low": float("nan"), "high": float("nan")}
    point_denom = np.nanmean(matrix_c) - np.nanmean(matrix_a)
    point = float((np.nanmean(matrix_b) - np.nanmean(matrix_a)) / point_denom) if abs(point_denom) > 1e-9 else float("nan")
    rng = np.random.default_rng(seed)
    chunk = 500
    values = []
    for start in range(0, BOOTSTRAP_RESAMPLES, chunk):
        count = min(chunk, BOOTSTRAP_RESAMPLES - start)
        sample_idx = rng.integers(0, n_probes, size=(count, n_probes))
        sampled_a = matrix_a[:, sample_idx]
        sampled_b = matrix_b[:, sample_idx]
        sampled_c = matrix_c[:, sample_idx]
        acc_a = np.nanmean(sampled_a, axis=(0, 2))
        acc_b = np.nanmean(sampled_b, axis=(0, 2))
        acc_c = np.nanmean(sampled_c, axis=(0, 2))
        denom = acc_c - acc_a
        numer = acc_b - acc_a
        pcr = np.divide(numer, denom, out=np.full_like(numer, np.nan), where=np.abs(denom) > 1e-9)
        values.append(pcr)
    samples = np.concatenate(values)
    return {
        "point": point,
        "median": float(np.nanmedian(samples)),
        "low": float(np.nanpercentile(samples, 2.5)),
        "high": float(np.nanpercentile(samples, 97.5)),
    }


def bootstrap_rescue_rate(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    seed: int,
) -> Dict[str, float]:
    a = matrix_a.flatten()
    b = matrix_b.flatten()
    valid = ~np.isnan(a) & ~np.isnan(b)
    a = a[valid]
    b = b[valid]
    failures = np.where(a < 0.5)[0]
    if failures.size == 0:
        return {"point": float("nan"), "low": float("nan"), "high": float("nan"), "denom": 0}
    rescued = b[failures]
    point = float(np.nanmean(rescued))
    rng = np.random.default_rng(seed)
    samples = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample_idx = rng.integers(0, failures.size, size=failures.size)
        samples.append(float(np.nanmean(rescued[sample_idx])))
    low, high = np.nanpercentile(samples, [2.5, 97.5])
    return {"point": point, "low": float(low), "high": float(high), "denom": int(failures.size)}


def build_pair_map(
    results: Sequence[Dict[str, object]],
    condition: str,
    stage: int,
    allowed_models: Optional[Sequence[str]] = None,
) -> Dict[Tuple[str, str], int]:
    allowed = set(allowed_models or [])
    pairs = {}
    for record in results:
        if int(record["stage"]) != stage or str(record["condition"]) != condition:
            continue
        if allowed and str(record["model"]) not in allowed:
            continue
        key = (str(record["model"]), str(record["probe_id"]))
        pairs[key] = 1 if record["correct"] else 0
    return pairs


def mcnemar_test(
    results_a: Sequence[Dict[str, object]],
    results_b: Sequence[Dict[str, object]],
    condition_a: str,
    condition_b: str,
    stage: int,
    allowed_models: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    pairs_a = build_pair_map(results_a, condition_a, stage, allowed_models)
    pairs_b = build_pair_map(results_b, condition_b, stage, allowed_models)
    common = sorted(set(pairs_a) & set(pairs_b))
    b = 0
    c = 0
    for key in common:
        a_val = pairs_a[key]
        b_val = pairs_b[key]
        if a_val == 1 and b_val == 0:
            b += 1
        elif a_val == 0 and b_val == 1:
            c += 1
    if b + c == 0:
        chi_sq = 0.0
        p_value = 1.0
    else:
        chi_sq = (abs(b - c) - 1.0) ** 2 / (b + c)
        p_value = math.erfc(math.sqrt(max(chi_sq, 0.0) / 2.0))
    return {"b": b, "c": c, "chi_sq": chi_sq, "p": p_value, "n_pairs": len(common)}


def categorize_error(record: Dict[str, object], probe: Dict[str, object]) -> str:
    response = normalize_text(str(record["response"]))
    expected = normalize_text(str(probe["expected"]))
    actual_aliases = [normalize_text(alias) for alias in probe.get("actual_aliases", []) if alias]
    if actual_aliases and any(alias in response for alias in actual_aliases) and expected not in response:
        return "reality_bias"
    if probe["template"] == "knowledge_who_knows":
        if any(token in response for token in ("left", "right", probe.get("agent2_name", ""), "the one who left")):
            return "wrong_agent"
    if probe["template"] in {"sally_anne_search", "sally_anne_belief", "knowledge_belief"}:
        other_agent = normalize_text(str(probe.get("agent2_name", "")))
        if other_agent and other_agent in response:
            return "wrong_agent"
    word_count = len(response.split())
    if word_count > 6 and not response_contains_any(response, probe["expected_aliases"]):
        return "format_error"
    return "other"


def analysis_report_lines(analysis: Dict[str, object]) -> List[str]:
    lines = []
    lines.append("Perception Before Belief — V5 analysis")
    lines.append("")
    lines.append("Counts")
    for stage, count in analysis["counts"]["probe_counts_by_stage"].items():
        lines.append(f"  Stage {stage}: {count} probes")
    lines.append("")
    lines.append("Main Results")
    for row in analysis["main_table"]:
        lines.append(
            f"  Stage {row['stage']} {row['stage_name']}: "
            f"A={row['A']['text']} | "
            f"B_auto={row['B_auto']['text']} | "
            f"B_percept={row['B_percept']['text']} | "
            f"B_full={row['B_full']['text']} | "
            f"C={row['C']['text']} | "
            f"B_auto-A={row['B_auto_minus_A']['text']} | "
            f"PCR={row['PCR_auto']['text']}"
        )
    lines.append("")
    lines.append("Stage 3 per model")
    for row in analysis["stage3_per_model"]:
        lines.append(
            f"  {row['model']}: A={row['A']['text']} | B_auto={row['B_auto']['text']} | "
            f"B_percept={row['B_percept']['text']} | B_full={row['B_full']['text']} | "
            f"C={row['C']['text']} | B_auto-A={row['B_auto_minus_A']['text']} | "
            f"rescue={row['rescue_rate']['text']}"
        )
    lines.append("")
    lines.append("Scaffold ablation")
    for row in analysis["scaffold_ablation"]:
        lines.append(f"  {row['condition']}: {row['accuracy_text']} ({row['delta_text']})")
    lines.append("")
    lines.append("Control baselines")
    for row in analysis["control_table"]:
        lines.append(f"  {row['condition']}: {row['accuracy_text']}")
    lines.append("")
    lines.append("Prompt robustness")
    for row in analysis["prompt_robustness"]["rows"]:
        lines.append(f"  {row['paraphrase']}: {row['accuracy_text']} ({row['delta_text']})")
    lines.append(
        f"  mean={analysis['prompt_robustness']['mean_text']} | std={analysis['prompt_robustness']['std_text']}"
    )
    lines.append("")
    lines.append("McNemar tests")
    for row in analysis["mcnemar_tests"]:
        lines.append(
            f"  {row['comparison']}: chi^2={row['chi_sq']:.3f}, p={row['p']:.3g}, "
            f"p_bonf={row['p_bonf']:.3g}, significant={row['significant']}"
        )
    lines.append("")
    lines.append("Template difficulty")
    for row in analysis["template_difficulty"]:
        lines.append(
            f"  {row['template']}: A={row['accuracy_text']} | parser={row['parser_accuracy_text']}"
        )
    lines.append("")
    lines.append("Tier PCR")
    for row in analysis["tier_pcr"]:
        lines.append(
            f"  {row['tier']}: auto={row['PCR_auto_text']} | percept={row['PCR_percept_text']} | full={row['PCR_full_text']}"
        )
    return lines


def run_analysis(probes: List[Dict[str, object]], selected_models: List[str]) -> None:
    phase1_results = load_results(RESULTS_DIR / "v5_phase1_results.json")
    phase2_results = load_results(RESULTS_DIR / "v5_phase2_results.json")
    robustness_results = load_results(RESULTS_DIR / "v5_robustness_results.json")
    all_phase_results = phase1_results + phase2_results
    probe_index = build_probe_index(probes)
    model_slugs = [slug_model(model) for model in selected_models]

    counts = {
        "probe_counts_by_stage": {
            str(stage): sum(1 for probe in probes if int(probe["stage"]) == stage) for stage in (1, 2, 3)
        },
        "template_counts": {template: sum(1 for probe in probes if str(probe["template"]) == template) for template in ALL_TEMPLATES},
        "phase1_rows": len(phase1_results),
        "phase2_rows": len(phase2_results),
        "robustness_rows": len(robustness_results),
        "models": model_slugs,
    }

    main_table = []
    for stage in (1, 2, 3):
        probe_ids = stage_probe_ids(probes, stage)
        matrices = {
            condition: matrix_for_condition(phase1_results, probe_ids, model_slugs, condition, stage=stage)
            for condition in ("A", "B_auto", "B_percept", "B_full", "C")
        }
        row = {"stage": stage, "stage_name": STAGE_NAMES[stage]}
        for offset, condition in enumerate(("A", "B_auto", "B_percept", "B_full", "C")):
            point, low, high = bootstrap_metric(matrices[condition], seed=1000 * stage + offset)
            row[condition] = {"point": point, "low": low, "high": high, "text": percent_ci(point, low, high)}
        gap_point, gap_low, gap_high = bootstrap_difference(matrices["A"], matrices["B_auto"], seed=7000 + stage)
        row["B_auto_minus_A"] = {
            "point": gap_point,
            "low": gap_low,
            "high": gap_high,
            "text": percent_ci(gap_point, gap_low, gap_high),
        }
        row["PCR_auto"] = bootstrap_pcr(matrices["A"], matrices["B_auto"], matrices["C"], seed=8000 + stage)
        row["PCR_auto"]["text"] = (
            f"{row['PCR_auto']['median']:.3f} [{row['PCR_auto']['low']:.3f}, {row['PCR_auto']['high']:.3f}]"
        )
        main_table.append(row)

    stage3_probe_ids = stage_probe_ids(probes, 3)
    stage3_per_model = []
    for idx, model_slug in enumerate(model_slugs):
        row = {"model": model_slug}
        single_model = [model_slug]
        matrices = {
            condition: matrix_for_condition(phase1_results, stage3_probe_ids, single_model, condition, stage=3)
            for condition in ("A", "B_auto", "B_percept", "B_full", "C")
        }
        for offset, condition in enumerate(("A", "B_auto", "B_percept", "B_full", "C")):
            point, low, high = bootstrap_metric(matrices[condition], seed=9000 + 100 * idx + offset)
            row[condition] = {"point": point, "low": low, "high": high, "text": percent_ci(point, low, high)}
        gap_point, gap_low, gap_high = bootstrap_difference(matrices["A"], matrices["B_auto"], seed=9500 + idx)
        row["B_auto_minus_A"] = {
            "point": gap_point,
            "low": gap_low,
            "high": gap_high,
            "text": percent_ci(gap_point, gap_low, gap_high),
        }
        rescue = bootstrap_rescue_rate(matrices["A"], matrices["B_auto"], seed=9800 + idx)
        rescue["text"] = percent_ci(rescue["point"], rescue["low"], rescue["high"])
        row["rescue_rate"] = rescue
        stage3_per_model.append(row)

    stage3_matrices = {
        condition: matrix_for_condition(phase1_results, stage3_probe_ids, model_slugs, condition, stage=3)
        for condition in ("A", "B_auto", "B_percept", "B_full", "C")
    }
    pcr = {
        "PCR_auto": bootstrap_pcr(stage3_matrices["A"], stage3_matrices["B_auto"], stage3_matrices["C"], seed=11000),
        "PCR_percept": bootstrap_pcr(stage3_matrices["A"], stage3_matrices["B_percept"], stage3_matrices["C"], seed=11001),
        "PCR_full": bootstrap_pcr(stage3_matrices["A"], stage3_matrices["B_full"], stage3_matrices["C"], seed=11002),
    }

    scaffold_ablation = []
    for condition in ("A", "B_auto", "B_percept", "B_full", "C"):
        point, low, high = bootstrap_metric(stage3_matrices[condition], seed=12000 + hash(condition) % 1000)
        delta = point - float(np.nanmean(stage3_matrices["A"]))
        scaffold_ablation.append(
            {
                "condition": condition,
                "accuracy": point,
                "low": low,
                "high": high,
                "accuracy_text": percent_ci(point, low, high),
                "delta_vs_A": delta,
                "delta_text": "baseline" if condition == "A" else f"{delta:+.1%}",
            }
        )

    stage3_models_phase2 = sorted({str(record["model"]) for record in phase2_results if int(record["stage"]) == 3})
    control_probe_ids = stage3_probe_ids
    control_sources = {
        "A": (phase1_results, "A"),
        "D_mismatch": (phase2_results, "D_mismatch"),
        "D_shuffled": (phase2_results, "D_shuffled"),
        "D_noise": (phase2_results, "D_noise"),
        "B_auto": (phase1_results, "B_auto"),
    }
    control_table = []
    for condition, (source, source_condition) in control_sources.items():
        allowed_models = stage3_models_phase2 if condition.startswith("D_") or condition == "A" else stage3_models_phase2 or model_slugs
        models_for_condition = allowed_models or model_slugs
        matrix = matrix_for_condition(source, control_probe_ids, models_for_condition, source_condition, stage=3)
        point, low, high = bootstrap_metric(matrix, seed=13000 + hash(condition) % 1000)
        control_table.append(
            {
                "condition": condition,
                "accuracy": point,
                "low": low,
                "high": high,
                "accuracy_text": percent_ci(point, low, high),
            }
        )

    robustness_probe_ids = [
        str(probe["id"])
        for probe in probes
        if int(probe["stage"]) == 3 and int(probe["variant_index"]) < min(20, max(1, max(int(p["variant_index"]) for p in probes if int(p["stage"]) == 3) + 1))
    ]
    robustness_probe_ids.sort()
    original_models = sorted({str(record["model"]) for record in robustness_results}) or model_slugs
    original_matrix = matrix_for_condition(phase1_results, robustness_probe_ids, original_models, "A", stage=3)
    original_point = float(np.nanmean(original_matrix))
    prompt_rows = []
    paraphrase_points = []
    for offset, paraphrase in enumerate(("restructured", "vocabulary", "distractor")):
        matrix = matrix_for_condition(
            robustness_results,
            robustness_probe_ids,
            original_models,
            "A",
            stage=3,
            paraphrase=paraphrase,
        )
        point, low, high = bootstrap_metric(matrix, seed=14000 + offset)
        paraphrase_points.append(point)
        prompt_rows.append(
            {
                "paraphrase": paraphrase,
                "accuracy": point,
                "low": low,
                "high": high,
                "accuracy_text": percent_ci(point, low, high),
                "delta": point - original_point,
                "delta_text": f"{point - original_point:+.1%}",
            }
        )
    prompt_robustness = {
        "rows": prompt_rows,
        "original": original_point,
        "mean": float(np.nanmean(paraphrase_points)) if paraphrase_points else float("nan"),
        "std": float(np.nanstd(paraphrase_points)) if paraphrase_points else float("nan"),
        "mean_text": percent(float(np.nanmean(paraphrase_points)) if paraphrase_points else float("nan")),
        "std_text": percent(float(np.nanstd(paraphrase_points)) if paraphrase_points else float("nan")),
    }

    stage3_phase1 = [record for record in phase1_results if int(record["stage"]) == 3]
    stage3_phase2 = [record for record in phase2_results if int(record["stage"]) == 3]
    mcnemar_specs = [
        ("A vs B_auto", stage3_phase1, stage3_phase1, "A", "B_auto", None),
        ("A vs B_percept", stage3_phase1, stage3_phase1, "A", "B_percept", None),
        ("A vs B_full", stage3_phase1, stage3_phase1, "A", "B_full", None),
        ("A vs C", stage3_phase1, stage3_phase1, "A", "C", None),
        ("B_auto vs B_percept", stage3_phase1, stage3_phase1, "B_auto", "B_percept", None),
        ("B_auto vs B_full", stage3_phase1, stage3_phase1, "B_auto", "B_full", None),
        ("B_percept vs B_full", stage3_phase1, stage3_phase1, "B_percept", "B_full", None),
        ("A vs D_mismatch", stage3_phase1, stage3_phase2, "A", "D_mismatch", stage3_models_phase2),
        ("B_auto vs D_mismatch", stage3_phase1, stage3_phase2, "B_auto", "D_mismatch", stage3_models_phase2),
        ("B_auto vs D_noise", stage3_phase1, stage3_phase2, "B_auto", "D_noise", stage3_models_phase2),
        ("B_auto vs D_shuffled", stage3_phase1, stage3_phase2, "B_auto", "D_shuffled", stage3_models_phase2),
        ("A vs D_noise", stage3_phase1, stage3_phase2, "A", "D_noise", stage3_models_phase2),
    ]
    mcnemar_rows = []
    for comparison, source_a, source_b, cond_a, cond_b, allowed_models in mcnemar_specs:
        stats = mcnemar_test(source_a, source_b, cond_a, cond_b, stage=3, allowed_models=allowed_models)
        adjusted = min(stats["p"] * 12.0, 1.0)
        mcnemar_rows.append(
            {
                "comparison": comparison,
                "chi_sq": stats["chi_sq"],
                "p": stats["p"],
                "p_bonf": adjusted,
                "n_pairs": stats["n_pairs"],
                "b": stats["b"],
                "c": stats["c"],
                "significant": adjusted < 0.05 and stats["p"] < BONFERRONI_THRESHOLD,
            }
        )

    template_difficulty = []
    for template in STAGE3_TEMPLATES:
        template_probe_ids = [str(probe["id"]) for probe in probes if str(probe["template"]) == template]
        matrix = matrix_for_condition(stage3_phase1, template_probe_ids, model_slugs, "A", stage=3)
        point, low, high = bootstrap_metric(matrix, seed=15000 + ALL_TEMPLATES.index(template))
        parser_values = [
            1.0 if probe_index[probe_id].get("parser_agent_all_correct") else 0.0 for probe_id in template_probe_ids
        ]
        parser_point = safe_mean(parser_values)
        template_difficulty.append(
            {
                "template": template,
                "accuracy": point,
                "low": low,
                "high": high,
                "accuracy_text": percent_ci(point, low, high),
                "parser_accuracy": parser_point,
                "parser_accuracy_text": percent(parser_point),
            }
        )
    template_difficulty.sort(key=lambda row: row["accuracy"])

    error_typing_counts = defaultdict(int)
    total_errors = 0
    for record in stage3_phase1:
        if str(record["condition"]) != "A" or record["correct"]:
            continue
        total_errors += 1
        error_typing_counts[categorize_error(record, probe_index[str(record["probe_id"])])] += 1
    error_typing = {
        "counts": dict(error_typing_counts),
        "total": total_errors,
        "fractions": {
            key: (value / total_errors if total_errors else float("nan")) for key, value in error_typing_counts.items()
        },
    }

    tier_rows = []
    tier_to_models = defaultdict(list)
    for model_slug in model_slugs:
        tier_to_models[MODEL_TIERS.get(model_slug, "unknown")].append(model_slug)
    for tier, tier_models in sorted(tier_to_models.items()):
        matrices = {
            condition: matrix_for_condition(phase1_results, stage3_probe_ids, tier_models, condition, stage=3)
            for condition in ("A", "B_auto", "B_percept", "B_full", "C")
        }
        pcr_auto = bootstrap_pcr(matrices["A"], matrices["B_auto"], matrices["C"], seed=16000 + len(tier_models))
        pcr_percept = bootstrap_pcr(matrices["A"], matrices["B_percept"], matrices["C"], seed=16100 + len(tier_models))
        pcr_full = bootstrap_pcr(matrices["A"], matrices["B_full"], matrices["C"], seed=16200 + len(tier_models))
        tier_rows.append(
            {
                "tier": tier,
                "models": tier_models,
                "PCR_auto": pcr_auto,
                "PCR_percept": pcr_percept,
                "PCR_full": pcr_full,
                "PCR_auto_text": f"{pcr_auto['median']:.3f} [{pcr_auto['low']:.3f}, {pcr_auto['high']:.3f}]",
                "PCR_percept_text": f"{pcr_percept['median']:.3f} [{pcr_percept['low']:.3f}, {pcr_percept['high']:.3f}]",
                "PCR_full_text": f"{pcr_full['median']:.3f} [{pcr_full['low']:.3f}, {pcr_full['high']:.3f}]",
            }
        )

    analysis = {
        "counts": counts,
        "main_table": main_table,
        "stage3_per_model": stage3_per_model,
        "pcr": pcr,
        "scaffold_ablation": scaffold_ablation,
        "control_table": control_table,
        "prompt_robustness": prompt_robustness,
        "mcnemar_tests": mcnemar_rows,
        "template_difficulty": template_difficulty,
        "error_typing": error_typing,
        "tier_pcr": tier_rows,
        "bonferroni_threshold": BONFERRONI_THRESHOLD,
    }
    (RESULTS_DIR / "v5_analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    (RESULTS_DIR / "v5_analysis.txt").write_text("\n".join(analysis_report_lines(analysis)) + "\n", encoding="utf-8")
    print("Saved analysis to results_v5/v5_analysis.json and results_v5/v5_analysis.txt")


def dry_run_summary(phase: str, variants: int, models) -> None:
    probes = ensure_probe_set(variants)
    if phase == "all":
        total = 0
        for phase_name in ("phase1", "phase2", "robustness"):
            tasks = build_phase_tasks(phase_name, probes, models, variants)
            cost = len(tasks) * PER_CALL_COST
            total += len(tasks)
            print(f"{phase_name}: {len(tasks)} calls, estimated cost ${cost:.2f}")
        print(f"all: {total} calls, estimated cost ${total * PER_CALL_COST:.2f}")
    elif phase == "analysis":
        print("analysis: 0 API calls")
    else:
        run_phase(phase, probes, models, variants, dry_run=True, api_keys=[], parallel_workers=1)
    stage_counts = {stage: sum(1 for probe in probes if int(probe["stage"]) == stage) for stage in (1, 2, 3)}
    parser_ok = sum(1 for probe in probes if probe.get("parser_agent_all_correct"))
    print(f"Generated probes: {len(probes)} total | stage counts={stage_counts} | stage3 parser-perfect={parser_ok}")


def main() -> None:
    global RESULTS_DIR, STIMULI_DIR, PROBES_PATH
    parser = argparse.ArgumentParser(description="Abstract 2D experiment runner")
    parser.add_argument("--phase", choices=["phase1", "phase2", "robustness", "analysis", "all"], required=True)
    parser.add_argument("--models", type=int, default=7, help="Number of models to use (default 7)")
    parser.add_argument("--model-ids", nargs="+", default=None, help="Explicit model IDs to run; overrides --models")
    parser.add_argument("--variants", type=int, default=50, help="Variants per template (default 50)")
    parser.add_argument("--data-dir", type=Path, default=STIMULI_DIR, help="Probe and image directory")
    parser.add_argument(
        "--api-key-file",
        action="append",
        default=[],
        help="Additional OpenRouter API key file(s) to use for parallel workers",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR),
        help="Directory for result bundles and per-model intermediates",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Maximum concurrent workers; capped by the number of available API keys",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate probes and report call count without API calls")
    args = parser.parse_args()

    if args.variants <= 0:
        raise SystemExit("--variants must be positive")
    if args.parallel_workers <= 0:
        raise SystemExit("--parallel-workers must be positive")
    RESULTS_DIR = Path(args.output_dir)
    STIMULI_DIR = args.data_dir.resolve()
    PROBES_PATH = STIMULI_DIR / "probes.json"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    models = select_models(args.models, args.model_ids)

    if args.dry_run:
        dry_run_summary(args.phase, args.variants, models)
        return

    probes = ensure_probe_set(args.variants)
    for probe in probes:
        probe["image_path"] = str(STIMULI_DIR / probe["image_path"])
    if args.phase == "analysis":
        run_analysis(probes, models)
        return
    api_keys = load_api_keys(args.api_key_file, require=True)
    effective_workers = min(args.parallel_workers, len(api_keys))
    print(
        f"Using {effective_workers} worker(s) across {len(api_keys)} API key(s)"
        f" (requested workers={args.parallel_workers})"
    )
    if args.phase == "all":
        for phase_name in ("phase1", "phase2", "robustness"):
            run_phase(
                phase_name,
                probes,
                models,
                args.variants,
                dry_run=False,
                api_keys=api_keys,
                parallel_workers=args.parallel_workers,
            )
        run_analysis(probes, models)
        return
    run_phase(
        args.phase,
        probes,
        models,
        args.variants,
        dry_run=False,
        api_keys=api_keys,
        parallel_workers=args.parallel_workers,
    )


if __name__ == "__main__":
    main()
