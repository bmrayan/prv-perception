from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _paths import ROOT, REPO_ROOT, BENCHMARK_ROOT, resolve_path


import argparse
import concurrent.futures
import importlib.util
import json
import os
import re
import subprocess
from collections import Counter
from typing import Dict, List, Sequence

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from mindpower.benchmark_remainder import (
    DEFAULT_REMAINDER_PREP_DIR,
    build_storyboard_from_images,
    extract_storyboard_bundle_generic,
)
from mindpower.common import ROOT, now_iso, read_jsonl, relative_to_root, slugify, write_json


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def ensure_virtualhome_available() -> None:
    if importlib.util.find_spec("virtualhome") is None:
        raise RuntimeError("The 'virtualhome' Python package is not installed in this environment.")


def load_rows(path: Path) -> List[dict]:
    rows = read_jsonl(path)
    rows.sort(key=lambda row: str(row["sample_id"]))
    return rows


def select_rows(rows: Sequence[dict], sample_ids: Sequence[str], limit_samples: int) -> List[dict]:
    selected = list(rows)
    if sample_ids:
        sample_set = {str(sample_id) for sample_id in sample_ids}
        selected = [row for row in selected if str(row["sample_id"]) in sample_set]
    if limit_samples > 0:
        selected = selected[:limit_samples]
    return selected


def patch_script_text(
    script_text: str,
    *,
    virtualhome_exe: str,
    port: int,
    work_dir: Path,
    x_display: str | None,
) -> str:
    patched = script_text
    simulation_pkg = BENCHMARK_ROOT / "external" / "virtualhome" / "virtualhome" / "simulation"
    patched = patched.replace(
        "from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication",
        (
            f"import sys\n"
            f"sys.path.insert(0, r{str(simulation_pkg)!r})\n"
            f"from unity_simulator.comm_unity import UnityCommunication"
        ),
    )
    for variable_name in ["YOUR_FILE_NAME", "unity_executable"]:
        patched = re.sub(
            rf"(?m)^\s*{re.escape(variable_name)}\s*=.*$",
            f"{variable_name} = r{virtualhome_exe!r}",
            patched,
        )
    for variable_name in ["port", "PORT"]:
        patched = re.sub(
            rf'(?m)^\s*{re.escape(variable_name)}\s*=.*$',
            f'{variable_name} = "{port}"',
            patched,
        )

    def rewrite_unity_call(match: re.Match[str]) -> str:
        args = match.group(1)
        if re.search(r"\bfile_name\s*=", args):
            args = re.sub(r"\bfile_name\s*=\s*[^,\n)]+", f"file_name=r{virtualhome_exe!r}", args)
        else:
            args = args.rstrip() + f", file_name=r{virtualhome_exe!r}"
        if re.search(r"\bport\s*=", args):
            args = re.sub(r"\bport\s*=\s*[^,\n)]+", f'port="{port}"', args)
        else:
            args = args.rstrip() + f', port="{port}"'
        if x_display and "x_display" not in args:
            args = args.rstrip() + f", x_display='{x_display}'"
        if "timeout_wait" not in args:
            args = args.rstrip() + ", timeout_wait=300"
        return f"UnityCommunication({args})"

    patched = re.sub(
        r"UnityCommunication\(([^)]*)\)",
        rewrite_unity_call,
        patched,
        count=1,
    )
    prelude = (
        "import os\n"
        "os.chdir(r{work_dir})\n"
        "import sys\n"
        "venv_prefix = os.path.realpath(sys.prefix)\n"
        "cleaned_sys_path = []\n"
        "for entry in sys.path:\n"
        "    if not entry:\n"
        "        cleaned_sys_path.append(entry)\n"
        "        continue\n"
        "    real_entry = os.path.realpath(entry)\n"
        "    if 'site-packages' in real_entry and not real_entry.startswith(venv_prefix):\n"
        "        continue\n"
        "    cleaned_sys_path.append(entry)\n"
        "sys.path[:] = cleaned_sys_path\n"
        "sys.path.insert(0, r{simulation_pkg})\n"
    ).format(
        work_dir=repr(str(work_dir)),
        simulation_pkg=repr(str(simulation_pkg)),
    )
    return prelude + patched


def build_child_env(python_bin: str, x_display: str | None) -> Dict[str, str]:
    env = os.environ.copy()
    python_path = Path(python_bin).expanduser()
    if not python_path.is_absolute():
        python_path = (Path.cwd() / python_path).resolve()
    venv_root = python_path.parent.parent
    if not (venv_root / "pyvenv.cfg").exists():
        resolved_path = python_path.resolve()
        resolved_candidate = resolved_path.parent.parent
        if (resolved_candidate / "pyvenv.cfg").exists():
            venv_root = resolved_candidate
    env["VIRTUAL_ENV"] = str(venv_root)
    env["PATH"] = os.pathsep.join([str(python_path.parent), env.get("PATH", "")]).rstrip(os.pathsep)
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env.pop("CONDA_PREFIX", None)
    env.pop("CONDA_DEFAULT_ENV", None)
    env.pop("CONDA_PROMPT_MODIFIER", None)
    env["PYTHONNOUSERSITE"] = "1"
    if x_display:
        env["DISPLAY"] = f":{x_display}" if not str(x_display).startswith(":") else str(x_display)
    return env


def discover_media_files(work_dir: Path) -> Dict[str, List[Path]]:
    videos: List[Path] = []
    images: List[Path] = []
    for path in work_dir.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            videos.append(path)
        elif suffix in IMAGE_EXTENSIONS:
            images.append(path)
    videos.sort(key=lambda path: (path.stat().st_mtime, str(path)))
    images.sort()
    return {"videos": videos, "images": images}


def summarize_results(path: Path) -> dict:
    rows = read_jsonl(path)
    by_status = Counter(str(row.get("status", "unknown")) for row in rows)
    return {
        "generated_at_utc": now_iso(),
        "result_count": len(rows),
        "status_counts": dict(sorted(by_status.items())),
    }


def load_completed_sample_ids(results_path: Path) -> set[str]:
    if not results_path.exists():
        return set()
    completed: set[str] = set()
    for row in read_jsonl(results_path):
        sample_id = row.get("sample_id")
        if sample_id:
            completed.add(str(sample_id))
    return completed


def render_one_row(
    row: dict,
    *,
    row_index: int,
    run_root: Path,
    virtualhome_exe: str | None,
    python_bin: str,
    port_base: int,
    x_display: str | None,
    dry_run: bool,
) -> dict:
    bundle_id = slugify(str(row["sample_id"]))
    work_dir = run_root / "workdirs" / bundle_id
    work_dir.mkdir(parents=True, exist_ok=True)
    original_script = resolve_path(str(row["script_path"]))
    patched_script_path = work_dir / "patched_script.py"
    patched_script_path.write_text(
        patch_script_text(
            original_script.read_text(encoding="utf-8", errors="ignore"),
            virtualhome_exe=virtualhome_exe or "",
            port=port_base + row_index,
            work_dir=work_dir,
            x_display=x_display,
        ),
        encoding="utf-8",
    )

    record = {
        "timestamp_utc": now_iso(),
        "sample_id": str(row["sample_id"]),
        "source_sample_id": str(row.get("source_sample_id") or row["sample_id"]),
        "script_path": str(row["script_path"]),
        "patched_script_path": relative_to_root(patched_script_path),
        "work_dir": relative_to_root(work_dir),
        "port": port_base + row_index,
        "status": "prepared" if dry_run else "unknown",
    }

    if not dry_run:
        process = subprocess.run(
            [python_bin, str(patched_script_path)],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=900,
            env=build_child_env(python_bin, x_display),
        )
        record["returncode"] = process.returncode
        record["stdout_tail"] = process.stdout[-2000:]
        record["stderr_tail"] = process.stderr[-2000:]
        record["status"] = "ok" if process.returncode == 0 else "error"

    media = discover_media_files(work_dir)
    record["video_paths"] = [relative_to_root(path) for path in media["videos"]]
    record["image_paths"] = [relative_to_root(path) for path in media["images"][:24]]

    storyboard = None
    if media["videos"]:
        storyboard = extract_storyboard_bundle_generic(
            media["videos"][-1],
            bundle_id=bundle_id,
            prep_dir=run_root,
        )
    elif media["images"]:
        storyboard = build_storyboard_from_images(
            media["images"],
            bundle_id=bundle_id,
            prep_dir=run_root,
        )
    if storyboard:
        record["storyboard_path"] = storyboard["contact_sheet_path"]
        record["storyboard_frame_paths"] = storyboard["frame_paths"]
        if record["status"] == "error":
            text = f"{record.get('stderr_tail') or ''}\n{record.get('stdout_tail') or ''}"
            if "Read timed out" in text:
                record["status"] = "usable_timeout"
            else:
                record["status"] = "usable_error"
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the VirtualHome script-backed MindPower remainder.")
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_REMAINDER_PREP_DIR / "virtualhome_script_manifest.jsonl",
    )
    parser.add_argument("--render-dir", type=Path, default=DEFAULT_REMAINDER_PREP_DIR / "virtualhome_renders")
    parser.add_argument("--run-name", type=str, default="virtualhome_render")
    parser.add_argument("--sample-ids", nargs="*", default=[])
    parser.add_argument("--limit-samples", type=int, default=0)
    parser.add_argument("--virtualhome-exe", type=Path, default=None)
    parser.add_argument("--python-bin", type=str, default=sys.executable)
    parser.add_argument("--port-base", type=int, default=8080)
    parser.add_argument("--x-display", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.dry_run:
        if args.virtualhome_exe is None:
            raise SystemExit("--virtualhome-exe is required unless --dry-run is set.")
        ensure_virtualhome_available()

    rows = select_rows(load_rows(args.manifest_path), args.sample_ids, args.limit_samples)
    if not rows:
        raise SystemExit("No script-backed samples selected.")

    run_root = args.render_dir / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)
    results_path = run_root / "render_results.jsonl"
    summary_path = run_root / "render_summary.json"
    config_path = run_root / "render_config.json"
    completed_sample_ids = load_completed_sample_ids(results_path)
    pending_rows = [row for row in rows if str(row["sample_id"]) not in completed_sample_ids]

    config = {
        "generated_at_utc": now_iso(),
        "manifest_path": relative_to_root(args.manifest_path),
        "render_dir": relative_to_root(run_root),
        "sample_ids": [str(row["sample_id"]) for row in rows],
        "virtualhome_exe": str(args.virtualhome_exe) if args.virtualhome_exe else None,
        "python_bin": args.python_bin,
        "port_base": args.port_base,
        "x_display": args.x_display,
        "num_workers": args.num_workers,
        "dry_run": bool(args.dry_run),
        "completed_sample_ids_skipped": len(completed_sample_ids),
        "pending_sample_ids": [str(row["sample_id"]) for row in pending_rows],
    }
    write_json(config_path, config)

    if pending_rows:
        pending_with_index = [
            (row_index, row)
            for row_index, row in enumerate(rows)
            if str(row["sample_id"]) not in completed_sample_ids
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
            future_to_sample = {
                executor.submit(
                    render_one_row,
                    row,
                    row_index=row_index,
                    run_root=run_root,
                    virtualhome_exe=str(args.virtualhome_exe) if args.virtualhome_exe else None,
                    python_bin=args.python_bin,
                    port_base=args.port_base,
                    x_display=args.x_display,
                    dry_run=bool(args.dry_run),
                ): str(row["sample_id"])
                for row_index, row in pending_with_index
            }
            for future in concurrent.futures.as_completed(future_to_sample):
                record = future.result()
                with results_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=True) + "\n")

    summary = summarize_results(results_path)
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
