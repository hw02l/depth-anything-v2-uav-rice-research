"""Persistent CPU preparation state; validation never opens image/GT pixel payloads."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path

SCHEMA_VERSION = 1
CODE_DIR = Path(__file__).resolve().parent
DEFAULT_SETTINGS = {
    "TRAINING_MODES": ["relative_encoder"], "EPOCHS": 1, "IMAGE_SIZE": 518,
    "BATCH_SIZE": 1, "GRAD_ACCUMULATION": 4, "USE_AMP": True, "SEED": 42,
    "WORKERS": 2, "MAX_DEPTH_M": None, "RUN_LABEL": "trial_s42_res518_bs1_acc4",
    "RUN_KIND": "trial", "RESUME_TRAINING": False,
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_file(root: Path, relative: str) -> Path:
    root = Path(root).resolve()
    rel = Path(relative)
    result = (root / rel).resolve()
    if rel.is_absolute() or ".." in rel.parts or not result.is_relative_to(root):
        raise ValueError(f"保存先外のパスです: {relative}")
    return result


def preparation_context(manifests, rgb_root, depth_root, physical_splits,
                        kind="audit", parameters=None, extra_files=None):
    """Compare manifests, roots, code, conditions, and file size/mtime on every reuse."""
    manifests = {k: Path(v) for k, v in manifests.items()}
    code_names = ["runtime_state.py", "data.py", "audit_data.py" if kind == "audit" else "analyze_gt.py"]
    inventory = {}
    problems = []
    for split in physical_splits:
        with manifests[split].open(encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"空のmanifest: {split}")
        for row in rows:
            pairs = [(depth_root, "depth_relpath", "depth_size_bytes")]
            if rgb_root is not None:
                pairs.append((rgb_root, "image_relpath", "image_size_bytes"))
            for root, key, size_key in pairs:
                path = safe_file(Path(root), row[key])
                if not path.is_file():
                    problems.append(f"未取得: {path}")
                    continue
                stat = path.stat()
                expected = row.get(size_key)
                if expected and stat.st_size != int(expected):
                    problems.append(f"サイズ不一致: {path} (期待{expected}, 実際{stat.st_size})")
                inventory[f"{split}:{key}:{row[key]}"] = {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if problems:
        raise ValueError(f"必要ファイルの検査に失敗しました（{len(problems)}件）。" + "\n" + "\n".join(problems[:5]))
    packages = {}
    for name in ("numpy", "tifffile", "opencv-python-headless", "opencv-python"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {
        "schema": SCHEMA_VERSION, "kind": kind,
        "manifest_sha256": {k: sha256(v) for k, v in manifests.items()},
        "rgb_root": str(Path(rgb_root).resolve()) if rgb_root is not None else None,
        "depth_root": str(Path(depth_root).resolve()), "physical_splits": sorted(physical_splits),
        "parameters": parameters or {}, "code_sha256": {n: sha256(CODE_DIR / n) for n in code_names},
        "extra_sha256": {k: sha256(Path(p)) if p is not None and Path(p).is_file() else None
                         for k, p in (extra_files or {}).items()},
        "packages": packages, "files": inventory,
    }


def reuse_report(report, **context_args):
    try:
        if report.get("status") != "pass":
            return False, "正常完了したレポートではありません。"
        old = report.get("reuse_context")
        if not old:
            return False, "再利用検証情報がありません。新しい仕様で再実行してください。"
        current = preparation_context(**context_args)
        changed = [key for key in current if current[key] != old.get(key)]
        return (False, "再利用条件が変わりました: " + ", ".join(changed)) if changed else (True, "manifest・保存先・サイズ・更新日時・仕様・条件が一致しました。")
    except (OSError, ValueError, KeyError) as exc:
        return False, str(exc)


def find_report(directory, pattern, **context_args):
    reasons = []
    for path in sorted(Path(directory).glob(pattern), key=lambda p: p.stat().st_mtime_ns, reverse=True):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
            ok, reason = reuse_report(report, **context_args)
            if ok:
                return path, report, reason
            reasons.append(f"{path.name}: {reason}")
        except (OSError, ValueError) as exc:
            reasons.append(f"{path.name}: {exc}")
    return None, None, reasons[0] if reasons else "有効な保存済みレポートがありません。"


def validate_settings(settings):
    config = dict(DEFAULT_SETTINGS) | settings
    if not isinstance(config["TRAINING_MODES"], list) or not config["TRAINING_MODES"] or len(set(config["TRAINING_MODES"])) != len(config["TRAINING_MODES"]) or not set(config["TRAINING_MODES"]) <= {"relative_encoder", "metric_outdoor"}:
        raise ValueError("TRAINING_MODESを確認してください。")
    for key in ("EPOCHS", "IMAGE_SIZE", "BATCH_SIZE", "GRAD_ACCUMULATION"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"{key}は正の整数で指定してください。")
    if config["IMAGE_SIZE"] < 56 or config["IMAGE_SIZE"] % 14:
        raise ValueError("解像度は56以上の14の倍数で指定してください。")
    if config["RUN_KIND"] not in ("trial", "full") or (config["RUN_KIND"] == "trial" and config["EPOCHS"] != 1):
        raise ValueError("試験runはEPOCHS=1です。本学習はRUN_KIND='full'と新しいRUN_LABELを設定してください。")
    if not isinstance(config["RUN_LABEL"], str) or not config["RUN_LABEL"] or config["RUN_LABEL"] in (".", "..") or any(x in config["RUN_LABEL"] for x in ("/", "\\")):
        raise ValueError("RUN_LABELには単一の新しいフォルダー名を指定してください。")
    if config["MAX_DEPTH_M"] is not None and (not isinstance(config["MAX_DEPTH_M"], (int, float)) or not math.isfinite(config["MAX_DEPTH_M"]) or config["MAX_DEPTH_M"] <= 0):
        raise ValueError("MAX_DEPTH_Mはtrain GT解析後に正のメートル値で指定してください。")
    if type(config["SEED"]) is not int or not 0 <= config["SEED"] < 2**32:
        raise ValueError("SEEDは0以上2**32未満の整数で指定してください。")
    if type(config["WORKERS"]) is not int or config["WORKERS"] < 0:
        raise ValueError("WORKERSは0以上の整数で指定してください。")
    if type(config["USE_AMP"]) is not bool or type(config["RESUME_TRAINING"]) is not bool:
        raise ValueError("USE_AMPとRESUME_TRAININGはTrue/Falseで指定してください。")
    return config


def load_settings(path):
    path = Path(path)
    try:
        return validate_settings(json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {})
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Driveの学習設定を読み込めません: {path}。JSONと設定値を確認してください。原因: {exc}") from exc


def save_settings(path, settings):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(validate_settings(settings), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def dataset_status(plan_dir, rgb_root, depth_root, splits=("train", "validation")):
    result = {"pairs": 0, "complete_pairs": 0, "expected_files": 0, "present_files": 0,
              "missing_files": 0, "conflicting_files": [], "planned_bytes": 0, "new_bytes": 0}
    for split in splits:
        with (Path(plan_dir) / f"{split}.csv").open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                result["pairs"] += 1
                complete = True
                for root, key, sizekey in ((rgb_root, "image_relpath", "image_size_bytes"),
                                           (depth_root, "depth_relpath", "depth_size_bytes")):
                    target = safe_file(root, row[key])
                    size = int(row[sizekey])
                    result["planned_bytes"] += size
                    result["expected_files"] += 1
                    if not target.is_file():
                        result["missing_files"] += 1
                        result["new_bytes"] += size
                        complete = False
                    elif target.stat().st_size != size:
                        result["conflicting_files"].append(str(target))
                        complete = False
                    else:
                        result["present_files"] += 1
                result["complete_pairs"] += int(complete)
    result["complete"] = result["complete_pairs"] == result["pairs"] and not result["conflicting_files"]
    return result
