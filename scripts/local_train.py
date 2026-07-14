from __future__ import annotations

import os
import subprocess
from pathlib import Path

# ========== 完全移除 modal 所有导入与相关对象 ==========
APP_NAME = "hw4-llm-rl"
NETRC_PATH = Path("~/_netrc").expanduser()
PROJECT_DIR = str(Path(__file__).resolve().parent.parent)  # 本地项目根目录
VOLUME_PATH = PROJECT_DIR  # 本地直接用项目目录替代挂载卷
DEFAULT_TIMEOUT_SECONDS = 60 * 60 * 24
DEFAULT_VOLUME_COMMIT_INTERVAL_SECONDS = 300

# 本地无需Modal Volume，直接空对象占位，仅保留接口兼容
class LocalVolumeDummy:
    def commit(self):
        pass

volume = LocalVolumeDummy()


def load_gitignore_patterns() -> list[str]:
    """Translate .gitignore entries into Modal ignore globs.（本地该函数不再被使用，保留原函数不删）"""
    root = Path(__file__).resolve().parents[1]
    gitignore_path = root / ".gitignore"
    if not gitignore_path.is_file():
        return []

    patterns: list[str] = []
    for line in gitignore_path.read_text(encoding="utf-8").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or entry.startswith("!"):
            continue
        entry = entry.lstrip("/")
        if entry.endswith("/"):
            entry = entry.rstrip("/")
            patterns.append(f"**/{entry}/**")
        else:
            patterns.append(f"**/{entry}")
    return patterns


def _to_volume_path(path_value: str) -> str:
    p = Path(path_value)
    if p.is_absolute():
        p_str = str(p)
        if p_str != VOLUME_PATH and not p_str.startswith(f"{VOLUME_PATH}/"):
            print(
                f"[local][warning] path '{p_str}' is outside '{VOLUME_PATH}'. "
                "Files written there may not persist after the run."
            )
        return path_value
    return str(Path(VOLUME_PATH) / p)


def _rewrite_path_flag(
    args: list[str], flag: str, *, default_relative_if_missing: str | None = None
) -> list[str]:
    out = list(args)
    found = False
    i = 0
    while i < len(out):
        token = out[i]
        if token == flag:
            found = True
            if i + 1 >= len(out):
                raise ValueError(f"Missing value for {flag}")
            out[i + 1] = _to_volume_path(out[i + 1])
            i += 2
            continue
        if token.startswith(f"{flag}="):
            found = True
            key, value = token.split("=", 1)
            out[i] = f"{key}={_to_volume_path(value)}"
        i += 1

    if not found and default_relative_if_missing is not None:
        out.extend([flag, _to_volume_path(default_relative_if_missing)])
    return out


def _normalize_modal_args(args: tuple[str, ...], *, is_eval: bool) -> list[str]:
    normalized = list(args)
    normalized = _rewrite_path_flag(
        normalized,
        "--output_dir",
        default_relative_if_missing=None if is_eval else "runs/default",
    )
    normalized = _rewrite_path_flag(normalized, "--adapter_path")
    return normalized


def _normalize_bundle_args(args: tuple[str, ...]) -> list[str]:
    normalized = list(args)
    normalized = _rewrite_path_flag(normalized, "--run_dir")
    normalized = _rewrite_path_flag(
        normalized,
        "--output_dir",
        default_relative_if_missing="submissions/hw4_gradescope_submission",
    )
    return normalized


def _is_wandb_enabled_for_train_args(args: tuple[str, ...] | list[str]) -> bool:
    enabled = True
    for token in args:
        if token == "--no-wandb_enabled":
            enabled = False
        elif token == "--wandb_enabled":
            enabled = True
    return enabled


def _assert_wandb_credentials_available_if_needed(args: tuple[str, ...] | list[str]) -> None:
    if not _is_wandb_enabled_for_train_args(args):
        return
    has_netrc = Path(Path.home() / ".netrc").is_file()
    has_api_key_env = bool(os.environ.get("WANDB_API_KEY"))
    if not has_netrc and not has_api_key_env:
        raise RuntimeError(
            "W&B logging is enabled for training, but no credentials were found locally. "
            "Run `wandb login` to generate ~/.netrc, or export WANDB_API_KEY, "
            "or pass `--no-wandb_enabled` to disable wandb."
        )


def _run_subprocess_with_periodic_volume_commits(cmd: list[str]) -> None:
    # 原逻辑完全保留，仅volume.commit变成空函数
    proc = subprocess.Popen(cmd, cwd=PROJECT_DIR, env=os.environ.copy())
    returncode: int | None = None
    try:
        while returncode is None:
            try:
                returncode = proc.wait(timeout=DEFAULT_VOLUME_COMMIT_INTERVAL_SECONDS)
            except subprocess.TimeoutExpired:
                volume.commit()
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        volume.commit()

    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


# 本地环境变量直接继承系统环境，只追加固定项
env = os.environ.copy()
env.update({
    "PYTHONPATH": PROJECT_DIR,
    "PYTHONUNBUFFERED": "1",
    "WANDB_DIR": f"{VOLUME_PATH}/wandb",
    "HF_HOME": f"{VOLUME_PATH}/hf",
    "HF_DATASETS_CACHE": f"{VOLUME_PATH}/hf/datasets",
    # 强制指定使用单GPU 0（你的5060一般是cuda:0）
    "CUDA_VISIBLE_DEVICES": "0"
})


# ========== 移除所有 @app.function 远端装饰器，改为普通本地函数 ==========
def train_remote(*args: str) -> None:
    normalized_args = _normalize_modal_args(args, is_eval=False)
    _assert_wandb_credentials_available_if_needed(normalized_args)
    cmd = ["python", "-u", "-m", "hw4.train", *normalized_args]
    _run_subprocess_with_periodic_volume_commits(cmd)


def eval_remote(*args: str) -> None:
    normalized_args = _normalize_modal_args(args, is_eval=True)
    cmd = ["python", "-u", "-m", "hw4.eval", *normalized_args]
    _run_subprocess_with_periodic_volume_commits(cmd)


def bundle_submission_remote(*args: str) -> None:
    normalized_args = _normalize_bundle_args(args)
    cmd = ["python", "-u", "-m", "hw4.gradescope_bundle", *normalized_args]
    _run_subprocess_with_periodic_volume_commits(cmd)


def main(*args: str) -> None:
    """Default entrypoint: forward args to train_remote."""
    if _is_wandb_enabled_for_train_args(args) and not NETRC_PATH.is_file() and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "W&B logging is enabled (default), but no credentials were detected locally. "
            "Run `wandb login` (creates ~/.netrc), or export WANDB_API_KEY before running, "
            "or pass `--no-wandb_enabled`."
        )
    train_remote(*args)


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:])
