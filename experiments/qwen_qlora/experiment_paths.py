from __future__ import annotations

from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[1]
DEFAULT_CONFIG = EXPERIMENT_DIR / "config.yaml"


def resolve_config_path(value: str | Path) -> Path:
    """支持绝对路径、当前目录相对路径和项目根目录相对路径。"""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path

    current_relative = (Path.cwd() / path).resolve()
    if current_relative.exists():
        return current_relative

    project_relative = (PROJECT_ROOT / path).resolve()
    if project_relative.exists():
        return project_relative

    return current_relative


def project_path(value: str | Path) -> Path:
    """配置文件中的相对路径统一相对于项目根目录。"""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

