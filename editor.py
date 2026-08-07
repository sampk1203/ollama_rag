from pathlib import Path
from config import WORKSPACE_DIR

SANDBOX = Path(WORKSPACE_DIR).resolve()

def _safe_path(rel_path):
    target = (SANDBOX / rel_path).resolve()
    if not str(target).startswith(str(SANDBOX)):
        raise PermissionError(f"Path escape attempt: {rel_path}")
    return target

def read_file(rel_path):
    p = _safe_path(rel_path)
    return p.read_text() if p.exists() else None

def write_file(rel_path, content):
    p = _safe_path(rel_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return str(p)

def list_workspace():
    return [str(f.relative_to(SANDBOX)) for f in SANDBOX.rglob("*") if f.is_file()]

def edit_file(rel_path, old_str, new_str):
    content = read_file(rel_path)
    if content is None:
        raise FileNotFoundError(rel_path)
    if old_str not in content:
        raise ValueError(f"String not found in {rel_path}")
    _safe_path(rel_path).write_text(content.replace(old_str, new_str, 1))
