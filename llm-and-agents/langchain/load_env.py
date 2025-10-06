from pathlib import Path
import os


def _find_repo_root(start: Path) -> Path:
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return candidate
    return start


def load_env_file() -> None:
    repo_root = _find_repo_root(Path.cwd())
    env_path = repo_root / ".env"
    example_path = repo_root / ".env.example"
    target = env_path if env_path.exists() else example_path
    if not target.exists():
        raise FileNotFoundError(
            f"Expected either {env_path} or {example_path} to exist."
        )

    with target.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            os.environ.setdefault(key, value)

    print(f"Loaded environment variables from {target.relative_to(repo_root)}")
