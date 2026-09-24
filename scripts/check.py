import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def run(*command: str) -> None:
    subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=True,
    )

def main() -> None:
    run("ruff", "check", ".")
    run("pyrefly", "check", "src", "evals")

if __name__ == "__main__":
    main()