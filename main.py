from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    app_path = Path(__file__).with_name("app.py").resolve()
    streamlit_options = [
        "--server.headless=false",
        "--browser.gatherUsageStats=false",
    ]
    print(f"Starting Jev evaluation UI from {app_path.parent}")
    print(f"Python: {sys.version.split()[0]} | Platform: {sys.platform}")

    try:
        from streamlit.web import cli as streamlit_cli
    except ModuleNotFoundError:
        uv = shutil.which("uv")
        if not uv:
            print("Streamlit is not installed and uv was not found on PATH.", file=sys.stderr)
            print("Install uv, then run: python main.py", file=sys.stderr)
            return 1
        child_env = os.environ.copy()
        child_env.pop("VIRTUAL_ENV", None)
        child_env.setdefault("UV_CACHE_DIR", str(app_path.parent / ".uv-cache"))
        return subprocess.call(
            [uv, "run", "python", "-m", "streamlit", "run", "app.py", *streamlit_options],
            cwd=app_path.parent,
            env=child_env,
        )

    os.chdir(app_path.parent)
    sys.argv = ["streamlit", "run", "app.py", *streamlit_options]
    return int(streamlit_cli.main() or 0)


if __name__ == "__main__":
    raise SystemExit(main())
