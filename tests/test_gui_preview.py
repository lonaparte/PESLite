"""Real-browser smoke test for the standalone GUI preview."""

import os
import shutil
import signal
import subprocess
from pathlib import Path

import pytest


pytestmark = [pytest.mark.full, pytest.mark.serial]

ROOT = Path(__file__).resolve().parents[1]
PREVIEW = ROOT / "src/peslite/gui/preview"


def _browser() -> str:
    configured = os.environ.get("PESLITE_CHROMIUM")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_file():
            pytest.fail(f"PESLITE_CHROMIUM does not name a file: {path}")
        return str(path)
    executable = shutil.which("chromium")
    if executable is None:
        pytest.skip("set PESLITE_CHROMIUM to a Chromium or headless-shell executable")
    return executable


def _run(command: list[str], *, cwd: Path, environment: dict[str, str],
         timeout: float) -> subprocess.CompletedProcess[str]:
    """Run one bounded process group so browser and build children cannot survive a timeout."""
    process = subprocess.Popen(
        command, cwd=cwd, env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, start_new_session=os.name == "posix",
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:  # pragma: no cover - exercised by Windows development environments
                process.terminate()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:  # pragma: no cover - exercised by Windows development environments
                    process.kill()
            except ProcessLookupError:
                pass
            process.wait()
        raise
    result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
    result.check_returncode()
    return result


def test_offline_preview_renders_in_headless_chromium(tmp_path):
    browser = _browser()
    npm = shutil.which("npm")
    if npm is None:
        pytest.skip("npm is required to build the browser bundle")

    workspace = tmp_path / "preview"
    shutil.copytree(PREVIEW, workspace, ignore=shutil.ignore_patterns(
        "node_modules", "dist", "test-output"))
    environment = os.environ.copy()
    environment["npm_config_cache"] = str(tmp_path / "npm-cache")
    try:
        _run([npm, "ci", "--ignore-scripts", "--no-audit", "--no-fund"],
             cwd=workspace, environment=environment, timeout=120)
        _run([npm, "run", "build"], cwd=workspace, environment=environment, timeout=120)
        page = _run([
            browser,
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--no-first-run",
            f"--user-data-dir={workspace / 'browser-profile'}",
            "--virtual-time-budget=1000",
            "--dump-dom",
            (workspace / "demo.html").as_uri(),
        ], cwd=workspace, environment=environment, timeout=30)

        assert "PESLite standalone preview" in page.stdout
        assert page.stdout.count('class="pp-node"') == 5
        assert 'aria-label="Event timeline"' in page.stdout
        assert 'role="slider"' in page.stdout
        assert "connect VSC" in page.stdout
        assert "workbench.css" in (workspace / "demo.html").read_text(encoding="utf-8")
        assert '<ul class="pp-diagnostics"' not in page.stdout
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
