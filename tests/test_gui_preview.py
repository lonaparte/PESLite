"""Real-browser smoke test for the standalone GUI preview."""

import os
import shutil
import signal
import subprocess
import threading
from pathlib import Path

import pytest

from peslite.gui.web_service import LocalServer


pytestmark = [pytest.mark.full, pytest.mark.serial]

ROOT = Path(__file__).resolve().parents[1]
GUI = ROOT / "src/peslite/gui"


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


def test_offline_preview_and_file_workbench_render_in_headless_chromium(tmp_path):
    browser = _browser()
    npm = shutil.which("npm")
    if npm is None:
        pytest.skip("npm is required to build the browser bundle")

    workspace = tmp_path / "gui"
    shutil.copytree(GUI, workspace, ignore=shutil.ignore_patterns(
        "node_modules", "dist", "test-output", "web_dist", "media", "out", "*.vsix"))
    preview = workspace / "preview"
    environment = os.environ.copy()
    environment["npm_config_cache"] = str(tmp_path / "npm-cache")
    try:
        _run([npm, "ci", "--ignore-scripts", "--no-audit", "--no-fund"],
             cwd=preview, environment=environment, timeout=120)
        _run([npm, "run", "build"], cwd=preview, environment=environment, timeout=120)
        page = _run([
            browser,
            "--headless",
            "--no-sandbox",
            "--disable-gpu",
            "--disable-dev-shm-usage",
            "--no-first-run",
            f"--user-data-dir={tmp_path / 'preview-profile'}",
            "--virtual-time-budget=1000",
            "--dump-dom",
            (preview / "demo.html").as_uri(),
        ], cwd=preview, environment=environment, timeout=30)

        assert "PESLite standalone preview" in page.stdout
        assert page.stdout.count('class="pp-node"') == 5
        assert 'aria-label="Event timeline"' in page.stdout
        assert 'role="slider"' in page.stdout
        assert "connect VSC" in page.stdout
        assert "workbench.css" in (preview / "demo.html").read_text(encoding="utf-8")
        assert '<ul class="pp-diagnostics"' not in page.stdout

        ui = workspace / "ui"
        _run([npm, "ci", "--ignore-scripts", "--no-audit", "--no-fund"],
             cwd=ui, environment=environment, timeout=120)
        _run([npm, "run", "build"], cwd=ui, environment=environment, timeout=120)
        project = tmp_path / "project"
        project.mkdir()
        (project / "case.pes").write_text("@circuit\nGrid -- Converter\n", encoding="utf-8")
        (project / "plant.csv").write_text("t,v\n0,1\n", encoding="utf-8")
        (project / "summary.json").write_text('{"stop_reason": null}', encoding="utf-8")
        server = LocalServer(0, project, assets=workspace / "web_dist")
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            workbench = _run([
                browser,
                "--headless",
                "--no-sandbox",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--no-first-run",
                f"--user-data-dir={tmp_path / 'workbench-profile'}",
                "--virtual-time-budget=2000",
                "--dump-dom",
                f"http://127.0.0.1:{server.server_port}/",
            ], cwd=workspace, environment=environment, timeout=30)
            assert "PESLite Workspace" in workbench.stdout
            assert "File-backed simulation workspace" in workbench.stdout
            assert all(name in workbench.stdout for name in
                       ("case.pes", "plant.csv", "summary.json"))
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=3)
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)
