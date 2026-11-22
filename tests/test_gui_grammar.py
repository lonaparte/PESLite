"""Keep the browser parser's generated lexical rules aligned with native PES syntax."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_browser_grammar_is_generated_from_native_syntax():
    script = ROOT / "scripts/sync_gui_grammar.py"
    spec = importlib.util.spec_from_file_location("sync_gui_grammar", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.TARGET.read_text(encoding="utf-8") == module.render()
