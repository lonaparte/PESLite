# PESLite file-backed UI

The frontend is optional and has no effect on the PESLite simulator, command line,
numerical outputs or `python -m peslite.frontend` JSON-lines protocol. Its
**two hosts** use the same React/TypeScript UI and the standalone native SVG
preview renderer in [`preview/`](preview/).

## Local Web

Install the `peslite-frontend` distribution and launch:

```sh
peslite-frontend --root ./my-project --port 8765
peslite-frontend --root ./my-project --no-browser
```

The server binds exclusively to `127.0.0.1`, validates the workspace root and
relative paths, rejects symlinks/traversal, and requires a per-session token for
all file and command requests. Original files remain authoritative. Text saves
require the matching SHA-256 version and use an atomic replacement; safe
parameter changes and supported component or connection edits are checked against
PESLite's existing parameter validator before applying one source-range edit.
The UI can add a validated component mapping, remove an unreferenced declaration
without discarding comments, and reconnect an explicit network terminal to an
existing bus. Ambiguous edits are rejected instead of rewriting YAML. Conflicting
file versions return an error.

The directory tree supports opening, creating, renaming and deleting files;
CodeMirror 6 edits original `.pes` text, and the shared SVG renderer shows a
preview of unsaved buffers. Existing and growing simulation CSV outputs can be
read in bounded chunks with resumable byte cursors, per-file time columns,
downsampling and Canvas plots. The viewer can overlay signals from another CSV
using that file's own time samples and can follow new rows written by another
process without loading the entire CSV. Summary JSON links to the corresponding resolved `simulation.pes`.

Run/Stop invokes the existing simulator CLI. The UI does **not** execute
project-local `PESaddons` while previewing; running a model with project-local
Python addons requires explicit trust. Full validation of custom project addons
requires installing/trusting them in the locally selected Python environment.

## Build the offline UI

From the repo root:

```sh
cd src/peslite/frontend/ui
npm install
npm run build
```

Vite writes a self-contained bundle into `../web_dist`. No CDN assets are used.
The Python distribution stages this bundle in `peslite.frontend`; the simulator
core has **no** React, editor or browser dependencies. Release packaging builds
the UI before wheel staging.

## VS Code

From `src/peslite/frontend/vscode`:

```sh
npm install
npm run build
```

The extension copies the exact same built UI to `media/`. Package with VS Code's
extension tooling after the build. The VSIX ships **no Python or numerical
runtime**. Select a Python interpreter containing PESLite using
`peslite.pythonPath`, open `.pes`, `.csv` or `summary.json`, then run
**PESLite: Preview File**. The extension uses the VS Code text editor for
unsaved YAML, applies validated graphical edits via `WorkspaceEdit` (preserving
undo/redo and dirty state), and uses the same lightweight message contract as
local Web. The extension uses VS Code's own Explorer and editor; it does not
embed a second CodeMirror editor.

## Legacy-YAML function addon

PESLite v0.5.1's regular simulation YAML is different from the new `@circuit`
preview notation. The shared TypeScript renderer remains browser-only. The
optional **function addon** [`../addons/functions/legacy_yaml.py`](../addons/functions/legacy_yaml.py)
converts a validated v0.5.1 model to its normalized graph model and supplies
source locations for safe graphical edits. Both host adapters invoke it
through lazy functions in `peslite.addons.functions` (`legacy_yaml_preview`,
`legacy_yaml_patch`, `legacy_yaml_add_component`,
`legacy_yaml_remove_component`, `legacy_yaml_rewire`). This uses the existing PESLite parser and topology
structures instead of duplicating the schema.

Unsupported source constructs remain in the text editor and cannot be
silently rewritten. The browser-only renderer stays fully independent of
PESLite's Python/NumPy/SciPy runtime.

## Validation

The frontend compatibility workflow checks the Python file and addon tests,
standalone SVG tests, Web UI and VS Code production builds, offline asset
staging, and the unchanged JSON-lines integration. Locally, the isolated
workspace checks can be run with:

```sh
python -m pytest -q tests/test_frontend_workspace_files.py
```

The other frontend tests require the installed PESLite v0.5.1 development
runtime. All source edits are optimistic: a graphical operation fails without
writing when the source was changed concurrently or when PESLite rejects the
resulting configuration. New `@circuit` syntax remains text-editable and is not
converted into legacy v0.5.1 simulation input by the frontend.
