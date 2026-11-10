# PES text-to-SVG preview

A small browser-only library for PESLite's `@circuit` / `@UnitName` views and YAML parameter domains. All source, build files, examples and tests live under `src/peslite/gui/preview/`. No Python, React, CodeMirror, uPlot, external web assets or simulation runtime are used by the renderer.

## Build and try it

From this folder:

```sh
npm ci
npm test
npm run build
```

Open `demo.html` in a browser, even through a `file://` URL. The demo uses only the local IIFE bundle in `dist/` and re-parses edited text without saving it. `dist/` is a build output, not committed source. Builds produce both `pes-preview.es.js` for imports and `pes-preview.iife.js` for script-tag embeds.

## Host API

```ts
import { createPESPreview } from './dist/pes-preview.es.js';

const preview = createPESPreview(container, {
  onSourceSelect: ({ line, column }) => console.log('Navigate to', line, column),
  onDiagnostics: errors => console.log(errors),
});

preview.update(editor.getText()); // accepts unsaved buffers, no file I/O
preview.navigate('circuit');
preview.getModel();
preview.destroy();
```

In local Web the host reads a file or text-editor buffer and passes that string to `update`. In a VS Code Webview, the extension host posts the active `TextDocument.getText()` (including dirty/unsaved edits) to the same bundle via `webview.postMessage`, and calls `update` in the Webview. Hosts own all file access, editors, simulation and permissions. The library never sends network requests or writes to source files; dynamic text is inserted via `textContent`, not `innerHTML`. Both hosts use the same built asset and receive the same graph/layout for the same input text.

The parser exposes `parsePES(text)`, `layoutView(view)` and `buildTimeline(events, duration)` separately from DOM rendering for host adapters and tests. Bad/incomplete input returns source-located diagnostics and a partial graph instead of rewriting YAML.

### Supported notation

- `@circuit` (default view) and `@VSC` (unit/control view); only rendered views use `@`
- `base:`, `simulation:`, `events:` and other parameter domains use YAML mappings and indentation
- `A -- B`, `A -> B`, chained paths and named ports `A.port -> B.port`; repeated names identify the same node within a view
- `Name: Type` and `Name: plant_type, control_type`, followed by indented parameter blocks; loop definitions live under their unit's parameter block
- `&Group: A, B` and nested `&Outer: &Group, C` boundaries, with independent folding
- Timeline segments at all `connect` / `disconnect` event times, including separate adjacent segments whose topologies match; selecting a segment only changes link state, not positions
- Text-order layout, pan/zoom, clickable unit navigation, folded parameter panels and optional source-location callbacks

### Deliberate scope limits

This is a **preview syntax parser**, not a general YAML parser or PESLite numerical validator. YAML parameter bodies are preserved as source snippets and parsed only as needed for global event/time metadata. Complex YAML anchors, implicit port inference, legacy `.pes` simulation YAML and component/connection editing are intentionally outside this library. Those belong to a separate host/adapter and the PESLite backend. No Python addon execution occurs when previewing text.
