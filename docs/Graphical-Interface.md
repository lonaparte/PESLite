# Graphical Interface

The PESLite graphical user interface is a source-first workbench for browsing, editing and
previewing simulation files. The `.pes` file remains the authoritative model: the interface reads
the current text buffer, and visual operations never create a separate simulation definition.

## Install and open

Install the GUI distribution directly or as part of `peslite-full`; the complete installation
matrix is listed in [Getting Started](Getting-Started.md#install).

```bash
python -m pip install peslite-gui
python -m peslite.gui.preview_host case.pes
```

Open the local address printed by the command. Passing a directory instead of a file opens that
directory as the workspace:

```bash
python -m peslite.gui.preview_host cases/
```

The host listens only on the local loopback interface.

## Workbench

The workbench combines a workspace explorer, text editor and graphical preview. Selecting a file
loads its source and preview together; editor changes are rendered immediately, and saving writes
only the edited source. `Ctrl+B` toggles the explorer and `Ctrl+S` saves the active file. The panes
can be resized independently.

The preview provides:

- tabs for the circuit and detailed Unit or subsystem views;
- collapsible groups, named ports and orthogonal connections;
- source navigation from displayed objects and source-located diagnostics;
- zoom, pan and fit controls;
- an event timeline that shows connect and disconnect states at the selected time.

Simulation-file and view notation is defined in
[Simulation Files](Simulation-Files.md#interface-views).

## Rendering

The preview uses one browser-side rendering pipeline:

1. The current text buffer is parsed into source-located views, objects, connections, groups,
   events and diagnostics.
2. Automatic layout assigns positions to objects and nested groups.
3. The router selects connection sides and produces orthogonal paths.
4. The renderer draws the resulting graph as SVG and updates it when the source, active view or
   event time changes.

Parsing, layout, routing and timeline calculations are independent of the DOM renderer. Incomplete
input therefore produces a partial preview with diagnostics instead of rewriting the source. The
same renderer is used by the local workbench and can be embedded by another host without changing
the simulation format.

## Layout state

The initial arrangement is automatic. Dragging an object or group records an offset from that
arrangement, with grid snapping and nearby alignment. These offsets are presentation state: they
do not modify the `.pes` file.

The local host keeps layout offsets in a temporary cache keyed by the file's workspace-relative
path. Closing the page or host removes the cache. **Fit** changes only the viewport, while
**Reset layout** discards the offsets and restores automatic placement.

## Validation boundary

The preview parser provides immediate structural diagnostics without loading Python addons or
running a simulation. Numerical validation and resolved parameter checks remain the responsibility
of `peslite inspect` and `peslite run`.
