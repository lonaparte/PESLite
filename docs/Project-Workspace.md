# Project Workspace

PESLite can treat the directory containing a `.pes` file as an extensible project workspace.

## Software add-ons and the core packages

The installed package keeps the simulation core and extensions separate:

```text
peslite/
├── control/                 built-in controller implementation
├── components/              built-in electrical components
└── addons/
    ├── control/             control-loop extension entry point
    ├── components/          component extension entry point
    └── functions/           optional user-facing functions
```

`peslite.addons.control` exposes the same public loop construction API and the same loop-type
registry as `peslite.control`. A loop registered from either location is consequently validated,
connected and executed in the same way. Similarly, `peslite.addons.components` exposes the public
component API and the same element-type registry as `peslite.components`. The add-on directories
are extension entry points; they do not introduce an add-on wrapper into the simulation hot path.

`peslite.addons.functions` has a different role. It contains optional capabilities around a
simulation, such as plotting completed CSV results. Functions are not controller loops or physical
components, do not enter the model registries and are imported only when requested.

## Project-local `PESaddons`

For a simulation file `/work/study/case.pes`, PESLite checks exactly:

```text
/work/study/PESaddons/
```

When that directory exists, its paths merge with the installed `peslite.addons` paths:

| Project path | Equivalent installed extension path | Behaviour |
|---|---|---|
| `PESaddons/control/` | `peslite/addons/control/` | Modules are imported before the `.pes` file is parsed and register loop types. |
| `PESaddons/components/` | `peslite/addons/components/` | Modules are imported before parsing and register element types. |
| `PESaddons/functions/` | `peslite/addons/functions/` | Modules become importable as optional functions and remain lazy. |

The name is case-sensitive and must be exactly `PESaddons`. The lookup is always relative to the
simulation file, not the current working directory. It does not search a parent directory and does
not add the project directory to global `sys.path`. Multiple `.pes` files in the same directory
share the same `PESaddons`; each available subdirectory is merged independently.

Project add-ons are ordinary Python code and execute while the simulation file is loaded. A project
containing `PESaddons` should therefore be treated as executable code rather than untrusted data.

## Workspace layout

A complete project may use this layout:

```text
converter-study/
├── case.pes
├── another-case.pes
├── PESaddons/
│   ├── control/
│   │   └── project_controller.py
│   ├── components/
│   │   └── project_component.py
│   └── functions/
│       └── postprocess.py
└── output/
    ├── case/
    │   ├── states.csv
    │   ├── summary.json
    │   └── simulation.pes
    └── another-case/
```

Run from the project directory to obtain this arrangement naturally:

```bash
cd converter-study
peslite case.pes
```

The three relevant paths follow two rules:

- `case.pes` anchors the `PESaddons/` lookup;
- the process working directory anchors the default `output/<case-name>/` result path.

Therefore this also finds the same project extensions:

```bash
cd /tmp
peslite /work/converter-study/case.pes
```

but its default results go to `/tmp/output/case/`. Select a project-local result directory
explicitly when running from elsewhere:

```bash
peslite /work/converter-study/case.pes \
  --out /work/converter-study/output/case
```

The Python API follows the same loading rule. `peslite.load("path/case.pes")` discovers the sibling
`PESaddons`, while `Simulation.run()` uses `output/run` unless an `out_dir` is supplied.
