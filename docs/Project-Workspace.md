# Project Workspace

The directory containing a `.pes` file may also contain project add-ons and result directories.

## Layout

```text
converter-study/
├── case.pes
├── another-case.pes
├── PESaddons/
│   ├── control/
│   ├── components/
│   └── functions/
└── output/
    ├── case/
    └── another-case/
```

| Project path | Purpose |
|---|---|
| `PESaddons/control/` | register project control loops |
| `PESaddons/components/` | register project circuit elements |
| `PESaddons/functions/` | provide optional user functions |

`PESaddons` is case-sensitive. PESLite checks only the directory beside the simulation file; it
does not search parent directories or add the project root to global `sys.path`. Add-on code is
executed while the simulation file is loaded and must therefore be treated as trusted Python code.

See [Extending PESLite](Extending-PESLite.md) for extension interfaces and
[Architecture](Architecture.md) for package ownership.

## Running a project

From the project directory:

```bash
cd converter-study
peslite case.pes
```

The default result directory is relative to the directory where the command runs:

```text
output/case/
```

Running the same file by path still finds its sibling `PESaddons`, but the default output remains
relative to the current directory:

```bash
cd /tmp
peslite /work/converter-study/case.pes
```

This writes `/tmp/output/case/`. Select a project-local destination explicitly when needed:

```bash
peslite /work/converter-study/case.pes \
  --out /work/converter-study/output/case
```

The Python API uses the same discovery rule:

```python
params = peslite.load("/work/converter-study/case.pes")
result = peslite.Simulation(params).run(out_dir="/work/converter-study/output/case")
```
