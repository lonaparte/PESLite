"""Command line: ``python peslite.py [config] [--set PATH=VALUE ...]`` (``--help`` for all options).

This folder is the ``peslite`` package. Run as a script, or imported from this folder, this file
loads the package from here under the name ``peslite``, so neither needs an installation and the
folder may have any name.
"""

if __package__:  # imported as a submodule of the installed package
    from .simulation import main  # noqa: F401
else:
    import importlib.util
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parent
    _spec = importlib.util.spec_from_file_location("peslite", _root / "__init__.py",
                                                   submodule_search_locations=[str(_root)])
    _package = importlib.util.module_from_spec(_spec)
    sys.modules["peslite"] = _package  # `import peslite` from this folder returns the package
    _spec.loader.exec_module(_package)
    if __name__ == "__main__":
        raise SystemExit(_package.main())
