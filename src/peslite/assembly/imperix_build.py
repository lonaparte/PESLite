"""Build an Imperix ELF executable with an explicit official-SDK command; standard library only.

This module is also emitted as build.py so a source-only export can be built on the SDK machine
without installing PESLite. Compiler, ABI, includes, linker script and BBOS libraries belong to
the installed vendor project, not to a second toolchain guessed by the simulator.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile


def _elf(path, machine):
    data = path.read_bytes()
    if len(data) < 52 or data[:4] != b"\x7fELF" or data[4] not in (1, 2) or data[5] not in (1, 2):
        raise RuntimeError("SDK build did not produce an ELF executable")
    order = "<" if data[5] == 1 else ">"
    actual = struct.unpack_from(order + "H", data, 18)[0]
    if machine not in (40, 183) or actual != machine:
        raise RuntimeError(f"SDK ELF machine mismatch: expected ARM=40 or AArch64=183 ({machine}), got {actual}")
    if data[4] != (1 if machine == 40 else 2):
        raise RuntimeError("SDK ELF word size does not match the requested architecture")
    kind = struct.unpack_from(order + "H", data, 16)[0]
    if kind != 2:
        raise RuntimeError("SDK artifact must be an ELF executable, not an object or host library")
    offsets = (24, 28, 32, 42, 44, 46, 48) if data[4] == 1 else (24, 32, 40, 54, 56, 58, 60)
    address = "I" if data[4] == 1 else "Q"
    entry, phoff, shoff = [struct.unpack_from(order + address, data, at)[0] for at in offsets[:3]]
    phsize, phnum, shsize, shnum = [struct.unpack_from(order + "H", data, at)[0] for at in offsets[3:]]
    if (entry == 0 or phnum == 0 or shnum == 0
            or phsize < (32 if data[4] == 1 else 56) or shsize < (40 if data[4] == 1 else 64)
            or phoff + phsize * phnum > len(data) or shoff + shsize * shnum > len(data)):
        raise RuntimeError("SDK ELF is missing its entry point, program or symbol/debug section table")
    if not any(struct.unpack_from(order + "I", data, phoff + index * phsize)[0] == 1
               for index in range(phnum)):
        raise RuntimeError("SDK ELF has no loadable segment")
    if not any(struct.unpack_from(order + "I", data, shoff + index * shsize + 4)[0] == 2
               for index in range(shnum)):
        raise RuntimeError("SDK ELF was stripped; Cockpit requires retained symbols")
    names_index = struct.unpack_from(order + "H", data, 50 if data[4] == 1 else 62)[0]
    if names_index >= shnum:
        raise RuntimeError("SDK ELF has an invalid section-name table")
    names_header = shoff + names_index * shsize
    names_offset, names_size = struct.unpack_from(order + address * 2, data,
                                                 names_header + (16 if data[4] == 1 else 24))
    if names_offset + names_size > len(data):
        raise RuntimeError("SDK ELF section names extend past the artifact")
    names = data[names_offset:names_offset + names_size]
    sections = []
    for index in range(shnum):
        offset = struct.unpack_from(order + "I", data, shoff + index * shsize)[0]
        sections.append(names[offset:].split(b"\0", 1)[0])
    if not any(name in sections for name in (b".debug_info", b".zdebug_info")):
        raise RuntimeError("SDK ELF has no debug information; retain the SDK project's debug flags")
    return hashlib.sha256(data).hexdigest()


def build_project(directory):
    directory = Path(directory).resolve()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(status="building", artifact=None, sha256=None)
    manifest.pop("error", None)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    try:
        profile = json.loads((directory / "imperix-build.json").read_text(encoding="utf-8"))
        if not isinstance(profile, dict):
            raise ValueError("imperix-build.json must contain a mapping")
        unknown = profile.keys() - {"name", "sdk", "command", "elf_machine", "timeout"}
        if unknown:
            raise ValueError(f"unknown SDK build fields: {sorted(unknown)}")
        sdk = profile.get("sdk") or os.environ.get("PESLITE_IMPERIX_SDK")
        if not sdk or not Path(sdk).is_dir():
            raise ValueError("an installed official Imperix SDK directory is required; "
                             "set build.sdk, --imperix-sdk or PESLITE_IMPERIX_SDK; "
                             "use --imperix-source-only to generate sources without building")
        command = profile.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(arg, str) and arg for arg in command):
            raise ValueError("build.command must be an argument list for the official SDK build")
        if not any("{elf}" in arg for arg in command):
            raise ValueError("build.command must write its fresh output to {elf}")
        machine = profile.get("elf_machine")
        if type(machine) is not int or machine not in (40, 183):
            raise ValueError("build.elf_machine must match the installed SDK: ARM=40 or AArch64=183")
        timeout = profile.get("timeout", 300.0)
        if isinstance(timeout, bool) or not isinstance(timeout, (float, int)) or not 0 < timeout <= 3600:
            raise ValueError("build.timeout must be in (0, 3600] seconds")
        name = profile.get("name", "run")
        if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
            raise ValueError("the export name must be a single filename stem")
        with tempfile.TemporaryDirectory(prefix=".build-", dir=directory) as staging:
            elf = Path(staging) / (name + ".elf")
            values = dict(sdk=str(Path(sdk).resolve()), source=str(directory), elf=str(elf))
            args = [arg.format_map(values) for arg in command]
            (directory / "build-command.json").write_text(json.dumps(args, indent=2) + "\n", encoding="utf-8")
            with (directory / "build.log").open("w", encoding="utf-8") as stream:
                result = subprocess.run(args, cwd=directory, stdout=stream, stderr=subprocess.STDOUT,
                                        timeout=timeout, check=False)
            if result.returncode:
                raise RuntimeError(f"official SDK build failed ({result.returncode}); inspect build.log")
            if not elf.is_file():
                raise RuntimeError("official SDK build returned success without producing a fresh ELF")
            digest = _elf(elf, machine)
            artifact = directory / (name + ".elf")
            elf.replace(artifact)
        manifest.update(status="built", artifact=artifact.name, sha256=digest,
                        sources={key: hashlib.sha256((directory / key).read_bytes()).hexdigest()
                                 for key in ("user.cpp", "user.h", "peslite_control.hpp")})
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return artifact
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError, struct.error) as error:
        manifest.update(status="build_failed", artifact=None, error=str(error))
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        raise RuntimeError(f"Imperix: {error}") from error


if __name__ == "__main__":
    import sys
    try:
        print(build_project(Path(__file__).resolve().parent))
    except RuntimeError as error:
        print(error, file=sys.stderr)
        sys.exit(1)
