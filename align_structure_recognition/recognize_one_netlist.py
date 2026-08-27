#!/usr/bin/env python3
"""Recognize structures in one SPICE netlist using the packaged ALIGN subset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

PACKAGE_ROOT = Path(__file__).resolve().parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from align.compiler.structure_recognition import identify_structures_from_file_to_json, identify_structures_from_file


def logical_lines(netlist_text: str) -> List[str]:
    lines: List[str] = []
    current = ""
    for raw_line in netlist_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("*"):
            continue
        if line.startswith("+"):
            current += " " + line[1:].strip()
        else:
            if current:
                lines.append(current)
            current = line
    if current:
        lines.append(current)
    return lines


def parse_subckts(netlist_text: str) -> List[str]:
    names: List[str] = []
    for line in logical_lines(netlist_text):
        tokens = line.split()
        if len(tokens) >= 2 and tokens[0].upper() == ".SUBCKT":
            names.append(tokens[1])
    return names


def choose_top_subckt(netlist_path: Path, requested: Optional[str]) -> str:
    text = netlist_path.read_text(encoding="utf-8", errors="replace")
    subckts = parse_subckts(text)
    if requested:
        lookup = {name.upper(): name for name in subckts}
        found = lookup.get(requested.upper())
        if found is None:
            available = ", ".join(subckts) or "<none>"
            raise SystemExit(f"requested --top {requested!r} not found; available: {available}")
        return found
    if not subckts:
        raise SystemExit(f"no .SUBCKT found in {netlist_path}")
    by_upper = {name.upper(): name for name in subckts}
    for candidate in (netlist_path.stem, netlist_path.parent.name):
        found = by_upper.get(candidate.upper())
        if found is not None:
            return found
    return subckts[0]


def write_json_or_stdout(result, output_path: Optional[Path]) -> None:
    if output_path is None or str(output_path) == "-":
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recognize one SPICE netlist using the packaged ALIGN template recognizer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", type=Path, help="Input SPICE netlist file.")
    parser.add_argument("-o", "--output", type=Path, help="Output JSON path. Omit or use '-' for stdout.")
    parser.add_argument("--top", help="Top .SUBCKT name. If omitted, choose file-stem/parent/first .SUBCKT.")
    parser.add_argument("--summary", action="store_true", help="Print a one-line summary to stderr after recognition.")
    parser.add_argument("--pdk-dir", type=Path, default=PACKAGE_ROOT / "pdks" / "FinFET14nm_Mock_PDK", help="PDK directory for models/templates.")
    parser.add_argument("--config-dir", type=Path, default=PACKAGE_ROOT / "align" / "config", help="Template config directory.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.input.exists() or not args.input.is_file():
        raise SystemExit(f"input netlist does not exist or is not a file: {args.input}")

    top = choose_top_subckt(args.input, args.top)
    if args.output is None or str(args.output) == "-":
        result = identify_structures_from_file(
            args.input,
            top,
            pdk_dir=args.pdk_dir,
            config_path=args.config_dir,
        )
        write_json_or_stdout(result, None)
    else:
        result = identify_structures_from_file_to_json(
            args.input,
            top,
            args.output,
            pdk_dir=args.pdk_dir,
            config_path=args.config_dir,
        )

    if args.summary:
        counts = {}
        for structure in result.get("structures", []):
            typ = structure.get("type", "<missing>")
            counts[typ] = counts.get(typ, 0) + 1
        print(
            f"recognized {len(result.get('structures', []))} structures in top={top}; counts={dict(sorted(counts.items()))}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
