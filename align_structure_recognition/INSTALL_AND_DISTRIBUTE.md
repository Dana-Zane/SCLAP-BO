# Install and distribute

This directory is a pip-installable package for the extracted ALIGN structure-recognition parity subset.

## Install directly from this directory

```powershell
python -m pip install .
```

After installation, use the console command:

```powershell
align-recognize-structure path\to\netlist.sp --top TOP_SUBCKT -o path\to\result.structure.json --summary
```

Or run the module script directly from the source tree:

```powershell
python recognize_one_netlist.py path\to\netlist.sp --top TOP_SUBCKT -o result.structure.json --summary
```

## Build distributable artifacts

Preferred:

```powershell
python -m pip install build
python -m build
```

If `build` is unavailable, build a wheel with setuptools:

```powershell
python setup.py bdist_wheel
```

The generated files will be under:

```text
dist/
```

## Install a wheel

```powershell
python -m pip install dist\align_structure_recognition_parity-0.1.0-py3-none-any.whl
```

## Runtime dependencies

The wheel declares these dependencies:

```text
pydantic>=1.9.2,<2.0
networkx
pyyaml
z3-solver
flatdict
more-itertools
colorlog
```

Use Pydantic v1. Pydantic v2 is not compatible with this extracted ALIGN code path.

## Parity note

This package runs the extracted original ALIGN `align.compiler.structure_recognition` implementation and the extracted template files. It is intended for exact output parity with the source ALIGN checkout used to generate it.
