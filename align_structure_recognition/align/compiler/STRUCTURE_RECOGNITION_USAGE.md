# Structure Recognition Module Usage

This document is for agents or scripts that need to call the lightweight ALIGN netlist structure recognizer.

## Purpose

Use `align.compiler.structure_recognition` when you need **hierarchical netlist structure information only**:

- recognized analog structures, e.g. `DP_NMOS_B`, `SCM_NMOS`, `SCM_PMOS`, `CMC_*`, `CCP_*`, `INV`, `CASCODED_*`;
- each structure's original member device names;
- optional per-device model / pin / parameter details;
- module-level hierarchy and instance information.

Do **not** use this module for layout generation, GDS generation, primitive layout generation, placement, routing, or constraint export.

## Environment

Use the project-local virtual environment:

```powershell
.\.venv\Scripts\python.exe
```

The venv is intentionally minimal. It is configured for topology / structure recognition only and avoids full ALIGN layout dependencies such as `gdspy`.

Required minimal packages are:

```text
pydantic>=1.9.2,<2.0
networkx
pyyaml
z3-solver
flatdict
more-itertools
colorlog
```

If running from a shell command, prefer:

```powershell
.\.venv\Scripts\python.exe your_script.py
```

Do not use the system Python 3.14 for this module; ALIGN's schema code requires Pydantic 1.x.

## Main APIs

```python
from align.compiler.structure_recognition import (
    identify_structures,
    identify_structures_from_file,
    identify_structures_from_file_to_json,
    identify_structures_from_text,
    identify_structures_from_text_to_json,
    write_structure_json,
)
```

### From in-memory SPICE text

```python
from align.compiler.structure_recognition import identify_structures_from_text

netlist = """
.subckt CKT ibias vccx vssx von vin vip
mn1 ibias ibias vssx vssx n w=360e-9 nf=2 m=8
mn2 tail  ibias vssx vssx n w=360e-9 nf=2 m=8
mn3 vop vip tail vssx n w=360e-9 nf=2 m=16
mn4 von vin tail vssx n w=360e-9 nf=2 m=16
mp5 vop vop vccx vccx p w=360e-9 nf=2 m=4
mp6 von vop vccx vccx p w=360e-9 nf=2 m=4
.ends CKT
"""

result = identify_structures_from_text(netlist, "CKT")

for structure in result["structures"]:
    print(structure["type"], structure["devices"])
```

Expected style of output:

```text
SCM_NMOS ['MN1', 'MN2']
SCM_PMOS ['MP5', 'MP6']
DP_NMOS_B ['MN3', 'MN4']
```

### From a SPICE file

```python
from align.compiler.structure_recognition import identify_structures_from_file

result = identify_structures_from_file("path/to/netlist.sp", "TOP_SUBCKT")
```

### Write recognition result directly to JSON

All APIs can write the same returned dictionary to JSON by passing `output_json_path`:

```python
from align.compiler.structure_recognition import identify_structures_from_file

result = identify_structures_from_file(
    "path/to/netlist.sp",
    "TOP_SUBCKT",
    output_json_path="path/to/structure_result.json",
)
```

Convenience wrappers are also available:

```python
from align.compiler.structure_recognition import (
    identify_structures_from_file_to_json,
    identify_structures_from_text_to_json,
)

result = identify_structures_from_file_to_json(
    "path/to/netlist.sp",
    "TOP_SUBCKT",
    "path/to/structure_result.json",
)

result = identify_structures_from_text_to_json(
    netlist_text,
    "TOP_SUBCKT",
    "path/to/structure_result.json",
)
```

The JSON file contains the exact same object returned by the function.
Parent directories are created automatically.

### Generic API

```python
from align.compiler.structure_recognition import identify_structures

result = identify_structures(
    netlist="path/to/netlist.sp",  # or raw SPICE text
    design_name="TOP_SUBCKT",
)
```

## Optional arguments

```python
result = identify_structures_from_text(
    netlist,
    "CKT",
    constraints=[
        {"constraint": "PowerPorts", "ports": ["VCCX"]},
        {"constraint": "GroundPorts", "ports": ["VSSX"]},
    ],
    infer_power_ground=True,
    include_device_details=True,
)
```

Arguments:

- `constraints`: optional ALIGN-style constraints used only to guide topology recognition. They are not returned as layout constraints. Most useful for `PowerPorts` and `GroundPorts`.
- `infer_power_ground`: default `True`. Automatically infers common supply/ground pin names such as `VDD`, `VSS`, `VCCX`, `VSSX`, `VDDA`, `VSSA`, `GND`.
- `include_device_details`: default `True`. Includes model, pins, and parameters for each original device inside a recognized structure.
- `pdk_dir`: optional PDK directory. Defaults to the repo mock FinFET PDK.
- `config_path`: optional template directory. Defaults to `align/config`.

## Return format

The return value is JSON-serializable:

```python
{
    "design_name": "CKT",
    "top_module": {...},
    "modules": [...],
    "structures": [...],
}
```

### Top-level `structures`

`result["structures"]` is a flat list of all recognized structures across modules. This is usually the most convenient field for agents.

Each structure has this shape:

```python
{
    "module": "CKT",
    "name": "X_MN3_MN4",
    "type": "DP_NMOS_B",
    "model": "DP_NMOS_B_77995687",
    "pins": {
        "DA": "VOP",
        "DB": "VON",
        "GA": "VIP",
        "GB": "VIN",
        "S": "TAIL",
        "B": "VSSX",
    },
    "devices": ["MN3", "MN4"],
    "device_details": [
        {
            "name": "MN3",
            "model": "N",
            "pins": {"D": "VOP", "G": "VIP", "S": "TAIL", "B": "VSSX"},
            "parameters": {"W": "360E-9", "NF": "2", "M": "16", ...},
        },
        ...
    ],
}
```

Important fields:

- `type`: recognized structure template name, e.g. `DP_NMOS_B`, `SCM_PMOS`.
- `devices`: original netlist device names contained in the structure.
- `device_details`: original device model, pins, and parameters.
- `pins`: structure-level formal pin to actual net mapping.
- `model`: concrete internal hierarchy name. It may include a hash suffix; use `type` for the stable structure class.

### Module records

Each module in `result["modules"]` has:

```python
{
    "name": "CKT",
    "pins": [...],
    "instances": [...],
    "structures": [...],
}
```

`instances` contains both devices and recognized structure instances:

```python
{
    "name": "X_MN3_MN4",
    "kind": "structure",
    "structure_type": "DP_NMOS_B",
    "model": "DP_NMOS_B_77995687",
    "pins": {...},
    "devices": ["MN3", "MN4"],
}
```

or:

```python
{
    "name": "MN0",
    "kind": "device",
    "model": "N",
    "pins": {...},
    "parameters": {...},
}
```

## Recommended agent workflow

1. Use `identify_structures_from_text` if the netlist content is already in memory.
2. Use `identify_structures_from_file` if the netlist is stored on disk.
3. Inspect `result["structures"]` first.
4. Use `structure["type"]` as the stable structure label.
5. Use `structure["devices"]` as the original device membership.
6. Ignore layout constraints, PnR files, GDS files, and primitive generation APIs.

## Minimal command-line smoke test

From the repository root:

```powershell
$code = @'
from align.compiler.structure_recognition import identify_structures_from_text
netlist = '''
.subckt CKT ibias vccx vssx von vin vip
mn1 ibias ibias vssx vssx n w=360e-9 nf=2 m=8
mn2 tail  ibias vssx vssx n w=360e-9 nf=2 m=8
mn3 vop vip tail vssx n w=360e-9 nf=2 m=16
mn4 von vin tail vssx n w=360e-9 nf=2 m=16
mp5 vop vop vccx vccx p w=360e-9 nf=2 m=4
mp6 von vop vccx vccx p w=360e-9 nf=2 m=4
.ends CKT
'''
r = identify_structures_from_text(netlist, 'CKT')
print([(s['type'], s['devices']) for s in r['structures']])
'@
.\.venv\Scripts\python.exe -c $code
```

Expected output:

```text
[('SCM_NMOS', ['MN1', 'MN2']), ('SCM_PMOS', ['MP5', 'MP6']), ('DP_NMOS_B', ['MN3', 'MN4'])]
```

## Known limitations

- This module uses ALIGN's existing template-based topology annotation. Structures are recognized only when they match templates in `align/config/basic_template.sp`, `align/config/user_template.sp`, or any override template files provided via `config_path` / PDK.
- It is not a full layout flow. Do not expect GDS, LEF, placement, routing, or primitive geometry output.
- The concrete `model` names may contain hash suffixes. Prefer `type` when comparing structure classes.
- If recognition seems worse than expected, pass explicit `PowerPorts` and `GroundPorts` constraints to improve source/drain normalization.
