"""Netlist structure recognition helpers.

This module exposes a small Python-callable API around ALIGN's topology
identification pass.  It reuses the library-based annotation flow, but returns
only hierarchy/structure membership data instead of layout constraints or PnR
collateral.
"""

from __future__ import annotations

import copy
import json
import pathlib
import tempfile
from typing import Any, Dict, Iterable, List, Optional, Union

from align.schema import SubCircuit, constraint

from .compiler import annotate_library, compiler_input


PathLike = Union[str, pathlib.Path]


_DEFAULT_PDK_DIR = pathlib.Path(__file__).resolve().parents[2] / "pdks" / "FinFET14nm_Mock_PDK"
_DEFAULT_CONFIG_DIR = pathlib.Path(__file__).resolve().parents[1] / "config"

_POWER_NAMES = {
    "VDD",
    "VDD!",
    "VCC",
    "VCC!",
    "VCCX",
    "VDDA",
    "AVDD",
    "DVDD",
    "VP",
    "VPWR",
    "PWR",
}
_GROUND_NAMES = {
    "VSS",
    "VSS!",
    "GND",
    "GND!",
    "VSSX",
    "VSSA",
    "AVSS",
    "DVSS",
    "VN",
    "VGND",
    "AGND",
    "DGND",
}


def identify_structures(
    netlist: Union[str, pathlib.Path],
    design_name: str,
    *,
    pdk_dir: Optional[PathLike] = None,
    config_path: Optional[PathLike] = None,
    constraints: Optional[List[Dict[str, Any]]] = None,
    netlist_is_text: Optional[bool] = None,
    infer_power_ground: bool = True,
    include_device_details: bool = True,
    output_json_path: Optional[PathLike] = None,
) -> Dict[str, Any]:
    """Recognize hierarchical structures in a SPICE netlist.

    Args:
        netlist: Either a path to a SPICE file or the SPICE netlist text.
        design_name: Top-level ``.SUBCKT`` name to analyze.
        pdk_dir: PDK directory. Defaults to the repository FinFET mock PDK.
        config_path: Template/config directory. Defaults to ``align/config``.
        constraints: Optional ALIGN constraints for ``design_name``. These are
            used only to guide topology identification; they are not returned as
            layout constraints. Typical examples are ``PowerPorts`` and
            ``GroundPorts``.
        netlist_is_text: Force interpretation of ``netlist`` as text/path. If
            omitted, existing filesystem paths are treated as paths; everything
            else is treated as text.
        infer_power_ground: If true, add inferred ``PowerPorts``/``GroundPorts``
            constraints when they are not already provided and matching supply
            pins are present in the top subcircuit line. This improves MOS
            source/drain normalization before matching.
        include_device_details: Include per-device model/pins/parameters for
            the devices contained in each recognized structure.
        output_json_path: Optional path. When provided, the recognition result is
            also written to this JSON file.

    Returns:
        A JSON-serializable dictionary with ``modules`` and ``structures``. The
        ``structures`` list is flat across all modules; each structure contains
        its recognized type/template and original member device names.
    """

    pdk_dir = pathlib.Path(pdk_dir) if pdk_dir is not None else _DEFAULT_PDK_DIR
    config_path = pathlib.Path(config_path) if config_path is not None else _DEFAULT_CONFIG_DIR
    design_name = design_name.upper()
    constraints = copy.deepcopy(constraints or [])

    with _netlist_file(netlist, design_name, constraints, netlist_is_text, infer_power_ground) as netlist_path:
        ckt_data, primitive_library = compiler_input(
            pathlib.Path(netlist_path),
            design_name,
            pdk_dir,
            config_path,
        )
        original_elements = _snapshot_elements(ckt_data)
        annotate_library(ckt_data, primitive_library)

    modules = []
    all_structures = []
    for subckt in ckt_data:
        if not isinstance(subckt, SubCircuit):
            continue
        if _has_generator(subckt):
            continue
        module = _module_to_dict(subckt, original_elements, include_device_details)
        modules.append(module)
        all_structures.extend(module["structures"])

    top = next((m for m in modules if m["name"] == design_name), None)
    result = {
        "design_name": design_name,
        "top_module": top,
        "modules": modules,
        "structures": all_structures,
    }
    if output_json_path is not None:
        write_structure_json(result, output_json_path)
    return result


def identify_structures_from_file(
    netlist_path: PathLike,
    design_name: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Recognize structures from a SPICE file path."""

    return identify_structures(
        pathlib.Path(netlist_path),
        design_name,
        netlist_is_text=False,
        **kwargs,
    )


def identify_structures_from_file_to_json(
    netlist_path: PathLike,
    design_name: str,
    output_json_path: PathLike,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Recognize structures from a SPICE file and write the result to JSON."""

    return identify_structures_from_file(
        netlist_path,
        design_name,
        output_json_path=output_json_path,
        **kwargs,
    )


def identify_structures_from_text(
    netlist_text: str,
    design_name: str,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Recognize structures from an in-memory SPICE netlist string."""

    return identify_structures(
        netlist_text,
        design_name,
        netlist_is_text=True,
        **kwargs,
    )


def identify_structures_from_text_to_json(
    netlist_text: str,
    design_name: str,
    output_json_path: PathLike,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Recognize structures from SPICE text and write the result to JSON."""

    return identify_structures_from_text(
        netlist_text,
        design_name,
        output_json_path=output_json_path,
        **kwargs,
    )


def write_structure_json(result: Dict[str, Any], output_json_path: PathLike) -> pathlib.Path:
    """Write a structure-recognition result dictionary to a JSON file.

    Parent directories are created when needed. The same path is returned for
    convenient logging by callers.
    """

    path = pathlib.Path(output_json_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)
        fp.write("\n")
    return path


class _netlist_file:
    def __init__(
        self,
        netlist: Union[str, pathlib.Path],
        design_name: str,
        constraints: List[Dict[str, Any]],
        netlist_is_text: Optional[bool],
        infer_power_ground: bool,
    ):
        self.netlist = netlist
        self.design_name = design_name
        self.constraints = constraints
        self.netlist_is_text = netlist_is_text
        self.infer_power_ground = infer_power_ground
        self._tmpdir = None

    def __enter__(self) -> pathlib.Path:
        if self.netlist_is_text is None:
            candidate = pathlib.Path(str(self.netlist))
            self.netlist_is_text = not candidate.exists()

        if not self.netlist_is_text:
            path = pathlib.Path(self.netlist)
            if self.constraints or self.infer_power_ground:
                text = path.read_text()
                return self._write_temp(text)
            return path

        return self._write_temp(str(self.netlist))

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._tmpdir is not None:
            self._tmpdir.cleanup()

    def _write_temp(self, netlist_text: str) -> pathlib.Path:
        self._tmpdir = tempfile.TemporaryDirectory(prefix="align_structure_")
        tmpdir = pathlib.Path(self._tmpdir.name)
        netlist_path = tmpdir / f"{self.design_name}.sp"
        netlist_path.write_text(netlist_text)

        constraints = copy.deepcopy(self.constraints)
        if self.infer_power_ground:
            constraints = _with_inferred_power_ground(netlist_text, self.design_name, constraints)
        if constraints:
            const_json = tmpdir / "const.json"
            const_json.write_text(
                _json_dumps([
                    {"subcircuit": self.design_name, "constraints": constraints}
                ])
            )
        return netlist_path


def _json_dumps(value: Any) -> str:
    return json.dumps(value, indent=2, ensure_ascii=False)


def _with_inferred_power_ground(
    netlist_text: str,
    design_name: str,
    constraints: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if any(c.get("constraint") == "PowerPorts" for c in constraints) and any(
        c.get("constraint") == "GroundPorts" for c in constraints
    ):
        return constraints

    pins = _top_subckt_pins(netlist_text, design_name)
    if not pins:
        return constraints

    power = [pin for pin in pins if pin.upper() in _POWER_NAMES]
    ground = [pin for pin in pins if pin.upper() in _GROUND_NAMES]
    if power and not any(c.get("constraint") == "PowerPorts" for c in constraints):
        constraints.append({"constraint": "PowerPorts", "ports": power})
    if ground and not any(c.get("constraint") == "GroundPorts" for c in constraints):
        constraints.append({"constraint": "GroundPorts", "ports": ground})
    return constraints


def _top_subckt_pins(netlist_text: str, design_name: str) -> List[str]:
    logical_lines = []
    current = ""
    for raw_line in netlist_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("*"):
            continue
        if line.startswith("+"):
            current += " " + line[1:].strip()
        else:
            if current:
                logical_lines.append(current)
            current = line
    if current:
        logical_lines.append(current)

    target = design_name.upper()
    for line in logical_lines:
        tokens = line.split()
        if len(tokens) >= 2 and tokens[0].upper() == ".SUBCKT" and tokens[1].upper() == target:
            pins = []
            for token in tokens[2:]:
                if "=" in token:
                    break
                pins.append(token.upper())
            return pins
    return []


def _snapshot_elements(ckt_data: Iterable[Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    snapshot: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for subckt in ckt_data:
        if not isinstance(subckt, SubCircuit):
            continue
        snapshot[subckt.name] = {ele.name: _instance_to_dict(ele) for ele in subckt.elements}
    return snapshot


def _module_to_dict(
    subckt: SubCircuit,
    original_elements: Dict[str, Dict[str, Dict[str, Any]]],
    include_device_details: bool,
) -> Dict[str, Any]:
    group_by_name = {
        const.instance_name: const
        for const in subckt.constraints
        if isinstance(const, constraint.GroupBlocks)
    }
    structures = []
    instances = []

    for ele in subckt.elements:
        group = group_by_name.get(ele.name)
        if group is not None:
            structure = _structure_to_dict(
                subckt,
                ele,
                group,
                original_elements.get(subckt.name, {}),
                include_device_details,
            )
            structures.append(structure)
            instances.append(
                {
                    "name": ele.name,
                    "kind": "structure",
                    "structure_type": structure["type"],
                    "model": ele.model,
                    "pins": dict(ele.pins),
                    "devices": list(structure["devices"]),
                }
            )
        else:
            instances.append(_annotated_instance_to_dict(ele, subckt))

    # Nested structures live in the newly-created template-specific subcircuits
    # (for example DP_NMOS_B_12345678).  Report them once from their parent
    # module by expanding the matching child hierarchy.
    for ele in subckt.elements:
        child = subckt.parent.find(ele.model) if subckt.parent is not None else None
        if not isinstance(child, SubCircuit) or _has_generator(child):
            continue
        child_group_names = {
            const.instance_name
            for const in child.constraints
            if isinstance(const, constraint.GroupBlocks)
        }
        for child_ele in child.elements:
            if child_ele.name not in child_group_names:
                continue
            parent_device_order = [
                name for name in original_elements.get(subckt.name, {})
            ]
            nested_devices = sorted(
                _collect_leaf_devices(child_ele, child, original_elements),
                key=lambda name: parent_device_order.index(name) if name in parent_device_order else len(parent_device_order),
            )
            structures.append(
                {
                    "module": subckt.name,
                    "name": f"{ele.name}/{child_ele.name}",
                    "type": _base_template_name(child_ele.model),
                    "model": child_ele.model,
                    "pins": dict(child_ele.pins),
                    "devices": nested_devices,
                    **(
                        {
                            "device_details": [
                                original_elements.get(subckt.name, {}).get(name, {"name": name})
                                for name in nested_devices
                            ]
                        }
                        if include_device_details
                        else {}
                    ),
                }
            )

    return {
        "name": subckt.name,
        "pins": list(subckt.pins or []),
        "instances": instances,
        "structures": structures,
    }


def _structure_to_dict(
    subckt: SubCircuit,
    ele: Any,
    group: Any,
    original_elements: Dict[str, Dict[str, Any]],
    include_device_details: bool,
) -> Dict[str, Any]:
    devices = list(group.instances)
    result = {
        "module": subckt.name,
        "name": group.instance_name,
        "type": group.template_name,
        "model": ele.model,
        "pins": dict(ele.pins),
        "devices": devices,
    }
    if include_device_details:
        result["device_details"] = [
            original_elements.get(name, {"name": name}) for name in devices
        ]
    return result


def _annotated_instance_to_dict(ele: Any, subckt: SubCircuit) -> Dict[str, Any]:
    model = subckt.parent.find(ele.model) if subckt.parent is not None else None
    kind = "subcircuit" if isinstance(model, SubCircuit) and not _has_generator(model) else "device"
    return {
        "name": ele.name,
        "kind": kind,
        "model": ele.model,
        "pins": dict(ele.pins),
        "parameters": dict(ele.parameters or {}),
    }


def _collect_leaf_devices(
    ele: Any,
    owning_subckt: SubCircuit,
    original_elements: Dict[str, Dict[str, Dict[str, Any]]],
) -> List[str]:
    """Map a possibly nested recognized instance back to original device names."""

    if owning_subckt.parent is None:
        return [ele.name]
    child = owning_subckt.parent.find(ele.model)
    if not isinstance(child, SubCircuit) or _has_generator(child):
        return [ele.name]

    leaf_names: List[str] = []
    template_by_name = {template_ele.name: template_ele for template_ele in child.elements}
    for original_name, original_ele in original_elements.get(owning_subckt.name, {}).items():
        template_ele = template_by_name.get(original_ele["model"])
        if template_ele is None:
            continue
        if _pins_match_template(original_ele["pins"], ele.pins, template_ele.pins):
            leaf_names.append(original_name)
    return leaf_names or [ele.name]


def _pins_match_template(
    original_pins: Dict[str, str],
    parent_instance_pins: Dict[str, str],
    template_pins: Dict[str, str],
) -> bool:
    for formal, template_net in template_pins.items():
        original_net = original_pins.get(formal)
        if original_net is None:
            return False
        expected_net = parent_instance_pins.get(template_net, template_net)
        if original_net != expected_net:
            return False
    return True


def _base_template_name(model_name: str) -> str:
    parts = model_name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return model_name


def _instance_to_dict(ele: Any) -> Dict[str, Any]:
    return {
        "name": ele.name,
        "model": ele.model,
        "pins": dict(ele.pins),
        "parameters": dict(ele.parameters or {}),
    }


def _has_generator(subckt: SubCircuit) -> bool:
    return any(isinstance(const, constraint.Generator) for const in subckt.constraints)
