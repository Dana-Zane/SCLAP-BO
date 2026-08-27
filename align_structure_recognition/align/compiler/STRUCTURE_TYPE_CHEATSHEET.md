# Structure Type Cheatsheet

This cheatsheet explains common `type` values emitted by `align.compiler.structure_recognition`.

When reading recognition JSON, prefer the stable `type` field over `model`. The `model` field may contain a hash suffix such as `DP_NMOS_B_77995687`.

## Quick Reference Table

| Type | Meaning | Chinese | Typical device count | Notes |
|---|---|---|---:|---|
| `NMOS_4T` | 4-terminal NMOS | 四端 NMOS | 1 | D/G/S/B all exposed |
| `PMOS_4T` | 4-terminal PMOS | 四端 PMOS | 1 | D/G/S/B all exposed |
| `NMOS_S` | NMOS with bulk tied to source | bulk 接 source 的 NMOS | 1 | Common 3-terminal NMOS form |
| `PMOS_S` | PMOS with bulk tied to source | bulk 接 source 的 PMOS | 1 | Common 3-terminal PMOS form |
| `NMOS_G` | NMOS with bulk tied to gate | bulk 接 gate 的 NMOS | 1 | Less common |
| `PMOS_G` | PMOS with bulk tied to gate | bulk 接 gate 的 PMOS | 1 | Less common |
| `DUMMY_NMOS` | Dummy NMOS | dummy NMOS | 1 | Dummy device |
| `DUMMY_PMOS` | Dummy PMOS | dummy PMOS | 1 | Dummy device |
| `DUMMY_NMOS_S` | Dummy NMOS, source/body tied | source/body 相连 dummy NMOS | 1 | Dummy device |
| `DUMMY_PMOS_S` | Dummy PMOS, source/body tied | source/body 相连 dummy PMOS | 1 | Dummy device |
| `DUMMY1_NMOS` | Single-terminal dummy NMOS | 单端 dummy NMOS | 1 | D/G/S may be shorted |
| `DUMMY1_PMOS` | Single-terminal dummy PMOS | 单端 dummy PMOS | 1 | D/G/S may be shorted |
| `DCAP_NMOS` | Decap NMOS | NMOS 去耦电容 | 1 | MOS capacitor / decap |
| `DCAP_PMOS` | Decap PMOS | PMOS 去耦电容 | 1 | MOS capacitor / decap |
| `DCAP_NMOS_B` | Decap NMOS with body pin | 带 body 的 NMOS 去耦电容 | 1 | Body exposed |
| `DCAP_PMOS_B` | Decap PMOS with body pin | 带 body 的 PMOS 去耦电容 | 1 | Body exposed |
| `DCL_NMOS` | Diode-connected NMOS | 二极管连接 NMOS | 1 | D = G |
| `DCL_PMOS` | Diode-connected PMOS | 二极管连接 PMOS | 1 | D = G |
| `DCL_NMOS_S` | Diode-connected NMOS, source/body tied | source/body 相连的二极管连接 NMOS | 1 | D = G, B = S |
| `DCL_PMOS_S` | Diode-connected PMOS, source/body tied | source/body 相连的二极管连接 PMOS | 1 | D = G, B = S |
| `SCM_NMOS` | Simple current mirror NMOS | 简单 NMOS 电流镜 | 2 | One diode-connected reference + one output device |
| `SCM_PMOS` | Simple current mirror PMOS | 简单 PMOS 电流镜 | 2 | One diode-connected reference + one output device |
| `SCM_NMOS_B` | Simple current mirror NMOS with body pin | 带 body 端口的 NMOS 简单电流镜 | 2 | Body exposed |
| `SCM_PMOS_B` | Simple current mirror PMOS with body pin | 带 body 端口的 PMOS 简单电流镜 | 2 | Body exposed |
| `CMC_NMOS` | Common-centroid / matched current mirror NMOS | NMOS 匹配电流镜对 | 2 | Matched two-device mirror-like cell |
| `CMC_PMOS` | Common-centroid / matched current mirror PMOS | PMOS 匹配电流镜对 | 2 | Common in active loads |
| `CMC_NMOS_B` | CMC NMOS with body pin | 带 body 的 NMOS 匹配电流镜对 | 2 | Body exposed |
| `CMC_S_NMOS` | CMC NMOS with separate sources | 源端分开的 NMOS 匹配对 | 2 | Separate `SA` / `SB` |
| `CMC_S_PMOS` | CMC PMOS with separate sources | 源端分开的 PMOS 匹配对 | 2 | Separate `SA` / `SB` |
| `CMC_S_NMOS_B` | CMC NMOS, separate sources, body pin | 源端分开且带 body 的 NMOS 匹配对 | 2 | `SA` / `SB` / `B` |
| `CMC_S_PMOS_B` | CMC PMOS, separate sources, body pin | 源端分开且带 body 的 PMOS 匹配对 | 2 | `SA` / `SB` / `B` |
| `DP_NMOS` | NMOS differential pair | NMOS 差分对 | 2 | Shared source, no explicit body port |
| `DP_PMOS` | PMOS differential pair | PMOS 差分对 | 2 | Shared source, no explicit body port |
| `DP_NMOS_B` | NMOS differential pair with body pin | 带 body 的 NMOS 差分对 | 2 | Common output for 4-terminal MOS netlists |
| `DP_PMOS_B` | PMOS differential pair with body pin | 带 body 的 PMOS 差分对 | 2 | Body exposed |
| `CCP_NMOS` | Cross-coupled pair NMOS | NMOS 交叉耦合对 | 2 | M1 gate to M2 drain, M2 gate to M1 drain |
| `CCP_PMOS` | Cross-coupled pair PMOS | PMOS 交叉耦合对 | 2 | Common in latches / comparators |
| `CCP_NMOS_B` | Cross-coupled pair NMOS with body pin | 带 body 的 NMOS 交叉耦合对 | 2 | Body exposed |
| `CCP_PMOS_B` | Cross-coupled pair PMOS with body pin | 带 body 的 PMOS 交叉耦合对 | 2 | Body exposed |
| `CCP_S_NMOS_B` | Cross-coupled pair NMOS, separate sources, body pin | 源端分开且带 body 的 NMOS 交叉耦合对 | 2 | Common in comparators |
| `CCP_S_PMOS_B` | Cross-coupled pair PMOS, separate sources, body pin | 源端分开且带 body 的 PMOS 交叉耦合对 | 2 | Common in comparators / latches |
| `LS_S_NMOS_B` | Level-shifter / latch-style NMOS structure | 源端分开的 NMOS LS 类结构 | 2 | `LS` template family |
| `LS_S_PMOS_B` | Level-shifter / latch-style PMOS structure | 源端分开的 PMOS LS 类结构 | 2 | `LS` template family |
| `INV` | CMOS inverter | CMOS 反相器 | 2 | NMOS + PMOS, shared gate and drain |
| `INV_B` | CMOS inverter with explicit body pin | 带 body 端口的反相器 | 2 | Body exposed in template |
| `stage2_inv` | Two-stage inverter | 两级反相器 | 4 | Two cascaded inverters |
| `CASCODED_CMC_NMOS` | Cascoded common-centroid current mirror NMOS | NMOS cascode 匹配电流镜 | 4 | Cascode current mirror |
| `CASCODED_CMC_PMOS` | Cascoded common-centroid current mirror PMOS | PMOS cascode 匹配电流镜 | 4 | Cascode current mirror |
| `CASCODED_SCM_NMOS` | Cascoded simple current mirror NMOS | NMOS cascode 简单电流镜 | 3 | User-template structure |
| `CASCODED_SCM_PMOS` | Cascoded simple current mirror PMOS | PMOS cascode 简单电流镜 | 3 | User-template structure |
| `CASCODED_CMB_NMOS_2` | Cascoded current mirror bank NMOS, 2 outputs | NMOS cascode 电流镜阵列 | 4 | Multi-output mirror |
| `CASCODED_CMB_PMOS_2` | Cascoded current mirror bank PMOS, 2 outputs | PMOS cascode 电流镜阵列 | 4 | Multi-output mirror |
| `CASCODED_CMB_PMOS_3` | Cascoded current mirror bank PMOS, 3 outputs | PMOS cascode 电流镜阵列 | 5 | Multi-output mirror |
| `CMB_NMOS_2` | Current mirror bank NMOS, 2 outputs | NMOS 多输出电流镜 | 3 | 1 reference + 2 outputs |
| `CMB_PMOS_2` | Current mirror bank PMOS, 2 outputs | PMOS 多输出电流镜 | 3 | 1 reference + 2 outputs |
| `CMB_NMOS_3` | Current mirror bank NMOS, 3 outputs | NMOS 多输出电流镜 | 4 | 1 reference + 3 outputs |
| `CMB_PMOS_3` | Current mirror bank PMOS, 3 outputs | PMOS 多输出电流镜 | 4 | 1 reference + 3 outputs |
| `CMB_NMOS_4` | Current mirror bank NMOS, 4 outputs | NMOS 多输出电流镜 | 5 | 1 reference + 4 outputs |
| `CMB_PMOS_4` | Current mirror bank PMOS, 4 outputs | PMOS 多输出电流镜 | 5 | 1 reference + 4 outputs |
| `LSB_NMOS_2` | LSB current mirror bank NMOS, 2 outputs | NMOS LSB 多输出镜 | 3 | DAC / array-like structure |
| `LSB_NMOS_7` | LSB current mirror bank NMOS, 7 outputs | NMOS LSB 多输出镜 | 8 | DAC / array-like structure |
| `LSB_PMOS_2` | LSB current mirror bank PMOS, 2 outputs | PMOS LSB 多输出镜 | 3 | DAC / array-like structure |
| `DP_PAIR_PMOS` | PMOS differential-pair-like group | PMOS 差分对组合结构 | 4 | User-template structure |
| `tgate` | Transmission gate | 传输门 | 2 | NMOS + PMOS parallel switch |
| `switched_capacitor_combination` | Switched capacitor combination | 开关电容组合结构 | Multiple | User-template structure |
| `SCM_NMOS_C` | Simple current mirror NMOS with capacitor | 带电容的 NMOS 电流镜 | 3 | 2 MOS + 1 capacitor |
| `SCM_PMOS_C` | Simple current mirror PMOS with capacitor | 带电容的 PMOS 电流镜 | 3 | 2 MOS + 1 capacitor |
| `SCM_NMOS_RC` | Simple current mirror NMOS with RC network | 带 RC 的 NMOS 电流镜 | Multiple | User-template structure |
| `SCM_PMOS_RC` | Simple current mirror PMOS with RC network | 带 RC 的 PMOS 电流镜 | Multiple | User-template structure |

## Naming Rules

### `_B`

Usually means the body / bulk terminal is explicitly exposed.

Examples:

```text
DP_NMOS_B
SCM_PMOS_B
CMC_S_NMOS_B
```

### `_S`

Usually means source-related specialization. Exact meaning depends on the template family:

- `NMOS_S` / `PMOS_S`: body tied to source.
- `CMC_S_*` / `CCP_S_*`: separate sources, usually `SA` and `SB`.

### `DCL`

`DCL` means diode-connected:

```text
D = G
```

Examples:

```text
DCL_NMOS
DCL_PMOS
```

### `DCAP`

`DCAP` means decap / MOS capacitor.

### `SCM`

`SCM` means simple current mirror:

```text
one diode-connected reference + one output device
```

### `CMC`

`CMC` can be read as common-centroid / matched current mirror cell. It usually denotes a matched two-device structure that carries symmetry intent in ALIGN templates.

### `CMB`

`CMB` means current mirror bank:

```text
CMB_NMOS_2 -> 1 reference + 2 outputs
CMB_NMOS_3 -> 1 reference + 3 outputs
```

### `DP`

`DP` means differential pair.

### `CCP`

`CCP` means cross-coupled pair.

### `INV`

`INV` means inverter.

### `CASCODED_*`

`CASCODED_*` means cascode structure, usually a cascode current mirror or cascode matched pair.

Examples:

```text
CASCODED_CMC_NMOS
CASCODED_SCM_PMOS
```

## Recommended JSON Reading Strategy

When reading a structure entry like:

```json
{
  "type": "DP_NMOS_B",
  "devices": ["MN3", "MN4"],
  "pins": {
    "DA": "VOP",
    "DB": "VON",
    "GA": "VIP",
    "GB": "VIN",
    "S": "TAIL",
    "B": "VSSX"
  }
}
```

Use this order:

1. Read `type` to identify the structure class.
2. Read `devices` to see the original netlist devices contained in the structure.
3. Read `pins` to see how the structure's formal pins connect to actual nets.
4. Use `device_details` if you need original model / pin / parameter information.
5. Ignore hash suffixes in `model` for human structure interpretation.
