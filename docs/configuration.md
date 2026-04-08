# Configuration Guide

This document describes the configuration parameters available in `run_step_to_gprmax.py` which control the behaviour of the STEP-to-gprMax conversion workflow.

## Configuration philosophy

Most users only need to modify a small number of parameters:

- `STEP_PATH`
- `VOXEL_SIZE`
- `SUPERSAMPLE`
- `RUN_MATERIALS_WORKFLOW`
- `WRITE_GPRMAX_IN`

Other parameters control advanced behaviour and normally do not require modification.

---

# Required input

## STEP file

| Parameter | Description |
|-----------|-------------|
| STEP_PATH | Path to the STEP CAD file to be processed |

---

# Materials workflow

| Parameter | Description |
|-----------|-------------|
| RUN_MATERIALS_WORKFLOW | Enables the materials assignment workflow |
| MATERIALS_GROUP_MODE | Controls grouping of geometrically similar parts ("auto" groups similar parts, "none" keeps all parts separate) |
| MATERIALS_REL_BIN | Relative tolerance used when grouping similar parts |
| MATERIALS_FORCE_INIT | Forces creation of a new materials CSV template, overwriting any existing file. Intended only for initial template generation. Should be set to False after assigning materials to avoid unintentionally resetting the workflow. |
| MATERIALS_SHOW_ALL_GROUPED_PART_NAMES | Debug option to list all grouped part names |

---

# gprMax export

| Parameter | Description |
|-----------|-------------|
| WRITE_GPRMAX_IN | Enables generation of gprMax input files |
| GPRMAX_TIME_WINDOW | Simulation time window written to the `.in` file |
| GPRMAX_PAD_CELLS | Number of padding cells added around the geometry |

---

# Selection controls

| Parameter | Description |
|-----------|-------------|
| VOXELISE_ALL | If True, voxelises the full assembly |
| TARGET_PARTS | List of part names to voxelise if VOXELISE_ALL is False (names use sanitised identifiers where spaces are replaced with underscores) |
| SORT_MODE | Controls ordering of parts before voxelisation ("volume_desc" = largest to smallest, recommended; "volume_asc" = smallest to largest; "name" = alphabetical; "none" = original STEP order). Processing order can influence results when using priority merging. |

---

# Voxelisation controls

| Parameter | Description |
|-----------|-------------|
| VOXEL_SIZE | Grid resolution in metres (smaller values increase accuracy and runtime) |
| SUPERSAMPLE | Controls sampling density for interior classification (3–5 typically suitable) |
| pad | Number of padding cells added around the voxel grid |
| dx, dy, dz | Internal resolution values derived from VOXEL_SIZE (do not modify directly) |
| MERGE_MODE | Controls how overlapping voxel assignments are resolved ("priority" uses processing order and is recommended; "last_wins" assigns voxels to the last processed part; "first_wins" keeps the first assignment) |
| PARALLEL_SLICES | Reserved for future parallelisation. Currently inactive and has no effect on runtime or behaviour |

---

# Tessellation controls

| Parameter | Description |
|-----------|-------------|
| P_LINEAR_DEFLECTION | Linear tessellation tolerance controlling mesh fidelity |
| P_ANGULAR_DEFLECTION | Angular tessellation tolerance |
| P_IS_RELATIVE_DEFLECTION | Enables relative tessellation tolerance scaling |

---

# Performance and caching

| Parameter | Description |
|-----------|-------------|
| USE_VOXEL_CACHE | Enables voxel caching to speed repeated runs |
| USE_TESS_CACHE | Enables tessellation caching |
| ENABLE_TIMERS | Displays runtime timing information |
| VOXEL_PIPELINE_VERSION | Internal voxel cache version identifier |
| TESS_PIPELINE_VERSION | Internal tessellation cache version identifier |

---

# Visualisation

| Parameter | Description |
|-----------|-------------|
| SHOW_3D | Displays the voxelised model (typically used with materials workflow disabled) |
| SHOW_CUTAWAY | Displays a cutaway section of the voxel model |
| CUT_AXIS | Axis used for cutaway view |
| CUT_FRAC | Fractional position of the cut plane |
| SHOW_SLICES | Displays 2D slice occupancy plots |
| SLICE_AXIS | Axis used for slice plots |
| SLICE_STEP | Interval between displayed slices |

*Note:* `SHOW_3D` and `SHOW_CUTAWAY` should be used with `RUN_MATERIALS_WORKFLOW=False` and `WRITE_GPRMAX_IN=False`.

---

# Output folders

| Parameter | Description |
|-----------|-------------|
| OUTPUT_DIR | Main output directory |
| CACHE_DIR | Cache directory location |
| MATERIALS_DIR | Materials output directory |
| GPRMAX_DIR | gprMax output directory |

---

# Parser options (advanced)

These parameters control geometry processing behaviour and normally do not require modification unless specific behaviour is required.

| Parameter | Description |
|-----------|-------------|
| P_VERBOSE_PER_PART | Prints bounding box, volume and surface area for each part |
| P_APPLY_OCC_LOCATIONS | Applies OpenCascade assembly transformations |
| P_AUTO_EXPLODE_SOLIDS | Automatically extracts individual solids |
| P_FORCE_UNITS_TO_METRES | Converts geometry units to metres |

---

# Global rotation (optional)

| Parameter | Description |
|-----------|-------------|
| P_ENABLE_GLOBAL_ROTATION | Enables global rotation of the model |
| P_ROTATE_DEG | Rotation angles in degrees |
| P_ROTATE_ABOUT | Rotation reference location |
| P_ROTATE_ORDER | Rotation order |

---

# Triangle cleanup

| Parameter | Description |
|-----------|-------------|
| P_EPS_AREA2 | Area tolerance used when filtering degenerate triangles |
| P_COMPACT_TESSELLATION | Enables tessellation cleanup |

---

# Optional STL export

| Parameter | Description |
|-----------|-------------|
| P_EXPORT_STL | Enables STL export |
| P_STL_OUTNAME | STL output filename |

---

## Configuration used for dissertation validation

The following settings were used for the validation examples presented in the dissertation:

```python
VOXEL_SIZE = (0.001, 0.001, 0.001)
SUPERSAMPLE = 3
MATERIALS_GROUP_MODE = "auto"
MERGE_MODE = "priority"
USE_VOXEL_CACHE = True
USE_TESS_CACHE = True
```

These settings provided a suitable balance between geometric fidelity and runtime for the models tested.
