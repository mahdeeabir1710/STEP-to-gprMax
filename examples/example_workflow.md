# Example voxelisation output and workflow

This folder contains example outputs generated using the STEP-to-gprMax framework.

## Example geometry

The figures below show an example engineering assembly and the corresponding voxelised representation produced using this framework.

### Original CAD model
<p align="center">
<img src="Engine_CAD.png" width="900">
</p>

### Voxelised model
<p align="center">
<img src="Engine_Voxelised.png" width="900">
</p>

The main entry point for the framework is:

```bash
python src/run_step_to_gprmax.py
```

A typical workflow is:

1. Set the STEP file path in `run_step_to_gprmax.py`.

2. Configure the required settings, including voxel resolution and output options.

   **STEP export note:**  
   Due to differences in how CAD software exports STEP assemblies, component names may not always be preserved. When exported from some software, parts may appear with generic identifiers (for example NAUO labels).

   This does not affect voxelisation, material grouping, or simulation preparation. All geometry will still be processed correctly. The only impact is reduced readability when assigning material properties.

   If CAD component names are required, the STEP file can be opened in FreeCAD and re-exported without modification. This generally preserves assembly naming while requiring no changes to the geometry.

3. If the aim is to prepare a model for gprMax, leave the materials workflow and gprMax input generation enabled.
4. Run the script once. This creates an output directory containing:
   - `cache/`
   - `gprmax/`
   - `materials/`
5. Open `materials/materials_grouped.csv` (generated after first run) in a CSV editor such as Microsoft Excel.
6. Fill in the required electromagnetic material properties and material names.
7. Save the CSV file and run the script again.
8. The framework then generates:
   - a `materials.txt` file in the `materials/` folder
   - a gprMax `.in` input file in the `gprmax/` folder

If the user only wants to inspect the voxelised geometry rather than prepare a gprMax model, the materials workflow and gprMax input generation can be disabled and the visualisation options enabled instead.

## Notes

- The generated `.in` file includes the geometry import setup and a computed simulation domain based on the model bounding box.
- Source, receiver, and waveform definitions are not added automatically, as these depend on the intended simulation.
- Once the `.in` file and material definitions have been generated, the remaining simulation setup and execution should follow the standard gprMax workflow. See the [gprMax project](https://github.com/gprMax/gprMax) for further details.
