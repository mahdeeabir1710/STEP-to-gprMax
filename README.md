# STEP-to-gprMax

Computational framework for converting STEP CAD assemblies into voxel grids for gprMax electromagnetic simulation.

## Overview

This repository contains the source code developed as part of an MEng dissertation project at the University of Edinburgh. The project focuses on automating the conversion of STEP CAD assemblies into voxelised models suitable for use in gprMax electromagnetic simulations.

The framework was developed to reduce the manual effort traditionally required to prepare complex engineering geometries for FDTD-based simulation, enabling reproducible conversion of multi-component CAD assemblies into simulation-ready models.

## Quick start

The framework can be run in three simple steps:

1. Install the required Python dependencies:

```bash
pip install -r requirements.txt
```

2. Set the STEP file path and configure options in:

```text
src/run_step_to_gprmax.py
```

3. Run the workflow:

```bash
python src/run_step_to_gprmax.py
```

After the first run, the framework will generate the required output folders and a materials CSV template. After defining material properties, running the script again will generate the final gprMax input files.

## Repository structure

src/step_parser.py  
Parses STEP assemblies and extracts geometry and assembly hierarchy.

src/voxeliser.py  
Voxelises tessellated geometry into structured grids.

src/materials_builder.py  
Handles material grouping and assignment.

src/gprmax_input_builder.py  
Generates gprMax-compatible geometry input.

src/visualisation_utilities.py  
Provides utilities for visualising voxelised results.

src/run_step_to_gprmax.py  
Main runner script controlling the overall workflow.

## Main workflow

The framework follows the general sequence:

1. Parse STEP geometry and assembly structure  
2. Tessellate the geometry  
3. Convert the tessellated representation into a voxel grid  
4. Assign materials to voxelised components  
5. Export geometry and inputs for gprMax  
6. Optionally visualise the voxelised result  

## Requirements

This project was developed and tested using Python 3.11 in a Conda-based scientific Python environment. Key dependencies include:

NumPy  
h5py  
pythonocc-core (Open Cascade Python bindings)  
PyVista  
Matplotlib  

A requirements.txt file is provided listing the required dependencies. Some dependencies, particularly PythonOCC, may require environment-specific installation depending on the platform.

## Usage

The main entry point for the workflow is:

```bash
python src/run_step_to_gprmax.py
```

A typical workflow is:

1. Set the STEP file path and configure options in `src/run_step_to_gprmax.py`
2. Run the script once to generate the initial output folders and materials CSV template
3. Edit the generated materials CSV and define the required electromagnetic material properties
4. Run the script again to generate the final gprMax input files

If the aim is only to inspect the voxelised geometry, the materials workflow and gprMax input generation can be disabled and the visualisation options enabled instead.

See `examples/example_workflow.md` for an example output and a more detailed workflow description.

## Example model

An example STEP CAD model is provided in `test_models/` to demonstrate usage of the framework.

This model can be used to reproduce the voxelisation example shown in the `examples/` directory.

## Notes

- Confidential CAD models used for validation are not included.
- Generated outputs and cached voxel files are excluded.
- This repository documents the dissertation implementation rather than a packaged software release.
- The `PARALLEL_SLICES` parameter is reserved for future development. In the current version it is inactive and enabling it has no effect on runtime, output or program behaviour.

## Acknowledgement of prior work

The overall STEP-to-gprMax framework was developed as part of the associated dissertation project.

The voxelisation component was informed by and adapted from ideas in Christian Pederkoff’s STL-to-voxel work:

https://github.com/cpederkoff/stl-to-voxel

This work was extended to support STEP assemblies and integration within the broader CAD-to-gprMax workflow.

## Author

Mahdee Abir  
MEng Mechanical Engineering  
University of Edinburgh  
2026
