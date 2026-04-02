#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 14:34:25 2026

@author: mahdeeabir1710
"""

from __future__ import annotations

import os
from typing import Sequence


def _fmt3(vals: Sequence[float]) -> str:
    return " ".join(f"{float(v):.12g}" for v in vals)


def _relpath_from(base_dir: str, target_path: str) -> str:
    """
    Return a relative path from base_dir to target_path, using forward slashes
    so the written .in file is clean/cross-platform.
    """
    rel = os.path.relpath(os.path.abspath(target_path), start=os.path.abspath(base_dir))
    return rel.replace("\\", "/")


def write_gprmax_input_file(
    *,
    path_in: str,
    grid,
    geometry_h5_path: str,
    materials_txt_path: str,
    title: str = "Imported voxel geometry",
    time_window: float = 1e-8,
    pad_cells: int = 20,
    geometry_view_name: str = "imported_geometry",
) -> None:
    """
    Write a minimal gprMax .in file for geometry import / sanity checking.

    This function does NOT try to define sources, receivers, waveforms, or
    scan modes. It only creates a valid import-ready model that:
      - defines domain and discretisation
      - imports geometry.h5 + materials.txt
      - writes a geometry_view command

    Parameters
    ----------
    path_in : str
        Output path for the .in file.
    grid : GridSpec-like
        Must expose:
          - nx, ny, nz
          - dxyz_world -> (dx, dy, dz)
    geometry_h5_path : str
        Path to geometry HDF5 file.
    materials_txt_path : str
        Path to generated materials.txt file.
    title : str
        Title line for gprMax input.
    time_window : float
        Placeholder/default time window. User can later change this.
    pad_cells : int
        Number of cells of padding to add on EACH side of the imported geometry
        when defining the simulation domain.
    geometry_view_name : str
        Output stem for geometry_view.
    """
    if pad_cells < 0:
        raise ValueError("pad_cells must be >= 0")
    if time_window <= 0:
        raise ValueError("time_window must be > 0")

    dx, dy, dz = map(float, grid.dxyz_world)
    nx, ny, nz = int(grid.nx), int(grid.ny), int(grid.nz)

    # Geometry size
    geom_lx = nx * dx
    geom_ly = ny * dy
    geom_lz = nz * dz

    # Domain padding in metres
    pad_x = pad_cells * dx
    pad_y = pad_cells * dy
    pad_z = pad_cells * dz

    # Final domain
    domain = (
        geom_lx + 2.0 * pad_x,
        geom_ly + 2.0 * pad_y,
        geom_lz + 2.0 * pad_z,
    )

    # Import offset (place geometry away from boundary/PML)
    import_origin = (pad_x, pad_y, pad_z)

    out_dir = os.path.dirname(os.path.abspath(path_in))
    os.makedirs(out_dir, exist_ok=True)

    geofile_rel = _relpath_from(out_dir, geometry_h5_path)
    matfile_rel = _relpath_from(out_dir, materials_txt_path)

    lines = [
        f"#title: {title}",
        f"#domain: {_fmt3(domain)}",
        f"#dx_dy_dz: {_fmt3((dx, dy, dz))}",
        f"#time_window: {float(time_window):.12g}",
        "",
        "## Geometry/material import generated automatically.",
        "## Edit this file to add waveform/source/rx commands for your scenario.",
        f"#geometry_objects_read: {_fmt3(import_origin)} {geofile_rel} {matfile_rel}",
        "",
        f"#geometry_view: 0 0 0 {_fmt3(domain)} {_fmt3((dx, dy, dz))} {geometry_view_name} n",
        "",
    ]

    tmp = path_in + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines))
    os.replace(tmp, path_in)