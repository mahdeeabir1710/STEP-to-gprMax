#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 22 14:12:51 2026

@author: mahdeeabir1710
"""

from __future__ import annotations

from typing import Optional
import numpy as np


def show_voxels_3d(mat_grid: np.ndarray, grid, threshold: int = 0) -> None:
    import pyvista as pv

    mat_grid = np.asarray(mat_grid)
    nx, ny, nz = mat_grid.shape

    filled_count = int((mat_grid >= threshold).sum())
    print(f"Filled voxels (mat>={threshold}): {filled_count}")
    if filled_count == 0:
        print("Nothing to render.")
        return

    ug = pv.ImageData()
    ug.dimensions = (nx + 1, ny + 1, nz + 1)
    ug.origin = tuple(map(float, grid.origin_world))
    ug.spacing = tuple(map(float, grid.dxyz_world))

    # IMPORTANT: keep order="F" to match voxel grid convention
    ug.cell_data["mat"] = mat_grid.ravel(order="F")

    filled = ug.threshold(value=threshold - 0.5, scalars="mat")

    pl = pv.Plotter()
    pl.add_mesh(filled, show_edges=False)
    pl.add_axes()
    pl.show_grid()
    pl.show(title="Voxelised geometry (cell-based)")


def show_voxels_cutaway(
    mat_grid: np.ndarray,
    grid,
    axis: str = "z",
    frac: float = 0.5,
    threshold: int = 0,
) -> None:
    import pyvista as pv

    mat_grid = np.asarray(mat_grid)
    nx, ny, nz = mat_grid.shape

    filled_count = int((mat_grid >= threshold).sum())
    if filled_count == 0:
        print("Nothing to render.")
        return

    ug = pv.ImageData()
    ug.dimensions = (nx + 1, ny + 1, nz + 1)
    ug.origin = tuple(map(float, grid.origin_world))
    ug.spacing = tuple(map(float, grid.dxyz_world))

    ug.cell_data["mat"] = mat_grid.ravel(order="F")

    filled = ug.threshold(value=threshold - 0.5, scalars="mat")

    axis = axis.lower()
    bounds = filled.bounds

    if axis == "x":
        x0 = bounds[0] + frac * (bounds[1] - bounds[0])
        cut = filled.clip(normal=(1, 0, 0), origin=(x0, 0, 0))
    elif axis == "y":
        y0 = bounds[2] + frac * (bounds[3] - bounds[2])
        cut = filled.clip(normal=(0, 1, 0), origin=(0, y0, 0))
    elif axis == "z":
        z0 = bounds[4] + frac * (bounds[5] - bounds[4])
        cut = filled.clip(normal=(0, 0, 1), origin=(0, 0, z0))
    else:
        raise ValueError("axis must be x, y, or z")

    pl = pv.Plotter()
    pl.add_mesh(cut, show_edges=False)
    pl.add_axes()
    pl.show_grid()
    pl.show(title=f"Cutaway axis={axis}, frac={frac}")


def debug_plot_slice(
    mat_grid: np.ndarray,
    grid,
    axis: str = "z",
    index: Optional[int] = None,
) -> None:
    import matplotlib.pyplot as plt

    axis = axis.lower()
    nx, ny, nz = mat_grid.shape

    if axis == "z":
        k = nz // 2 if index is None else int(index)
        img = (mat_grid[:, :, k] >= 0).T
        title = f"Occupancy slice z={k}"
    elif axis == "y":
        k = ny // 2 if index is None else int(index)
        img = (mat_grid[:, k, :] >= 0).T
        title = f"Occupancy slice y={k}"
    elif axis == "x":
        k = nx // 2 if index is None else int(index)
        img = (mat_grid[k, :, :] >= 0).T
        title = f"Occupancy slice x={k}"
    else:
        raise ValueError("axis must be x/y/z")

    print(title, "fill ratio:", float(img.mean()))

    plt.figure()
    plt.imshow(img, origin="lower", interpolation="nearest")
    plt.title(title)
    plt.show()