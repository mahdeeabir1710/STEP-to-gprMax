#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 20 18:12:39 2026

@author: mahdeeabir1710
"""

import os
import json
import hashlib

import numpy as np
import time

import step_parser as sp

from voxeliser import (
    GridSpec,
    TriangleMesh,
    voxelise_material_grid,
    write_gprmax_hdf5,
    write_voxel_cache_hdf5,
    read_voxel_cache_hdf5,
    validate_cached_voxel_grid,
)

import materials_builder as mb

import gprmax_input_builder as gib

from visualisation_utilities import show_voxels_3d, show_voxels_cutaway, debug_plot_slice

def _file_fingerprint(path: str) -> dict:
    st = os.stat(path)
    return {"path": os.path.abspath(path), "size": int(st.st_size), "mtime": float(st.st_mtime)}

def _stable_hash(obj: dict) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def _safe_name(s: str) -> str:
    s = s or "unnamed"
    return "".join(c if (c.isalnum() or c in ("_", "-", ".")) else "_" for c in s)

def _tess_cache_path(tess_dir: str, tess_key: str, part_name: str, part_uid: int) -> str:
    safe = _safe_name(part_name)
    return os.path.join(tess_dir, f"{safe}__uid{part_uid}__{tess_key[:12]}.npz")

def _load_tess_npz(path_npz: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path_npz) as d:
        V = d["V"]
        T = d["T"]
    if V.ndim != 2 or V.shape[1] != 3:
        raise ValueError(f"Cached V has shape {V.shape}, expected (N,3)")
    if T.ndim != 2 or T.shape[1] != 3:
        raise ValueError(f"Cached T has shape {T.shape}, expected (M,3)")
    return V, T

def _save_tess_npz(path_npz: str, V: np.ndarray, T: np.ndarray) -> None:
    tmp = path_npz + ".tmp"     # no extension here
    np.savez_compressed(tmp, V=V, T=T)   # writes tmp + ".npz"
    os.replace(tmp + ".npz", path_npz)

def _remap_material_grid(mat_grid: np.ndarray, old_to_new: dict[int, int]) -> np.ndarray:
    """
    Remap voxel material IDs from per-part IDs to compact grouped IDs.

    Any value < 0 is preserved (e.g. -1 for empty voxels).
    """
    src = np.asarray(mat_grid, dtype=np.int16)
    dst = src.copy()

    positive_ids = np.unique(src[src >= 0])
    for old_id in positive_ids:
        if int(old_id) not in old_to_new:
            raise ValueError(f"material_id {int(old_id)} exists in mat_grid but not in compact mapping.")
        dst[src == old_id] = int(old_to_new[int(old_id)])

    return dst

def _fmt_time(seconds: float) -> str:
    m = int(seconds // 60)
    s = seconds % 60
    return f"{m} min {s:05.2f} s" if m > 0 else f"{s:.3f} s"

if __name__ == "__main__":
    # ============================================================
    # (Optional) start overall timer as early as possible
    # ============================================================
    t_total0 = time.perf_counter()

    # ============================================================
    # USER CONFIG
    # ============================================================

    # --- Paths / I/O ---
    STEP_PATH = r"Provide file path to STEP here"
    
    # --- Materials workflow ---
    RUN_MATERIALS_WORKFLOW = True
    MATERIALS_GROUP_MODE = "auto"   # "auto" or "none"
    MATERIALS_REL_BIN = 0.01        # 1% binning
    MATERIALS_FORCE_INIT = False    # True to overwrite CSV template
    MATERIALS_SHOW_ALL_GROUPED_PART_NAMES = False
    
    # --- gprMax input file output ---
    WRITE_GPRMAX_IN = True
    GPRMAX_TIME_WINDOW = 3e-9
    GPRMAX_PAD_CELLS = 20

    # --- Selection ---
    VOXELISE_ALL = True
    TARGET_PARTS = ["piston_v1", "pin"]  # sanitised name (spaces get replaced with underscores)
    SORT_MODE = "volume_desc"           # "volume_desc", "volume_asc", "name", "none"; "volume_desc" default and recommended

    # --- Voxelisation controls ---
    VOXEL_SIZE = (0.001, 0.001, 0.001)  # metres
    dx, dy, dz = VOXEL_SIZE # Internal, do not edit. Edit VOXEL_SIZE to control voxel discretisation
    pad = 2
    SUPERSAMPLE = 3
    
    # --- Tessellation quality ---
    P_LINEAR_DEFLECTION = 0.001
    P_ANGULAR_DEFLECTION = 0.1
    P_IS_RELATIVE_DEFLECTION = True

    MERGE_MODE = "priority"             # "priority", "last_wins", "first_wins"; "priority" is default and recommended
    PARALLEL_SLICES = False

    # --- Output folders ---
    OUTPUT_DIR = "output"
    CACHE_DIR = os.path.join(OUTPUT_DIR, "cache")
    MATERIALS_DIR = os.path.join(OUTPUT_DIR, "materials")
    GPRMAX_DIR = os.path.join(OUTPUT_DIR, "gprmax")
    
    # --- Caching controls ---
    USE_VOXEL_CACHE = True
    USE_TESS_CACHE = True
    
    VOXEL_PIPELINE_VERSION = "voxel_cache_v1"
    TESS_PIPELINE_VERSION = "tess_cache_v1"

    # --- Visualisation controls ---
    SHOW_3D = False         # Shows full voxelised model if True. If True, toggle off "RUN_MATERIALS_WORKFLOW"
                            # to prevent potential bugs with viewer.
    
    SHOW_CUTAWAY = False    # Shows section cut at specified point of model. If True, toggle off ""RUN_MATERIALS_WORKFLOW"
                            # to prevent potential bugs with viewer.
    CUT_AXIS = "z" 
    CUT_FRAC = 0.5
    
    SHOW_SLICES = False
    SLICE_AXIS = "z"   
    SLICE_STEP = 1  # show every Nth slice (1 = all)

    # --- Timing ---
    ENABLE_TIMERS = True

    # --- Parser / assembly options ---
    P_VERBOSE_PER_PART = False          # If true, prints bbox, volume and surface area of each part in STEP model      
    P_APPLY_OCC_LOCATIONS = True
    P_AUTO_EXPLODE_SOLIDS = True
    P_FORCE_UNITS_TO_METRES = True

    # --- Global rotation ---
    P_ENABLE_GLOBAL_ROTATION = False
    P_ROTATE_DEG = (0.0, 0.0, 0.0)
    P_ROTATE_ABOUT = "bbox_global"
    P_ROTATE_ORDER = "xyz"

    # --- Triangle cleanup ---
    P_EPS_AREA2 = 1e-30
    P_COMPACT_TESSELLATION = True

    # --- Optional STL export ---
    P_EXPORT_STL = False
    P_STL_OUTNAME = "assembly.stl"

    # ============================================================
    # BUILD CONFIG OBJECTS (derived from user config)
    # ============================================================
    cfg = sp.ParserConfig(
        verbose_per_part=P_VERBOSE_PER_PART,
        apply_occurrence_locations=P_APPLY_OCC_LOCATIONS,
        auto_explode_solids=P_AUTO_EXPLODE_SOLIDS,
        force_units_to_metres=P_FORCE_UNITS_TO_METRES,

        enable_global_rotation=P_ENABLE_GLOBAL_ROTATION,
        rotate_deg=P_ROTATE_DEG,
        rotate_about=P_ROTATE_ABOUT,
        rotate_order=P_ROTATE_ORDER,

        linear_deflection=P_LINEAR_DEFLECTION,
        angular_deflection=P_ANGULAR_DEFLECTION,
        is_relative_deflection=P_IS_RELATIVE_DEFLECTION,

        eps_area2=P_EPS_AREA2,
        compact_tessellation=P_COMPACT_TESSELLATION,

        export_stl=P_EXPORT_STL,
        stl_outname=P_STL_OUTNAME,
    )

    # ============================================================
    # CACHE BOOKKEEPING (derived; do not edit unless needed)
    # ============================================================
    # Ensure directories exist
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(MATERIALS_DIR, exist_ok=True)
    os.makedirs(GPRMAX_DIR, exist_ok=True)
    
    # Internal output filenames
    MATERIALS_MANIFEST_NAME = "materials_manifest.json"
    MANIFEST_JSON = os.path.join(MATERIALS_DIR, MATERIALS_MANIFEST_NAME)
    
    OUT_H5 = os.path.join(GPRMAX_DIR, "geometry.h5")
    OUT_IN = os.path.join(GPRMAX_DIR, "model.in")
    
    # Internal behaviour:
    # If gprMax output is wanted, geometry.h5 is required.
    WRITE_H5 = WRITE_GPRMAX_IN
    
    CACHE_H5 = os.path.join(CACHE_DIR, "voxel_cache.h5")
    CACHE_META = os.path.join(CACHE_DIR, "voxel_cache_meta.json")
    
    TESS_DIR = os.path.join(CACHE_DIR, "tess_cache")
    if USE_TESS_CACHE:
        os.makedirs(TESS_DIR, exist_ok=True)

    # Fingerprint STEP file to invalidate caches if file changes
    step_fp = _file_fingerprint(STEP_PATH)

    # Tessellation cache key: depends on STEP + geometry-affecting parser ops + tessellation params
    tess_inputs = {
        "pipeline_version": TESS_PIPELINE_VERSION,
        "step": step_fp,
        "parser_cfg_geom": {
            "apply_occurrence_locations": cfg.apply_occurrence_locations,
            "auto_explode_solids": cfg.auto_explode_solids,
            "force_units_to_metres": cfg.force_units_to_metres,
            "enable_global_rotation": cfg.enable_global_rotation,
            "rotate_deg": tuple(cfg.rotate_deg),
            "rotate_about": cfg.rotate_about,
            "rotate_order": cfg.rotate_order,
        },
        "tess_cfg": {
            "linear_deflection": cfg.linear_deflection,
            "angular_deflection": cfg.angular_deflection,
            "is_relative_deflection": cfg.is_relative_deflection,
            "eps_area2": cfg.eps_area2,
            "compact_tessellation": cfg.compact_tessellation,
        },
    }
    tess_key = _stable_hash(tess_inputs)

    # Voxel cache key: depends on STEP + parser/tess settings + selection + voxeliser settings
    cache_inputs = {
        "pipeline_version": VOXEL_PIPELINE_VERSION,
        "step": step_fp,
        "parser_cfg": {
            "apply_occurrence_locations": cfg.apply_occurrence_locations,
            "auto_explode_solids": cfg.auto_explode_solids,
            "force_units_to_metres": cfg.force_units_to_metres,

            "enable_global_rotation": cfg.enable_global_rotation,
            "rotate_deg": tuple(cfg.rotate_deg),
            "rotate_about": cfg.rotate_about,
            "rotate_order": cfg.rotate_order,

            "linear_deflection": cfg.linear_deflection,
            "angular_deflection": cfg.angular_deflection,
            "is_relative_deflection": cfg.is_relative_deflection,
            "eps_area2": cfg.eps_area2,
            "compact_tessellation": cfg.compact_tessellation,
        },
        "selection": {
            "voxelise_all": bool(VOXELISE_ALL),
            "target_parts": [] if VOXELISE_ALL else list(TARGET_PARTS),
            "sort_mode": SORT_MODE,
        },
        "voxeliser_cfg": {
            "dx": float(dx), "dy": float(dy), "dz": float(dz),
            "pad": int(pad),
            "merge_mode": MERGE_MODE,
            "parallel_slices": bool(PARALLEL_SLICES),
            "supersample": int(SUPERSAMPLE),
        },
    }
    cache_key = _stable_hash(cache_inputs)

    # Small debug prints (handy for cache mismatch debugging)
    print(f"Tess key:  {tess_key[:12]}...  Tess dir:  {TESS_DIR}")
    print(f"Cache key: {cache_key[:12]}...  Cache file: {CACHE_H5}")

    # ----------------------------
    # 1) Run STEP parser
    # ----------------------------
    parts = sp.main(STEP_PATH, cfg)
    
    if not parts:
        raise RuntimeError("Parser returned 0 parts.")

    # ----------------------------
    # 2) Choose parts to voxelise
    # ----------------------------
    if VOXELISE_ALL:
        selected_parts = parts
        print(f"Voxelising ALL parts: {len(selected_parts)}")
    
    else:
        if not TARGET_PARTS:
            raise ValueError("VOXELISE_ALL=False but TARGET_PARTS is empty. Provide at least 1 part name.")
    
        parts_by_name = {(p.name or ""): p for p in parts}
    
        missing = [name for name in TARGET_PARTS if name not in parts_by_name]
        if missing:
            available = sorted(n for n in parts_by_name.keys() if n)
            raise ValueError(
                "These TARGET_PARTS were not found:\n"
                f"  {missing}\n\n"
                "Available part names include:\n"
                "  " + "\n  ".join(available[:50]) + ("\n  ..." if len(available) > 50 else "")
            )
    
        selected_parts = [parts_by_name[name] for name in TARGET_PARTS]
    
        print(f"Voxelising {len(selected_parts)} selected part(s)")
        
    # ----------------------------
    # 3) Optional ordering
    # ----------------------------
    def _part_volume_or_zero(p):
        cad = getattr(p, "cad", None) or {}
        v = cad.get("vol_m3", None)
        return float(v) if v is not None else 0.0

    if SORT_MODE == "volume_desc":
        selected_parts = sorted(selected_parts, key=_part_volume_or_zero, reverse=True)
    elif SORT_MODE == "volume_asc":
        selected_parts = sorted(selected_parts, key=_part_volume_or_zero, reverse=False)
    elif SORT_MODE == "name":
        selected_parts = sorted(selected_parts, key=lambda p: (p.name or ""))
    elif SORT_MODE == "none":
        pass
    else:
        raise ValueError(f"Unknown SORT_MODE: {SORT_MODE}")
    
    # ----------------------------
    # 3.1) Stable material_id mapping (needed even on voxel cache HIT)
    # ----------------------------
    uid_to_material_id = {int(p.uid): i for i, p in enumerate(selected_parts)}
    
    # ----------------------------
    # 3.2) Write materials manifest for materials workflow
    # ----------------------------
    if RUN_MATERIALS_WORKFLOW:
        selected_uids = {int(p.uid) for p in selected_parts}
    
        manifest_parts = []
        for p in parts:
            cad = getattr(p, "cad", None)
    
            manifest_parts.append({
                "uid": int(p.uid),
                "name": str(p.name),
                "selected": (int(p.uid) in selected_uids),
                "material_id": (uid_to_material_id[int(p.uid)] if int(p.uid) in selected_uids else None),
                "cad": cad,
            })
    
        payload = {
            "manifest_version": "materials_manifest_v1",
            "cache_key": cache_key,  # ties to STEP+configs used by this run
            "step": step_fp,
            "cache_dir": os.path.abspath(CACHE_DIR),
            "materials_dir": os.path.abspath(MATERIALS_DIR),
            "selection": {
                "voxelise_all": bool(VOXELISE_ALL),
                "target_parts": [] if VOXELISE_ALL else list(TARGET_PARTS),
                "sort_mode": str(SORT_MODE),
            },
            "parts": manifest_parts,
        }
    
        tmp = MANIFEST_JSON + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        os.replace(tmp, MANIFEST_JSON)
        print(f"🧾 Wrote materials manifest: {MANIFEST_JSON} (parts={len(parts)}, selected={len(selected_uids)})")
        
    # ----------------------------
    # 4) Try to load voxel cache BEFORE tessellation
    # ----------------------------
    loaded_from_cache = False

    if USE_VOXEL_CACHE and os.path.exists(CACHE_H5):
        try:
            cached_grid, attrs = read_voxel_cache_hdf5(CACHE_H5)
            cached_key = attrs.get("cache_key", "")

            if cached_key == cache_key and validate_cached_voxel_grid(cached_grid, attrs):
                origin = np.array(attrs["origin_xyz"], dtype=np.float64)
                dxyz = np.array(attrs["dx_dy_dz"], dtype=np.float64)
                nxyz = np.array(attrs["shape_nxyz"], dtype=np.int32)

                grid = GridSpec(origin_world=origin, dxyz_world=dxyz, nxyz=nxyz)
                mat_grid = cached_grid
                loaded_from_cache = True
                print(f"✅ Loaded voxel grid from cache: {CACHE_H5}")
            else:
                print("ℹ️ Cache exists but key mismatch or failed validation → will voxelise.")
        except Exception as e:
            print(f"ℹ️ Failed to load cache ({e}) → will voxelise.")
    
    if loaded_from_cache:
        print("✅ Cache HIT → skipping tessellation + voxelisation.")
    else:
        print("ℹ️ Cache MISS → tessellating + voxelising.")
    
    # ----------------------------
    # 5) Tessellate per part -> TriangleMesh list
    # ----------------------------
    if not loaded_from_cache:
        # ----------------------------
        # Timing: tessellation start
        # ----------------------------
        if ENABLE_TIMERS:
            t_tess0 = time.perf_counter()
            tess_hit = 0
            tess_miss = 0
            tess_write = 0
    
        meshes = []
        mat_name_map = {}
    
        vols = {p.uid: _part_volume_or_zero(p) for p in selected_parts}
        sorted_by_small = sorted(selected_parts, key=lambda p: (vols[p.uid], p.name or ""))
        priority_rank = {p.uid: (len(sorted_by_small) - i) for i, p in enumerate(sorted_by_small)}
    
        for p in selected_parts:
            shp = sp.shape_for_ops(p, cfg)
    
            cache_npz = _tess_cache_path(TESS_DIR, tess_key, p.name or "", p.uid)
    
            V = None
            T = None
    
            # 1) Try load cached triangles
            if USE_TESS_CACHE and os.path.exists(cache_npz):
                try:
                    Vc, Tc = _load_tess_npz(cache_npz)
                    V = np.asarray(Vc, dtype=np.float32)
                    T = np.asarray(Tc, dtype=np.int32)
                    if ENABLE_TIMERS:
                        tess_hit += 1
                except Exception as e:
                    print(f"  - tess cache FAIL ({p.name}): {e} → re-tessellating")
                    V = None
                    T = None
    
            # 2) Cache miss → tessellate and store
            if V is None or T is None:
                if ENABLE_TIMERS:
                    tess_miss += 1
                verts, tris = sp.tessellate_shape(shp, cfg)
    
                if not verts or not tris:
                    print(f"  - Skipping (no triangles): {p.name}")
                    continue
    
                V = np.asarray(verts, dtype=np.float32)
                T = np.asarray(tris, dtype=np.int32)
    
                if USE_TESS_CACHE:
                    try:
                        _save_tess_npz(cache_npz, V, T)
                        if ENABLE_TIMERS:
                            tess_write += 1
                    except Exception as e:
                        print(f"  - tess cache WRITE FAIL ({p.name}): {e}")
    
            # material_id comes from stable mapping
            mid = uid_to_material_id[int(p.uid)]
    
            prio = int(priority_rank.get(p.uid, 0)) if MERGE_MODE == "priority" else 0
    
            meshes.append(
                TriangleMesh(
                    vertices_world=V,
                    triangles=T,
                    material_id=mid,
                    priority=prio,
                    name=(p.name or f"Part_{mid}")
                )
            )
            mat_name_map[mid] = p.name or f"Part_{mid}"
    
            print(f"  - mat_id={mid:3d} prio={prio:4d} | {mat_name_map[mid]} | verts={V.shape[0]} tris={T.shape[0]}")
    
        if not meshes:
            raise RuntimeError("No tessellated meshes produced from selected parts.")
    
        print(f"Voxeliser supersample = {SUPERSAMPLE}")
        
        # ----------------------------
        # Timing: tessellation end
        # ----------------------------
        if ENABLE_TIMERS:
            t_tess1 = time.perf_counter()
        
        # ----------------------------
        # Timing: voxelisation start
        # ----------------------------
        if ENABLE_TIMERS:
            t_vox0 = time.perf_counter()
    
        # ----------------------------
        # 6) Voxelise into one material grid (cache miss path)
        # ----------------------------
        mat_grid, grid = voxelise_material_grid(
            meshes,
            dx=dx, dy=dy, dz=dz,
            pad=pad,
            merge_mode=MERGE_MODE,
            parallel_slices=PARALLEL_SLICES,
            supersample=SUPERSAMPLE,
        )
        
        if ENABLE_TIMERS:
            t_vox1 = time.perf_counter()
    
        if USE_VOXEL_CACHE:
            meta_to_store = {"cache_inputs": cache_inputs}
    
            write_voxel_cache_hdf5(
                CACHE_H5,
                mat_grid,
                grid,
                cache_key=cache_key,
                meta=meta_to_store,
                compression="gzip",
            )
    
            tmp_meta = CACHE_META + ".tmp"
            with open(tmp_meta, "w", encoding="utf-8") as f:
                json.dump({"cache_key": cache_key, **meta_to_store}, f, indent=2, sort_keys=True)
            os.replace(tmp_meta, CACHE_META)
    
            print(f"💾 Wrote voxel cache: {CACHE_H5}")
    
    if "mat_grid" not in locals() or "grid" not in locals():
        raise RuntimeError("No voxel grid available (cache load failed and voxelisation did not run).")
        
    
    # ----------------------------
    # Timing summary
    # ----------------------------
    if ENABLE_TIMERS:
        t_total1 = time.perf_counter()
    
        print("\n================ Timing Summary ================")
        print(f"Total runtime: {_fmt_time(t_total1 - t_total0)}")
    
        if loaded_from_cache:
            print("Tessellation: skipped (voxel cache HIT)")
            print("Voxelisation: skipped (voxel cache HIT)")
        else:
            if 't_tess0' in locals():
                print(
                    f"Tessellation: {_fmt_time(t_tess1 - t_tess0)} "
                    f"(hit={tess_hit}, miss={tess_miss}, wrote={tess_write})"
                )
    
            if 't_vox0' in locals():
                print(f"Voxelisation: {_fmt_time(t_vox1 - t_vox0)}")
    
        print("================================================\n")
        
    # ----------------------------
    # 7) Materials workflow
    # ----------------------------
    result = None
    if RUN_MATERIALS_WORKFLOW:
        result = mb.run_materials_workflow(
            manifest_path=MANIFEST_JSON,
            out_dir=MATERIALS_DIR,
            group_mode=MATERIALS_GROUP_MODE,
            rel_bin=MATERIALS_REL_BIN,
            force_init=MATERIALS_FORCE_INIT,
            csv_name="material_groups.csv",
            txt_name="materials.txt",
            show_all_grouped_part_names=MATERIALS_SHOW_ALL_GROUPED_PART_NAMES,
        )

        if result["action"] == "init_csv":
            print(f"✅ Materials CSV created: {result['csv_path']}")
            print("➡️  Fill the CSV columns (material_name, relative_permittivity, conductivity, etc.) then run again.")
        else:
            print(f"✅ materials.txt written: {result['txt_path']}")

    # ----------------------------
    # 7.1) Optional: write compacted gprMax HDF5
    # ----------------------------
    export_grid = None

    if WRITE_H5:
        materials_txt_path = os.path.join(MATERIALS_DIR, "materials.txt")

        if result is None:
            print("ℹ️ Skipping geometry.h5 export because materials workflow did not run.")
        elif result["action"] != "build_txt":
            print("ℹ️ Skipping geometry.h5 export because materials.txt is not ready yet.")
        else:
            old_to_new = result["material_id_map"]
            export_grid = _remap_material_grid(mat_grid, old_to_new)

            write_gprmax_hdf5(OUT_H5, export_grid, grid)
            print(f"✅ Wrote compacted gprMax geometry: {OUT_H5}")
    else:
        print("WRITE_H5=False → not writing any .h5 file.")

    # ----------------------------
    # 7.2) Optional: write gprMax input file (.in)
    # ----------------------------
    materials_txt_path = os.path.join(MATERIALS_DIR, "materials.txt")

    if WRITE_GPRMAX_IN:
        if not os.path.exists(OUT_H5):
            print(f"ℹ️ Skipping model.in because geometry file does not exist: {OUT_H5}")
        elif not os.path.exists(materials_txt_path):
            print(f"ℹ️ Skipping model.in because materials file does not exist yet: {materials_txt_path}")
        else:
            gib.write_gprmax_input_file(
                path_in=OUT_IN,
                grid=grid,
                geometry_h5_path=OUT_H5,
                materials_txt_path=materials_txt_path,
                time_window=GPRMAX_TIME_WINDOW,
                pad_cells=GPRMAX_PAD_CELLS,
            )
            print(f"✅ Wrote gprMax input file: {OUT_IN}")
            
    # ----------------------------
    # 8) Visualisations
    # ----------------------------
    if SHOW_3D:
        show_voxels_3d(mat_grid, grid, threshold=0)

    if SHOW_CUTAWAY:
        show_voxels_cutaway(mat_grid, grid, axis=CUT_AXIS, frac=CUT_FRAC, threshold=0)

    if SHOW_SLICES:
        axis = SLICE_AXIS.lower()
        n = getattr(grid, f"n{axis}")
    
        step = max(1, int(SLICE_STEP))
    
        print(f"\nShowing {axis}-slices every {step} step(s) (total {n})")
    
        for i in range(0, n, step):
            debug_plot_slice(mat_grid, grid, axis=axis, index=i)