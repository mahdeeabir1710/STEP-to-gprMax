#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 12:34:14 2026

@author: mahdeeabir1710
"""

#!/usr/bin/env python3

from __future__ import annotations

"""
materials_workflow.py

Reads runner-produced materials_manifest.json and manages:
  1) group/material CSV creation (auto grouping or no grouping)
  2) materials.txt generation for gprMax (#geometry_objects_read)

Workflow:
  - Run once (CSV missing) -> creates CSV template.
  - User edits CSV -> fills material_name, relative_permittivity, conductivity, etc.
  - Run again -> validates and writes materials.txt.

Assumptions:
  - manifest JSON contains:
      payload["cache_key"], payload["parts"] list
      each part has: uid, name, selected(bool), material_id(int or None), cad(dict or None)
  - cad dict contains:
      ok_bbox, ok_props, bbox_dims_xyz (dx,dy,dz), vol_m3, area_m2
"""

import argparse
import csv
import os
import sys
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import math


# -----------------------------
# Config (safe defaults)
# -----------------------------

DEFAULT_CSV_NAME = "material_groups.csv"
DEFAULT_TXT_NAME = "materials.txt"

# Grouping:
#  - "auto" : numeric signature grouping (recommended)
#  - "none" : each selected part becomes its own group
DEFAULT_GROUP_MODE = "auto"

# Quantisation tolerances for auto-grouping
# Example: 0.01 = 1% bins; 0.005 = 0.5% bins.
DEFAULT_REL_BIN = 0.01

# For extremely tiny values, relative binning can be unstable; use absolute floors
DEFAULT_ABS_FLOOR_VOL = 1e-18
DEFAULT_ABS_FLOOR_AREA = 1e-18
DEFAULT_ABS_FLOOR_LEN = 1e-12

# gprMax defaults
DEFAULT_REL_PERMEABILITY = 1.0
DEFAULT_MAGNETIC_LOSS = 0.0


# -----------------------------
# Data structures
# -----------------------------

@dataclass(frozen=True)
class PartRec:
    uid: int
    name: str
    selected: bool
    material_id: Optional[int]
    cad: Dict[str, Any]


@dataclass
class GroupRec:
    group_id: int
    group_key: str
    part_count: int
    example_names: str
    vol_m3_med: Optional[float]
    area_m2_med: Optional[float]
    bbox_dx_m_med: Optional[float]
    bbox_dy_m_med: Optional[float]
    bbox_dz_m_med: Optional[float]
    # selected material_ids in this group (for writing materials.txt)
    material_ids: List[int]


# -----------------------------
# Helpers
# -----------------------------

def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _write_atomic_text(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    os.replace(tmp, path)

def _write_atomic_csv(path: str, header: List[str], rows: List[List[Any]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    os.replace(tmp, path)

def _median(xs: List[float]) -> Optional[float]:
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return None
    xs.sort()
    n = len(xs)
    mid = n // 2
    if n % 2 == 1:
        return xs[mid]
    return 0.5 * (xs[mid - 1] + xs[mid])

def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except Exception:
        return None

def _quant_rel(x: float, rel_bin: float, abs_floor: float) -> Tuple[int, int]:
    """
    Relative quantisation using scientific notation:
      x = sign * m * 10^e, with m in [1,10)
    We quantise m in steps of rel_bin and keep exponent e.

    Returns: (qmant_signed, exponent)
    """
    x = float(x)
    ax = abs(x)

    # tiny values: fall back to absolute binning around abs_floor
    if ax < abs_floor:
        q = int(round(x / abs_floor))
        return (q, 0)

    sign = -1 if x < 0 else 1
    e = int(math.floor(math.log10(ax)))
    m = ax / (10.0 ** e)  # mantissa in [1,10)

    qmant = int(round(m / rel_bin))
    return (sign * qmant, e)

def _signature_from_cad(
    cad: Dict[str, Any],
    rel_bin: float,
    abs_floor_len: float,
    abs_floor_vol: float,
    abs_floor_area: float,
) -> Optional[str]:
    """
    Create a stable signature string from bbox dims (sorted), volume, area.
    Returns None if insufficient CAD metrics.
    """
    if not cad:
        return None
    if not cad.get("ok_bbox", False) or not cad.get("ok_props", False):
        return None

    dims = cad.get("bbox_dims_xyz", None)
    vol = cad.get("vol_m3", None)
    area = cad.get("area_m2", None)

    if dims is None or vol is None or area is None:
        return None

    try:
        dx, dy, dz = [float(d) for d in dims]
    except Exception:
        return None

    # sort dims to be axis-agnostic
    a, b, c = sorted([dx, dy, dz])

    qa_m, qa_e = _quant_rel(a, rel_bin, abs_floor_len)
    qb_m, qb_e = _quant_rel(b, rel_bin, abs_floor_len)
    qc_m, qc_e = _quant_rel(c, rel_bin, abs_floor_len)
    qv_m, qv_e = _quant_rel(float(vol), rel_bin, abs_floor_vol)
    qs_m, qs_e = _quant_rel(float(area), rel_bin, abs_floor_area)
    
    return (
        f"bbox({qa_m}e{qa_e},{qb_m}e{qb_e},{qc_m}e{qc_e})_"
        f"vol({qv_m}e{qv_e})_area({qs_m}e{qs_e})"
    )

def _load_parts_from_manifest(manifest_path: str) -> Tuple[Dict[str, Any], List[PartRec]]:
    payload = _read_json(manifest_path)
    parts_raw = payload.get("parts", [])
    parts: List[PartRec] = []
    for pr in parts_raw:
        uid = int(pr.get("uid"))
        name = str(pr.get("name") or "")
        selected = bool(pr.get("selected", False))
        mid = pr.get("material_id", None)
        material_id = int(mid) if mid is not None else None
        cad = pr.get("cad", None) or {}
        parts.append(PartRec(uid=uid, name=name, selected=selected, material_id=material_id, cad=cad))
    return payload, parts

def _assert_manifest_compatible(payload: Dict[str, Any]) -> None:
    # Minimal checks only (don’t over-reject).
    if "parts" not in payload:
        raise ValueError("Manifest missing 'parts' field.")
    if "cache_key" not in payload:
        # Not strictly required, but good to have
        print("⚠️ Manifest missing 'cache_key' (recommended).", file=sys.stderr)

def _group_parts(
    parts: List[PartRec],
    group_mode: str,
    rel_bin: float,
    abs_floor_len: float,
    abs_floor_vol: float,
    abs_floor_area: float,
    show_all_grouped_part_names: bool = False,
) -> List[GroupRec]:
    """
    Returns list of GroupRec (only for selected parts).
    """
    selected = [p for p in parts if p.selected]
    if not selected:
        raise ValueError("No selected parts found in manifest. (Did you run runner with VOXELISE_ALL=True or valid TARGET_PARTS?)")

    if group_mode not in ("auto", "none"):
        raise ValueError("group_mode must be 'auto' or 'none'.")

    buckets: Dict[str, List[PartRec]] = defaultdict(list)

    if group_mode == "none":
        # each part is its own group, key by uid
        for p in selected:
            buckets[f"uid({p.uid})"].append(p)
    else:
        # auto
        for p in selected:
            sig = _signature_from_cad(
                p.cad, rel_bin=rel_bin,
                abs_floor_len=abs_floor_len,
                abs_floor_vol=abs_floor_vol,
                abs_floor_area=abs_floor_area,
            )
            if sig is None:
                # fallback: isolate if metrics missing
                sig = f"uid({p.uid})"
            buckets[sig].append(p)

    # Deterministic ordering:
    # sort groups by (size desc, key asc) so big repeated items appear near top
    group_items = sorted(
        buckets.items(),
        key=lambda kv: (-len(kv[1]), str(kv[0]))
    )

    groups: List[GroupRec] = []
    for gid, (gkey, plist) in enumerate(group_items, start=1):
        names = [p.name for p in plist if p.name]
        names_sorted = sorted(names)
        if show_all_grouped_part_names:
            example = "; ".join(names_sorted)
        else:
            example = "; ".join(names_sorted[:5])

        vols = []
        areas = []
        dxs = []
        dys = []
        dzs = []

        mids: List[int] = []
        for p in plist:
            if p.material_id is None:
                # selected parts should have material_id; if not, something is off upstream.
                raise ValueError(f"Selected part uid={p.uid} has material_id=None in manifest.")
            mids.append(int(p.material_id))

            cad = p.cad or {}
            v = _safe_float(cad.get("vol_m3"))
            a = _safe_float(cad.get("area_m2"))
            dims = cad.get("bbox_dims_xyz")
            if v is not None:
                vols.append(v)
            if a is not None:
                areas.append(a)
            if isinstance(dims, (list, tuple)) and len(dims) == 3:
                dxs.append(float(dims[0]))
                dys.append(float(dims[1]))
                dzs.append(float(dims[2]))

        groups.append(
            GroupRec(
                group_id=gid,
                group_key=str(gkey),
                part_count=len(plist),
                example_names=example,
                vol_m3_med=_median(vols),
                area_m2_med=_median(areas),
                bbox_dx_m_med=_median(dxs),
                bbox_dy_m_med=_median(dys),
                bbox_dz_m_med=_median(dzs),
                material_ids=sorted(set(mids)),
            )
        )

    return groups

def _csv_header_v1() -> List[str]:
    return [
        "group_id",
        "part_count",
        "example_part_names",
        # gprMax order:
        "relative_permittivity",
        "conductivity",
        "relative_permeability",
        "magnetic_loss",
        "material_name",
        # internal mapping
        "material_ids",
    ]

def _group_to_csv_row(g: GroupRec) -> List[Any]:
    return [
        g.group_id,
        g.part_count,
        g.example_names,
        "",                        # relative_permittivity (user)
        "",                        # conductivity (user)
        DEFAULT_REL_PERMEABILITY,  # default mu_r
        DEFAULT_MAGNETIC_LOSS,     # default magnetic loss
        "",                        # material_name (user)
        "|".join(str(m) for m in g.material_ids),
    ]

def _parse_float_required(value: str, field: str, group_id: str) -> float:
    try:
        return float(value)
    except Exception:
        raise ValueError(f"CSV group_id={group_id}: field '{field}' must be a number, got '{value}'")

def _build_material_table_from_csv(csv_path: str) -> Dict[int, Dict[str, Any]]:
    """
    Reads CSV and returns mapping:
      material_id -> dict(material_name, eps_r, sigma, mu_r, mag_loss)
    """
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        required_cols = set(_csv_header_v1())
        got = set(r.fieldnames or [])
        missing = sorted(required_cols - got)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        mat_by_id: Dict[int, Dict[str, Any]] = {}

        for row in r:
            gid = str(row.get("group_id", "")).strip() or "?"
            mat_name = str(row.get("material_name", "")).strip()
            eps_r_raw = str(row.get("relative_permittivity", "")).strip()
            sigma_raw = str(row.get("conductivity", "")).strip()
            mu_r_raw = str(row.get("relative_permeability", "")).strip()
            ml_raw = str(row.get("magnetic_loss", "")).strip()
            
            missing_fields = []
            if mat_name == "":
                missing_fields.append("material_name")
            if eps_r_raw == "":
                missing_fields.append("relative_permittivity")
            if sigma_raw == "":
                missing_fields.append("conductivity")
            if mu_r_raw == "":
                missing_fields.append("relative_permeability")
            if ml_raw == "":
                missing_fields.append("magnetic_loss")
            
            if missing_fields:
                raise ValueError(
                    f"CSV group_id={gid}: missing value(s) for {', '.join(missing_fields)}. "
                    "Fill all material fields before running the build step."
                )
            
            eps_r = _parse_float_required(eps_r_raw, "relative_permittivity", gid)
            sigma = _parse_float_required(sigma_raw, "conductivity", gid)
            mu_r = _parse_float_required(mu_r_raw, "relative_permeability", gid)
            mag_loss = _parse_float_required(ml_raw, "magnetic_loss", gid)

            mids_raw = str(row.get("material_ids", "")).strip()
            if not mids_raw:
                raise ValueError(f"CSV group_id={gid}: material_ids is blank (internal field).")
            try:
                mids = [int(x) for x in mids_raw.split("|") if x.strip() != ""]
            except Exception:
                raise ValueError(f"CSV group_id={gid}: could not parse material_ids='{mids_raw}'")

            for mid in mids:
                if mid in mat_by_id:
                    # This should not happen if grouping is sane; but guard anyway.
                    raise ValueError(
                        f"material_id={mid} appears in multiple CSV rows. "
                        "This indicates a CSV edit error (material_ids overlaps)."
                    )
                mat_by_id[mid] = {
                    "material_name": mat_name,
                    "relative_permittivity": eps_r,
                    "conductivity": sigma,
                    "relative_permeability": mu_r,
                    "magnetic_loss": mag_loss,
                }

    return mat_by_id

def _build_compacted_material_table_from_csv(csv_path: str) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, int]]:
    """
    Reads CSV and returns:

      compact_mat_by_new_id:
          new_compact_id -> dict(material_name, eps_r, sigma, mu_r, mag_loss)

      old_to_new_id:
          old_material_id -> new_compact_id

    Compaction is by CSV row/group, NOT by raw EM values.
    That means each CSV row becomes exactly one exported material.
    """
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        required_cols = set(_csv_header_v1())
        got = set(r.fieldnames or [])
        missing = sorted(required_cols - got)
        if missing:
            raise ValueError(f"CSV missing required columns: {missing}")

        compact_mat_by_new_id: Dict[int, Dict[str, Any]] = {}
        old_to_new_id: Dict[int, int] = {}

        new_id = 0
        for row in r:
            gid = str(row.get("group_id", "")).strip() or "?"
            mat_name = str(row.get("material_name", "")).strip()
            eps_r_raw = str(row.get("relative_permittivity", "")).strip()
            sigma_raw = str(row.get("conductivity", "")).strip()
            mu_r_raw = str(row.get("relative_permeability", "")).strip()
            ml_raw = str(row.get("magnetic_loss", "")).strip()
            
            missing_fields = []
            if mat_name == "":
                missing_fields.append("material_name")
            if eps_r_raw == "":
                missing_fields.append("relative_permittivity")
            if sigma_raw == "":
                missing_fields.append("conductivity")
            if mu_r_raw == "":
                missing_fields.append("relative_permeability")
            if ml_raw == "":
                missing_fields.append("magnetic_loss")
            
            if missing_fields:
                raise ValueError(
                    f"CSV group_id={gid}: missing value(s) for {', '.join(missing_fields)}. "
                    "Fill all material fields before running the build step."
                )
            
            eps_r = _parse_float_required(eps_r_raw, "relative_permittivity", gid)
            sigma = _parse_float_required(sigma_raw, "conductivity", gid)
            mu_r = _parse_float_required(mu_r_raw, "relative_permeability", gid)
            mag_loss = _parse_float_required(ml_raw, "magnetic_loss", gid)

            mids_raw = str(row.get("material_ids", "")).strip()
            if not mids_raw:
                raise ValueError(f"CSV group_id={gid}: material_ids is blank (internal field).")
            try:
                old_ids = [int(x) for x in mids_raw.split("|") if x.strip() != ""]
            except Exception:
                raise ValueError(f"CSV group_id={gid}: could not parse material_ids='{mids_raw}'")

            # One compact material per CSV row/group
            compact_mat_by_new_id[new_id] = {
                "material_name": mat_name,
                "relative_permittivity": eps_r,
                "conductivity": sigma,
                "relative_permeability": mu_r,
                "magnetic_loss": mag_loss,
            }

            for old_id in old_ids:
                if old_id in old_to_new_id:
                    raise ValueError(
                        f"old material_id={old_id} appears in multiple CSV rows. "
                        "This indicates a CSV edit error (material_ids overlaps)."
                    )
                old_to_new_id[old_id] = new_id

            new_id += 1

    return compact_mat_by_new_id, old_to_new_id

def _write_gprmax_materials_txt(out_path: str, mat_by_id: Dict[int, Dict[str, Any]]) -> None:
    """
    Writes a compacted materials file where line order matches compact material_id order.
    """
    if not mat_by_id:
        raise ValueError("No materials found to write.")

    max_id = max(mat_by_id.keys())
    missing = [i for i in range(0, max_id + 1) if i not in mat_by_id]
    if missing:
        raise ValueError(
            "Compacted materials are missing definitions for some material_id values. "
            f"Missing ids: {missing[:50]}" + ("..." if len(missing) > 50 else "")
        )

    lines = []
    lines.append("## gprMax materials generated by materials_builder.py")
    lines.append("## Order matters: material_id == line index (0-based)")
    lines.append("")

    for mid in range(0, max_id + 1):
        m = mat_by_id[mid]
        eps_r = m["relative_permittivity"]
        sigma = m["conductivity"]
        mu_r = m.get("relative_permeability", DEFAULT_REL_PERMEABILITY)
        mag_loss = m.get("magnetic_loss", DEFAULT_MAGNETIC_LOSS)
        name = str(m["material_name"]).strip().replace(" ", "_")

        lines.append(f"#material: {eps_r:g} {sigma:g} {mu_r:g} {mag_loss:g} {name}")

    lines.append("")
    _write_atomic_text(out_path, "\n".join(lines))
    print(f"✅ Wrote materials.txt: {out_path} (materials={max_id + 1})")

def run_materials_workflow(
    *,
    manifest_path: str,
    out_dir: str | None = None,
    csv_name: str = DEFAULT_CSV_NAME,
    txt_name: str = DEFAULT_TXT_NAME,
    group_mode: str = DEFAULT_GROUP_MODE,   # "auto" or "none"
    rel_bin: float = DEFAULT_REL_BIN,
    force_init: bool = False,
    show_all_grouped_part_names: bool = False,
) -> dict:
    """
    Runner-friendly entry point.

    Behaviour:
      - If force_init=True OR CSV does not exist -> create CSV template and return.
      - Else -> read CSV and write materials.txt.

    Returns a dict with paths + what action was taken.
    """
    manifest_path = os.path.abspath(manifest_path)
    payload, parts = _load_parts_from_manifest(manifest_path)
    _assert_manifest_compatible(payload)

    out_dir = os.path.abspath(out_dir) if out_dir else os.path.dirname(manifest_path)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, csv_name)
    txt_path = os.path.join(out_dir, txt_name)

    # Init path
    if force_init or (not os.path.exists(csv_path)):
        groups = _group_parts(
            parts,
            group_mode=group_mode,
            rel_bin=float(rel_bin),
            abs_floor_len=DEFAULT_ABS_FLOOR_LEN,
            abs_floor_vol=DEFAULT_ABS_FLOOR_VOL,
            abs_floor_area=DEFAULT_ABS_FLOOR_AREA,
            show_all_grouped_part_names=show_all_grouped_part_names,
        )

        header = _csv_header_v1()
        rows = [_group_to_csv_row(g) for g in groups]
        _write_atomic_csv(csv_path, header, rows)

        return {
            "action": "init_csv",
            "csv_path": csv_path,
            "txt_path": txt_path,
            "group_count": len(groups),
            "selected_part_count": sum(1 for p in parts if p.selected),
        }

    # Build path
    compact_mat_by_new_id, old_to_new_id = _build_compacted_material_table_from_csv(csv_path)
    _write_gprmax_materials_txt(txt_path, compact_mat_by_new_id)
    
    return {
        "action": "build_txt",
        "csv_path": csv_path,
        "txt_path": txt_path,
        "material_count": len(compact_mat_by_new_id),
        "material_id_map": old_to_new_id,
    }

# -----------------------------
# Main entry
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="Path to materials_manifest.json produced by runner")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: directory containing manifest)")
    ap.add_argument("--csv-name", default=DEFAULT_CSV_NAME, help="CSV filename (default: material_groups.csv)")
    ap.add_argument("--txt-name", default=DEFAULT_TXT_NAME, help="materials.txt filename (default: materials.txt)")
    ap.add_argument("--group-mode", default=DEFAULT_GROUP_MODE, choices=["auto", "none"], help="Grouping mode")
    ap.add_argument("--rel-bin", type=float, default=DEFAULT_REL_BIN, help="Relative quantisation bin (e.g. 0.01 = 1%)")
    ap.add_argument("--force-init", action="store_true", help="Overwrite CSV template even if it exists")
    args = ap.parse_args()

    manifest_path = os.path.abspath(args.manifest)
    payload, parts = _load_parts_from_manifest(manifest_path)
    _assert_manifest_compatible(payload)

    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.dirname(manifest_path)
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, args.csv_name)
    txt_path = os.path.join(out_dir, args.txt_name)

    # Init path (create CSV)
    if args.force_init or (not os.path.exists(csv_path)):
        groups = _group_parts(
            parts,
            group_mode=args.group_mode,
            rel_bin=float(args.rel_bin),
            abs_floor_len=DEFAULT_ABS_FLOOR_LEN,
            abs_floor_vol=DEFAULT_ABS_FLOOR_VOL,
            abs_floor_area=DEFAULT_ABS_FLOOR_AREA,
            show_all_grouped_part_names=False,
        )

        header = _csv_header_v1()
        rows = [_group_to_csv_row(g) for g in groups]
        _write_atomic_csv(csv_path, header, rows)

        print(f"✅ Wrote CSV template: {csv_path}")
        print(f"   Groups: {len(groups)} | Selected parts: {sum(1 for p in parts if p.selected)}")
        print("➡️  Now fill: material_name, relative_permittivity, conductivity (and optionally permeability/loss).")
        return 0

    # Build path (CSV exists)
    mat_by_id = _build_material_table_from_csv(csv_path)
    _write_gprmax_materials_txt(txt_path, mat_by_id)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
    