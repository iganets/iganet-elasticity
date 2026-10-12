#!/usr/bin/env python3
"""
Append one row per training run to results/run_log.xlsx.

Called by the two multipatch examples right after they write their result, so
every run leaves a record of the settings it used together with what came out.
The bone and the parametric chain keep their own sheet, because they describe
their boundary conditions differently: the bone lists them per patch under
patches_3d, the chain states them globally per side.

Usage:
  python3 -m std_collocation_python.log_run <sheet> <config.json> <result.json>

A missing result file is not an error - the settings are still recorded, the
outcome columns stay empty.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter

from .run_std_coll import load_json_with_line_comments

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = REPO_ROOT / "results" / "run_log.xlsx"

COLUMNS = [
    "Zeitpunkt",
    "Lauf-ID",
    "Epochen",
    "Hidden Layers",
    "E (global)",
    "nu (global)",
    "Material je Patch",
    "Volumenkraft",
    "Dirichlet",
    "Kraft",
    "kraftfrei",
    "Geometrie",
    "Verlust Start",
    "Verlust Ende",
    "Dauer [s]",
]


def _compact(value) -> str:
    """Lists and dicts as compact JSON, so one cell holds them readably."""
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(", ", ": "), ensure_ascii=False)
    return str(value)


def _boundary_conditions(cfg: dict) -> tuple[str, str, str]:
    """Returns (dirichlet, force, traction-free) however the config states them.

    The chain keeps them globally under multipatch.boundary_conditions, the
    bone per patch under patches_3d. Both are folded into one text per kind.
    """
    mp = cfg.get("multipatch", {})
    if "boundary_conditions" in mp:
        bc = mp["boundary_conditions"]
        return (_compact(bc.get("diri_sides")),
                _compact(bc.get("force_sides")),
                _compact(bc.get("tfbc_sides")))

    diri, force, tfbc = {}, {}, {}
    for patch in cfg.get("patches_3d", []):
        pid = patch.get("patch_id")
        bc = patch.get("boundary_conditions", {})
        if bc.get("diri_sides"):
            diri[pid] = bc["diri_sides"]
        if bc.get("force_sides"):
            force[pid] = bc["force_sides"]
        if bc.get("tfbc_sides"):
            tfbc[pid] = bc["tfbc_sides"]
    return _compact(diri), _compact(force), _compact(tfbc)


def _row(cfg: dict, result: dict) -> list:
    material = cfg.get("material", {})
    per_patch = material.get("patches")
    diri, force, tfbc = _boundary_conditions(cfg)
    geometry = cfg.get("geometry", {})
    xml = geometry.get("multipatch_xml_path")
    if xml:
        geometry_text = Path(xml).name
    else:
        geometry_text = f"{cfg.get('multipatch', {}).get('num_patches', '?')} Wuerfel"

    return [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        # Same id the run wrote into its result file and that the show script
        # puts into the figure names, so a row and its screenshots match up.
        result.get("run_id", ""),
        cfg.get("simulation", {}).get("max_epoch"),
        _compact(cfg.get("network", {}).get("hidden_layers")),
        material.get("young_modulus"),
        material.get("poisson_ratio"),
        _compact(per_patch) if per_patch else "einheitlich",
        _compact(cfg.get("multipatch", {}).get("body_force")),
        diri,
        force,
        tfbc,
        geometry_text,
        result.get("loss_initial"),
        result.get("loss_final"),
        round(result["elapsed_seconds"], 1) if "elapsed_seconds" in result else None,
    ]


def _sheet(book: Workbook, name: str):
    if name in book.sheetnames:
        return book[name]
    sheet = book.create_sheet(name)
    sheet.append(COLUMNS)
    for cell in sheet[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(vertical="top")
    widths = [19, 17, 9, 14, 11, 12, 34, 16, 30, 26, 16, 26, 14, 14, 10]
    for index, width in enumerate(widths, start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    sheet.freeze_panes = "A2"
    return sheet


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 2:
        raise SystemExit(
            "usage: python3 -m std_collocation_python.log_run "
            "<sheet> <config.json> [result.json] [log.xlsx]")

    sheet_name = args[0]
    config_path = Path(args[1])
    result_path = Path(args[2]) if len(args) > 2 else None
    log_path = Path(args[3]) if len(args) > 3 else DEFAULT_LOG
    for path in (config_path, log_path):
        if not path.is_absolute():
            path = REPO_ROOT / path

    cfg = load_json_with_line_comments(str(config_path))

    result = {}
    if result_path is not None and result_path.exists():
        raw = json.load(open(result_path))
        result = raw.get("multipatch_elasticity", raw)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    if log_path.exists():
        book = load_workbook(log_path)
    else:
        book = Workbook()
        book.remove(book.active)          # drop the default empty sheet

    _sheet(book, sheet_name).append(_row(cfg, result))
    try:
        book.save(log_path)
    except OSError as error:
        # By far the most common case: the workbook is open in Excel, which
        # locks the file. Say so plainly instead of dumping a traceback.
        raise SystemExit(
            f"Konnte {log_path} nicht schreiben: {error}\n"
            "Ist die Datei gerade in Excel geoeffnet? Dann sperrt Excel sie. "
            "Schliessen und den Lauf wiederholen, oder den Eintrag nachtragen "
            f"mit: python3 -m std_collocation_python.log_run {sheet_name} "
            "<config.json> <result.json>")
    print(f"Lauf protokolliert: {log_path} (Blatt {sheet_name})")


if __name__ == "__main__":
    main()
