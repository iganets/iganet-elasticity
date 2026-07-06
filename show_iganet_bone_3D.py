import json
from pathlib import Path

import numpy as np

try:
    import gustaf as gus
    import splinepy
except ModuleNotFoundError as exc:
    raise SystemExit(
        "This visualization needs the same Python visualization stack as "
        "show_iganet_2D.py: install/activate splinepy and gustaf first."
    ) from exc


RESULT_PATH = Path("result_bone_3D.json")


def has_key_nonempty(d, key):
    return (key in d) and (d[key] is not None) and (len(d[key]) > 0)


def latest_result(data):
    latest = dict(data)
    if not has_key_nonempty(latest, "net_CtrlPts") and has_key_nonempty(data, "net_Snapshots"):
        latest.update(data["net_Snapshots"][-1])
    return latest


def as_ctrlpts(data, key):
    arr = np.asarray(data[key], dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"'{key}' has shape {arr.shape}; expected (n, 3).")
    return arr


def split_patch_ctrlpts(ctrlpts, patch_meta):
    patches = []
    offset = 0
    for meta in patch_meta:
        ncoeffs = tuple(int(v) for v in meta["ncoeffs"])
        count = int(np.prod(ncoeffs))
        patches.append(ctrlpts[offset : offset + count])
        offset += count

    if offset != len(ctrlpts):
        raise ValueError(
            f"Patch metadata accounts for {offset} control points, "
            f"but result contains {len(ctrlpts)}."
        )
    return patches


def make_patch_spline(meta, ctrlpts, translation=None):
    points = np.array(ctrlpts, dtype=float, copy=True)
    if translation is not None:
        points += np.asarray(translation, dtype=float)

    spline = splinepy.BSpline(
        degrees=tuple(int(v) for v in meta["degrees"]),
        knot_vectors=[list(kv) for kv in meta["knot_vectors"]],
        control_points=points,
    )
    spline.show_options["control_mesh"] = False
    spline.show_options["control_points"] = False
    spline.show_options["control_point_ids"] = False
    return spline


def make_multipatch(data, key, translation=None):
    patch_meta = data["net_PatchMeta"]
    patch_ctrlpts = split_patch_ctrlpts(as_ctrlpts(data, key), patch_meta)
    patches = [
        make_patch_spline(meta, ctrlpts, translation)
        for meta, ctrlpts in zip(patch_meta, patch_ctrlpts)
    ]
    multipatch = splinepy.Multipatch(patches)
    try:
        multipatch.determine_interfaces()
    except Exception as exc:
        print(f"Warning: could not determine visualization interfaces: {exc}")
    return multipatch


def side_by_side_translation(origin_ctrlpts):
    mins = origin_ctrlpts.min(axis=0)
    maxs = origin_ctrlpts.max(axis=0)
    width = maxs[0] - mins[0]
    return np.array([1.35 * width, 0.0, 0.0])


with RESULT_PATH.open("r") as file:
    data = latest_result(json.load(file))

if not has_key_nonempty(data, "net_PatchMeta"):
    raise ValueError("Result file does not contain 'net_PatchMeta'. Run the bone executable first.")

if not has_key_nonempty(data, "net_OriginCtrlPts") or not has_key_nonempty(data, "net_CtrlPts"):
    raise ValueError("Result file does not yet contain geometry results. Let the bone run reach an export epoch.")

origin_ctrlpts = as_ctrlpts(data, "net_OriginCtrlPts")
translation = side_by_side_translation(origin_ctrlpts)

original = make_multipatch(data, "net_OriginCtrlPts")
deformed = make_multipatch(data, "net_CtrlPts", translation=translation)

epoch = data.get("net_Epoch", "unknown")
loss = data.get("net_TotalLoss", None)
label_suffix = f"epoch {epoch}"
if loss is not None:
    label_suffix += f", loss {loss:.4g}"

gus.show(
    ["Ausgangsgeometrie", original],
    [f"Verformte Geometrie ({label_suffix})", deformed],
    control_mesh=False,
    control_point_ids=False,
)
