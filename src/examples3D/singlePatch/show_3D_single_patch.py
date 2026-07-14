"""Visualize the final result of the 3D single-patch example.

The script mirrors the 2D single-patch visualizer, but reconstructs a 3D
spline object from the stored control points and renders the deformed result
next to the reference geometry when available.
"""

import json
import argparse
from pathlib import Path

import numpy as np
import splinepy


SCRIPT_DIR = Path(__file__).resolve().parent


def find_repo_root(start: Path) -> Path:
    for current in [start, *start.parents]:
        if (
            (current / "CMakeLists.txt").exists()
            and (current / "src").exists()
            and (current / "results").exists()
        ):
            return current
    raise RuntimeError(f"Could not locate repository root from script path: {start}")


REPO_ROOT = find_repo_root(SCRIPT_DIR)
DEFAULT_RESULT_PATH = REPO_ROOT / "results" / "result_iganet_lin_elasticity_3D.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "iganet_lin_elasticity_3D.png"

# These helpers convert JSON arrays back into splinepy-friendly spline data.
def load_result(path):
    with open(path, "r") as file:
        return json.load(file)


def has_key_nonempty(data, key):
    return key in data and data[key] is not None and len(data[key]) > 0


def as_ctrlpts(data, key):
    arr = np.asarray(data[key], dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"'{key}' has unexpected shape {arr.shape}. Expected (n, dim>=2).")
    return arr


def infer_parametric_dim(ctrlpts, key):
    npts = ctrlpts.shape[0]
    phys_dim = ctrlpts.shape[1]

    if phys_dim == 2:
        n1d = int(round(np.sqrt(npts)))
        if n1d * n1d != npts:
            raise ValueError(
                f"'{key}' has {npts} control points; expected a perfect square for a 2D tensor grid."
            )
        return 2, n1d

    if phys_dim == 3:
        n1d = int(round(npts ** (1.0 / 3.0)))
        if n1d * n1d * n1d != npts:
            raise ValueError(
                f"'{key}' has {npts} control points; expected a perfect cube for a 3D tensor grid."
            )
        return 3, n1d

    raise ValueError(f"Unsupported physical dimension {phys_dim} in '{key}'.")


def generate_knot_vectors(degree, nr_ctrl_pts_1d, para_dim):
    num_knots = nr_ctrl_pts_1d + degree + 1
    num_repeats = degree + 1
    num_inner_knots = num_knots - 2 * num_repeats

    knot_vector = [0.0] * num_repeats
    if num_inner_knots > 0:
        step = 1.0 / (nr_ctrl_pts_1d - degree)
        knot_vector.extend(i * step for i in range(1, num_inner_knots + 1))
    knot_vector.extend([1.0] * num_repeats)

    return tuple(knot_vector for _ in range(para_dim))


def make_bspline(control_points, degree, key):
    para_dim, nr_ctrl_pts_1d = infer_parametric_dim(control_points, key)
    knot_vectors = generate_knot_vectors(degree, nr_ctrl_pts_1d, para_dim)
    return splinepy.BSpline(
        degrees=(degree,) * para_dim,
        knot_vectors=knot_vectors,
        control_points=control_points,
    )


def make_iganet_object(data, degree):
    if has_key_nonempty(data, "net_CtrlPts"):
        return make_bspline(as_ctrlpts(data, "net_CtrlPts"), degree, "net_CtrlPts")

    patch_keys = sorted(
        key for key in data.keys() if key.startswith("net_patch") and key.endswith("_CtrlPts")
    )
    if not patch_keys:
        raise KeyError("Could not find 'net_CtrlPts' or any 'net_patch*_CtrlPts' entries.")

    patches = [make_bspline(as_ctrlpts(data, key), degree, key) for key in patch_keys]
    multipatch = splinepy.Multipatch(patches)
    multipatch.determine_interfaces()
    return multipatch


def make_reference_object(data, degree):
    if not has_key_nonempty(data, "stdCollCtrlPts"):
        raise KeyError("Could not find 'stdCollCtrlPts' in result file.")
    return make_bspline(as_ctrlpts(data, "stdCollCtrlPts"), degree, "stdCollCtrlPts")


def main():
    # Read the result file, rebuild spline objects, then render/save them.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result",
        nargs="?",
        default=str(DEFAULT_RESULT_PATH),
        help="Path to the single-patch 3D result json",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to the output PNG file",
    )
    args = parser.parse_args()

    result_path = Path(args.result)
    data = load_result(result_path)

    if "net_Degree" not in data:
        raise KeyError("Could not find 'net_Degree' in result file.")

    degree = int(data["net_Degree"])
    reference = make_reference_object(data, degree)
    iganet_solution = make_iganet_object(data, degree)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    splinepy.show(
        ["Reference Solution", reference],
        ["IgANet Solution", iganet_solution],
        offscreen=True,
        interactive=False,
        close=True,
    )

    try:
        import vedo

        if hasattr(vedo, "screenshot"):
            vedo.screenshot(str(output_path), scale=2)
        else:
            raise AttributeError("vedo.screenshot not available")
    except Exception as exc:
        print(f"Could not save screenshot with vedo: {exc}")
        raise

    splinepy.show(
        ["Reference Solution", reference],
        ["IgANet Solution", iganet_solution],
        control_mesh=False,
        control_point_ids=False,
    )


if __name__ == "__main__":
    main()
