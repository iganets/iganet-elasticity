"""Visualize the 2D multi-patch elasticity result with splinepy.

Besides the patch geometry itself, this script also shows diagnostic point
sets such as interior collocation points, traction-free points, force points,
and interface points. That makes it useful for debugging the collocation
placement as well as the final deformation.
"""

import argparse
import json
from pathlib import Path

import gustaf as gus
import splinepy


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_RESULT_PATH = REPO_ROOT / "results" / "result_iganet_lin_elasticity_2D_multipatch_parametric.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "iganet_lin_elasticity_2D_multipatch_parametric.png"

# Helper functions keep the actual plotting logic in main() compact.
def load_result(path: Path):
    with path.open() as f:
        data = json.load(f)
    return data["multipatch_elasticity_2d"]


def make_patch(patch_data, deformed=False):
    # The result file stores a full spline description per patch, so the
    # visualization can be reconstructed without reopening the source XML.
    degrees = patch_data["degrees"]
    knot_vectors = patch_data["knot_vectors"]
    control_points = (
        patch_data["deformed_control_points"]
        if deformed
        else patch_data["control_points"]
    )

    return splinepy.BSpline(
        degrees=degrees,
        knot_vectors=knot_vectors,
        control_points=control_points,
    )


def style_patches(patches):
    # Showing control points is useful here because the script is often used
    # for debugging patch interfaces and collocation placement.
    for patch in patches:
        if hasattr(patch, "show_options"):
            patch.show_options["control_points"] = True
            patch.show_options["control_mesh"] = False


def flatten_points(entries, key="points"):
    # Point data is stored patch-wise or interface-wise in the result file.
    # For visualization we flatten it into one simple list per category.
    points = []
    for entry in entries:
        points.extend(entry.get(key, []))
    return points


def make_vertices(points, color, radius=12):
    if not points:
        return None
    vertices = gus.Vertices(points)
    vertices.show_options.update(c=color, r=radius)
    return vertices


def main():
    # Rebuild reference and deformed patches, then overlay point diagnostics.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result",
        nargs="?",
        default=str(DEFAULT_RESULT_PATH),
        help="Path to the 2D multipatch result json",
    )
    parser.add_argument(
        "--deformed-only",
        action="store_true",
        help="Show only the deformed multipatch",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to the output PNG file",
    )
    args = parser.parse_args()

    result_path = Path(args.result)
    result = load_result(result_path)
    patches = result["patches"]
    collocation = result.get("collocation_points", {})

    reference = [make_patch(p, deformed=False) for p in patches]
    deformed = [make_patch(p, deformed=True) for p in patches]

    style_patches(reference)
    style_patches(deformed)

    interior_points = flatten_points(collocation.get("interior", []))
    tfbc_points = flatten_points(collocation.get("traction_free", []))
    force_points = flatten_points(collocation.get("force", []))
    interface_points = flatten_points(collocation.get("interface", []), key="points_patch1")

    overlays = []
    # The colors below are intentionally semantic:
    # black = interior PDE points,
    # blue  = traction-free boundary points,
    # red   = prescribed-force boundary points,
    # orange = interface points.
    for obj in [
        make_vertices(interior_points, "black", 10),
        make_vertices(tfbc_points, "blue", 14),
        make_vertices(force_points, "red", 14),
        make_vertices(interface_points, "orange", 16),
    ]:
        if obj is not None:
            overlays.append(obj)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.deformed_only:
        splinepy.show(
            ["deformed", *deformed, *overlays],
            offscreen=True,
            interactive=False,
            close=True,
        )
    else:
        splinepy.show(
            ["reference", *reference, *overlays],
            ["deformed", *deformed, *overlays],
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


if __name__ == "__main__":
    main()
