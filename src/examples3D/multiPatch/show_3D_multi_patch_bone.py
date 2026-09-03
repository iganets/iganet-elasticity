"""Visualize the iganet_lin_elasticity_3D_multipatch_bone result with splinepy.

Rebuilds the reference and deformed patch set from that program's JSON
result file and renders them side by side. This script is deliberately
near-identical to show_3D_multi_patch_parametrized.py (which does the same
for the other 3D multipatch example) rather than sharing a common module.
"""

import argparse
import colorsys
import json
from pathlib import Path

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
DEFAULT_RESULT_PATH = REPO_ROOT / "results" / "result_iganet_lin_elasticity_3D_multipatch_bone.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "iganet_lin_elasticity_3D_multipatch_bone.png"

# Helper functions keep the main plotting routine compact.
def derive_output_path(result_path: Path) -> Path:
    stem = result_path.stem
    if stem.startswith("result_"):
        stem = stem[len("result_"):]
    return REPO_ROOT / "results" / f"{stem}.png"


def resolve_result_paths(result_arg):
    if result_arg is not None:
        return [Path(result_arg)]

    return [DEFAULT_RESULT_PATH]


def load_result(path: Path):
    with path.open() as file:
        data = json.load(file)
    return data["multipatch_elasticity"]


def make_patch(patch_data, deformed=False):
    # Each patch already carries all spline data needed for reconstruction.
    control_points = (
        patch_data["deformed_control_points"]
        if deformed
        else patch_data["control_points"]
    )
    return splinepy.BSpline(
        degrees=patch_data["degrees"],
        knot_vectors=patch_data["knot_vectors"],
        control_points=control_points,
    )



_GOLDEN_RATIO_CONJUGATE = 0.6180339887498949


def patch_color(index: int):
    hue = (index * _GOLDEN_RATIO_CONJUGATE) % 1.0
    return colorsys.hsv_to_rgb(hue, 1.0, 1.0)


def style_patches(patches):
    for index, patch in enumerate(patches):
        if not hasattr(patch, "show_options"):
            continue
        patch.show_options["control_points"] = False
        patch.show_options["control_mesh"] = False
        patch.show_options["control_point_ids"] = False
        patch.show_options["c"] = patch_color(index)


def print_patch_legend(patches_data):
    print("Patch legend (xml_id: RGB color):")
    for index, patch_data in enumerate(patches_data):
        xml_id = patch_data.get("xml_id", 100 + index)
        r, g, b = patch_color(index)
        print(f"  {xml_id}: ({r:.2f}, {g:.2f}, {b:.2f})")


def render_result(result_path: Path, output_path: Path, deformed_only: bool, show_patch_ids: bool):
    result = load_result(result_path)
    patches = result["patches"]

    reference = [make_patch(patch_data, deformed=False) for patch_data in patches]
    deformed = [make_patch(patch_data, deformed=True) for patch_data in patches]

    style_patches(reference)
    style_patches(deformed)

    if show_patch_ids:
        print_patch_legend(patches)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if deformed_only:
        splinepy.show(
            [*deformed],
            offscreen=True,
            interactive=False,
            close=True,
        )
    else:
        splinepy.show(
            [*reference],
            [*deformed],
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
        print(f"Could not save screenshot with vedo for {result_path}: {exc}")
        raise

    print(f"Showing {result_path}")
    if deformed_only:
        splinepy.show(
            [*deformed],
            control_mesh=False,
            control_point_ids=False,
        )
        return

    splinepy.show(
        [*reference],
        [*deformed],
        control_mesh=False,
        control_point_ids=False,
    )


def main():
    # Read one or more stored patch sets, rebuild spline objects, and render them.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result",
        nargs="?",
        default=None,
        help="Optional path to a specific 3D multipatch result json",
    )
    parser.add_argument(
        "--deformed-only",
        action="store_true",
        help="Show only the deformed multipatch",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to the output PNG file (only used when one result file is shown)",
    )
    parser.add_argument(
        "--show-patch-ids",
        action="store_true",
        help="Print a console legend mapping each patch's xml_id to its render color",
    )
    args = parser.parse_args()

    result_paths = resolve_result_paths(args.result)
    if not result_paths:
        raise FileNotFoundError("Could not find any matching 3D multipatch result json files.")

    for result_path in result_paths:
        output_path = Path(args.output) if args.output is not None else derive_output_path(result_path)
        render_result(Path(result_path), output_path, args.deformed_only, args.show_patch_ids)


if __name__ == "__main__":
    main()
