"""Visualize the 3D multi-patch elasticity result with splinepy.

This is the simplest of the show scripts: it rebuilds the reference and
deformed patch set from the JSON file and renders them side by side.
"""

import argparse
import json
from pathlib import Path

import gustaf as gus
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
DEFAULT_RESULT_PATH = REPO_ROOT / "results" / "result_iganet_lin_elasticity_3D_multipatch_parametrized.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "iganet_lin_elasticity_3D_multipatch_parametrized.png"

# Helper functions keep the main plotting routine compact.
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


def style_patches(patches):
    # For 3D screenshots the surface itself is usually the important signal,
    # not the underlying control mesh. We therefore hide those extras here.
    for patch in patches:
        if not hasattr(patch, "show_options"):
            continue
        patch.show_options["control_points"] = False
        patch.show_options["control_mesh"] = False
        patch.show_options["control_point_ids"] = False


def main():
    # Read the stored patch data, rebuild spline objects, and render them.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result",
        nargs="?",
        default=str(DEFAULT_RESULT_PATH),
        help="Path to the 3D multipatch result json",
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

    result = load_result(Path(args.result))
    patches = result["patches"]

    reference = [make_patch(patch_data, deformed=False) for patch_data in patches]
    deformed = [make_patch(patch_data, deformed=True) for patch_data in patches]

    style_patches(reference)
    style_patches(deformed)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.deformed_only:
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
        print(f"Could not save screenshot with vedo: {exc}")
        raise

    if args.deformed_only:
        gus.show([*deformed], control_mesh=False, control_point_ids=False)
        return

    gus.show(
        [*reference],
        [*deformed],
        control_mesh=False,
        control_point_ids=False,
    )


if __name__ == "__main__":
    main()
