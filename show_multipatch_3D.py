import argparse
import json
from pathlib import Path

import gustaf as gus
import splinepy


def load_result(path: Path):
    with path.open() as file:
        data = json.load(file)
    return data["multipatch_elasticity"]


def make_patch(patch_data, deformed=False):
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
    for patch in patches:
        if not hasattr(patch, "show_options"):
            continue
        patch.show_options["control_points"] = False
        patch.show_options["control_mesh"] = False
        patch.show_options["control_point_ids"] = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result",
        nargs="?",
        default="results/result_multipatch_parametric.json",
        help="Path to the 3D multipatch result json",
    )
    parser.add_argument(
        "--deformed-only",
        action="store_true",
        help="Show only the deformed multipatch",
    )
    args = parser.parse_args()

    result = load_result(Path(args.result))
    patches = result["patches"]

    reference = [make_patch(patch_data, deformed=False) for patch_data in patches]
    deformed = [make_patch(patch_data, deformed=True) for patch_data in patches]

    style_patches(reference)
    style_patches(deformed)

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
