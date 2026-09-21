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


def make_patch(patch_data, deformed=False, exaggerate=1.0):
    # Each patch already carries all spline data needed for reconstruction.
    reference = patch_data["control_points"]
    if not deformed:
        control_points = reference
    elif exaggerate == 1.0:
        control_points = patch_data["deformed_control_points"]
    else:
        # Scale only the displacement (deformed - reference), not the
        # absolute position, so exaggerate=1.0 always matches the raw result.
        deformed_cp = patch_data["deformed_control_points"]
        control_points = [
            [r + exaggerate * (d - r) for r, d in zip(ref_pt, def_pt)]
            for ref_pt, def_pt in zip(reference, deformed_cp)
        ]
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


def derive_loss_plot_path(output_path: Path) -> Path:
    return output_path.with_name(output_path.stem + "_loss" + output_path.suffix)


def plot_loss_history(loss_history, output_path: Path = None, interactive: bool = True, title: str = "Training Loss", max_epoch: int = None, enabled: bool = True):
    """Plot the per-epoch training loss in its own window, opened after the
    3D solution view is closed (or saved to a PNG when non-interactive).
    Does nothing if the result has no loss_history, e.g. a classical
    reference solution that was solved directly rather than trained."""
    if not enabled:
        return
    if not loss_history:
        print("No loss_history in this result (not an iteratively trained "
              "result?); skipping loss plot.")
        return

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        print(f"Could not plot loss history (matplotlib not installed): {exc}")
        return

    # The optimizer (LBFGS) evaluates the loss several times per epoch and
    # every evaluation is recorded, so the history is longer than max_epoch.
    # If the result stores max_epoch, plot against real epochs; otherwise
    # label the axis honestly as loss evaluations.
    if max_epoch:
        evals_per_epoch = len(loss_history) / max_epoch
        epochs = [(i + 1) / evals_per_epoch for i in range(len(loss_history))]
        xlabel = "Epoch"
    else:
        epochs = range(1, len(loss_history) + 1)
        xlabel = "Epoch"
    fig, ax = plt.subplots(num=title)
    ax.plot(epochs, loss_history)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Total loss")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Loss plot saved to {output_path}")

    if interactive:
        plt.show()
    else:
        plt.close(fig)


def render_result(result_path: Path, output_path: Path, deformed_only: bool, show_patch_ids: bool, interactive: bool = True, exaggerate: float = 1.0, plot_loss: bool = True, save_screenshot: bool = False):
    result = load_result(result_path)
    patches = result["patches"]

    reference = [make_patch(patch_data, deformed=False) for patch_data in patches]
    deformed = [make_patch(patch_data, deformed=True, exaggerate=exaggerate) for patch_data in patches]

    style_patches(reference)
    style_patches(deformed)

    if show_patch_ids:
        print_patch_legend(patches)

    if save_screenshot:
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

        print(f"Screenshot saved to {output_path}")

    print(f"Showing {result_path}")
    loss_history = result.get("loss_history")
    loss_output_path = derive_loss_plot_path(output_path) if save_screenshot else None

    if not interactive:
        print("Skipping interactive window (--no-interactive).")
        plot_loss_history(loss_history, enabled=plot_loss, max_epoch=result.get("max_epoch"), output_path=loss_output_path, interactive=False)
        return

    if deformed_only:
        splinepy.show(
            [*deformed],
            control_mesh=False,
            control_point_ids=False,
        )
    else:
        splinepy.show(
            [*reference],
            [*deformed],
            control_mesh=False,
            control_point_ids=False,
        )

    plot_loss_history(loss_history, enabled=plot_loss, max_epoch=result.get("max_epoch"), output_path=loss_output_path, interactive=True)


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
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip opening the final interactive window (useful when no display is "
             "available, e.g. over SSH without X forwarding); add --screenshot if you "
             "still want a PNG saved",
    )
    parser.add_argument(
        "--exaggerate",
        type=float,
        default=1.0,
        help="Visually scale the displacement (deformed - reference) by this factor "
             "for the deformed plot only; does not change the underlying data (default: 1.0)",
    )
    parser.add_argument(
        "--no-loss-plot",
        action="store_true",
        help="Skip the separate training-loss plot window shown after the 3D solution",
    )
    parser.add_argument(
        "--screenshot",
        action="store_true",
        help="Save a PNG screenshot of the 3D view (to --output, or its default path) and the loss plot (same name with _loss suffix). "
             "Off by default; combine with --no-interactive for a purely headless run "
             "that still produces an image.",
    )
    args = parser.parse_args()

    result_paths = resolve_result_paths(args.result)
    if not result_paths:
        raise FileNotFoundError("Could not find any matching 3D multipatch result json files.")

    for result_path in result_paths:
        output_path = Path(args.output) if args.output is not None else derive_output_path(result_path)
        render_result(
            Path(result_path),
            output_path,
            args.deformed_only,
            args.show_patch_ids,
            interactive=not args.no_interactive,
            exaggerate=args.exaggerate,
            plot_loss=not args.no_loss_plot,
            save_screenshot=args.screenshot,
        )


if __name__ == "__main__":
    main()
