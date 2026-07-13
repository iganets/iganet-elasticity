import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import gustaf as gus
import numpy as np
import splinepy


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_RESULT_PATH = REPO_ROOT / "results" / "result_iganet_lin_elasticity_2D_video.json"
DEFAULT_REFERENCE_RESULT_PATH = REPO_ROOT / "results" / "result_iganet_lin_elasticity_2D.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "iganet_lin_elasticity_2D_video.png"


def load_video_result(path: Path):
    with path.open() as file:
        data = json.load(file)
    if "multipatch_training_video" in data:
        return {"format": "multipatch_3d", "data": data["multipatch_training_video"]}
    if "frames" in data and "degree" in data:
        return {"format": "single_patch_2d", "data": data}
    raise RuntimeError(f"Unsupported video JSON format in {path}")


def load_reference_result(path: Path):
    with path.open() as file:
        return json.load(file)


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


def infer_num_ctrl_pts_per_direction(control_points):
    nctrl = len(control_points)
    n1d = int(round(nctrl ** 0.5))
    if n1d * n1d != nctrl:
        raise RuntimeError(
            f"Could not infer a square 2D control grid from {nctrl} control points."
        )
    return n1d


def make_patch(patch_data, deformed):
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


def make_single_patch_2d(control_points, degree, nr_ctrl_pts_1d):
    return splinepy.BSpline(
        degrees=[degree, degree],
        knot_vectors=generate_knot_vectors(degree, nr_ctrl_pts_1d, 2),
        control_points=control_points,
    )


def build_runtime_objects(video_format, video_data, frames, result_path):
    if video_format == "multipatch_3d":
        reference_objects = [
            make_patch(patch_data, deformed=False)
            for patch_data in video_data.get("reference_patches", [])
        ]
        animated_objects = [
            make_patch(patch_data, deformed=True)
            for patch_data in frames[0]["patches"]
        ]
        style_patches(reference_objects)
        style_patches(animated_objects)

        def update(frame):
            for patch, patch_data in zip(animated_objects, frame["patches"]):
                patch.cps[:] = np.asarray(
                    patch_data["deformed_control_points"], dtype=float
                )

        return {
            "reference": reference_objects,
            "animated": animated_objects,
            "update": update,
        }

    if video_format == "single_patch_2d":
        degree = int(video_data["degree"])
        reference_control_points = video_data.get(
            "initial_control_points", frames[0]["control_points"]
        )
        reference_result_path = DEFAULT_REFERENCE_RESULT_PATH
        if reference_result_path.exists():
            reference_result = load_reference_result(reference_result_path)
            if "stdCollCtrlPts" in reference_result:
                reference_control_points = reference_result["stdCollCtrlPts"]

        nr_ctrl_pts_1d = int(
            video_data.get(
                "num_ctrl_pts_per_direction",
                infer_num_ctrl_pts_per_direction(reference_control_points),
            )
        )

        reference_patch = make_single_patch_2d(
            reference_control_points,
            degree,
            nr_ctrl_pts_1d,
        )
        animated_patch = make_single_patch_2d(
            frames[0]["control_points"],
            degree,
            nr_ctrl_pts_1d,
        )
        style_patches([reference_patch])
        style_patches([animated_patch])

        def update(frame):
            animated_patch.cps[:] = np.asarray(frame["control_points"], dtype=float)

        return {
            "reference": [reference_patch],
            "animated": [animated_patch],
            "update": update,
        }

    raise RuntimeError(f"Unsupported video format: {video_format}")

def show_sections(plotter, runtime_objects, show_reference, interactive):
    reference_objects, animated_objects = runtime_objects["reference"], runtime_objects["animated"]
    if show_reference and reference_objects:
        return gus.show(
            [*reference_objects],
            [*animated_objects],
            vedoplot=plotter,
            interactive=interactive,
            close=not interactive,
        )
    return gus.show(
        [*animated_objects],
        vedoplot=plotter,
        interactive=interactive,
        close=not interactive,
    )


def save_mp4(result_path, runtime_objects, frames, show_reference, dt_ms, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fps = max(1, round(1000.0 / max(dt_ms, 1)))

    with tempfile.TemporaryDirectory(prefix="iganet_video_frames_", dir=result_path.parent) as tmpdir:
        frame_dir = Path(tmpdir)

        runtime_objects["update"](frames[0])
        plotter = show_sections(
            plotter=None,
            runtime_objects=runtime_objects,
            show_reference=show_reference,
            interactive=False,
        )
        plotter.screenshot(str(frame_dir / "frame_000000.png"), scale=1)

        for index, frame in enumerate(frames[1:], start=1):
            runtime_objects["update"](frame)
            plotter = show_sections(
                plotter=plotter,
                runtime_objects=runtime_objects,
                show_reference=show_reference,
                interactive=False,
            )
            plotter.screenshot(str(frame_dir / f"frame_{index:06d}.png"), scale=1)

        plotter.close()

        ffmpeg_cmd = [
            shutil.which("ffmpeg") or "ffmpeg",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%06d.png"),
            "-pix_fmt",
            "yuv420p",
            "-vcodec",
            "libx264",
            str(output_path),
        ]
        subprocess.run(ffmpeg_cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "result",
        nargs="?",
        default=str(DEFAULT_RESULT_PATH),
        help="Path to the training-video JSON file",
    )
    parser.add_argument(
        "--dt",
        type=int,
        default=60,
        help="Milliseconds between frames while playing",
    )
    parser.add_argument(
        "--hide-reference",
        action="store_true",
        help="Show only the deformed patches",
    )
    parser.add_argument(
        "--mp4",
        help="Write the animation to an MP4 file instead of opening the final interactive viewer",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Path to the output PNG file for the final frame",
    )
    args = parser.parse_args()

    result_path = Path(args.result)
    loaded = load_video_result(result_path)
    video_format = loaded["format"]
    video_data = loaded["data"]
    frames = video_data.get("frames", [])
    if not frames:
        raise RuntimeError(f"No frames found in {result_path}")

    show_reference = not args.hide_reference
    runtime_objects = build_runtime_objects(
        video_format, video_data, frames, result_path
    )

    if args.mp4:
        save_mp4(
            result_path=result_path,
            runtime_objects=runtime_objects,
            frames=frames,
            show_reference=show_reference,
            dt_ms=args.dt,
            output_path=args.mp4,
        )
        return

    runtime_objects["update"](frames[0])
    plotter = show_sections(
        plotter=None,
        runtime_objects=runtime_objects,
        show_reference=show_reference,
        interactive=False,
    )

    delay_seconds = max(args.dt, 1) / 1000.0
    for frame in frames[1:]:
        runtime_objects["update"](frame)
        plotter = show_sections(
            plotter=plotter,
            runtime_objects=runtime_objects,
            show_reference=show_reference,
            interactive=False,
        )
        time.sleep(delay_seconds)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter.screenshot(str(output_path), scale=2)

    show_sections(
        plotter=plotter,
        runtime_objects=runtime_objects,
        show_reference=show_reference,
        interactive=True,
    )


if __name__ == "__main__":
    main()
