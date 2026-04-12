# Extended argparse for SimplerEnv visual prompting evaluation.
# Adds SAM3, VLM, overlay video, and training-matching arguments
# on top of the base SimplerEnv argparse.

import argparse

import numpy as np
from sapien.core import Pose
from transforms3d.euler import euler2quat

from simpler_env.utils.io import DictAction


def parse_range_tuple(t):
    return np.linspace(t[0], t[1], int(t[2]))


def get_args():
    parser = argparse.ArgumentParser()

    # ===== Base SimplerEnv arguments (same as custom_argparse.py) =====
    parser.add_argument(
        "--policy-model", type=str, default="rt1",
        help="Policy model type; e.g., 'rt1', 'octo-base', 'octo-small'",
    )
    parser.add_argument(
        "--policy-setup", type=str, default="google_robot",
        help="Policy model setup; e.g., 'google_robot', 'widowx_bridge'",
    )
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--env-name", type=str, required=True)
    parser.add_argument(
        "--additional-env-save-tags", type=str, default=None,
        help="Additional tags to save the environment eval results",
    )
    parser.add_argument("--scene-name", type=str, default="google_pick_coke_can_1_v4")
    parser.add_argument("--enable-raytracing", action="store_true")
    parser.add_argument("--robot", type=str, default="google_robot_static")
    parser.add_argument(
        "--obs-camera-name", type=str, default=None,
        help="Obtain image observation from this camera for policy input. None = default",
    )
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--control-freq", type=int, default=3)
    parser.add_argument("--sim-freq", type=int, default=513)
    parser.add_argument("--max-episode-steps", type=int, default=80)
    parser.add_argument("--rgb-overlay-path", type=str, default=None)
    parser.add_argument(
        "--robot-init-x-range", type=float, nargs=3, default=[0.35, 0.35, 1],
        help="[xmin, xmax, num]",
    )
    parser.add_argument(
        "--robot-init-y-range", type=float, nargs=3, default=[0.20, 0.20, 1],
        help="[ymin, ymax, num]",
    )
    parser.add_argument(
        "--robot-init-rot-quat-center", type=float, nargs=4, default=[1, 0, 0, 0],
        help="[x, y, z, w]",
    )
    parser.add_argument(
        "--robot-init-rot-rpy-range", type=float, nargs=9,
        default=[0, 0, 1, 0, 0, 1, 0, 0, 1],
        help="[rmin, rmax, rnum, pmin, pmax, pnum, ymin, ymax, ynum]",
    )
    parser.add_argument(
        "--obj-variation-mode", type=str, default="xy", choices=["xy", "episode"],
        help="Whether to vary the xy position of a single object, or to vary predetermined episodes",
    )
    parser.add_argument("--obj-episode-range", type=int, nargs=2, default=[0, 60], help="[start, end]")
    parser.add_argument(
        "--obj-init-x-range", type=float, nargs=3, default=[-0.35, -0.12, 5],
        help="[xmin, xmax, num]",
    )
    parser.add_argument(
        "--obj-init-y-range", type=float, nargs=3, default=[-0.02, 0.42, 5],
        help="[ymin, ymax, num]",
    )
    parser.add_argument(
        "--additional-env-build-kwargs", nargs="+", action=DictAction,
        help="Additional env build kwargs in xxx=yyy format.",
    )
    parser.add_argument("--logging-dir", type=str, default="./results")
    parser.add_argument("--tf-memory-limit", type=int, default=3072, help="Tensorflow memory limit")
    parser.add_argument("--octo-init-rng", type=int, default=0, help="Octo init rng seed")
    parser.add_argument("--async-freq", type=int, default=1)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=10093)

    # ===== SAM3 segmentation arguments =====
    parser.add_argument("--use-sam3", action="store_true", help="Enable SAM3 visual prompt overlay")
    parser.add_argument("--sam3-host", type=str, default="127.0.0.1")
    parser.add_argument("--sam3-port", type=int, default=10094)
    parser.add_argument("--sam3-overlay-alpha", type=float, default=0.5)
    parser.add_argument(
        "--target-object-prompt-mode", type=str, default="crosshair",
        help="SAM3 overlay mode for target object (must match training: crosshair)",
    )
    parser.add_argument(
        "--target-location-prompt-mode", type=str, default="box",
        help="SAM3 overlay mode for target location (must match training: box)",
    )
    parser.add_argument("--sam3-threshold", type=float, default=0.5)
    parser.add_argument("--sam3-mask-threshold", type=float, default=0.5)

    # ===== VLM subtask detection arguments =====
    parser.add_argument("--use-vlm", action="store_true", help="Enable VLM subtask detection")
    parser.add_argument("--vlm-host", type=str, default="127.0.0.1")
    parser.add_argument("--vlm-port", type=int, default=10095)
    parser.add_argument(
        "--vlm-query-on-gripper-change", action="store_true", default=True,
        help="Query VLM when gripper state changes (default, matches training)",
    )
    parser.add_argument(
        "--no-vlm-query-on-gripper-change", dest="vlm_query_on_gripper_change", action="store_false",
        help="Query VLM only once on first frame",
    )
    parser.add_argument("--vlm-query-rate-limit-steps", type=int, default=0,
                        help="Minimum steps between VLM queries (0 = no limit)")
    parser.add_argument("--vlm-proceed-cooldown-steps", type=int, default=0,
                        help="Min steps after subtask transition before next VLM query")
    parser.add_argument("--vlm-decompose-subtasks", action="store_true", default=False,
                        help="Use VLM to decompose task into subtasks instead of rule-based regex")

    # ===== Overlay video (debugging) =====
    parser.add_argument("--save-overlay-video", action="store_true", default=False,
                        help="Save overlay visualization videos for debugging")
    parser.add_argument("--overlay-video-dir", type=str, default="./overlay_videos")

    args = parser.parse_args()

    # ===== Post-processing (same as base) =====
    args.robot_init_xs = parse_range_tuple(args.robot_init_x_range)
    args.robot_init_ys = parse_range_tuple(args.robot_init_y_range)
    args.robot_init_quats = []
    for r in parse_range_tuple(args.robot_init_rot_rpy_range[:3]):
        for p in parse_range_tuple(args.robot_init_rot_rpy_range[3:6]):
            for y in parse_range_tuple(args.robot_init_rot_rpy_range[6:]):
                args.robot_init_quats.append(
                    (Pose(q=euler2quat(r, p, y)) * Pose(q=args.robot_init_rot_quat_center)).q
                )
    if args.obj_variation_mode == "xy":
        args.obj_init_xs = parse_range_tuple(args.obj_init_x_range)
        args.obj_init_ys = parse_range_tuple(args.obj_init_y_range)
    if args.obs_camera_name is not None:
        if args.additional_env_save_tags is None:
            args.additional_env_save_tags = f"obs_camera_{args.obs_camera_name}"
        else:
            args.additional_env_save_tags = args.additional_env_save_tags + f"_obs_camera_{args.obs_camera_name}"

    return args
