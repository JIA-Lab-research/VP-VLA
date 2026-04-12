import os

from examples.SimplerEnv.eval_files.custom_argparse_visual_prompt import get_args
from examples.SimplerEnv.eval_files.model2simpler_interface_visual_prompt import ModelClientVP
from simpler_env.evaluation.maniskill2_evaluator import maniskill2_evaluator

import numpy as np


def start_debugpy_once():
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    args = get_args()

    os.environ["DISPLAY"] = ""
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

    if os.getenv("DEBUG", False):
        start_debugpy_once()

    model = ModelClientVP(
        policy_ckpt_path=args.ckpt_path,
        policy_setup=args.policy_setup,
        port=args.port,
        action_scale=args.action_scale,
        cfg_scale=1.5,
        # SAM3 options
        use_sam3=args.use_sam3,
        sam3_host=args.sam3_host,
        sam3_port=args.sam3_port,
        sam3_overlay_alpha=args.sam3_overlay_alpha,
        target_object_prompt_mode=args.target_object_prompt_mode,
        target_location_prompt_mode=args.target_location_prompt_mode,
        sam3_threshold=args.sam3_threshold,
        sam3_mask_threshold=args.sam3_mask_threshold,
        # VLM options
        use_vlm=args.use_vlm,
        vlm_host=args.vlm_host,
        vlm_port=args.vlm_port,
        vlm_query_on_gripper_change=args.vlm_query_on_gripper_change,
        vlm_query_rate_limit_steps=args.vlm_query_rate_limit_steps,
        vlm_proceed_cooldown_steps=args.vlm_proceed_cooldown_steps,
        vlm_decompose_subtasks=args.vlm_decompose_subtasks,
        # Overlay video
        save_overlay_video=args.save_overlay_video,
        overlay_video_dir=args.overlay_video_dir,
    )

    success_arr = maniskill2_evaluator(model, args)
    print(args)
    print(" " * 10, "Average success", np.mean(success_arr))
