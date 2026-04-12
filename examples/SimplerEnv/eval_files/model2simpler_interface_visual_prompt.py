from collections import deque
from typing import Optional, Sequence, Dict, List
import os
import sys
import re
import logging
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from transforms3d.euler import euler2axangle
from pathlib import Path

# Add Robocasa visual_prompt_utility to path for SAM3 client imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "Robocasa_tabletop" / "visual_prompt_utility"))

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler
from starVLA.model.tools import read_mode_config

# SimplerEnv gripper threshold: unnormalized value > 0.5 => OPEN, <= 0.5 => CLOSE
GRIPPER_OPEN_THRESHOLD = 0.5


def parse_explicit_task(task: str) -> List[str]:
    """Parse task description into subtasks (e.g. 'pick X and place on Y' -> ['pick X', 'place on Y'])."""
    parts = re.split(r",\s*and\s*|,\s*|\s+and\s+", task.lower())
    return [p.strip() for p in parts if p.strip()]


class ModelClientVP:
    """
    Visual-prompting-aware ModelClient for SimplerEnv evaluation.

    Extends the baseline ModelClient with SAM3 segmentation overlay and VLM
    subtask detection, adapted for SimplerEnv's 7-DOF single-gripper actions.
    """

    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "widowx_bridge",
        horizon: int = 0,
        action_ensemble_horizon: Optional[int] = None,
        image_size: list[int] = [224, 224],
        action_scale: float = 1.0,
        cfg_scale: float = 1.5,
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        action_ensemble=True,
        adaptive_ensemble_alpha=0.1,
        host="0.0.0.0",
        port=10093,
        # SAM3 segmentation options
        use_sam3: bool = False,
        sam3_host: str = "127.0.0.1",
        sam3_port: int = 10094,
        sam3_overlay_alpha: float = 0.5,
        target_object_prompt_mode: str = "crosshair",
        target_location_prompt_mode: str = "box",
        sam3_threshold: float = 0.5,
        sam3_mask_threshold: float = 0.5,
        # VLM options
        use_vlm: bool = False,
        vlm_host: str = "127.0.0.1",
        vlm_port: int = 10095,
        vlm_query_on_gripper_change: bool = True,
        vlm_query_rate_limit_steps: int = 0,
        vlm_proceed_cooldown_steps: int = 0,
        # Overlay video saving
        save_overlay_video: bool = False,
        overlay_video_dir: str = "./overlay_videos",
        # VLM-based task decomposition (alternative to rule-based parse_explicit_task)
        vlm_decompose_subtasks: bool = False,
    ) -> None:
        self.client = WebsocketClientPolicy(host, port)

        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        if policy_setup == "widowx_bridge":
            unnorm_key = "oxe_bridge" if unnorm_key is None else unnorm_key
            if action_ensemble_horizon is None:
                action_ensemble_horizon = 7
            self.sticky_gripper_num_repeat = 1
        elif policy_setup == "google_robot":
            unnorm_key = "oxe_rt1" if unnorm_key is None else unnorm_key
            if action_ensemble_horizon is None:
                action_ensemble_horizon = 2
            self.sticky_gripper_num_repeat = 10
        else:
            raise NotImplementedError(f"Policy setup {policy_setup} not supported.")

        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.cfg_scale = cfg_scale
        self.image_size = image_size
        self.action_scale = action_scale
        self.horizon = horizon
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)

        # ===== SAM3 client =====
        self.use_sam3 = use_sam3
        self.sam3_client = None
        if use_sam3:
            from sam3_client import SAM3Client, extract_target_from_task
            self.sam3_client = SAM3Client(
                host=sam3_host,
                port=sam3_port,
                overlay_alpha=sam3_overlay_alpha,
                overlay_mode_open=target_object_prompt_mode,
                overlay_mode_closed=target_location_prompt_mode,
                threshold_open=sam3_threshold,
                mask_threshold_open=sam3_mask_threshold,
            )
            self._extract_target_from_task = extract_target_from_task
            print(f"*** SAM3 enabled: target_object={target_object_prompt_mode}, target_location={target_location_prompt_mode} ***")

        # ===== VLM client =====
        self.use_vlm = use_vlm
        self.vlm_query_on_gripper_change = vlm_query_on_gripper_change
        self.vlm_query_rate_limit_steps = vlm_query_rate_limit_steps
        self.vlm_proceed_cooldown_steps = vlm_proceed_cooldown_steps
        self.vlm_client = None
        self._vlm_queried = False
        self._cached_grasp_object = ""
        self._cached_place_target = None
        self._step_count = 0
        self._last_vlm_query_step = -1
        self._last_proceed_step: int = -999

        # Single gripper state tracking (SimplerEnv has only one gripper)
        self._recorded_gripper_state: str = "OPEN"
        self._last_action_gripper: str = "OPEN"
        self._is_first_frame = True
        self._reference_image: Optional[np.ndarray] = None

        # Subtask tracking
        self._subtasks: List[str] = []
        self._current_subtask_idx: int = 0
        self._current_subtask: str = ""

        self.vlm_decompose_subtasks = vlm_decompose_subtasks

        if use_vlm:
            from vlm_client_simpler import VLMClientSimpler
            self.vlm_client = VLMClientSimpler(host=vlm_host, port=vlm_port)
            mode_str = "gripper-change mode" if vlm_query_on_gripper_change else "query-once mode"
            rate_limit_str = f" (rate limit: {vlm_query_rate_limit_steps} steps)" if vlm_query_rate_limit_steps > 0 else " (no rate limit)"
            cooldown_str = f", proceed cooldown: {vlm_proceed_cooldown_steps} steps" if vlm_proceed_cooldown_steps > 0 else ""
            print(f"*** VLM enabled ({mode_str}{rate_limit_str}{cooldown_str}) ***")

        # Overlay video saving
        self.save_overlay_video = save_overlay_video
        self.overlay_video_dir = overlay_video_dir
        self._overlay_frames = []
        self._overlay_episode_count = 0
        if save_overlay_video:
            os.makedirs(overlay_video_dir, exist_ok=True)

        # Track gripper action history for visualization
        self.gripper_action_history = []

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)

    def _extract_gripper_from_action(self, raw_actions: np.ndarray) -> str:
        """
        Extract gripper state from predicted action (7-DOF SimplerEnv).
        Action layout: [x, y, z, roll, pitch, yaw, gripper].
        Gripper at index 6, value > 0.5 => OPEN, <= 0.5 => CLOSED.
        """
        gripper_value = raw_actions[0, 6]
        return "OPEN" if gripper_value > GRIPPER_OPEN_THRESHOLD else "CLOSED"

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.num_image_history = 0

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

        # Reset VLM state
        self._vlm_queried = False
        self._cached_grasp_object = ""
        self._cached_place_target = None
        if self.use_vlm and self.vlm_client is not None:
            self.vlm_client.reset()

        # Reset single gripper state tracking
        self._recorded_gripper_state = "OPEN"
        self._last_action_gripper = "OPEN"
        self._is_first_frame = True
        self._reference_image = None

        # Reset step tracking for rate limiting
        self._step_count = 0
        self._last_vlm_query_step = -1
        self._last_proceed_step = -999

        # Reset subtask tracking
        if self.vlm_decompose_subtasks and self.use_vlm and self.vlm_client is not None:
            vlm_subtasks = self.vlm_client.decompose_task_to_subtasks(task_description)
            if vlm_subtasks:
                self._subtasks = vlm_subtasks
                print(f"[SUBTASK] VLM decomposition: '{task_description}' -> {self._subtasks}")
            else:
                self._subtasks = parse_explicit_task(task_description)
                print(f"[SUBTASK] VLM decomposition failed, falling back to rule-based: '{task_description}' -> {self._subtasks}")
        else:
            self._subtasks = parse_explicit_task(task_description)

        if not self._subtasks:
            self._subtasks = [task_description]
        self._current_subtask_idx = 0
        self._current_subtask = self._subtasks[0] if self._subtasks else ""
        print(
            f"[SUBTASK] task_description='{task_description}' -> "
            f"subtasks={self._subtasks} (vlm_decompose={self.vlm_decompose_subtasks})"
        )

        # Save previous episode's overlay video and reset
        if self.save_overlay_video and len(self._overlay_frames) > 0:
            self._save_overlay_video_episode()
            self._overlay_episode_count += 1
        self._overlay_frames = []

        # Reset gripper action history
        self.gripper_action_history = []

    def step(
        self, image: np.ndarray, task_description: Optional[str] = None, *args, **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Input:
            image: np.ndarray of shape (H, W, 3), uint8
            task_description: Optional[str]
        Output:
            raw_action: dict with world_vector, rotation_delta, open_gripper
            action: dict with world_vector, rot_axangle, gripper, terminate_episode
        """
        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)

        assert image.dtype == np.uint8
        self._add_image_to_history(self._resize_image(image))

        # Increment step count for rate limiting
        self._step_count += 1

        # ===== Visual prompting: SAM3 overlay + VLM subtask detection =====
        if self.use_sam3 and self.sam3_client is not None:
            original_image = image.copy()

            current_gripper = self._last_action_gripper
            gripper_state_changed = (current_gripper != self._recorded_gripper_state)

            # Determine if we should query VLM
            should_query_vlm = False
            if self.use_vlm and self.vlm_client is not None:
                steps_since_last_query = self._step_count - self._last_vlm_query_step
                rate_limit_satisfied = (
                    self.vlm_query_rate_limit_steps == 0 or
                    steps_since_last_query >= self.vlm_query_rate_limit_steps
                )

                steps_since_proceed = self._step_count - self._last_proceed_step
                proceed_cooldown_satisfied = (
                    self.vlm_proceed_cooldown_steps == 0 or
                    steps_since_proceed >= self.vlm_proceed_cooldown_steps
                )

                if self.vlm_query_on_gripper_change:
                    if self._is_first_frame:
                        should_query_vlm = True
                    elif gripper_state_changed and rate_limit_satisfied and proceed_cooldown_satisfied:
                        should_query_vlm = True
                    elif gripper_state_changed and not proceed_cooldown_satisfied:
                        logging.debug(
                            f"[VLM Proceed Cooldown] Skipping query at step {self._step_count}: "
                            f"only {steps_since_proceed} steps since last proceed "
                            f"(cooldown: {self.vlm_proceed_cooldown_steps} steps)"
                        )
                    elif gripper_state_changed and not rate_limit_satisfied:
                        logging.debug(
                            f"[VLM Rate Limit] Skipping query at step {self._step_count}: "
                            f"only {steps_since_last_query} steps since last query "
                            f"(limit: {self.vlm_query_rate_limit_steps} steps)"
                        )
                else:
                    should_query_vlm = not self._vlm_queried

            if should_query_vlm:
                if self._is_first_frame:
                    query_reason = "first frame"
                else:
                    query_reason = f"gripper change (step {self._step_count})"
                logging.info(f"[VLM Query] Querying VLM at step {self._step_count}: {query_reason}")

                vlm_current_subtask = self._subtasks[self._current_subtask_idx] if self._subtasks else ""

                next_subtask = (
                    self._subtasks[self._current_subtask_idx + 1]
                    if self._current_subtask_idx + 1 < len(self._subtasks)
                    else None
                )

                vlm_result = self.vlm_client.detect_subtask(
                    reference_image=self._reference_image,
                    current_image=image,
                    task_description=self.task_description,
                    current_subtask=vlm_current_subtask,
                    next_subtask=next_subtask,
                    gripper_state=current_gripper,
                    gripper_state_changed=gripper_state_changed,
                    previous_gripper_state=self._recorded_gripper_state if gripper_state_changed else None,
                )

                self._current_subtask = vlm_current_subtask

                if vlm_result.get("target_object"):
                    self._cached_grasp_object = vlm_result["target_object"]
                if vlm_result.get("target_location"):
                    self._cached_place_target = vlm_result["target_location"]

                vlm_decision = vlm_result.get("decision", "continue")
                self._last_vlm_query_step = self._step_count

                if self._is_first_frame:
                    self._is_first_frame = False
                    self._reference_image = image.copy()
                    self._vlm_queried = True
                    if not self._cached_grasp_object:
                        self._cached_grasp_object = self._extract_target_from_task(self.task_description)
                elif gripper_state_changed:
                    if vlm_decision == "proceed" and self._current_subtask_idx + 1 < len(self._subtasks):
                        self._current_subtask_idx += 1
                        self._reference_image = image.copy()
                        self._last_proceed_step = self._step_count
                        logging.info(
                            f"[VLM] Proceeding to subtask {self._current_subtask_idx}: "
                            f"{self._subtasks[self._current_subtask_idx]} "
                            f"(language stays at '{self._current_subtask}' until next VLM query)"
                        )
                    # Update recorded gripper state
                    self._recorded_gripper_state = current_gripper

            elif not self.use_vlm and not self._vlm_queried and self.use_sam3:
                self._cached_grasp_object = self._extract_target_from_task(self.task_description)
                self._vlm_queried = True

            # Apply SAM3 overlay
            image = self.sam3_client.segment_and_overlay_both(
                image,
                grasp_object=self._cached_grasp_object,
                place_target=self._cached_place_target,
                top_score_only=True,
            )

            if self.save_overlay_video:
                self._overlay_frames.append(image.copy())

            image_list = [self._resize_image(original_image), self._resize_image(image)]
        else:
            image_list = [self._resize_image(image)]

        # ===== Build language instruction =====
        current_instruction = self.task_description

        current_instruction = (
            "You are given two images: the first is the original robot observation, "
            "and the second has visual prompts overlaid highlighting the target object "
            "and target location. " + current_instruction
        )

        # ===== Build VLA input and get action =====
        example = {
            "image": image_list,
            "lang": current_instruction,
        }

        vla_input = {
            "examples": [example],
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }

        response = self.client.predict_action(vla_input)

        # Unnormalize the action
        normalized_actions = response["data"]["normalized_actions"]  # B, chunk, D
        normalized_actions = normalized_actions[0]

        raw_actions = self.unnormalize_actions(
            normalized_actions=normalized_actions,
            action_norm_stats=self.action_norm_stats,
        )

        if self.action_ensemble:
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]

        raw_action = {
            "world_vector": np.array(raw_actions[0, :3]),
            "rotation_delta": np.array(raw_actions[0, 3:6]),
            "open_gripper": np.array(raw_actions[0, 6:7]),
        }

        # Update single gripper state from action for next step's VLM re-query check
        self._last_action_gripper = self._extract_gripper_from_action(raw_actions)

        # Track gripper value for visualization
        self.gripper_action_history.append(raw_actions[0, 6])

        # ===== Process raw_action for ManiSkill2 environment =====
        action = {}
        action["world_vector"] = raw_action["world_vector"] * self.action_scale
        action_rotation_delta = np.asarray(raw_action["rotation_delta"], dtype=np.float64)
        roll, pitch, yaw = action_rotation_delta
        axes, angles = euler2axangle(roll, pitch, yaw)
        action_rotation_axangle = axes * angles
        action["rot_axangle"] = action_rotation_axangle * self.action_scale

        if self.policy_setup == "google_robot":
            action["gripper"] = 0
            current_gripper_action = raw_action["open_gripper"]
            if self.previous_gripper_action is None:
                relative_gripper_action = np.array([0])
                self.previous_gripper_action = current_gripper_action
            else:
                relative_gripper_action = self.previous_gripper_action - current_gripper_action

            if np.abs(relative_gripper_action) > 0.5 and (not self.sticky_action_is_on):
                self.sticky_action_is_on = True
                self.sticky_gripper_action = relative_gripper_action
                self.previous_gripper_action = current_gripper_action

            if self.sticky_action_is_on:
                self.gripper_action_repeat += 1
                relative_gripper_action = self.sticky_gripper_action

            if self.gripper_action_repeat == self.sticky_gripper_num_repeat:
                self.sticky_action_is_on = False
                self.gripper_action_repeat = 0
                self.sticky_gripper_action = 0.0

            action["gripper"] = relative_gripper_action

        elif self.policy_setup == "widowx_bridge":
            action["gripper"] = 2.0 * (raw_action["open_gripper"] > 0.5) - 1.0

        action["terminate_episode"] = np.array([0.0])
        return raw_action, action

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        normalized_actions[:, 6] = np.where(normalized_actions[:, 6] < 0.5, 0, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)
        return norm_stats[unnorm_key]["action"]

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def finalize_overlay_video(self) -> None:
        """Call at end of evaluation to save last episode's overlay video."""
        if self.save_overlay_video and len(self._overlay_frames) > 0:
            self._save_overlay_video_episode()
            self._overlay_frames = []

    def _save_overlay_video_episode(self) -> None:
        if not self.save_overlay_video or len(self._overlay_frames) == 0:
            return

        import av

        video_path = os.path.join(
            self.overlay_video_dir,
            f"episode_{self._overlay_episode_count:03d}_overlay.mp4"
        )
        h, w = self._overlay_frames[0].shape[:2]
        fps = 10

        container = av.open(video_path, mode="w")
        stream = container.add_stream("h264", rate=fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.codec_context.options = {"crf": "18", "profile:v": "high"}

        for frame_data in self._overlay_frames:
            frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)

        for packet in stream.encode():
            container.mux(packet)

        container.close()
        print(f"[OVERLAY] Saved: {video_path} ({len(self._overlay_frames)} frames, {len(self._overlay_frames)/fps:.1f}s at {fps}fps)")

        gripper_plot_path = os.path.join(
            self.overlay_video_dir,
            f"episode_{self._overlay_episode_count:03d}_gripper_actions.png"
        )
        self._visualize_gripper_actions(gripper_plot_path)

    def _visualize_gripper_actions(self, save_path: str) -> None:
        if len(self.gripper_action_history) == 0:
            return

        data = np.array(self.gripper_action_history)
        timesteps = np.arange(len(data))

        plt.figure(figsize=(12, 4))
        plt.plot(timesteps, data, 'b-', linewidth=1.5, label='Gripper (dim 6)')
        plt.axhline(y=GRIPPER_OPEN_THRESHOLD, color='r', linestyle='--', alpha=0.5, label='Open/Close threshold')
        plt.xlabel('Timestep')
        plt.ylabel('Action Value')
        plt.title('Gripper Action (raw_actions[:, 6])')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
