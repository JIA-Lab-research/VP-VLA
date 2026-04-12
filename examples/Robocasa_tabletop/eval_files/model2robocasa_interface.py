from collections import deque
from typing import Optional, Sequence, Dict, List
import os
import sys
import re
import logging
import av
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add visual_prompt_utility to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "visual_prompt_utility"))

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy

from examples.Robocasa_tabletop.eval_files.adaptive_ensemble import AdaptiveEnsembler

from starVLA.model.framework.share_tools import read_mode_config

# ===============================
# GRIPPER CONFIGURATION (from process_dataset_parallel.py)
# ===============================

# Gripper state values for GR1 robot
# Actions output strictly 1.5 (CLOSED) or -1.5 (OPEN).
# Any other value is considered TRANSIT (gripper in motion) and is ignored
# for VLM query triggering.
GRIPPER_VALUE_CLOSED = 1.5
GRIPPER_VALUE_OPEN = -1.5


def parse_explicit_task(task: str) -> List[str]:
    """Parse task description into subtasks."""
    parts = re.split(r",\s*and\s*|,\s*|\s+and\s+", task.lower())
    return [p.strip() for p in parts if p.strip()]


def check_gripper_state_changed(
    recorded_gripper: Dict[str, str],
    current_gripper: Dict[str, str],
) -> bool:
    """Check if gripper state has changed (TRANSIT states are ignored)."""
    for side in ("left", "right"):
        current = current_gripper[side]
        if current == "TRANSIT":
            continue  # Gripper in motion, not a settled state
        if current != recorded_gripper[side]:
            return True
    return False



class PolicyWarper:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "franka",
        horizon: int = 0,
        action_ensemble = False, # @Jinhui
        action_ensemble_horizon: Optional[int] = 3, # different cross sim
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha = 0.1,
        host="0.0.0.0",
        port=10095,
        n_action_steps=2,
        # SAM3 segmentation options
        use_sam3: bool = False,
        sam3_host: str = "127.0.0.1",
        sam3_port: int = 10094,
        sam3_overlay_alpha: float = 0.5,
        target_object_prompt_mode: str = "crosshair",
        target_location_prompt_mode: str = "crosshair",
        sam3_threshold: float = 0.5,
        sam3_mask_threshold: float = 0.5,
        # VLM options
        use_vlm: bool = False,
        vlm_host: str = "127.0.0.1",
        vlm_port: int = 10095,
        vlm_query_on_gripper_change: bool = True,  # Query VLM on gripper state change (like training)
        # Overlay video saving
        save_overlay_video: bool = False,
        overlay_video_dir: str = "./overlay_videos",
        # VLM-based task decomposition (alternative to rule-based parse_explicit_task)
        vlm_decompose_subtasks: bool = False,  # If True, use VLM to decompose task into subtasks instead of regex
    ) -> None:
        
        # build client to connect server policy
        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key

        print(f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key} ***")
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon #0
        self.action_ensemble = action_ensemble
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None
        self.n_action_steps = n_action_steps

        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        if self.action_ensemble:
            self.action_ensembler = AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
        else:
            self.action_ensembler = None
        self.num_image_history = 0

        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path=policy_ckpt_path)
        
        # SAM3 client
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
        
        # VLM client
        self.use_vlm = use_vlm
        self.vlm_query_on_gripper_change = vlm_query_on_gripper_change
        self.vlm_client = None
        self._vlm_queried = False  # Track if VLM has been queried at least once
        self._cached_grasp_object = ""
        self._cached_place_target = None
        self._step_count = 0  # Track steps within an episode
        
        # Gripper state tracking (for VLM re-query on gripper change)
        self._recorded_gripper_state: Dict[str, str] = {"left": "OPEN", "right": "OPEN"}
        self._last_action_gripper: Dict[str, str] = {"left": "OPEN", "right": "OPEN"}  # Gripper from previous action
        self._is_first_frame = True
        self._reference_image: Optional[np.ndarray] = None
        
        # Subtask tracking
        self._subtasks: List[str] = []
        self._current_subtask_idx: int = 0
        self._current_subtask: str = ""
        
        self.vlm_decompose_subtasks = vlm_decompose_subtasks
        
        if use_vlm:
            from vlm_client import VLMClient
            self.vlm_client = VLMClient(host=vlm_host, port=vlm_port)
            mode_str = "gripper-change mode" if vlm_query_on_gripper_change else "query-once mode"
            print(f"*** VLM enabled ({mode_str}) ***")
        
        # Overlay video saving
        self.save_overlay_video = save_overlay_video
        self.overlay_video_dir = overlay_video_dir
        self._overlay_frames = []
        self._overlay_episode_count = 0
        if save_overlay_video:
            os.makedirs(overlay_video_dir, exist_ok=True)
        
        # Track hand action values for visualization
        self.left_hand_first_dim_history = []
        self.right_hand_first_dim_history = []

    def _add_image_to_history(self, image: np.ndarray) -> None:
        self.image_history.append(image)
        self.num_image_history = min(self.num_image_history + 1, self.horizon)
    
    def _extract_gripper_state(self, observations: Dict, batch_idx: int = 0) -> Dict[str, str]:
        """
        Extract gripper state from the PREVIOUS action prediction.
        
        This returns the gripper state from the last action, which is used to detect
        gripper state changes for VLM re-querying (matching training data processing).
        
        Args:
            observations: Dict containing state observations (not used, kept for API compatibility)
            batch_idx: Index in the batch (not used, kept for API compatibility)
            
        Returns:
            Dict with 'left' and 'right' states ('OPEN' or 'CLOSED')
        """
        # Return the gripper state from the previous action prediction
        # This is initialized to OPEN and updated after each action prediction
        return self._last_action_gripper
    
    def _extract_gripper_from_action(self, raw_actions: np.ndarray, batch_idx: int = 0) -> Dict[str, str]:
        """
        Extract gripper state from predicted action.
        
        Uses the first dimension of left_hand and right_hand actions.
        Actions output strictly 1.5 (CLOSED) or -1.5 (OPEN).
        Any other value is TRANSIT (gripper in motion) and won't trigger VLM queries.
        
        Args:
            raw_actions: Raw action array of shape (B, chunk, D) where D=29
                        D=14: left_hand starts, D=20: right_hand starts
            batch_idx: Index in the batch
            
        Returns:
            Dict with 'left' and 'right' states ('OPEN', 'CLOSED', or 'TRANSIT')
        """
        # Extract first dimension of hand actions (gripper indicator)
        # raw_actions layout: [left_arm(7), right_arm(7), left_hand(6), right_hand(6), waist(3)]
        left_hand_gripper = raw_actions[batch_idx, 0, 14]  # First dim of left_hand
        right_hand_gripper = raw_actions[batch_idx, 0, 20]  # First dim of right_hand
        
        # Determine left gripper state (strict value matching)
        if left_hand_gripper == GRIPPER_VALUE_CLOSED:
            left_state = "CLOSED"
        elif left_hand_gripper == GRIPPER_VALUE_OPEN:
            left_state = "OPEN"
        else:
            left_state = "TRANSIT"
        
        # Determine right gripper state (strict value matching)
        if right_hand_gripper == GRIPPER_VALUE_CLOSED:
            right_state = "CLOSED"
        elif right_hand_gripper == GRIPPER_VALUE_OPEN:
            right_state = "OPEN"
        else:
            right_state = "TRANSIT"
        
        return {
            "left": left_state,
            "right": right_state,
        }
    

    def reset(self, task_description: str or tuple) -> None:
       
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
        
        # Reset gripper state tracking
        self._recorded_gripper_state = {"left": "OPEN", "right": "OPEN"}
        self._last_action_gripper = {"left": "OPEN", "right": "OPEN"}
        self._is_first_frame = True
        self._reference_image = None
        
        self._step_count = 0
        
        # Reset subtask tracking
        task_str = task_description if isinstance(task_description, str) else (
            task_description[0] if task_description else ""
        )
        
        if self.vlm_decompose_subtasks and self.use_vlm and self.vlm_client is not None:
            vlm_subtasks = self.vlm_client.decompose_task_to_subtasks(task_str)
            if vlm_subtasks:
                self._subtasks = vlm_subtasks
                print(f"[SUBTASK] VLM decomposition: '{task_str}' -> {self._subtasks}")
            else:
                self._subtasks = parse_explicit_task(task_str)
                print(f"[SUBTASK] VLM decomposition failed, falling back to rule-based: '{task_str}' -> {self._subtasks}")
        else:
            self._subtasks = parse_explicit_task(task_str)
        
        if not self._subtasks:
            self._subtasks = [task_str]
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

        # Reset hand action history
        self.left_hand_first_dim_history = []
        self.right_hand_first_dim_history = []

    def step(
        self, 
        observations,
        **kwargs
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        执行一步推理
        :param image: 输入图像 (H, W, 3) uint8格式
        :param task_description: 任务描述文本
        :return: (原始动作, 处理后的动作)
        """

        task_description = observations['annotation.human.coarse_action'][0] # tuple       
        ego_view = observations['video.ego_view']  # (N, 1, H, W, 3)
        state = {}
        state['left_arm'] = observations['state.left_arm']     
        state['right_arm'] = observations['state.right_arm']             # (N, 1, 7)
        state['left_hand'] = observations['state.left_hand']              # (N, 1, 6)
        state['right_hand'] = observations['state.right_hand']            # (N, 1, 6)
        state['waist'] = observations['state.waist']                      # (N, 1, 3)
        

        state = self.normalize_state(state)
        input_state = []
        for key in state.keys():
            input_state.append(state[key])
        input_state = np.concatenate(input_state, axis=-1)

        if task_description is not None:
            if task_description != self.task_description:
                self.reset(task_description)
        
        # Increment step count for rate limiting
        self._step_count += 1

        # Apply visual prompting to each image in batch
        if self.use_sam3 and self.sam3_client is not None:
            processed_images = []
            original_images = []
            for b in range(len(ego_view)):
                image = ego_view[b][0]  # Get first view, shape (H, W, 3)
                original_images.append(image.copy())
                
                # Get current gripper state from observations
                current_gripper = self._extract_gripper_state(observations, b)
                gripper_state_changed = check_gripper_state_changed(
                    self._recorded_gripper_state, current_gripper
                )
                
                # Determine if we should query VLM
                should_query_vlm = False
                if self.use_vlm and self.vlm_client is not None:
                    if self.vlm_query_on_gripper_change:
                        if self._is_first_frame:
                            should_query_vlm = True
                        elif gripper_state_changed:
                            should_query_vlm = True
                    else:
                        should_query_vlm = not self._vlm_queried
                
                if should_query_vlm:
                    if self._is_first_frame:
                        query_reason = "first frame"
                    else:
                        query_reason = f"gripper change (step {self._step_count})"
                    logging.info(f"[VLM Query] Querying VLM at step {self._step_count}: {query_reason}")
                    
                    # Use subtask index to determine what VLM evaluates
                    # (index may have advanced from a previous "proceed", but _current_subtask
                    #  for language prompt stays at the old subtask until this VLM query)
                    vlm_current_subtask = self._subtasks[self._current_subtask_idx] if self._subtasks else ""
                    
                    # Get next subtask
                    next_subtask = (
                        self._subtasks[self._current_subtask_idx + 1]
                        if self._current_subtask_idx + 1 < len(self._subtasks)
                        else None
                    )
                    
                    # Query VLM for subtask detection
                    task_desc_str = self.task_description if isinstance(self.task_description, str) else str(self.task_description)
                    print(
                        f"[VLM INPUT] step={self._step_count}, "
                        f"task_description='{task_desc_str}', "
                        f"current_subtask='{vlm_current_subtask}' (idx={self._current_subtask_idx}), "
                        f"next_subtask='{next_subtask}', "
                        f"gripper=L:{current_gripper['left']}/R:{current_gripper['right']}, "
                        f"changed={gripper_state_changed}"
                    )

                    vlm_result = self.vlm_client.detect_subtask(
                        reference_image=self._reference_image,
                        current_image=image,
                        task_description=task_desc_str,
                        current_subtask=vlm_current_subtask,
                        next_subtask=next_subtask,
                        gripper_state={"left": current_gripper["left"], "right": current_gripper["right"]},
                        gripper_state_changed=gripper_state_changed,
                        previous_gripper_state=self._recorded_gripper_state if gripper_state_changed else None,
                    )

                    print(
                        f"[VLM OUTPUT] decision='{vlm_result.get('decision')}', "
                        f"target_object='{vlm_result.get('target_object')}', "
                        f"target_location='{vlm_result.get('target_location')}', "
                        f"reasoning='{vlm_result.get('reasoning', '')[:100]}'"
                    )
                    
                    # Update language prompt to match what VLM just evaluated.
                    # CRITICAL: This must happen BEFORE the subtask advance below, so
                    # that the language prompt matches the VLM entry -- exactly as in
                    # training data where vlm_results are saved BEFORE advancing the
                    # subtask index (process_dataset_parallel.py lines 684-702).
                    self._current_subtask = vlm_current_subtask
                    
                    # Update targets from VLM
                    if vlm_result.get("target_object"):
                        self._cached_grasp_object = vlm_result["target_object"]
                    if vlm_result.get("target_location"):
                        self._cached_place_target = vlm_result["target_location"]
                    
                    vlm_decision = vlm_result.get("decision", "continue")
                    
                    # Handle subtask transitions
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
                            logging.info(f"[VLM] Proceeding to subtask {self._current_subtask_idx}: {self._subtasks[self._current_subtask_idx]} (language stays at '{self._current_subtask}' until next VLM query)")
                        # Update recorded gripper state (only OPEN/CLOSED, never TRANSIT)
                        for side in ("left", "right"):
                            if current_gripper[side] != "TRANSIT":
                                self._recorded_gripper_state[side] = current_gripper[side]
                
                elif not self.use_vlm and not self._vlm_queried:
                    # Static extraction without VLM (first time only)
                    self._cached_grasp_object = self._extract_target_from_task(self.task_description)
                    self._vlm_queried = True
                
                # Apply SAM3 overlay
                image = self.sam3_client.segment_and_overlay_both(
                    image,
                    grasp_object=self._cached_grasp_object,
                    place_target=self._cached_place_target,
                    top_score_only=True,
                )
                
                # Collect overlay frame for video
                if self.save_overlay_video:
                    self._overlay_frames.append(image.copy())
                
                processed_images.append([original_images[b], image])
            
            ego_view = np.array(processed_images)

        images = [[self._resize_image(img) for img in sample] for sample in ego_view] # (B, N_view, H, W, 3)
        input_state = [input_s for input_s in input_state] # B, state_dim*(sin, cos)

        examples = []
        batch_size = len(images)
        current_instruction = self.task_description if isinstance(self.task_description, str) else str(self.task_description)
        
        current_instruction = (
            "You are given two images: the first is the original robot observation, "
            "and the second has visual prompts overlaid highlighting the target object "
            "and target location. " + current_instruction
        )
        for b in range(batch_size):
            example = {
                "image": images[b],  # A list of multi-view images for a single sample
                "lang": current_instruction,
                "state": input_state[b],  # N_history, 58 #Hack BUG
            }
            examples.append(example)
        
        vla_input = {
            "examples": examples,
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }
        
        response = self.client.predict_action(vla_input)
        
        
        # unnormalize the action
        normalized_actions = response["data"]["normalized_actions"] # B, chunk, D        
        
        # unnormalize actions in batch form
        raw_actions = self.unnormalize_actions(normalized_actions=normalized_actions, action_norm_stats=self.action_norm_stats)
        
        # raw_actions shape: (B, chunk, D)
        if self.action_ensemble:
            # 对batch中的每个样本进行ensemble
            batch_size = raw_actions.shape[0]
            ensembled_actions = []
            for b in range(batch_size):
                ensembled = self.action_ensembler.ensemble_action(raw_actions[b])[None]  # (1, D)
                ensembled_actions.append(ensembled)
            raw_actions = np.stack(ensembled_actions, axis=0)  # (B, 1, D)

        raw_action = {
            "action.left_arm": raw_actions[:, :self.n_action_steps, :7],      # (B, n_action_steps, 7)
            "action.right_arm": raw_actions[:, :self.n_action_steps, 7:14],   # (B, n_action_steps, 7)
            "action.left_hand": raw_actions[:, :self.n_action_steps, 14:20],  # (B, n_action_steps, 6)
            "action.right_hand": raw_actions[:, :self.n_action_steps, 20:26], # (B, n_action_steps, 6)
            "action.waist": raw_actions[:, :self.n_action_steps, 26:29],      # (B, n_action_steps, 3)
        }

        # Track the first dimension of left_hand and right_hand actions for visualization
        # raw_actions[:, :self.n_action_steps, 14] is the first dim of left_hand
        # raw_actions[:, :self.n_action_steps, 20] is the first dim of right_hand
        # We take the first batch and first action step for simplicity
        left_hand_first_dim = raw_actions[0, 0, 14]  # scalar
        right_hand_first_dim = raw_actions[0, 0, 20]  # scalar
        self.left_hand_first_dim_history.append(left_hand_first_dim)
        self.right_hand_first_dim_history.append(right_hand_first_dim)

        # Update gripper state from action for next step's VLM re-query check
        self._last_action_gripper = self._extract_gripper_from_action(raw_actions, batch_idx=0)

        return {"actions": raw_action}

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        """
        Args:
            normalized_actions: shape (B, chunk, D) (chunk, D)
            action_norm_stats:
        Returns:
            actions
        """
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        
        normalized_actions = np.clip(normalized_actions, -1, 1)
        
        actions = np.where(
            mask,
            (normalized_actions + 1) / 2 * (action_high - action_low) + action_low,
            normalized_actions,
        )
        
        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        """
        Duplicate stats accessor (retained for backward compatibility).
        """
        policy_ckpt_path = Path(policy_ckpt_path)
        model_config, norm_stats = read_mode_config(policy_ckpt_path)  # read config and norm_stats

        unnorm_key = PolicyWarper._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]



    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def visualize_epoch(
        self, predicted_raw_actions: Sequence[np.ndarray], images: Sequence[np.ndarray], save_path: str
    ) -> None:
        images = [self._resize_image(image) for image in images]
        ACTION_DIM_LABELS = ["x", "y", "z", "roll", "pitch", "yaw", "grasp"]

        img_strip = np.concatenate(np.array(images[::3]), axis=1)

        # set up plt figure
        figure_layout = [["image"] * len(ACTION_DIM_LABELS), ACTION_DIM_LABELS]
        plt.rcParams.update({"font.size": 12})
        fig, axs = plt.subplot_mosaic(figure_layout)
        fig.set_size_inches([45, 10])

        # plot actions
        pred_actions = np.array(
            [
                np.concatenate([a["world_vector"], a["rotation_delta"], a["open_gripper"]], axis=-1)
                for a in predicted_raw_actions
            ]
        )
        for action_dim, action_label in enumerate(ACTION_DIM_LABELS):
            # actions have batch, horizon, dim, in this example we just take the first action for simplicity
            axs[action_label].plot(pred_actions[:, action_dim], label="predicted action")
            axs[action_label].set_title(action_label)
            axs[action_label].set_xlabel("Time in one episode")

        axs["image"].imshow(img_strip)
        axs["image"].set_xlabel("Time in one episode (subsampled)")
        plt.legend()
        plt.savefig(save_path)
    
    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        """
        Duplicate helper (retained for backward compatibility).
        See primary _check_unnorm_key above.
        """
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key
    
    def normalize_state(self, state: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Normalize the state
        """
        for key in state.keys():
            sin_state = np.sin(state[key])
            cos_state = np.cos(state[key])
            state[key] = np.concatenate([sin_state, cos_state], axis=-1)
        return state
    
    def _save_overlay_video_episode(self) -> None:
        """Save the overlay video for the current episode.
        
        Saves every overlaid image that was passed to the model (one per policy step).
        """
        if not self.save_overlay_video or len(self._overlay_frames) == 0:
            return
        
        video_path = os.path.join(
            self.overlay_video_dir,
            f"episode_{self._overlay_episode_count:03d}_overlay.mp4"
        )
        h, w = self._overlay_frames[0].shape[:2]
        # Use fps=10 to match the main video's VideoConfig.fps
        # The overlay video will be shorter since frames are collected per policy step, not per env step
        fps = 10
        
        # Use PyAV with h264 codec (same as VideoRecordingWrapper)
        container = av.open(video_path, mode="w")
        stream = container.add_stream("h264", rate=fps)
        stream.width = w
        stream.height = h
        stream.pix_fmt = "yuv420p"
        stream.codec_context.options = {"crf": "18", "profile:v": "high"}
        
        for frame_data in self._overlay_frames:
            # PyAV expects RGB format (frames are already in RGB)
            frame = av.VideoFrame.from_ndarray(frame_data, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        
        # Flush stream
        for packet in stream.encode():
            container.mux(packet)
        
        container.close()
        print(f"[OVERLAY] Saved: {video_path} ({len(self._overlay_frames)} frames, {len(self._overlay_frames)/fps:.1f}s at {fps}fps)")
        
        # Also save hand action visualization for this episode
        hand_action_plot_path = os.path.join(
            self.overlay_video_dir,
            f"episode_{self._overlay_episode_count:03d}_hand_actions.png"
        )
        self.visualize_hand_actions(hand_action_plot_path)

    def finalize_overlay_video(self) -> None:
        """Call at end of evaluation to save last episode's overlay video."""
        if self.save_overlay_video and len(self._overlay_frames) > 0:
            self._save_overlay_video_episode()
            self._overlay_frames = []

    def visualize_hand_actions(self, save_path: str) -> None:
        """
        Visualize the first dimension of left_hand and right_hand actions over the episode.
        
        Args:
            save_path: Path to save the figure (e.g., '/path/to/video_hand_actions.png')
        """
        if len(self.left_hand_first_dim_history) == 0:
            print("Warning: No hand action data to visualize.")
            return
        
        left_hand_data = np.array(self.left_hand_first_dim_history)
        right_hand_data = np.array(self.right_hand_first_dim_history)
        timesteps = np.arange(len(left_hand_data))
        
        plt.figure(figsize=(12, 6))
        
        # Plot left hand first dimension
        plt.subplot(2, 1, 1)
        plt.plot(timesteps, left_hand_data, 'b-', linewidth=1.5, label='Left Hand (dim 0)')
        plt.xlabel('Timestep')
        plt.ylabel('Action Value')
        plt.title('Left Hand Action - First Dimension (raw_actions[:, :, 14])')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot right hand first dimension
        plt.subplot(2, 1, 2)
        plt.plot(timesteps, right_hand_data, 'r-', linewidth=1.5, label='Right Hand (dim 0)')
        plt.xlabel('Timestep')
        plt.ylabel('Action Value')
        plt.title('Right Hand Action - First Dimension (raw_actions[:, :, 20])')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Hand action visualization saved to: {save_path}")
