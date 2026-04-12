#!/usr/bin/env python3
"""
Parallel Dataset Processing Script for Visual Prompt Generation

This script processes episodes in a specified range, using VLM+SAM servers
to generate segmentation masks and save them as .npz files.

Uses the golden logic from combined_vlm_sam.py:
- VLM is queried on first frame and when gripper state changes
- SAM segments target_object and target_location on every frame
- Outputs .npz files with masks, boxes, and scores

Example usage:
    # Process episodes 0-249 of a task
    python process_dataset_parallel.py \
        --dataset-root "/path/to/dataset" \
        --task "gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000" \
        --episode-start 0 \
        --episode-end 249 \
        --sam-port 10094 \
        --vlm-port 10102 \
        --output-dir "/path/to/output"

Output structure:
    output_dir/
        task_folder_name/
            episode_000000.npz
            episode_000001.npz
            ...
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import websockets.sync.client

# Add the utility directory to path for msgpack_utils
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(os.path.dirname(SCRIPT_DIR), "..")
UTILITY_DIR = os.path.join(PROJECT_ROOT, "examples", "Robocasa_tabletop", "visual_prompt_utility")
sys.path.insert(0, UTILITY_DIR)

from msgpack_utils import packb, unpackb

# Import decord for fast video loading
try:
    import decord
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False

# Import PIL for loading pre-extracted frames
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ===============================
# GRIPPER CONFIGURATION (from combined_vlm_sam.py)
# ===============================

GRIPPER_LEFT_ACTION_INDEX = 7
GRIPPER_RIGHT_ACTION_INDEX = 29
GRIPPER_THRESHOLD = 0.0


# ===============================
# GRIPPER STATE FUNCTIONS (from combined_vlm_sam.py)
# ===============================

def check_gripper_state_changed(
    recorded_gripper: Dict[str, str],
    current_gripper: Dict[str, str],
) -> bool:
    """Check if gripper state has changed."""
    left_changed = recorded_gripper["left"] != current_gripper["left"]
    right_changed = recorded_gripper["right"] != current_gripper["right"]
    return left_changed or right_changed


def load_gripper_states_from_parquet(parquet_path: str, num_frames: int) -> Dict[int, Dict[str, str]]:
    """Load gripper states from parquet file.
    
    Args:
        parquet_path: Path to the parquet file containing action data.
        num_frames: Number of frames in the episode.
    
    Returns:
        Dictionary mapping frame index to gripper state.
    
    Raises:
        ValueError: If the number of frames doesn't match the action data length.
    """
    try:
        if not os.path.exists(parquet_path):
            logging.warning(f"Parquet file not found: {parquet_path}")
            return _get_default_gripper_states(num_frames)
        
        df = pd.read_parquet(parquet_path)
        
        # Validate alignment: frames and action data must be strictly aligned
        if num_frames != len(df):
            raise ValueError(
                f"Frame count mismatch: {num_frames} frames but {len(df)} action entries in parquet. "
                f"Frames and action data must be strictly aligned. Parquet: {parquet_path}"
            )
        
        gripper_states = {}
        
        for frame_idx in range(num_frames):
            action = df.iloc[frame_idx]['action']
            left_value = float(action[GRIPPER_LEFT_ACTION_INDEX])
            right_value = float(action[GRIPPER_RIGHT_ACTION_INDEX])
            
            left_state = "OPEN" if left_value < GRIPPER_THRESHOLD else "CLOSED"
            right_state = "OPEN" if right_value < GRIPPER_THRESHOLD else "CLOSED"
            
            gripper_states[frame_idx] = {
                "left": left_state,
                "right": right_state,
                "left_raw": left_value,
                "right_raw": right_value,
            }
        
        return gripper_states
        
    except Exception as e:
        logging.warning(f"Failed to load gripper states: {e}")
        return _get_default_gripper_states(num_frames)


def _get_default_gripper_states(num_frames: int) -> Dict[int, Dict[str, str]]:
    """Return default gripper states (all OPEN)."""
    return {
        i: {"left": "OPEN", "right": "OPEN", "left_raw": -1.5, "right_raw": -1.5}
        for i in range(num_frames)
    }


def parse_explicit_task(task: str) -> List[str]:
    """Parse task description into subtasks."""
    parts = re.split(r",\s*and\s*|,\s*|\s+and\s+", task.lower())
    return [p.strip() for p in parts if p.strip()]


# ===============================
# DATA LOADING FUNCTIONS (from combined_vlm_sam.py)
# ===============================

def load_episode_metadata(task_folder_path: str, episode_index: int = 0) -> Dict:
    """Load episode metadata from meta/episodes.jsonl."""
    episodes_path = os.path.join(task_folder_path, "meta", "episodes.jsonl")
    
    if not os.path.exists(episodes_path):
        raise FileNotFoundError(f"Episodes file not found: {episodes_path}")
    
    with open(episodes_path, 'r') as f:
        for i, line in enumerate(f):
            if i == episode_index:
                return json.loads(line.strip())
    
    raise ValueError(f"Episode {episode_index} not found in {episodes_path}")


def construct_video_path(task_folder_path: str, episode_index: int = 0) -> str:
    """Construct video path for an episode."""
    chunk_num = episode_index // 1000
    video_path = os.path.join(
        task_folder_path,
        "videos",
        f"chunk-{chunk_num:03d}",
        "observation.images.ego_view",
        f"episode_{episode_index:06d}.mp4"
    )
    return video_path


def construct_parquet_path(task_folder_path: str, episode_index: int = 0) -> str:
    """Construct parquet path for an episode."""
    chunk_num = episode_index // 1000
    parquet_path = os.path.join(
        task_folder_path,
        "data",
        f"chunk-{chunk_num:03d}",
        f"episode_{episode_index:06d}.parquet"
    )
    return parquet_path


def extract_all_frames_numpy(video_path: str, video_backend: str = "decord") -> Tuple[List[np.ndarray], float]:
    """Extract all frames from video as numpy arrays using decord.
    
    Args:
        video_path: Path to the video file.
        video_backend: Backend for video reading. Currently only "decord" is supported.
                       Decord is faster than OpenCV and returns RGB directly.
    
    Returns:
        Tuple of (list of frames as numpy arrays in RGB format, fps)
    """
    if not DECORD_AVAILABLE:
        raise ImportError(
            "decord is not available. Please install it with: pip install decord"
        )
    
    # Use decord VideoReader - it's faster than OpenCV and returns RGB directly
    vr = decord.VideoReader(video_path)
    fps = vr.get_avg_fps()
    num_frames = len(vr)
    
    # Get all frames in a batch - more efficient than frame-by-frame
    # decord returns frames in RGB format, no need for color conversion
    frames_array = vr.get_batch(range(num_frames)).asnumpy()
    
    # Convert to list of frames for compatibility with existing code
    frames = [frames_array[i] for i in range(num_frames)]
    
    return frames, fps


def get_fps_from_metadata(task_folder_path: str) -> float:
    """Read fps from task's meta/info.json.
    
    Args:
        task_folder_path: Path to the task folder containing meta/info.json
    
    Returns:
        fps value from metadata, or 20.0 as default for robocasa dataset
    """
    info_path = os.path.join(task_folder_path, "meta", "info.json")
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
            return info.get("fps", 20.0)
    return 20.0  # Default for robocasa dataset


def construct_frames_dir(frames_root: str, task: str, episode_index: int) -> str:
    """Construct the directory path for pre-extracted frames.
    
    Args:
        frames_root: Root directory containing pre-extracted frames
        task: Task folder name
        episode_index: Index of the episode
    
    Returns:
        Path to the directory containing frame_XXXXXX.jpg files
    """
    chunk_num = episode_index // 1000
    frames_dir = os.path.join(
        frames_root,
        task,
        f"chunk-{chunk_num:03d}",
        "observation.images.ego_view",
        f"episode_{episode_index:06d}"
    )
    return frames_dir


def load_frames_from_images(frames_dir: str, fps: float = 20.0) -> Tuple[List[np.ndarray], float]:
    """Load frames from pre-extracted JPEG images.
    
    This is much faster than decoding video on the fly, significantly reducing CPU usage.
    
    Args:
        frames_dir: Directory containing frame_XXXXXX.jpg files
        fps: Frames per second (read from meta/info.json)
    
    Returns:
        Tuple of (list of frames as numpy arrays in RGB format, fps)
    """
    if not PIL_AVAILABLE:
        raise ImportError(
            "PIL is not available. Please install it with: pip install Pillow"
        )
    
    if not os.path.exists(frames_dir):
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
    
    # Find all frame files and sort them
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.jpg')])
    
    if not frame_files:
        raise FileNotFoundError(f"No frame files found in {frames_dir}")
    
    frames = []
    for frame_file in frame_files:
        img_path = os.path.join(frames_dir, frame_file)
        img = Image.open(img_path)
        frames.append(np.array(img))  # PIL loads as RGB by default
    
    return frames, fps


# ===============================
# SERVER CLIENTS
# ===============================

class SAM3Client:
    """Client for SAM3 segmentation server."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 10094, max_retries: int = 3):
        self._uri = f"ws://{host}:{port}"
        self._max_retries = max_retries
        self._ws = None
        self._server_metadata = None
    
    def connect(self, timeout: float = 300) -> None:
        """Connect to SAM3 server."""
        start_time = time.time()
        
        # Clear proxy settings
        for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(k, None)
        
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Failed to connect to SAM3 server within {timeout} seconds")
            
            try:
                self._ws = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    open_timeout=150,
                    ping_interval=None,
                    ping_timeout=None,
                )
                self._server_metadata = unpackb(self._ws.recv())
                logging.info(f"Connected to SAM3 server: {self._server_metadata}")
                return
            except ConnectionRefusedError:
                logging.info(f"Waiting for SAM3 server {self._uri} ...")
                time.sleep(2)
    
    def _reconnect(self) -> None:
        """Reconnect to server."""
        for attempt in range(self._max_retries):
            try:
                if self._ws:
                    self._ws.close()
            except Exception:
                pass
            self._ws = None
            try:
                self.connect(timeout=60)
                return
            except Exception as e:
                logging.warning(f"Reconnect attempt {attempt + 1} failed: {e}")
                time.sleep(2)
        logging.error("Exceeded maximum SAM3 reconnect attempts")
    
    def close(self) -> None:
        """Close connection."""
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
    
    def segment(
        self,
        image: np.ndarray,
        text_prompt: str,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> Dict:
        """Request segmentation from server."""
        if self._ws is None:
            self.connect()
        
        request = {
            "type": "segment",
            "request_id": f"seg_{time.time()}",
            "image": image,
            "text_prompt": text_prompt,
            "threshold": threshold,
            "mask_threshold": mask_threshold,
        }
        
        try:
            self._ws.send(packb(request))
            response = unpackb(self._ws.recv())
        except Exception as e:
            logging.warning(f"SAM3 socket error: {e}. Reconnecting...")
            self._reconnect()
            self._ws.send(packb(request))
            response = unpackb(self._ws.recv())
        
        if not response.get("ok", False):
            error_msg = response.get("error", {}).get("message", "Unknown error")
            logging.warning(f"SAM3 segmentation failed: {error_msg}")
            return {
                "masks": np.zeros((0, image.shape[0], image.shape[1]), dtype=np.uint8),
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.array([], dtype=np.float32),
                "num_masks": 0,
                "failed": True,
                "error": error_msg,
            }
        
        data = response.get("data", {})
        data["failed"] = False
        return data


class VLMClient:
    """Client for VLM subtask detection server."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 10102, max_retries: int = 3):
        self._uri = f"ws://{host}:{port}"
        self._max_retries = max_retries
        self._ws = None
        self._server_metadata = None
    
    def connect(self, timeout: float = 300) -> None:
        """Connect to VLM server."""
        start_time = time.time()
        
        for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(k, None)
        
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Failed to connect to VLM server within {timeout} seconds")
            
            try:
                # Disable keepalive pings: VLM inference can take 1–5+ minutes; the server
                # often cannot respond to pings in time, causing spurious "keepalive ping
                # timeout" and ConnectionClosedError. ping_interval=None disables pings.
                self._ws = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    open_timeout=150,
                    ping_interval=None,
                    ping_timeout=None,
                )
                self._server_metadata = unpackb(self._ws.recv())
                logging.info(f"Connected to VLM server: {self._server_metadata}")
                return
            except ConnectionRefusedError:
                logging.info(f"Waiting for VLM server {self._uri} ...")
                time.sleep(2)
    
    def _reconnect(self) -> None:
        """Reconnect to server."""
        for attempt in range(self._max_retries):
            try:
                if self._ws:
                    self._ws.close()
            except Exception:
                pass
            self._ws = None
            try:
                self.connect(timeout=60)
                return
            except Exception as e:
                logging.warning(f"Reconnect attempt {attempt + 1} failed: {e}")
                time.sleep(2)
        logging.error("Exceeded maximum VLM reconnect attempts")
    
    def close(self) -> None:
        """Close connection."""
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
    
    def detect_subtask(
        self,
        reference_image: Optional[np.ndarray],
        current_image: np.ndarray,
        task_description: str,
        current_subtask: str,
        next_subtask: Optional[str],
        gripper_state: Dict[str, str],
        gripper_state_changed: bool = False,
        previous_gripper_state: Optional[Dict[str, str]] = None,
    ) -> Dict:
        """Request subtask detection from VLM server."""
        if self._ws is None:
            self.connect()
        
        request = {
            "type": "subtask_detection",
            "request_id": f"vlm_{time.time()}",
            "reference_image": reference_image,
            "current_image": current_image,
            "task_description": task_description,
            "current_subtask": current_subtask,
            "next_subtask": next_subtask,
            "gripper_state": gripper_state,
            "gripper_state_changed": gripper_state_changed,
            "previous_gripper_state": previous_gripper_state,
        }
        
        try:
            self._ws.send(packb(request))
            response = unpackb(self._ws.recv())
        except Exception as e:
            logging.warning(f"VLM socket error: {e}. Reconnecting...")
            self._reconnect()
            self._ws.send(packb(request))
            response = unpackb(self._ws.recv())
        
        if not response.get("ok", False):
            error_msg = response.get("error", {}).get("message", "Unknown error")
            logging.warning(f"VLM detection failed: {error_msg}")
            return {
                "reasoning": f"Error: {error_msg}",
                "decision": "continue",
                "target_object": "",
                "target_location": None,
                "failed": True,
                "error": error_msg,
            }
        
        data = response.get("data", {})
        data["failed"] = False
        return data


# ===============================
# EPISODE PROCESSING
# ===============================

def process_episode(
    task_folder_path: str,
    episode_index: int,
    sam_client: SAM3Client,
    vlm_client: VLMClient,
    output_dir: str,
    video_backend: str = "decord",
    frames_root: Optional[str] = None,
    task: Optional[str] = None,
) -> Dict:
    """
    Process a single episode following the golden logic from combined_vlm_sam.py.
    
    Args:
        task_folder_path: Path to the task folder.
        episode_index: Index of the episode to process.
        sam_client: Client for SAM3 segmentation server.
        vlm_client: Client for VLM subtask detection server.
        output_dir: Output directory for .npz files.
        video_backend: Backend for video reading ("decord", "pyav", "torchvision_av").
        frames_root: Optional path to pre-extracted frames directory. When provided,
                     loads frames from JPEG images instead of decoding video.
        task: Task folder name (required when using frames_root).
    
    Returns:
        Dict with processing status and metadata
    """
    output_path = os.path.join(output_dir, f"episode_{episode_index:06d}.npz")
    
    # Skip if already processed
    if os.path.exists(output_path):
        logging.info(f"Skipping episode {episode_index} (already processed)")
        return {"status": "skipped", "episode_index": episode_index}
    
    try:
        # Load episode metadata
        episode_meta = load_episode_metadata(task_folder_path, episode_index)
        task_description = episode_meta.get("remarks", "")
        
        # Construct paths
        parquet_path = construct_parquet_path(task_folder_path, episode_index)
        
        # Load frames: prefer pre-extracted images if frames_root is provided
        if frames_root and task:
            frames_dir = construct_frames_dir(frames_root, task, episode_index)
            if os.path.exists(frames_dir):
                # Get fps from metadata
                fps = get_fps_from_metadata(task_folder_path)
                frames, fps = load_frames_from_images(frames_dir, fps)
                logging.debug(f"Loaded {len(frames)} frames from pre-extracted images")
            else:
                logging.warning(f"Frames directory not found: {frames_dir}, falling back to video")
                video_path = construct_video_path(task_folder_path, episode_index)
                if not os.path.exists(video_path):
                    return {"status": "error", "episode_index": episode_index, "error": "Neither frames nor video found"}
                frames, fps = extract_all_frames_numpy(video_path, video_backend=video_backend)
        else:
            # Fall back to video loading
            video_path = construct_video_path(task_folder_path, episode_index)
            if not os.path.exists(video_path):
                logging.warning(f"Video not found: {video_path}")
                return {"status": "error", "episode_index": episode_index, "error": "Video not found"}
            frames, fps = extract_all_frames_numpy(video_path, video_backend=video_backend)
        
        num_frames = len(frames)
        
        if num_frames == 0:
            return {"status": "error", "episode_index": episode_index, "error": "No frames"}
        
        # Get image dimensions
        H, W = frames[0].shape[:2]
        
        # Load gripper states
        gripper_states = load_gripper_states_from_parquet(parquet_path, num_frames)
        
        # Parse subtasks
        subtasks = parse_explicit_task(task_description)
        if not subtasks:
            subtasks = [task_description]
        
        # Initialize tracking variables
        recorded_gripper_state = {"left": "OPEN", "right": "OPEN"}
        current_subtask_idx = 0
        current_subtask = subtasks[0]
        reference_image = None
        is_first_frame = True
        
        # Current targets (persist across frames)
        current_target_object = None
        current_target_location = None
        
        # Track SAM/VLM failures - if any occur, don't save this episode
        has_vlm_failure = False
        has_sam_failure = False
        failure_details = []
        
        # Storage arrays
        target_masks = np.zeros((num_frames, H, W), dtype=np.uint8)
        target_boxes = np.zeros((num_frames, 4), dtype=np.float32)
        target_scores = np.zeros(num_frames, dtype=np.float32)
        
        location_masks = np.zeros((num_frames, H, W), dtype=np.uint8)
        location_boxes = np.zeros((num_frames, 4), dtype=np.float32)
        location_scores = np.zeros(num_frames, dtype=np.float32)
        
        vlm_results = []
        
        # Process each frame
        for frame_idx, frame in enumerate(frames):
            curr_gripper = gripper_states[frame_idx]
            gripper_state_changed = check_gripper_state_changed(recorded_gripper_state, curr_gripper)
            
            # Determine if we should query VLM
            should_query_vlm = is_first_frame or gripper_state_changed
            
            if should_query_vlm:
                # Get next subtask
                next_subtask = subtasks[current_subtask_idx + 1] if current_subtask_idx + 1 < len(subtasks) else None
                
                # Query VLM
                vlm_output = vlm_client.detect_subtask(
                    reference_image=reference_image,
                    current_image=frame,
                    task_description=task_description,
                    current_subtask=current_subtask,
                    next_subtask=next_subtask,
                    gripper_state={"left": curr_gripper["left"], "right": curr_gripper["right"]},
                    gripper_state_changed=gripper_state_changed,
                    previous_gripper_state=recorded_gripper_state if gripper_state_changed else None,
                )
                
                # Check for VLM failure
                if vlm_output.get("failed", False):
                    has_vlm_failure = True
                    failure_details.append(f"VLM failed at frame {frame_idx}: {vlm_output.get('error', 'Unknown error')}")
                
                vlm_decision = vlm_output.get("decision", "continue")
                
                # Update targets from VLM
                if vlm_output.get("target_object"):
                    current_target_object = vlm_output["target_object"]
                if vlm_output.get("target_location"):
                    current_target_location = vlm_output["target_location"]
                
                # Record VLM result
                vlm_results.append({
                    "frame": frame_idx,
                    "subtask": current_subtask,
                    "subtask_idx": current_subtask_idx,
                    "decision": vlm_decision,
                    "target_object": current_target_object,
                    "target_location": current_target_location,
                    "reasoning": vlm_output.get("reasoning"),
                    "gripper_state_changed": gripper_state_changed,
                })
                
                # Handle subtask transitions
                if is_first_frame:
                    is_first_frame = False
                    reference_image = frame
                elif gripper_state_changed:
                    if vlm_decision == "proceed" and current_subtask_idx + 1 < len(subtasks):
                        current_subtask_idx += 1
                        current_subtask = subtasks[current_subtask_idx]
                        reference_image = frame
                    recorded_gripper_state = {"left": curr_gripper["left"], "right": curr_gripper["right"]}
            
            # SAM segmentation for target_object
            if current_target_object:
                obj_result = sam_client.segment(frame, current_target_object)
                if obj_result.get("failed", False):
                    has_sam_failure = True
                    failure_details.append(f"SAM failed at frame {frame_idx} for target_object '{current_target_object}': {obj_result.get('error', 'Unknown error')}")
                elif obj_result["num_masks"] > 0:
                    # Keep only top-scoring mask
                    top_idx = np.argmax(obj_result["scores"])
                    target_masks[frame_idx] = obj_result["masks"][top_idx]
                    target_boxes[frame_idx] = obj_result["boxes"][top_idx]
                    target_scores[frame_idx] = obj_result["scores"][top_idx]
            
            # SAM segmentation for target_location
            if current_target_location:
                loc_result = sam_client.segment(frame, current_target_location)
                if loc_result.get("failed", False):
                    has_sam_failure = True
                    failure_details.append(f"SAM failed at frame {frame_idx} for target_location '{current_target_location}': {loc_result.get('error', 'Unknown error')}")
                elif loc_result["num_masks"] > 0:
                    top_idx = np.argmax(loc_result["scores"])
                    location_masks[frame_idx] = loc_result["masks"][top_idx]
                    location_boxes[frame_idx] = loc_result["boxes"][top_idx]
                    location_scores[frame_idx] = loc_result["scores"][top_idx]
        
        # Check for SAM/VLM failures - don't save if any occurred
        if has_vlm_failure or has_sam_failure:
            failure_type = []
            if has_vlm_failure:
                failure_type.append("VLM")
            if has_sam_failure:
                failure_type.append("SAM")
            error_summary = f"{'/'.join(failure_type)} failure(s): {'; '.join(failure_details[:3])}"
            if len(failure_details) > 3:
                error_summary += f" ... and {len(failure_details) - 3} more"
            logging.error(f"Episode {episode_index} not saved due to {error_summary}")
            return {
                "status": "error",
                "episode_index": episode_index,
                "error": error_summary,
                "failure_count": len(failure_details),
            }
        
        # Save to .npz
        save_dict = {
            "episode_index": np.array(episode_index),
            "task_description": np.array(task_description),
            "num_frames": np.array(num_frames),
            "fps": np.array(fps),
            "target_object": np.array(current_target_object if current_target_object else ""),
            "target_location": np.array(current_target_location if current_target_location else ""),
            "target_masks": target_masks,
            "target_boxes": target_boxes,
            "target_scores": target_scores,
            "location_masks": location_masks,
            "location_boxes": location_boxes,
            "location_scores": location_scores,
            "vlm_results": np.array(json.dumps(vlm_results)),
        }
        
        np.savez_compressed(output_path, **save_dict)
        logging.info(f"Saved episode {episode_index} to {output_path}")
        
        return {
            "status": "success",
            "episode_index": episode_index,
            "num_frames": num_frames,
            "target_object": current_target_object,
            "target_location": current_target_location,
        }
        
    except Exception as e:
        logging.exception(f"Error processing episode {episode_index}")
        return {
            "status": "error",
            "episode_index": episode_index,
            "error": str(e),
        }


# ===============================
# MAIN
# ===============================

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        force=True,
    )
    
    parser = argparse.ArgumentParser(description="Parallel Dataset Processing for Visual Prompts")
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to dataset root"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Task folder name (e.g., gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000)"
    )
    parser.add_argument(
        "--episode-start",
        type=int,
        default=0,
        help="Start episode index (inclusive)"
    )
    parser.add_argument(
        "--episode-end",
        type=int,
        default=999,
        help="End episode index (inclusive)"
    )
    parser.add_argument(
        "--sam-host",
        type=str,
        default="127.0.0.1",
        help="SAM3 server host"
    )
    parser.add_argument(
        "--sam-port",
        type=int,
        default=10094,
        help="SAM3 server port"
    )
    parser.add_argument(
        "--vlm-host",
        type=str,
        default="127.0.0.1",
        help="VLM server host"
    )
    parser.add_argument(
        "--vlm-port",
        type=int,
        default=10200,
        help="VLM server port"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for .npz files"
    )
    parser.add_argument(
        "--video-backend",
        type=str,
        default="decord",
        choices=["decord", "pyav", "torchvision_av"],
        help="Video backend for frame extraction (decord is faster, default: decord)"
    )
    parser.add_argument(
        "--frames-root",
        type=str,
        default=None,
        help="Path to pre-extracted frames directory. When provided, loads frames from "
             "JPEG images instead of decoding video, significantly reducing CPU usage."
    )
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Process episodes in reverse order (from episode_end down to episode_start)"
    )
    parser.add_argument(
        "--middle-out",
        action="store_true",
        help="Process episodes from middle outward (500, 499, 501, 498, 502, ...)"
    )
    args = parser.parse_args()
    
    # Construct task folder path
    task_folder_path = os.path.join(args.dataset_root, args.task)
    if not os.path.exists(task_folder_path):
        raise FileNotFoundError(f"Task folder not found: {task_folder_path}")
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, args.task)
    os.makedirs(output_dir, exist_ok=True)
    
    logging.info(f"Processing task: {args.task}")
    logging.info(f"Episodes: {args.episode_start} to {args.episode_end}")
    logging.info(f"SAM server: {args.sam_host}:{args.sam_port}")
    logging.info(f"VLM server: {args.vlm_host}:{args.vlm_port}")
    if args.frames_root:
        logging.info(f"Frames root: {args.frames_root} (using pre-extracted images)")
    else:
        logging.info(f"Video backend: {args.video_backend}")
    logging.info(f"Output: {output_dir}")
    if args.reverse:
        logging.info("Processing in REVERSE order")
    elif args.middle_out:
        logging.info("Processing in MIDDLE-OUT order")
    
    # Connect to servers
    sam_client = SAM3Client(host=args.sam_host, port=args.sam_port)
    sam_client.connect()
    
    vlm_client = VLMClient(host=args.vlm_host, port=args.vlm_port)
    vlm_client.connect()
    
    # Process episodes
    results = {"success": 0, "skipped": 0, "error": 0}
    
    try:
        # Determine episode iteration order
        if args.reverse:
            episode_range = range(args.episode_end, args.episode_start - 1, -1)
        elif args.middle_out:
            # Generate middle-out order: start from middle, alternate left/right
            start, end = args.episode_start, args.episode_end
            middle = (start + end) // 2
            episode_list = [middle]
            offset = 1
            while True:
                added = False
                if middle - offset >= start:
                    episode_list.append(middle - offset)
                    added = True
                if middle + offset <= end:
                    episode_list.append(middle + offset)
                    added = True
                if not added:
                    break
                offset += 1
            episode_range = episode_list
        else:
            episode_range = range(args.episode_start, args.episode_end + 1)
        
        for episode_idx in episode_range:
            result = process_episode(
                task_folder_path=task_folder_path,
                episode_index=episode_idx,
                sam_client=sam_client,
                vlm_client=vlm_client,
                output_dir=output_dir,
                video_backend=args.video_backend,
                frames_root=args.frames_root,
                task=args.task,
            )
            
            results[result["status"]] += 1
            
            # Log progress every 10 episodes
            processed = episode_idx - args.episode_start + 1
            total = args.episode_end - args.episode_start + 1
            if processed % 10 == 0 or processed == total:
                logging.info(
                    f"Progress: {processed}/{total} "
                    f"(success={results['success']}, skipped={results['skipped']}, error={results['error']})"
                )
    
    finally:
        sam_client.close()
        vlm_client.close()
    
    # Final summary
    logging.info("=" * 60)
    logging.info("Processing complete!")
    logging.info(f"  Success: {results['success']}")
    logging.info(f"  Skipped: {results['skipped']}")
    logging.info(f"  Errors:  {results['error']}")
    logging.info(f"  Output:  {output_dir}")


if __name__ == "__main__":
    main()
