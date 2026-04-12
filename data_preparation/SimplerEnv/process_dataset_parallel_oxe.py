#!/usr/bin/env python3
"""
Parallel Dataset Processing Script for Visual Prompt Generation (OXE Datasets)

Adapted from the robocasa pipeline for OXE-style LeRobot datasets such as
bridge_orig_lerobot and fractal20220817_data_lerobot. Key differences from
the robocasa version:
  - Single gripper at action[6] (1 = CLOSED, 0 = OPEN)
  - Flat dataset structure (no task subfolders)
  - Task descriptions from episodes.jsonl["tasks"][0]
  - Camera key auto-detected from modality.json

Uses VLM+SAM servers to generate segmentation masks and save them as .npz files.

Example usage:
    # Bridge dataset
    python process_dataset_parallel_oxe.py \
        --dataset-root /path/to/bridge_orig_lerobot \
        --episode-start 0 --episode-end 999 \
        --sam-port 10094 --vlm-port 10200

    # Fractal dataset
    python process_dataset_parallel_oxe.py \
        --dataset-root /path/to/fractal20220817_data_lerobot \
        --episode-start 0 --episode-end 999 \
        --sam-port 10094 --vlm-port 10200

Output structure:
    output_dir/
        dataset_name/
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(os.path.dirname(SCRIPT_DIR), "..")
UTILITY_DIR = os.path.join(PROJECT_ROOT, "examples", "Robocasa_tabletop", "visual_prompt_utility")
sys.path.insert(0, UTILITY_DIR)

from msgpack_utils import packb, unpackb

import av

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


# ===============================
# GRIPPER CONFIGURATION (single gripper, OXE convention)
# action[6] == 1.0 => CLOSED
# action[6] == 0.0 => OPEN
# ===============================

GRIPPER_ACTION_INDEX = 6
GRIPPER_THRESHOLD = 0.5


# ===============================
# GRIPPER STATE FUNCTIONS
# ===============================

def check_gripper_state_changed(
    recorded_gripper: Dict[str, str],
    current_gripper: Dict[str, str],
) -> bool:
    """Check if single gripper state has changed."""
    return recorded_gripper["gripper"] != current_gripper["gripper"]


def load_gripper_states_from_parquet(parquet_path: str, num_frames: int) -> Dict[int, Dict[str, str]]:
    """Load single-gripper states from parquet file.

    Returns:
        Dictionary mapping frame index to gripper state dict with keys
        "gripper" ("OPEN"/"CLOSED") and "gripper_raw" (float).
    """
    try:
        if not os.path.exists(parquet_path):
            logging.warning(f"Parquet file not found: {parquet_path}")
            return _get_default_gripper_states(num_frames)

        df = pd.read_parquet(parquet_path)

        if num_frames != len(df):
            raise ValueError(
                f"Frame count mismatch: {num_frames} frames but {len(df)} action entries in parquet. "
                f"Parquet: {parquet_path}"
            )

        gripper_states = {}
        for frame_idx in range(num_frames):
            action = df.iloc[frame_idx]["action"]
            raw_value = float(action[GRIPPER_ACTION_INDEX])
            state = "CLOSED" if raw_value >= GRIPPER_THRESHOLD else "OPEN"
            gripper_states[frame_idx] = {
                "gripper": state,
                "gripper_raw": raw_value,
            }

        return gripper_states

    except Exception as e:
        logging.warning(f"Failed to load gripper states: {e}")
        return _get_default_gripper_states(num_frames)


def _get_default_gripper_states(num_frames: int) -> Dict[int, Dict[str, str]]:
    """Return default gripper states (all OPEN)."""
    return {
        i: {"gripper": "OPEN", "gripper_raw": 0.0}
        for i in range(num_frames)
    }


def parse_explicit_task(task: str) -> List[str]:
    """Parse task description into subtasks."""
    parts = re.split(r",\s*and\s*|,\s*|\s+and\s+", task.lower())
    return [p.strip() for p in parts if p.strip()]


# ===============================
# DATA LOADING FUNCTIONS
# ===============================

def detect_camera_key(dataset_root: str) -> str:
    """Auto-detect camera key from modality.json."""
    modality_path = os.path.join(dataset_root, "meta", "modality.json")
    if not os.path.exists(modality_path):
        raise FileNotFoundError(f"modality.json not found: {modality_path}")

    with open(modality_path, "r") as f:
        modality = json.load(f)

    video_section = modality.get("video", {})
    if not video_section:
        raise ValueError(f"No 'video' section in {modality_path}")

    first_key = next(iter(video_section))
    original_key = video_section[first_key].get("original_key", f"observation.images.{first_key}")
    return original_key


def load_episode_metadata(dataset_root: str, episode_index: int) -> Dict:
    """Load episode metadata from meta/episodes.jsonl."""
    episodes_path = os.path.join(dataset_root, "meta", "episodes.jsonl")

    if not os.path.exists(episodes_path):
        raise FileNotFoundError(f"Episodes file not found: {episodes_path}")

    with open(episodes_path, "r") as f:
        for i, line in enumerate(f):
            if i == episode_index:
                return json.loads(line.strip())

    raise ValueError(f"Episode {episode_index} not found in {episodes_path}")


def get_task_description(episode_meta: Dict) -> str:
    """Extract task description from episode metadata.

    OXE datasets store tasks in episodes.jsonl["tasks"] (a list of strings).
    """
    tasks = episode_meta.get("tasks", [])
    if tasks and isinstance(tasks, list) and tasks[0]:
        return tasks[0]
    return ""


def construct_video_path(dataset_root: str, camera_key: str, episode_index: int) -> str:
    """Construct video path for an episode."""
    chunk_num = episode_index // 1000
    return os.path.join(
        dataset_root,
        "videos",
        f"chunk-{chunk_num:03d}",
        camera_key,
        f"episode_{episode_index:06d}.mp4",
    )


def construct_parquet_path(dataset_root: str, episode_index: int) -> str:
    """Construct parquet path for an episode."""
    chunk_num = episode_index // 1000
    return os.path.join(
        dataset_root,
        "data",
        f"chunk-{chunk_num:03d}",
        f"episode_{episode_index:06d}.parquet",
    )


def extract_all_frames_numpy(video_path: str) -> Tuple[List[np.ndarray], float]:
    """Extract all frames from video as numpy arrays using PyAV.

    Uses the same PyAV approach as the dataloader in
    starVLA/dataloader/gr00t_lerobot/video.py: open the container, decode
    every video frame sequentially, and convert to rgb24 numpy arrays.
    """
    container = None
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate) if stream.average_rate else float(stream.guessed_rate)

        frames = []
        for frame in container.decode(video=0):
            frames.append(frame.to_ndarray(format="rgb24"))

        return frames, fps
    finally:
        if container is not None:
            container.close()


def get_fps_from_metadata(dataset_root: str) -> float:
    """Read fps from meta/info.json."""
    info_path = os.path.join(dataset_root, "meta", "info.json")
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info = json.load(f)
            return info.get("fps", 5.0)
    return 5.0


def construct_frames_dir(frames_root: str, dataset_name: str, camera_key: str, episode_index: int) -> str:
    """Construct directory path for pre-extracted frames."""
    chunk_num = episode_index // 1000
    return os.path.join(
        frames_root,
        dataset_name,
        f"chunk-{chunk_num:03d}",
        camera_key,
        f"episode_{episode_index:06d}",
    )


def load_frames_from_images(frames_dir: str, fps: float = 5.0) -> Tuple[List[np.ndarray], float]:
    """Load frames from pre-extracted JPEG images."""
    if not PIL_AVAILABLE:
        raise ImportError("PIL is not available. Install with: pip install Pillow")

    if not os.path.exists(frames_dir):
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    if not frame_files:
        raise FileNotFoundError(f"No frame files found in {frames_dir}")

    frames = []
    for frame_file in frame_files:
        img = Image.open(os.path.join(frames_dir, frame_file))
        frames.append(np.array(img))
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
        start_time = time.time()
        for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(k, None)

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Failed to connect to SAM3 server within {timeout}s")
            try:
                self._ws = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None,
                    open_timeout=150, ping_interval=None, ping_timeout=None,
                )
                self._server_metadata = unpackb(self._ws.recv())
                logging.info(f"Connected to SAM3 server: {self._server_metadata}")
                return
            except ConnectionRefusedError:
                logging.info(f"Waiting for SAM3 server {self._uri} ...")
                time.sleep(2)

    def _reconnect(self) -> None:
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
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass

    def segment(self, image: np.ndarray, text_prompt: str,
                threshold: float = 0.5, mask_threshold: float = 0.5) -> Dict:
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
        start_time = time.time()
        for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(k, None)

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Failed to connect to VLM server within {timeout}s")
            try:
                self._ws = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None,
                    open_timeout=150, ping_interval=None, ping_timeout=None,
                )
                self._server_metadata = unpackb(self._ws.recv())
                logging.info(f"Connected to VLM server: {self._server_metadata}")
                return
            except ConnectionRefusedError:
                logging.info(f"Waiting for VLM server {self._uri} ...")
                time.sleep(2)

    def _reconnect(self) -> None:
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
    dataset_root: str,
    episode_index: int,
    camera_key: str,
    sam_client: SAM3Client,
    vlm_client: VLMClient,
    output_dir: str,
    frames_root: Optional[str] = None,
    dataset_name: Optional[str] = None,
) -> Dict:
    """Process a single episode using VLM+SAM pipeline."""
    output_path = os.path.join(output_dir, f"episode_{episode_index:06d}.npz")

    if os.path.exists(output_path):
        logging.info(f"Skipping episode {episode_index} (already processed)")
        return {"status": "skipped", "episode_index": episode_index}

    try:
        episode_meta = load_episode_metadata(dataset_root, episode_index)
        task_description = get_task_description(episode_meta)

        if not task_description:
            logging.warning(f"Episode {episode_index} has no task description, skipping")
            return {"status": "error", "episode_index": episode_index, "error": "No task description"}

        parquet_path = construct_parquet_path(dataset_root, episode_index)

        if frames_root and dataset_name:
            frames_dir = construct_frames_dir(frames_root, dataset_name, camera_key, episode_index)
            if os.path.exists(frames_dir):
                fps = get_fps_from_metadata(dataset_root)
                frames, fps = load_frames_from_images(frames_dir, fps)
                logging.debug(f"Loaded {len(frames)} frames from pre-extracted images")
            else:
                logging.warning(f"Frames directory not found: {frames_dir}, falling back to video")
                video_path = construct_video_path(dataset_root, camera_key, episode_index)
                if not os.path.exists(video_path):
                    return {"status": "error", "episode_index": episode_index, "error": "Neither frames nor video found"}
                frames, fps = extract_all_frames_numpy(video_path)
        else:
            video_path = construct_video_path(dataset_root, camera_key, episode_index)
            if not os.path.exists(video_path):
                logging.warning(f"Video not found: {video_path}")
                return {"status": "error", "episode_index": episode_index, "error": "Video not found"}
            frames, fps = extract_all_frames_numpy(video_path)

        num_frames = len(frames)
        if num_frames == 0:
            return {"status": "error", "episode_index": episode_index, "error": "No frames"}

        H, W = frames[0].shape[:2]

        gripper_states = load_gripper_states_from_parquet(parquet_path, num_frames)

        subtasks = parse_explicit_task(task_description)
        if not subtasks:
            subtasks = [task_description]

        recorded_gripper_state = {"gripper": "OPEN"}
        current_subtask_idx = 0
        current_subtask = subtasks[0]
        reference_image = None
        is_first_frame = True

        current_target_object = None
        current_target_location = None

        has_vlm_failure = False
        has_sam_failure = False
        failure_details = []

        target_masks = np.zeros((num_frames, H, W), dtype=np.uint8)
        target_boxes = np.zeros((num_frames, 4), dtype=np.float32)
        target_scores = np.zeros(num_frames, dtype=np.float32)

        location_masks = np.zeros((num_frames, H, W), dtype=np.uint8)
        location_boxes = np.zeros((num_frames, 4), dtype=np.float32)
        location_scores = np.zeros(num_frames, dtype=np.float32)

        vlm_results = []

        for frame_idx, frame in enumerate(frames):
            curr_gripper = gripper_states[frame_idx]
            gripper_state_changed = check_gripper_state_changed(recorded_gripper_state, curr_gripper)

            should_query_vlm = is_first_frame or gripper_state_changed

            if should_query_vlm:
                next_subtask = subtasks[current_subtask_idx + 1] if current_subtask_idx + 1 < len(subtasks) else None

                vlm_output = vlm_client.detect_subtask(
                    reference_image=reference_image,
                    current_image=frame,
                    task_description=task_description,
                    current_subtask=current_subtask,
                    next_subtask=next_subtask,
                    gripper_state={"gripper": curr_gripper["gripper"]},
                    gripper_state_changed=gripper_state_changed,
                    previous_gripper_state=recorded_gripper_state if gripper_state_changed else None,
                )

                if vlm_output.get("failed", False):
                    has_vlm_failure = True
                    failure_details.append(f"VLM failed at frame {frame_idx}: {vlm_output.get('error', 'Unknown error')}")

                vlm_decision = vlm_output.get("decision", "continue")

                if vlm_output.get("target_object"):
                    current_target_object = vlm_output["target_object"]
                if vlm_output.get("target_location"):
                    current_target_location = vlm_output["target_location"]

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

                if is_first_frame:
                    is_first_frame = False
                    reference_image = frame
                elif gripper_state_changed:
                    if vlm_decision == "proceed" and current_subtask_idx + 1 < len(subtasks):
                        current_subtask_idx += 1
                        current_subtask = subtasks[current_subtask_idx]
                        reference_image = frame
                    recorded_gripper_state = {"gripper": curr_gripper["gripper"]}

            # SAM segmentation for target_object
            if current_target_object:
                obj_result = sam_client.segment(frame, current_target_object)
                if obj_result.get("failed", False):
                    has_sam_failure = True
                    failure_details.append(f"SAM failed at frame {frame_idx} for target_object '{current_target_object}': {obj_result.get('error', 'Unknown error')}")
                elif obj_result["num_masks"] > 0:
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

    parser = argparse.ArgumentParser(description="Parallel Dataset Processing for Visual Prompts (OXE Datasets)")
    parser.add_argument(
        "--dataset-root", type=str, required=True,
        help="Path to OXE LeRobot dataset root (e.g., .../bridge_orig_lerobot)",
    )
    parser.add_argument(
        "--camera-key", type=str, default=None,
        help="Camera observation key (e.g., observation.images.image_0). Auto-detected from modality.json if omitted.",
    )
    parser.add_argument("--episode-start", type=int, default=0, help="Start episode index (inclusive)")
    parser.add_argument("--episode-end", type=int, default=999, help="End episode index (inclusive)")
    parser.add_argument("--sam-host", type=str, default="127.0.0.1", help="SAM3 server host")
    parser.add_argument("--sam-port", type=int, default=10094, help="SAM3 server port")
    parser.add_argument("--vlm-host", type=str, default="127.0.0.1", help="VLM server host")
    parser.add_argument("--vlm-port", type=int, default=10200, help="VLM server port")
    parser.add_argument(
        "--output-dir", type=str,
        required=True,
        help="Output directory for .npz files",
    )
    parser.add_argument(
        "--frames-root", type=str, default=None,
        help="Path to pre-extracted frames directory. Loads JPEG images instead of decoding video.",
    )
    parser.add_argument("--reverse", action="store_true", help="Process episodes in reverse order")
    parser.add_argument("--middle-out", action="store_true", help="Process episodes from middle outward")
    args = parser.parse_args()

    dataset_root = os.path.abspath(args.dataset_root)
    if not os.path.exists(dataset_root):
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    dataset_name = os.path.basename(dataset_root)

    if args.camera_key:
        camera_key = args.camera_key
    else:
        camera_key = detect_camera_key(dataset_root)
    logging.info(f"Using camera key: {camera_key}")

    output_dir = os.path.join(args.output_dir, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    logging.info(f"Dataset: {dataset_name} ({dataset_root})")
    logging.info(f"Episodes: {args.episode_start} to {args.episode_end}")
    logging.info(f"SAM server: {args.sam_host}:{args.sam_port}")
    logging.info(f"VLM server: {args.vlm_host}:{args.vlm_port}")
    if args.frames_root:
        logging.info(f"Frames root: {args.frames_root} (using pre-extracted images)")
    else:
        logging.info("Video backend: pyav")
    logging.info(f"Output: {output_dir}")
    if args.reverse:
        logging.info("Processing in REVERSE order")
    elif args.middle_out:
        logging.info("Processing in MIDDLE-OUT order")

    sam_client = SAM3Client(host=args.sam_host, port=args.sam_port)
    sam_client.connect()

    vlm_client = VLMClient(host=args.vlm_host, port=args.vlm_port)
    vlm_client.connect()

    results = {"success": 0, "skipped": 0, "error": 0}

    try:
        if args.reverse:
            episode_range = range(args.episode_end, args.episode_start - 1, -1)
        elif args.middle_out:
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
                dataset_root=dataset_root,
                episode_index=episode_idx,
                camera_key=camera_key,
                sam_client=sam_client,
                vlm_client=vlm_client,
                output_dir=output_dir,
                frames_root=args.frames_root,
                dataset_name=dataset_name,
            )

            results[result["status"]] += 1

            processed = sum(results.values())
            total = args.episode_end - args.episode_start + 1
            if processed % 10 == 0 or processed == total:
                logging.info(
                    f"Progress: {processed}/{total} "
                    f"(success={results['success']}, skipped={results['skipped']}, error={results['error']})"
                )

    finally:
        sam_client.close()
        vlm_client.close()

    logging.info("=" * 60)
    logging.info("Processing complete!")
    logging.info(f"  Success: {results['success']}")
    logging.info(f"  Skipped: {results['skipped']}")
    logging.info(f"  Errors:  {results['error']}")
    logging.info(f"  Output:  {output_dir}")


if __name__ == "__main__":
    main()
