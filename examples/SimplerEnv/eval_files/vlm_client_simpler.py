# VLM Client for SimplerEnv (single gripper)
#
# Adapted from Robocasa's vlm_client.py.
# Key difference: all gripper-related APIs accept a plain string
# ("OPEN" / "CLOSED") instead of a {"left": ..., "right": ...} dict,
# matching SimplerEnv's single-gripper robots.

import logging
import time
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import websockets.sync.client

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR))),
    "examples", "Robocasa_tabletop", "visual_prompt_utility",
))
from msgpack_utils import packb, unpackb


class VLMClientSimpler:
    """VLM client for single-gripper SimplerEnv robots."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 10102,
    ):
        self._uri = f"ws://{host}:{port}"
        self._ws, self._server_metadata = self._wait_for_server()

    def _wait_for_server(self, timeout: float = 300) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        start_time = time.time()
        for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(k, None)

        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Failed to connect to VLM server within {timeout}s")
            try:
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None,
                    open_timeout=150, ping_interval=20, ping_timeout=20,
                )
                metadata = unpackb(conn.recv())
                logging.info(f"Connected to VLM SimplerEnv server: {metadata}")
                return conn, metadata
            except ConnectionRefusedError:
                logging.info(f"Still waiting for VLM server {self._uri} ...")
                time.sleep(2)

    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass
        
    def reset(self) -> None:
        """
        Reset the client state at the start of a new episode.
        """
        # Reset legacy detect_grasp state
        self._last_query_time = 0.0
        self._cached_grasped = False
        self._cached_sam_prompt = ""
        self._cached_raw_response = ""
        
        # Reset decompose_task state
        self._task_decomposed = False
        self._cached_grasp_object = ""
        self._cached_place_target = None
        self._cached_decomposition_raw_response = ""

    def decompose_task_to_subtasks(
        self,
        task_description: str,
    ) -> List[str]:
        """
        Use VLM to decompose a task description into sequential subtask strings.
        Text-only query (no images needed). Falls back to an empty list on failure.
        """
        request = {
            "type": "task_decomposition",
            "request_id": f"decompose_subtasks_{time.time()}",
            "task_description": task_description,
        }

        self._ws.send(packb(request))
        response = unpackb(self._ws.recv())

        if not response.get("ok", False):
            error_msg = response.get("error", {}).get("message", "Unknown error")
            logging.warning(f"VLM task decomposition to subtasks failed: {error_msg}")
            return []

        data = response.get("data", {})
        subtasks = data.get("subtasks", [])
        raw_response = data.get("raw_response", "")

        logging.info(
            f"VLM task decomposition: '{task_description}' -> {subtasks} "
            f"(raw: {raw_response[:200]})"
        )
        return subtasks

    def detect_subtask(
        self,
        reference_image: Optional[np.ndarray],
        current_image: np.ndarray,
        task_description: str,
        current_subtask: str,
        next_subtask: Optional[str],
        gripper_state: str,
        gripper_state_changed: bool = False,
        previous_gripper_state: Optional[str] = None,
    ) -> Dict:
        """
        Detect subtask progress using VLM server.

        Args:
            reference_image: (H, W, 3) uint8 RGB or None
            current_image:   (H, W, 3) uint8 RGB
            task_description: full task description
            current_subtask:  subtask being executed
            next_subtask:     next subtask or None
            gripper_state:    "OPEN" or "CLOSED"  (single string)
            gripper_state_changed: whether gripper state changed
            previous_gripper_state: "OPEN" or "CLOSED" if changed, else None

        Returns:
            dict with reasoning, decision, target_object, target_location,
            raw_response, from_cache
        """
        request = {
            "type": "subtask_detection",
            "request_id": f"subtask_{time.time()}",
            "reference_image": reference_image,
            "current_image": current_image,
            "task_description": task_description,
            "current_subtask": current_subtask,
            "next_subtask": next_subtask,
            "gripper_state": gripper_state,
            "gripper_state_changed": gripper_state_changed,
            "previous_gripper_state": previous_gripper_state,
        }

        self._ws.send(packb(request))
        response = unpackb(self._ws.recv())

        if not response.get("ok", False):
            error_msg = response.get("error", {}).get("message", "Unknown error")
            logging.warning(f"VLM subtask detection failed: {error_msg}")
            return {
                "reasoning": f"Error: {error_msg}",
                "decision": "continue",
                "target_object": "",
                "target_location": None,
                "raw_response": "",
                "from_cache": False,
                "error": error_msg,
            }

        data = response.get("data", {})
        data["from_cache"] = False

        logging.info(
            f"VLM subtask detection: decision='{data.get('decision')}', "
            f"target_object='{data.get('target_object')}', "
            f"target_location='{data.get('target_location')}'"
        )
        return data
