# VLM Grasp Detection Client
# Client to communicate with VLM server for grasp state detection and task decomposition

import logging
import time
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import websockets.sync.client

from msgpack_utils import packb, unpackb


class VLMClient:
    """Client for communicating with VLM grasp detection server."""
    
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 10095,
        query_interval: float = 1.0,  # Query at 1Hz (for legacy detect_grasp mode)
    ):
        """
        Initialize VLM client.
        
        Args:
            host: VLM server host
            port: VLM server port
            query_interval: Minimum interval between queries in seconds (default 1.0 for 1Hz)
                           Only used for legacy detect_grasp mode.
        """
        self._uri = f"ws://{host}:{port}"
        self._query_interval = query_interval
        
        # State caching for legacy detect_grasp mode
        self._last_query_time: float = 0.0
        self._cached_grasped: bool = False
        self._cached_sam_prompt: str = ""
        self._cached_raw_response: str = ""
        
        # State caching for decompose_task mode (query-once)
        self._task_decomposed: bool = False
        self._cached_grasp_object: str = ""
        self._cached_place_target: Optional[str] = None
        self._cached_decomposition_raw_response: str = ""
        
        self._ws, self._server_metadata = self._wait_for_server()
    
    def _wait_for_server(self, timeout: float = 300) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        start_time = time.time()
        
        # Clear proxy settings
        for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(k, None)
        
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Failed to connect to VLM server within {timeout} seconds")
            
            try:
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    open_timeout=150,
                    # VLM inference can take 30-60+ seconds; disable client ping to avoid
                    # "keepalive ping timeout" (client sends ping, server can't pong while inferencing)
                    ping_interval=None,
                    ping_timeout=None,
                )
                metadata = unpackb(conn.recv())
                logging.info(f"Connected to VLM server: {metadata}")
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
    
    def should_query(self) -> bool:
        """Check if enough time has passed since last query."""
        return time.time() - self._last_query_time >= self._query_interval
    
    def detect_grasp(
        self,
        current_image: np.ndarray,
        task_description: str,
        grasped_state: str = "open",
        force: bool = False,
    ) -> Dict:
        """
        Detect grasp state using VLM server.
        
        This method respects the query_interval for rate limiting.
        If called before the interval has passed, it returns cached results.
        
        Args:
            current_image: np.ndarray (H, W, 3), uint8 RGB - current image
            task_description: str - the task to complete
            grasped_state: str - "open" or "closed" indicating current gripper state
            force: bool - if True, bypass rate limiting and query immediately
            
        Returns:
            dict with:
                - grasped: bool - whether target object is grasped (from VLM response)
                - sam_prompt: str - the prompt to pass to SAM3 (target description)
                - raw_response: str - full VLM response for debugging
                - from_cache: bool - whether result is from cache
        """
        # Rate limiting - return cached result if not enough time has passed
        if not force and not self.should_query():
            return {
                "grasped": self._cached_grasped,
                "sam_prompt": self._cached_sam_prompt,
                "raw_response": self._cached_raw_response,
                "from_cache": True,
            }
        
        # Send request to VLM server
        request = {
            "type": "detect_grasp",
            "request_id": f"grasp_{time.time()}",
            "current_image": current_image,
            "task_description": task_description,
            "grasped_state": grasped_state,
        }
        
        self._ws.send(packb(request))
        response = unpackb(self._ws.recv())
        
        # Update query time
        self._last_query_time = time.time()
        
        if not response.get("ok", False):
            error_msg = response.get("error", {}).get("message", "Unknown error")
            logging.warning(f"VLM grasp detection failed: {error_msg}")
            # Return cached result on error
            return {
                "grasped": self._cached_grasped,
                "sam_prompt": self._cached_sam_prompt,
                "raw_response": self._cached_raw_response,
                "from_cache": True,
                "error": error_msg,
            }
        
        data = response.get("data", {})
        
        # Update cache
        self._cached_grasped = data.get("grasped", False)
        self._cached_sam_prompt = data.get("sam_prompt", "")
        self._cached_raw_response = data.get("raw_response", "")
        
        logging.info(f"VLM detection: grasped={self._cached_grasped}, sam_prompt='{self._cached_sam_prompt}'")
        
        return {
            "grasped": self._cached_grasped,
            "sam_prompt": self._cached_sam_prompt,
            "raw_response": self._cached_raw_response,
            "from_cache": False,
        }
    
    def get_sam_prompt(
        self,
        current_image: np.ndarray,
        task_description: str,
        grasped_state: str = "open",
        fallback_prompt: str = "",
    ) -> str:
        """
        Get the SAM3 prompt based on current grasp state.
        
        This is a convenience method that calls detect_grasp and returns just the SAM prompt.
        
        Args:
            current_image: np.ndarray (H, W, 3), uint8 RGB - current image
            task_description: str - the task to complete
            grasped_state: str - "open" or "closed" indicating current gripper state
            fallback_prompt: str - fallback prompt if VLM fails to provide one
            
        Returns:
            str - the prompt to pass to SAM3
        """
        result = self.detect_grasp(current_image, task_description, grasped_state)
        sam_prompt = result.get("sam_prompt", "")
        
        if not sam_prompt and fallback_prompt:
            logging.warning(f"VLM returned empty sam_prompt, using fallback: '{fallback_prompt}'")
            return fallback_prompt
        
        return sam_prompt
    
    @property
    def cached_grasped(self) -> bool:
        """Get the cached grasp state."""
        return self._cached_grasped
    
    @property
    def cached_sam_prompt(self) -> str:
        """Get the cached SAM prompt."""
        return self._cached_sam_prompt
    
    # =========================================================================
    # Task Decomposition Methods (Query-Once Mode)
    # =========================================================================
    
    def decompose_task(
        self,
        current_image: np.ndarray,
        task_description: str,
        force: bool = False,
    ) -> Dict:
        """
        Decompose task into grasp_object and place_target using VLM.
        This should be called once at the beginning of each episode.
        
        If already decomposed and force=False, returns cached results.
        
        Args:
            current_image: np.ndarray (H, W, 3), uint8 RGB - current/first image
            task_description: str - the task to decompose
            force: bool - if True, force re-query even if already decomposed
            
        Returns:
            dict with:
                - grasp_object: str - object to grasp
                - place_target: str or None - placement target
                - raw_response: str - full VLM response for debugging
                - from_cache: bool - whether result is from cache
        """
        # Return cached result if already decomposed
        if self._task_decomposed and not force:
            return {
                "grasp_object": self._cached_grasp_object,
                "place_target": self._cached_place_target,
                "raw_response": self._cached_decomposition_raw_response,
                "from_cache": True,
            }
        
        # Send request to VLM server
        request = {
            "type": "decompose_task",
            "request_id": f"decompose_{time.time()}",
            "current_image": current_image,
            "task_description": task_description,
        }
        
        self._ws.send(packb(request))
        response = unpackb(self._ws.recv())
        
        if not response.get("ok", False):
            error_msg = response.get("error", {}).get("message", "Unknown error")
            logging.warning(f"VLM task decomposition failed: {error_msg}")
            # Return cached result on error (may be empty)
            return {
                "grasp_object": self._cached_grasp_object,
                "place_target": self._cached_place_target,
                "raw_response": self._cached_decomposition_raw_response,
                "from_cache": True,
                "error": error_msg,
            }
        
        data = response.get("data", {})
        
        # Update cache
        self._task_decomposed = True
        self._cached_grasp_object = data.get("grasp_object", "")
        self._cached_place_target = data.get("place_target")
        self._cached_decomposition_raw_response = data.get("raw_response", "")
        
        logging.info(
            f"VLM task decomposition: grasp_object='{self._cached_grasp_object}', "
            f"place_target='{self._cached_place_target}'"
        )
        
        return {
            "grasp_object": self._cached_grasp_object,
            "place_target": self._cached_place_target,
            "raw_response": self._cached_decomposition_raw_response,
            "from_cache": False,
        }
    
    @property
    def task_decomposed(self) -> bool:
        """Check if task has been decomposed."""
        return self._task_decomposed
    
    @property
    def cached_grasp_object(self) -> str:
        """Get the cached grasp object prompt."""
        return self._cached_grasp_object
    
    @property
    def cached_place_target(self) -> Optional[str]:
        """Get the cached place target prompt."""
        return self._cached_place_target
    
    def decompose_task_to_subtasks(
        self,
        task_description: str,
    ) -> List[str]:
        """
        Use VLM to decompose a task description into sequential subtask strings.
        Text-only query (no images needed). Falls back to an empty list on failure.
        
        Args:
            task_description: Full task description (e.g., "pick up the potato and place it in the microwave, and close the microwave")
            
        Returns:
            List of subtask strings (e.g., ["pick up the potato", "place the potato in the microwave", "close the microwave"])
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
    
    # =========================================================================
    # Subtask Detection Methods (Query on Gripper Change Mode)
    # =========================================================================
    
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
        """
        Detect subtask progress using VLM server.
        
        This method follows the golden logic from process_dataset_parallel.py:
        - Query on first frame and when gripper state changes
        - Returns target_object and target_location for SAM segmentation
        
        Args:
            reference_image: np.ndarray (H, W, 3), uint8 RGB - frame when subtask started (optional)
            current_image: np.ndarray (H, W, 3), uint8 RGB - current frame
            task_description: str - full task description
            current_subtask: str - current subtask being executed
            next_subtask: str or None - next subtask in sequence
            gripper_state: dict with left/right states ("OPEN" or "CLOSED")
            gripper_state_changed: bool - whether gripper state changed
            previous_gripper_state: dict with previous left/right states (if changed)
            
        Returns:
            dict with:
                - reasoning: str - VLM's reasoning
                - decision: str - "continue" or "proceed"
                - target_object: str - object to segment
                - target_location: str or None - location to segment
                - raw_response: str - full VLM response for debugging
                - from_cache: bool - always False for this method
        """
        # Send request to VLM server
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


