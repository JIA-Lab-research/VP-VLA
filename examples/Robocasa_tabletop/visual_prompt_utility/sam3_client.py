# SAM3 Segmentation Client
# Client to communicate with SAM3 server and overlay masks on images

import logging
import time
import os
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import websockets.sync.client

from msgpack_utils import packb, unpackb


# Supported overlay modes
OVERLAY_MODES = ["mask", "box", "points", "contour", "crosshair", "contour_darken"]


class SAM3Client:
    """Client for communicating with SAM3 segmentation server."""
    
    def __init__(
        self, 
        host: str = "127.0.0.1", 
        port: int = 10094,
        overlay_alpha: float = 0.5,
        # Open gripper state configurations (for picking phase)
        robot_prompt_open: str = "claw",
        overlay_mode_open: str = "mask",
        threshold_open: float = 0.5,
        mask_threshold_open: float = 0.5,
        # Closed gripper state configurations (for placing phase)
        overlay_mode_closed: Optional[str] = None,
        threshold_closed: Optional[float] = None,
        mask_threshold_closed: Optional[float] = None,
    ):
        """
        Initialize SAM3 client.
        
        Args:
            host: SAM3 server host
            port: SAM3 server port
            overlay_alpha: Transparency for mask overlay (0-1)
            robot_prompt_open: Text prompt for robot arm segmentation when gripper is open
                              (default "claw"). When gripper is closed, the previously
                              targeted object becomes the robot_prompt (passed dynamically).
            overlay_mode_open: One of "mask", "box", "points", "contour", "crosshair", "contour_darken"
                              Used when gripper is open (picking phase)
            threshold_open: Confidence threshold for instance detection (when gripper is open)
            mask_threshold_open: Threshold for mask binarization (when gripper is open)
            overlay_mode_closed: Overlay mode when gripper is closed (placing phase).
                                If None, uses overlay_mode_open.
            threshold_closed: Confidence threshold when gripper is closed.
                             If None, uses threshold_open.
            mask_threshold_closed: Mask binarization threshold when gripper is closed.
                                  If None, uses mask_threshold_open.
        """
        self._uri = f"ws://{host}:{port}"
        self._overlay_alpha = overlay_alpha
        self._robot_prompt_open = robot_prompt_open
        
        # Open gripper (picking) configuration
        self._overlay_mode_open = overlay_mode_open
        self._threshold_open = threshold_open
        self._mask_threshold_open = mask_threshold_open
        
        # Closed gripper (placing) configuration - fall back to open config if not specified
        self._overlay_mode_closed = overlay_mode_closed if overlay_mode_closed is not None else overlay_mode_open
        self._threshold_closed = threshold_closed if threshold_closed is not None else threshold_open
        self._mask_threshold_closed = mask_threshold_closed if mask_threshold_closed is not None else mask_threshold_open
        
        # Validate overlay modes
        if self._overlay_mode_open not in OVERLAY_MODES:
            raise ValueError(f"Invalid overlay_mode_open '{self._overlay_mode_open}'. Must be one of {OVERLAY_MODES}")
        if self._overlay_mode_closed not in OVERLAY_MODES:
            raise ValueError(f"Invalid overlay_mode_closed '{self._overlay_mode_closed}'. Must be one of {OVERLAY_MODES}")
        
        self._ws, self._server_metadata = self._wait_for_server()
    
    def _wait_for_server(self, timeout: float = 300) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        start_time = time.time()
        
        # Clear proxy settings
        for k in ("HTTP_PROXY", "http_proxy", "HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
            os.environ.pop(k, None)
        
        while True:
            if time.time() - start_time > timeout:
                raise TimeoutError(f"Failed to connect to SAM3 server within {timeout} seconds")
            
            try:
                conn = websockets.sync.client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    open_timeout=150,
                    ping_interval=20,
                    ping_timeout=20,
                )
                metadata = unpackb(conn.recv())
                logging.info(f"Connected to SAM3 server: {metadata}")
                return conn, metadata
            except ConnectionRefusedError:
                logging.info(f"Still waiting for SAM3 server {self._uri} ...")
                time.sleep(2)
    
    def close(self) -> None:
        try:
            self._ws.close()
        except Exception:
            pass
    
    def segment(
        self, 
        image: np.ndarray, 
        text_prompt: str,
        threshold: Optional[float] = None,
        mask_threshold: Optional[float] = None,
    ) -> Dict:
        """
        Request segmentation from SAM3 server.
        
        Args:
            image: np.ndarray of shape (H, W, 3), uint8 RGB image
            text_prompt: str, text description of object to segment
            threshold: Confidence threshold (defaults to open gripper threshold)
            mask_threshold: Mask binarization threshold (defaults to open gripper threshold)
            
        Returns:
            dict with masks, boxes, scores, num_masks
        """
        if threshold is None:
            threshold = self._threshold_open
        if mask_threshold is None:
            mask_threshold = self._mask_threshold_open
        
        request = {
            "type": "segment",
            "request_id": f"seg_{time.time()}",
            "image": image,
            "text_prompt": text_prompt,
            "threshold": threshold,
            "mask_threshold": mask_threshold,
        }
        
        self._ws.send(packb(request))
        response = unpackb(self._ws.recv())
        
        if not response.get("ok", False):
            error_msg = response.get("error", {}).get("message", "Unknown error")
            logging.warning(f"SAM3 segmentation failed: {error_msg}")
            return {
                "masks": np.zeros((0, image.shape[0], image.shape[1]), dtype=np.uint8),
                "boxes": np.zeros((0, 4), dtype=np.float32),
                "scores": np.array([]),
                "num_masks": 0
            }
        
        return response.get("data", {})
    
    # =========================================================================
    # Overlay Methods
    # =========================================================================
    
    def overlay_masks(self, image: np.ndarray, masks: np.ndarray, alpha: float = None) -> np.ndarray:
        """
        Overlay segmentation masks on image with red semi-transparent overlay.
        
        Args:
            image: np.ndarray of shape (H, W, 3), uint8 RGB image
            masks: np.ndarray of shape (N, H, W), binary masks
            alpha: float, overlay transparency (0-1), defaults to self._overlay_alpha
            
        Returns:
            np.ndarray of shape (H, W, 3), uint8 RGB image with overlay
        """
        if alpha is None:
            alpha = self._overlay_alpha
        
        if masks is None or len(masks) == 0:
            return image
        
        # Create a copy to avoid modifying the original
        result = image.copy().astype(np.float32)
        
        # Use red color for all masks
        red_color = (255, 0, 0)
        
        for mask in masks:
            mask_bool = mask.astype(bool)
            for c in range(3):
                result[:, :, c] = np.where(
                    mask_bool,
                    result[:, :, c] * (1 - alpha) + red_color[c] * alpha,
                    result[:, :, c]
                )
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def overlay_boxes(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """
        Draw bounding boxes on the image.
        
        Args:
            image: np.ndarray of shape (H, W, 3), uint8 RGB image
            boxes: np.ndarray of shape (N, 4), bounding boxes in format (x1, y1, x2, y2)
            
        Returns:
            np.ndarray of shape (H, W, 3), uint8 RGB image with boxes
        """
        if boxes is None or len(boxes) == 0:
            return image
        
        result = image.copy()
        red_color = (255, 0, 0)  # RGB color
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            # Draw rectangle (cv2 uses BGR, but we're working with RGB)
            cv2.rectangle(result, (x1, y1), (x2, y2), red_color, 2)
        
        return result
    
    def overlay_points(self, image: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """
        Overlay center points of segmentation masks onto the image.
        
        Args:
            image: np.ndarray of shape (H, W, 3), uint8 RGB image
            masks: np.ndarray of shape (N, H, W), binary masks
            
        Returns:
            np.ndarray of shape (H, W, 3), uint8 RGB image with points
        """
        if masks is None or len(masks) == 0:
            return image
        
        result = image.copy()
        red_color = (255, 0, 0)
        radius = 4
        
        for mask in masks:
            coords = np.column_stack(np.where(mask.astype(bool)))
            if coords.size == 0:
                continue
            center_y, center_x = coords.mean(axis=0)
            center = (int(center_x), int(center_y))
            cv2.circle(result, center, radius, red_color, -1)  # -1 = filled
        
        return result
    
    def overlay_mask_contour(self, image: np.ndarray, masks: np.ndarray) -> np.ndarray:
        """
        Overlay contour (outline) of segmentation masks onto the image (red).
        
        Args:
            image: np.ndarray of shape (H, W, 3), uint8 RGB image
            masks: np.ndarray of shape (N, H, W), binary masks
            
        Returns:
            np.ndarray of shape (H, W, 3), uint8 RGB image with contours
        """
        if masks is None or len(masks) == 0:
            return image
        
        result = image.copy()
        red_color = (255, 0, 0)
        
        for mask in masks:
            mask_uint8 = (mask.astype(bool)).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, red_color, 2)
        
        return result
    
    def overlay_crosshair(
        self, 
        image: np.ndarray, 
        masks: np.ndarray,
        line_length: int = 10,
        gap: int = 6
    ) -> np.ndarray:
        """
        Overlay a red center point with green crosshair (horizontal & vertical lines)
        that leave a small gap around the center dot.
        
        Args:
            image: np.ndarray of shape (H, W, 3), uint8 RGB image
            masks: np.ndarray of shape (N, H, W), binary masks (uses first mask)
            line_length: Length of each crosshair arm in pixels
            gap: Gap between the edge of the red dot and the start of the crosshair arm
            
        Returns:
            np.ndarray of shape (H, W, 3), uint8 RGB image with crosshair
        """
        if masks is None or len(masks) == 0:
            return image
        
        result = image.copy()
        
        # Use first mask only (assume single instance)
        mask = masks[0]
        coords = np.column_stack(np.where(mask.astype(bool)))
        if coords.size == 0:
            return result
        
        center_y, center_x = coords.mean(axis=0)
        x_c, y_c = int(center_x), int(center_y)
        
        # Draw center point (red)
        red_color = (255, 0, 0)
        cv2.circle(result, (x_c, y_c), 2, red_color, -1)
        
        # Draw crosshair arms (green)
        green_color = (0, 255, 0)
        # Horizontal lines (left and right) with gap
        cv2.line(result, (x_c - gap - line_length, y_c), (x_c - gap, y_c), green_color, 2)
        cv2.line(result, (x_c + gap, y_c), (x_c + gap + line_length, y_c), green_color, 2)
        # Vertical lines (up and down) with gap
        cv2.line(result, (x_c, y_c - gap - line_length), (x_c, y_c - gap), green_color, 2)
        cv2.line(result, (x_c, y_c + gap), (x_c, y_c + gap + line_length), green_color, 2)
        
        return result
    
    def overlay_mask_contour_darken(
        self, 
        image: np.ndarray, 
        masks: np.ndarray,
        darken_factor: float = 0.4
    ) -> np.ndarray:
        """
        Overlay contour of segmentation masks and darken the rest of the image.
        The target object(s) remain at original brightness while the background is darkened.
        
        Args:
            image: np.ndarray of shape (H, W, 3), uint8 RGB image
            masks: np.ndarray of shape (N, H, W), binary masks
            darken_factor: Factor to darken the background (0.0 = black, 1.0 = original)
            
        Returns:
            np.ndarray of shape (H, W, 3), uint8 RGB image with contours and darkened background
        """
        if masks is None or len(masks) == 0:
            return image
        
        # Combine all masks into one (union of all objects)
        combined_mask = np.any(masks.astype(bool), axis=0)  # shape (H, W)
        
        # Create result array
        img_float = image.astype(np.float32)
        
        # Create darkened version
        darkened = img_float * darken_factor
        
        # Apply mask: use original where mask is True, darkened elsewhere
        result = np.where(combined_mask[..., None], img_float, darkened)
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Now draw contours on top
        red_color = (255, 0, 0)
        for mask in masks:
            mask_uint8 = (mask.astype(bool)).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, red_color, 2)
        
        return result
    
    # =========================================================================
    # Main Interface
    # =========================================================================
    
    def segment_and_overlay(
        self, 
        image: np.ndarray, 
        text_prompt: str,
        grasped_state: str = "open",
        robot_prompt_closed: Optional[str] = None,
    ) -> np.ndarray:
        """
        Segment image and apply the configured overlay mode based on gripper state.
        
        Args:
            image: np.ndarray of shape (H, W, 3), uint8 RGB image
            text_prompt: str, text description of object to segment
            grasped_state: str, "open" for picking phase, "closed" for placing phase
            robot_prompt_closed: str, when gripper is closed, the previously targeted object
                                becomes the robot_prompt (what's being held). If None when
                                closed, falls back to robot_prompt_open.
            
        Returns:
            np.ndarray of shape (H, W, 3), uint8 RGB image with overlay
        """
        # Select configuration based on gripper state
        if grasped_state == "closed":
            overlay_mode = self._overlay_mode_closed
            threshold = self._threshold_closed
            mask_threshold = self._mask_threshold_closed
            # When gripper is closed, use the previously targeted object as robot_prompt
            # (it's now being held by the gripper)
            robot_prompt = robot_prompt_closed if robot_prompt_closed else self._robot_prompt_open
        else:
            overlay_mode = self._overlay_mode_open
            threshold = self._threshold_open
            mask_threshold = self._mask_threshold_open
            # When gripper is open, use the default robot prompt (e.g., "claw")
            robot_prompt = self._robot_prompt_open
        
        # Get segmentation for target object
        result = self.segment(image, text_prompt, threshold=threshold, mask_threshold=mask_threshold)
        masks = result.get("masks", np.zeros((0,) + image.shape[:2], dtype=np.uint8))
        boxes = result.get("boxes", np.zeros((0, 4), dtype=np.float32))
        
        if result.get("num_masks", 0) > 0:
            logging.debug(f"Found {result['num_masks']} objects for prompt '{text_prompt}' (gripper: {grasped_state})")
        else:
            logging.debug(f"No objects found for prompt '{text_prompt}' (gripper: {grasped_state})")
        
        # For contour_darken mode, also segment robot arm and combine masks
        if overlay_mode == "contour_darken":
            robot_result = self.segment(image, robot_prompt, threshold=threshold, mask_threshold=mask_threshold)
            robot_masks = robot_result.get("masks", np.zeros((0,) + image.shape[:2], dtype=np.uint8))
            
            if robot_result.get("num_masks", 0) > 0:
                logging.debug(f"Found {robot_result['num_masks']} objects for robot prompt '{robot_prompt}'")
                # Merge all robot detections into one mask
                if len(robot_masks) > 0:
                    robot_masks = robot_masks.any(axis=0, keepdims=True).astype(np.uint8)
            
            # Combine target masks and robot mask
            if len(masks) > 0 and len(robot_masks) > 0:
                masks = np.concatenate([masks, robot_masks], axis=0)
            elif len(robot_masks) > 0:
                masks = robot_masks
        
        # Dispatch to appropriate overlay method
        if overlay_mode == "mask":
            return self.overlay_masks(image, masks)
        elif overlay_mode == "box":
            return self.overlay_boxes(image, boxes)
        elif overlay_mode == "points":
            return self.overlay_points(image, masks)
        elif overlay_mode == "contour":
            return self.overlay_mask_contour(image, masks)
        elif overlay_mode == "crosshair":
            return self.overlay_crosshair(image, masks)
        elif overlay_mode == "contour_darken":
            return self.overlay_mask_contour_darken(image, masks)
        else:
            # Fallback to mask overlay
            return self.overlay_masks(image, masks)
    
    def _filter_top_score(
        self,
        masks: np.ndarray,
        boxes: np.ndarray,
        scores: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter masks and boxes to keep only the one with the highest score.
        
        Args:
            masks: np.ndarray of shape (N, H, W), binary masks
            boxes: np.ndarray of shape (N, 4), bounding boxes
            scores: np.ndarray of shape (N,), confidence scores
            
        Returns:
            Tuple of (filtered_masks, filtered_boxes) with only the top-scoring instance
        """
        if len(masks) == 0 or len(scores) == 0:
            return masks, boxes
        
        # Find the index of the highest score
        top_idx = int(np.argmax(scores))
        
        # Return only the top-scoring mask and box
        filtered_masks = masks[top_idx:top_idx+1]  # Keep shape (1, H, W)
        filtered_boxes = boxes[top_idx:top_idx+1]  # Keep shape (1, 4)
        
        return filtered_masks, filtered_boxes
    
    def segment_and_overlay_both(
        self,
        image: np.ndarray,
        grasp_object: str,
        place_target: Optional[str] = None,
        overlay_mode_grasp: Optional[str] = None,
        overlay_mode_place: Optional[str] = None,
        top_score_only: bool = True,
    ) -> np.ndarray:
        """
        Segment and overlay BOTH grasp object and place target on the image.
        This method is used when VLM has decomposed the task at the beginning
        and we want to show both targets simultaneously.
        
        Args:
            image: np.ndarray of shape (H, W, 3), uint8 RGB image
            grasp_object: str, text description of object to grasp
            place_target: str or None, text description of placement target
            overlay_mode_grasp: str or None, overlay mode for grasp object (default: use open gripper config)
            overlay_mode_place: str or None, overlay mode for place target (default: use closed gripper config)
            top_score_only: bool, if True only overlay the highest-scoring detection for each target
            
        Returns:
            np.ndarray of shape (H, W, 3), uint8 RGB image with overlays
        """
        # Use default overlay modes if not specified
        if overlay_mode_grasp is None:
            overlay_mode_grasp = self._overlay_mode_open
        if overlay_mode_place is None:
            overlay_mode_place = self._overlay_mode_closed
        
        # Segment grasp object
        grasp_result = self.segment(
            image, 
            grasp_object, 
            threshold=self._threshold_open, 
            mask_threshold=self._mask_threshold_open
        )
        grasp_masks = grasp_result.get("masks", np.zeros((0,) + image.shape[:2], dtype=np.uint8))
        grasp_boxes = grasp_result.get("boxes", np.zeros((0, 4), dtype=np.float32))
        grasp_scores = grasp_result.get("scores", np.array([]))
        
        if grasp_result.get("num_masks", 0) > 0:
            logging.debug(f"Found {grasp_result['num_masks']} objects for grasp prompt '{grasp_object}'")
            # Filter to keep only the highest-scoring detection
            if top_score_only and len(grasp_scores) > 0:
                # It's just one mask/boxes
                grasp_masks, grasp_boxes = self._filter_top_score(grasp_masks, grasp_boxes, grasp_scores)
                logging.debug(f"Filtered to top-scoring grasp object (score: {grasp_scores.max():.3f})")
        else:
            logging.debug(f"No objects found for grasp prompt '{grasp_object}'")
        
        # Segment place target if provided
        place_masks = np.zeros((0,) + image.shape[:2], dtype=np.uint8)
        place_boxes = np.zeros((0, 4), dtype=np.float32)
        if place_target:
            place_result = self.segment(
                image, 
                place_target, 
                threshold=self._threshold_closed, 
                mask_threshold=self._mask_threshold_closed
            )
            place_masks = place_result.get("masks", np.zeros((0,) + image.shape[:2], dtype=np.uint8))
            place_boxes = place_result.get("boxes", np.zeros((0, 4), dtype=np.float32))
            place_scores = place_result.get("scores", np.array([]))
            
            if place_result.get("num_masks", 0) > 0:
                logging.debug(f"Found {place_result['num_masks']} objects for place prompt '{place_target}'")
                # Filter to keep only the highest-scoring detection
                if top_score_only and len(place_scores) > 0:
                    # It's just one mask/boxes
                    place_masks, place_boxes = self._filter_top_score(place_masks, place_boxes, place_scores)
                    logging.debug(f"Filtered to top-scoring place target (score: {place_scores.max():.3f})")
            else:
                logging.debug(f"No objects found for place prompt '{place_target}'")
        
        # Also segment robot arm for contour_darken mode
        robot_masks = np.zeros((0,) + image.shape[:2], dtype=np.uint8)
        if overlay_mode_grasp == "contour_darken" or overlay_mode_place == "contour_darken":
            robot_result = self.segment(
                image, 
                self._robot_prompt_open, 
                threshold=self._threshold_open, 
                mask_threshold=self._mask_threshold_open
            )
            robot_masks = robot_result.get("masks", np.zeros((0,) + image.shape[:2], dtype=np.uint8))
            if len(robot_masks) > 0:
                robot_masks = robot_masks.any(axis=0, keepdims=True).astype(np.uint8)
        
        # Apply overlays
        result = image.copy()
        
        # For contour_darken mode, we need to combine all masks first
        if overlay_mode_grasp == "contour_darken" or overlay_mode_place == "contour_darken":
            # Combine all masks for darkening
            all_masks = []
            if len(grasp_masks) > 0:
                all_masks.append(grasp_masks)
            if len(place_masks) > 0:
                all_masks.append(place_masks)
            if len(robot_masks) > 0:
                all_masks.append(robot_masks)
            
            if all_masks:
                combined_masks = np.concatenate(all_masks, axis=0)
                result = self.overlay_mask_contour_darken(result, combined_masks)
        else:
            # Apply grasp object overlay
            if len(grasp_masks) > 0:
                result = self._apply_single_overlay(result, grasp_masks, grasp_boxes, overlay_mode_grasp)
            
            # Apply place target overlay
            if len(place_masks) > 0:
                result = self._apply_single_overlay(result, place_masks, place_boxes, overlay_mode_place)
        
        return result
    
    def _apply_single_overlay(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        boxes: np.ndarray,
        overlay_mode: str,
    ) -> np.ndarray:
        """
        Apply a single overlay mode to the image.
        
        Args:
            image: np.ndarray of shape (H, W, 3), uint8 RGB image
            masks: np.ndarray of shape (N, H, W), binary masks
            boxes: np.ndarray of shape (N, 4), bounding boxes
            overlay_mode: str, one of OVERLAY_MODES
            
        Returns:
            np.ndarray of shape (H, W, 3), uint8 RGB image with overlay
        """
        if overlay_mode == "mask":
            return self.overlay_masks(image, masks)
        elif overlay_mode == "box":
            return self.overlay_boxes(image, boxes)
        elif overlay_mode == "points":
            return self.overlay_points(image, masks)
        elif overlay_mode == "contour":
            return self.overlay_mask_contour(image, masks)
        elif overlay_mode == "crosshair":
            return self.overlay_crosshair(image, masks)
        elif overlay_mode == "contour_darken":
            return self.overlay_mask_contour_darken(image, masks)
        else:
            return self.overlay_masks(image, masks)


def extract_target_from_task(task_description: str) -> str:
    """
    Extract the target object from task description for SAM3 prompt.
    
    Examples:
        "stack the green cube on the yellow cube" -> "green cube"
        "put carrot on plate" -> "carrot"
        "pick up the red block" -> "red block"
    
    Args:
        task_description: str, the task description
        
    Returns:
        str, the extracted target object
    """
    # Convert to lowercase for processing
    text = task_description.lower().strip()
    
    # Remove common action verbs
    action_verbs = [
        "stack", "put", "place", "pick up", "pick", "move", 
        "lift", "grab", "take", "get", "push", "pull"
    ]
    
    for verb in action_verbs:
        if text.startswith(verb + " "):
            text = text[len(verb) + 1:]
            break
    
    # Remove "the" at the beginning
    if text.startswith("the "):
        text = text[4:]
    
    # Find prepositions that indicate the target boundary
    prepositions = [" on ", " onto ", " in ", " into ", " to ", " near ", " beside ", " next to "]
    
    for prep in prepositions:
        if prep in text:
            text = text.split(prep)[0]
            break
    
    # Clean up
    text = text.strip()
    
    # If text is empty, return the original
    if not text:
        return task_description
    
    return text
