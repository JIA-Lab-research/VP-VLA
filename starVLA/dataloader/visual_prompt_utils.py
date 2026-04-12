# Visual Prompt Utilities for Training
# Provides overlay functions and coordinate utilities for visual prompt training

from typing import Optional, Tuple, List
import numpy as np
import cv2


# Supported visual prompt types
SUPPORTED_TARGET_OBJECT_TYPES = ["crosshair", "point"]
SUPPORTED_TARGET_LOCATION_TYPES = ["box"]


def validate_prompt_types(target_object_type: str, target_location_type: str) -> None:
    """
    Validate that the visual prompt types are supported.
    
    Args:
        target_object_type: Type of visual prompt for target object
        target_location_type: Type of visual prompt for target location
        
    Raises:
        ValueError: If unsupported visual prompt types are specified
    """
    if target_object_type not in SUPPORTED_TARGET_OBJECT_TYPES:
        raise ValueError(
            f"Unsupported target_object_prompt_type: '{target_object_type}'. "
            f"Only {SUPPORTED_TARGET_OBJECT_TYPES} are supported."
        )
    if target_location_type not in SUPPORTED_TARGET_LOCATION_TYPES:
        raise ValueError(
            f"Unsupported target_location_prompt_type: '{target_location_type}'. "
            f"Only {SUPPORTED_TARGET_LOCATION_TYPES} are supported."
        )


def is_valid_mask(mask: np.ndarray) -> bool:
    """
    Check if a mask has any positive (non-zero) pixels.
    
    Args:
        mask: np.ndarray of shape (H, W), binary mask
        
    Returns:
        bool: True if mask has at least one positive pixel
    """
    if mask is None:
        return False
    return np.any(mask > 0)


def get_crosshair_center(mask: np.ndarray) -> Optional[Tuple[float, float]]:
    """
    Compute the center point (centroid) of a mask.
    
    Args:
        mask: np.ndarray of shape (H, W), binary mask
        
    Returns:
        Tuple[float, float]: (x, y) center coordinates, or None if mask is invalid
    """
    if not is_valid_mask(mask):
        return None
    
    # Find all positive pixel coordinates
    coords = np.column_stack(np.where(mask.astype(bool)))
    if coords.size == 0:
        return None
    
    # Compute centroid (mean of coordinates)
    # np.where returns (row, col) = (y, x), so we need to swap
    center_y, center_x = coords.mean(axis=0)
    
    return (float(center_x), float(center_y))


def scale_coordinates_to_1000(
    coords: Tuple[float, ...], 
    image_size: Tuple[int, int]
) -> Tuple[int, ...]:
    """
    Scale coordinates to [0, 1000] range (Qwen3-VL format).
    
    Args:
        coords: Tuple of pixel coordinates (can be 2 values for point, or 4 for bbox)
        image_size: Tuple of (width, height)
        
    Returns:
        Tuple of scaled integer coordinates in [0, 1000] range
    """
    width, height = image_size
    
    if len(coords) == 2:
        # Point: (x, y)
        x, y = coords
        return (int(x / width * 1000), int(y / height * 1000))
    elif len(coords) == 4:
        # Bounding box: (x1, y1, x2, y2)
        x1, y1, x2, y2 = coords
        return (
            int(x1 / width * 1000), 
            int(y1 / height * 1000), 
            int(x2 / width * 1000), 
            int(y2 / height * 1000)
        )
    else:
        raise ValueError(f"Unexpected coordinates length: {len(coords)}")


def overlay_crosshair(
    image: np.ndarray,
    mask: np.ndarray,
    line_length: int = 10,
    gap: int = 6,
    center_color: Tuple[int, int, int] = (255, 0, 0),
    crosshair_color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    Overlay a crosshair on the image at the mask centroid.
    Draws a red center point with green crosshair lines.
    
    Args:
        image: np.ndarray of shape (H, W, 3), uint8 RGB image
        mask: np.ndarray of shape (H, W), binary mask
        line_length: Length of each crosshair arm in pixels
        gap: Gap between the center dot and the crosshair arms
        center_color: RGB color for center dot (default: red)
        crosshair_color: RGB color for crosshair lines (default: green)
        
    Returns:
        np.ndarray of shape (H, W, 3), uint8 RGB image with crosshair
    """
    if not is_valid_mask(mask):
        return image
    
    result = image.copy()
    
    # Get center coordinates
    center = get_crosshair_center(mask)
    if center is None:
        return result
    
    x_c, y_c = int(center[0]), int(center[1])
    
    # Draw center point (red by default)
    cv2.circle(result, (x_c, y_c), 2, center_color, -1)
    
    # Draw crosshair arms (green by default)
    # Horizontal lines (left and right) with gap
    cv2.line(result, (x_c - gap - line_length, y_c), (x_c - gap, y_c), crosshair_color, 2)
    cv2.line(result, (x_c + gap, y_c), (x_c + gap + line_length, y_c), crosshair_color, 2)
    # Vertical lines (up and down) with gap
    cv2.line(result, (x_c, y_c - gap - line_length), (x_c, y_c - gap), crosshair_color, 2)
    cv2.line(result, (x_c, y_c + gap), (x_c, y_c + gap + line_length), crosshair_color, 2)
    
    return result


def overlay_point(
    image: np.ndarray,
    mask: np.ndarray,
    radius: int = 4,
    color: Tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    """
    Overlay a single point (filled circle) on the image at the mask centroid.
    This is a simplified version of overlay_crosshair that draws only the
    red center dot without the green crosshair arms.
    
    Args:
        image: np.ndarray of shape (H, W, 3), uint8 RGB image
        mask: np.ndarray of shape (H, W), binary mask
        radius: Radius of the point in pixels (default: 4)
        color: RGB color for the point (default: red)
        
    Returns:
        np.ndarray of shape (H, W, 3), uint8 RGB image with point
    """
    if not is_valid_mask(mask):
        return image
    
    result = image.copy()
    
    # Get center coordinates
    center = get_crosshair_center(mask)
    if center is None:
        return result
    
    x_c, y_c = int(center[0]), int(center[1])
    
    # Draw filled circle at centroid
    cv2.circle(result, (x_c, y_c), radius, color, -1)
    
    return result


def overlay_box(
    image: np.ndarray,
    box: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw a bounding box on the image.
    
    Args:
        image: np.ndarray of shape (H, W, 3), uint8 RGB image
        box: np.ndarray of shape (4,), bounding box in format [x1, y1, x2, y2]
        color: RGB color for the box (default: red)
        thickness: Line thickness in pixels
        
    Returns:
        np.ndarray of shape (H, W, 3), uint8 RGB image with bounding box
    """
    if box is None or len(box) != 4:
        return image
    
    # Check if box is valid (has non-zero area)
    x1, y1, x2, y2 = box
    if x2 <= x1 or y2 <= y1:
        return image
    
    result = image.copy()
    
    # Convert to integers
    x1, y1, x2, y2 = map(int, box)
    
    # Draw rectangle
    cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)
    
    return result


def apply_visual_prompts(
    image: np.ndarray,
    target_object_mask: Optional[np.ndarray] = None,
    target_location_box: Optional[np.ndarray] = None,
    target_object_type: str = "crosshair",
    target_location_type: str = "box",
) -> np.ndarray:
    """
    Apply visual prompts to an image based on the specified types.
    
    Args:
        image: np.ndarray of shape (H, W, 3), uint8 RGB image
        target_object_mask: np.ndarray of shape (H, W), binary mask for target object
        target_location_box: np.ndarray of shape (4,), bounding box for target location
        target_object_type: Type of overlay for target object ("crosshair" or "point")
        target_location_type: Type of overlay for target location (only "box" supported)
        
    Returns:
        np.ndarray of shape (H, W, 3), uint8 RGB image with visual prompts
        
    Raises:
        ValueError: If unsupported visual prompt types are specified
    """
    # Validate prompt types
    validate_prompt_types(target_object_type, target_location_type)
    
    result = image.copy()
    
    # Apply target object overlay
    if target_object_mask is not None and is_valid_mask(target_object_mask):
        if target_object_type == "crosshair":
            result = overlay_crosshair(result, target_object_mask)
        elif target_object_type == "point":
            result = overlay_point(result, target_object_mask)
    
    # Apply target location overlay (bounding box)
    if target_location_box is not None:
        result = overlay_box(result, target_location_box)
    
    return result


def extract_visual_prompt_targets(
    target_object_mask: Optional[np.ndarray],
    target_location_box: Optional[np.ndarray],
    image_size: Tuple[int, int],
) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int, int, int]]]:
    """
    Extract visual prompt prediction targets in Qwen3-VL format (0-1000 scale).
    
    Args:
        target_object_mask: np.ndarray of shape (H, W), binary mask for target object
        target_location_box: np.ndarray of shape (4,), bounding box for target location
        image_size: Tuple of (width, height)
        
    Returns:
        Tuple of:
            - target_object_location: (x, y) in [0, 1000] scale or None
            - target_location_bbox: [x1, y1, x2, y2] in [0, 1000] scale or None
    """
    target_object_location = None
    target_location_bbox = None
    
    # Extract crosshair center for target object
    if target_object_mask is not None and is_valid_mask(target_object_mask):
        center = get_crosshair_center(target_object_mask)
        if center is not None:
            target_object_location = scale_coordinates_to_1000(center, image_size)
    
    # Extract bounding box for target location
    if target_location_box is not None and len(target_location_box) == 4:
        x1, y1, x2, y2 = target_location_box
        # Check if box is valid
        if x2 > x1 and y2 > y1:
            target_location_bbox = scale_coordinates_to_1000(
                (float(x1), float(y1), float(x2), float(y2)), 
                image_size
            )
    
    return target_object_location, target_location_bbox
