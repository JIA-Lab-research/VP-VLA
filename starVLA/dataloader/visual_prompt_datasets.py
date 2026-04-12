# Visual Prompt Datasets for Training
# Extends LeRobotMixtureDataset with visual prompt loading and overlay

import json
import os
import pickle
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from starVLA.dataloader.gr00t_lerobot.datasets import LeRobotSingleDataset, LeRobotMixtureDataset
from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.dataloader.gr00t_lerobot.data_config import ROBOT_TYPE_CONFIG_MAP
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import ROBOT_TYPE_TO_EMBODIMENT_TAG, EmbodimentTag
from torch.utils.data import Dataset

from starVLA.dataloader.visual_prompt_utils import (
    validate_prompt_types,
    is_valid_mask,
    apply_visual_prompts,
    extract_visual_prompt_targets,
)


def collate_fn(batch):
    """Simple collate function that returns list of dicts."""
    return batch


def _resolve_npz_path(visual_prompt_dir: Path, task_name: str, episode_index: int) -> Optional[Path]:
    """
    Resolve the .npz path for an episode, supporting both chunked and flat layouts.
    
    Tries chunked path first (OXE-style: task_name/chunk-XXX/episode_XXXXXX.npz),
    then falls back to flat path (Robocasa-style: task_name/episode_XXXXXX.npz).
    """
    filename = f"episode_{episode_index:06d}.npz"
    # Try chunked layout first
    chunk_idx = episode_index // 1000
    chunked_path = visual_prompt_dir / task_name / f"chunk-{chunk_idx:03d}" / filename
    if chunked_path.exists():
        return chunked_path
    # Fall back to flat layout
    flat_path = visual_prompt_dir / task_name / filename
    if flat_path.exists():
        return flat_path
    return None


def load_visual_prompt_data(visual_prompt_dir: Path, task_name: str, episode_index: int) -> Optional[Dict]:
    """
    Load visual prompt data directly from npz file (no caching).
    
    With 24,000 episodes, cache hit rate is too low to justify memory overhead.
    NPZ loading is fast enough for direct disk access.
    
    Supports both chunked (OXE-style) and flat (Robocasa-style) directory layouts.
    
    Args:
        visual_prompt_dir: Root directory containing visual prompt .npz files
        task_name: Name of the task folder
        episode_index: Index of the episode
        
    Returns:
        Dict with visual prompt data, or None if not found
    """
    if visual_prompt_dir is None:
        return None
    
    visual_prompt_dir = Path(visual_prompt_dir)
    npz_path = _resolve_npz_path(visual_prompt_dir, task_name, episode_index)
    
    if npz_path is None:
        return None
    
    try:
        npz_data = np.load(npz_path, allow_pickle=True)
        
        # Parse vlm_results
        vlm_results_str = str(npz_data['vlm_results'])
        vlm_results = json.loads(vlm_results_str)
        
        # Build subtask_frames mapping: frame_idx -> vlm_entry
        subtask_frames = {r["frame"]: r for r in vlm_results}
        
        return {
            'episode_index': int(npz_data['episode_index']),
            'task_description': str(npz_data['task_description']),
            'num_frames': int(npz_data['num_frames']),
            'fps': float(npz_data['fps']),
            'target_masks': npz_data['target_masks'],
            'target_boxes': npz_data['target_boxes'],
            'target_scores': npz_data['target_scores'],
            'location_masks': npz_data['location_masks'],
            'location_boxes': npz_data['location_boxes'],
            'location_scores': npz_data['location_scores'],
            'vlm_results': vlm_results,
            'subtask_frames': subtask_frames,
        }
        
    except Exception as e:
        print(f"Error loading visual prompt data from {npz_path}: {e}")
        return None


def has_visual_prompts(visual_prompt_dir: Path, task_name: str, episode_index: int) -> bool:
    """Check if visual prompt data exists for an episode (chunked or flat layout)."""
    if visual_prompt_dir is None:
        return False
    visual_prompt_dir = Path(visual_prompt_dir)
    return _resolve_npz_path(visual_prompt_dir, task_name, episode_index) is not None


class VisualPromptMixtureDataset(LeRobotMixtureDataset):
    """
    Dataset that extends LeRobotMixtureDataset with visual prompt support.
    
    - Loads preprocessed visual prompt data from .npz files
    - Applies visual prompt overlays using CURRENT frame's masks/boxes
    - Uses closest subtask (before or at current frame) as the language prompt
    - Does NOT include VP prediction targets (those are in VisualPromptPredictionDataset)
    """
    
    def __init__(
        self,
        *args,
        visual_prompt_dir: str = None,
        target_object_prompt_type: str = "crosshair",
        target_location_prompt_type: str = "box",
        feed_both_images: bool = False,
        **kwargs
    ):
        """
        Initialize the dataset.
        
        Args:
            *args: Arguments passed to LeRobotMixtureDataset
            visual_prompt_dir: Directory containing visual prompt .npz files
            target_object_prompt_type: Type of overlay for target object (only "crosshair")
            target_location_prompt_type: Type of overlay for target location (only "box")
            feed_both_images: If True, include both original and overlayed images for primary views
            **kwargs: Keyword arguments passed to LeRobotMixtureDataset
        """
        super().__init__(*args, **kwargs)
        
        # Validate visual prompt types
        validate_prompt_types(target_object_prompt_type, target_location_prompt_type)
        
        self.visual_prompt_dir = Path(visual_prompt_dir) if visual_prompt_dir else None
        self.target_object_prompt_type = target_object_prompt_type
        self.target_location_prompt_type = target_location_prompt_type
        self.feed_both_images = feed_both_images
        
        print(f"Initialized VisualPromptMixtureDataset with:")
        print(f"  visual_prompt_dir: {visual_prompt_dir}")
        print(f"  target_object_prompt_type: {target_object_prompt_type}")
        print(f"  target_location_prompt_type: {target_location_prompt_type}")
        print(f"  feed_both_images: {feed_both_images}")
    
    def _get_task_name_from_dataset(self, dataset: LeRobotSingleDataset) -> str:
        """Extract task name from dataset path."""
        return dataset.dataset_name
    
    def _find_closest_subtask_frame(
        self, 
        subtask_frames: Dict[int, Dict], 
        current_frame: int
    ) -> Optional[Dict]:
        """
        Find the most recent subtask frame entry for the current frame.
        This determines which subtask is active at the current frame.
        
        Args:
            subtask_frames: Dict mapping frame_idx to vlm_entry
            current_frame: Current frame index
            
        Returns:
            The vlm_entry for the most recent subtask, or None
        """
        if not subtask_frames:
            return None
        
        # Find all frames <= current_frame
        valid_frames = [f for f in subtask_frames.keys() if f <= current_frame]
        if not valid_frames:
            return None
        
        # Get the most recent one
        closest_frame = max(valid_frames)
        return subtask_frames[closest_frame]
    
    def __getitem__(self, index: int) -> dict:
        """
        Get the data for a single trajectory and start index.
        
        Returns dict with:
            - image: List[PIL.Image] with visual prompt overlays using CURRENT frame's masks/boxes
            - lang: str, subtask from closest subtask frame (before or at current frame)
            - action: np.ndarray
        
        Note: VP prediction targets are NOT included here. They are in VisualPromptPredictionDataset.
        """
        max_retries = 10
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # Sample a step with limited retries to avoid infinite loop
                video_search_retries = 0
                max_video_search = 100
                while video_search_retries < max_video_search:
                    dataset, trajectory_id, step = self.sample_step(index)
                    key = dataset.modality_keys["video"][0].replace("video.", "")
                    video_path = dataset.get_video_path(trajectory_id, key)
                    if os.path.exists(video_path):
                        break
                    index = random.randint(0, len(self) - 1)
                    video_search_retries += 1
                
                if video_search_retries >= max_video_search:
                    raise RuntimeError(f"Could not find valid video file after {max_video_search} attempts")
                
                # Get raw data and apply transforms
                raw_data = dataset.get_step_data(trajectory_id, step)
                data = dataset.transforms(raw_data)
                
                # Get task name for visual prompt lookup
                task_name = self._get_task_name_from_dataset(dataset)
                
                # Load visual prompt data directly (no cache)
                vp_data = load_visual_prompt_data(self.visual_prompt_dir, task_name, trajectory_id)
                
                # Find closest subtask for language prompt (subtask that's active at current step)
                current_vlm_entry = None
                
                if vp_data is not None:
                    subtask_frames = vp_data.get('subtask_frames', {})
                    # Find closest subtask frame <= current step
                    current_vlm_entry = self._find_closest_subtask_frame(subtask_frames, step)
                
                # Always use original task description as language prompt
                language = data[dataset.modality_keys["language"][0]][0]
                
                # Add two-image semantic context to language prompt when feeding both images
                if self.feed_both_images:
                    language = (
                        "You are given two images: the first is the original robot observation, "
                        "and the second has visual prompts overlaid highlighting the target object "
                        "and target location. " + language
                    )
                
                # Process video frames
                prim_images = []
                wrist_views = []
                
                for video_key in dataset.modality_keys["video"]:
                    image = data[video_key][0]  # Shape: (H, W, C)
                    image_original = image.copy()  # Keep original for dual image mode
                    
                    # Apply visual prompts to primary view (not wrist)
                    # Use CURRENT frame (step) for masks/boxes
                    if "wrist" not in video_key and vp_data is not None and current_vlm_entry is not None:
                        # Get target_object and target_location from current subtask's vlm_entry
                        frame_target_object = current_vlm_entry.get("target_object")
                        frame_target_location = current_vlm_entry.get("target_location")
                        
                        # Get masks and boxes for the CURRENT frame (step)
                        target_mask = None
                        target_box = None
                        
                        if frame_target_object is not None and step < len(vp_data['target_masks']):
                            target_mask = vp_data['target_masks'][step]
                        
                        if frame_target_location is not None and step < len(vp_data['location_boxes']):
                            # Check if location mask is valid for current frame
                            if step < len(vp_data['location_masks']) and is_valid_mask(vp_data['location_masks'][step]):
                                target_box = vp_data['location_boxes'][step]
                        
                        # Apply visual prompts to image
                        image = apply_visual_prompts(
                            image,
                            target_object_mask=target_mask if (target_mask is not None and is_valid_mask(target_mask)) else None,
                            target_location_box=target_box,
                            target_object_type=self.target_object_prompt_type,
                            target_location_type=self.target_location_prompt_type,
                        )
                    
                    # Resize to 224x224
                    if "wrist" not in video_key:
                        if self.feed_both_images:
                            # Always provide 2 images: original + overlayed (or original twice if no overlay)
                            pil_image_original = Image.fromarray(image_original).resize((224, 224))
                            prim_images.append(pil_image_original)
                        pil_image = Image.fromarray(image).resize((224, 224))
                        prim_images.append(pil_image)
                    else:
                        # Wrist views: always use original (no overlays applied)
                        pil_image = Image.fromarray(image).resize((224, 224))
                        wrist_views.append(pil_image)
                
                all_images = prim_images + wrist_views
                
                # Get action data
                action = []
                for action_key in dataset.modality_keys["action"]:
                    action.append(data[action_key])
                action = np.concatenate(action, axis=1).astype(np.float16)
                
                # Build result dict (NO VP prediction targets - those are in VisualPromptPredictionDataset)
                result = {
                    "action": action,
                    "image": all_images,
                    "lang": language,
                }
                
                # Add state if configured
                if self.data_cfg is not None and self.data_cfg.get("include_state", False) not in ["False", False]:
                    state = []
                    for state_key in dataset.modality_keys["state"]:
                        state.append(data[state_key])
                    state = np.concatenate(state, axis=1).astype(np.float16)
                    result["state"] = state
                
                return result
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1}/{max_retries} failed for index {index}: {e}")
                    print(f"Retrying with new sample...")
                    index = random.randint(0, len(self) - 1)
                else:
                    print(f"All {max_retries} attempts failed for index {index}")
                    print(f"Last error: {last_exception}")
                    raise last_exception


def make_VisualPromptSingleDataset(
    data_root_dir: Path | str,
    data_name: str,
    robot_type: str,
    delete_pause_frame: bool = False,
    data_cfg: dict | None = None,
) -> LeRobotSingleDataset:
    """
    Make a LeRobotSingleDataset object (same as original, reused for visual prompt dataset).
    """
    data_config = ROBOT_TYPE_CONFIG_MAP[robot_type]
    modality_config = data_config.modality_config()
    transforms = data_config.transform()
    dataset_path = data_root_dir / data_name
    
    if robot_type not in ROBOT_TYPE_TO_EMBODIMENT_TAG:
        print(f"Warning: Robot type {robot_type} not found in ROBOT_TYPE_TO_EMBODIMENT_TAG, using {EmbodimentTag.NEW_EMBODIMENT} as default")
        embodiment_tag = EmbodimentTag.NEW_EMBODIMENT
    else:
        embodiment_tag = ROBOT_TYPE_TO_EMBODIMENT_TAG[robot_type]
    
    video_backend = data_cfg.get("video_backend", "decord") if data_cfg else "decord"
    
    return LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        transforms=transforms,
        embodiment_tag=embodiment_tag,
        video_backend=video_backend,
        delete_pause_frame=delete_pause_frame,
        data_cfg=data_cfg,
    )


def get_vla_dataset(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> VisualPromptMixtureDataset:
    """
    Get a VisualPromptMixtureDataset object.
    
    Args:
        data_cfg: Dataset configuration with:
            - data_root_dir: Root directory of original dataset
            - data_mix: Name of dataset mixture
            - visual_prompt_dir: Directory containing visual prompt .npz files
            - target_object_prompt_type: Type of overlay for target object (default: "crosshair")
            - target_location_prompt_type: Type of overlay for target location (default: "box")
            - feed_both_images: If True, include both original and overlayed images for primary views (default: False)
        mode: "train" or "val"
        balance_dataset_weights: Whether to balance dataset weights
        balance_trajectory_weights: Whether to balance trajectory weights
        seed: Random seed
        
    Returns:
        VisualPromptMixtureDataset object
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    
    # Visual prompt configuration
    visual_prompt_dir = data_cfg.get("visual_prompt_dir", None)
    target_object_prompt_type = data_cfg.get("target_object_prompt_type", "crosshair")
    target_location_prompt_type = data_cfg.get("target_location_prompt_type", "box")
    feed_both_images = data_cfg.get("feed_both_images", False)
    
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    included_datasets, filtered_mixture_spec = set(), []
    
    for d_name, d_weight, robot_type in mixture_spec:
        dataset_key = (d_name, robot_type)
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue
        
        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))
    
    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset_mixture.append((
            make_VisualPromptSingleDataset(
                Path(data_root_dir), 
                d_name, 
                robot_type, 
                delete_pause_frame=delete_pause_frame, 
                data_cfg=data_cfg
            ), 
            d_weight
        ))
    
    return VisualPromptMixtureDataset(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        visual_prompt_dir=visual_prompt_dir,
        target_object_prompt_type=target_object_prompt_type,
        target_location_prompt_type=target_location_prompt_type,
        feed_both_images=feed_both_images,
        **kwargs,
    )


# ============================================================================
# Visual Prompt Prediction Dataset
# For VP prediction training - only samples from subtask-change frames
# ============================================================================

def format_visual_prompt_instruction(subtask: str, has_target_object: bool, has_target_location: bool) -> str:
    """
    Format the instruction for visual prompt location prediction (Qwen3-VL style).
    
    Args:
        subtask: The current subtask description
        has_target_object: Whether target object is present
        has_target_location: Whether target location is present
        
    Returns:
        Formatted instruction string
    """
    instruction = f"Your task is: {subtask}. "
    
    if has_target_object and has_target_location:
        instruction += 'Locate the target object (marked with crosshair) and target location (marked with bounding box). '
        instruction += 'Report the crosshair center point and bounding box coordinates in JSON format like: '
        instruction += '[{"point_2d": [x, y], "label": "target_object"}, {"bbox_2d": [x1, y1, x2, y2], "label": "target_location"}]'
    elif has_target_object:
        instruction += 'Locate the target object (marked with crosshair). '
        instruction += 'Report the point coordinates in JSON format like: [{"point_2d": [x, y], "label": "target_object"}]'
    elif has_target_location:
        instruction += 'Locate the target location (marked with bounding box). '
        instruction += 'Report the bbox coordinates in JSON format like: [{"bbox_2d": [x1, y1, x2, y2], "label": "target_location"}]'
    
    return instruction


def format_visual_prompt_answer(
    target_object_loc: Optional[Tuple[int, int]], 
    target_location_bbox: Optional[Tuple[int, int, int, int]]
) -> str:
    """
    Format the ground truth answer for visual prompt location in JSON format (Qwen3-VL style).
    Coordinates are in [0, 1000] scale.
    
    Args:
        target_object_loc: (x, y) coordinates in [0, 1000] scale or None
        target_location_bbox: [x1, y1, x2, y2] coordinates in [0, 1000] scale or None
        
    Returns:
        Formatted JSON answer string
    """
    result = []
    
    if target_object_loc is not None:
        x, y = target_object_loc
        result.append({"point_2d": [x, y], "label": "target_object"})
    
    if target_location_bbox is not None:
        x1, y1, x2, y2 = target_location_bbox
        result.append({"bbox_2d": [x1, y1, x2, y2], "label": "target_location"})
    
    return json.dumps(result)


class VisualPromptPredictionDataset(Dataset):
    """
    Dataset for VP prediction training.
    
    - Only samples from subtask-change frames
    - Loads pre-extracted frames for fast random access
    - Returns VLM-format data: image, instruction, answer
    - Caches subtask index to file for faster subsequent loads
    """
    
    def __init__(
        self,
        visual_prompt_dir: str,
        extracted_frames_dir: str,
        target_object_prompt_type: str = "crosshair",
        target_location_prompt_type: str = "box",
        index_cache_file: str = None,
        use_chunked_frames: bool = False,
    ):
        """
        Initialize the dataset.
        
        Args:
            visual_prompt_dir: Directory containing visual prompt .npz files
            extracted_frames_dir: Directory containing pre-extracted frame images
            target_object_prompt_type: Type of overlay for target object (only "crosshair")
            target_location_prompt_type: Type of overlay for target location (only "box")
            index_cache_file: Path to cache file for subtask index. If None, defaults to
                              {visual_prompt_dir}/subtask_index_cache.json
            use_chunked_frames: If True, compute chunk dir from episode index (episode // 1000).
                                If False (default), always use chunk-000.
        """
        self.visual_prompt_dir = Path(visual_prompt_dir)
        self.extracted_frames_dir = Path(extracted_frames_dir)
        self.target_object_prompt_type = target_object_prompt_type
        self.target_location_prompt_type = target_location_prompt_type
        self.use_chunked_frames = use_chunked_frames
        
        # Set default cache file path
        if index_cache_file is None:
            self.index_cache_file = self.visual_prompt_dir / "subtask_index_cache.pkl"
        else:
            self.index_cache_file = Path(index_cache_file)
        
        # Validate visual prompt types
        validate_prompt_types(target_object_prompt_type, target_location_prompt_type)
        
        # Load or build index of all subtask-change frames
        self.subtask_samples = self._load_or_build_subtask_index()
        print(f"Loaded {len(self.subtask_samples)} subtask-change frames")
    
    def _get_frame_path(self, task_name: str, episode: int, frame: int) -> Path:
        """Get path to extracted frame jpg file."""
        if self.use_chunked_frames:
            chunk_dir = f"chunk-{episode // 1000:03d}"
        else:
            chunk_dir = "chunk-000"
        return (self.extracted_frames_dir / task_name / chunk_dir / 
                "observation.images.ego_view" / f"episode_{episode:06d}" / 
                f"frame_{frame:06d}.jpg")
    
    def _load_or_build_subtask_index(self) -> List[Tuple[str, int, int]]:
        """
        Load subtask index from cache file if exists, otherwise build and save it.
        
        Returns:
            List of tuples (task_name, episode_idx, frame_idx)
        """
        # Try to load from cache
        if self.index_cache_file.exists():
            print(f"Loading subtask index from cache: {self.index_cache_file}")
            try:
                with open(self.index_cache_file, 'rb') as f:
                    cache_data = pickle.load(f)
                
                # Validate cache metadata
                if (cache_data.get('visual_prompt_dir') == str(self.visual_prompt_dir) and
                    cache_data.get('extracted_frames_dir') == str(self.extracted_frames_dir)):
                    samples = cache_data['samples']
                    print(f"Cache loaded successfully with {len(samples)} samples")
                    return samples
                else:
                    print("Cache metadata mismatch, rebuilding index...")
            except Exception as e:
                print(f"Error loading cache: {e}, rebuilding index...")
        
        # Build index from scratch
        samples = self._build_subtask_index()
        
        # Save to cache
        self._save_subtask_index(samples)
        
        return samples
    
    def _save_subtask_index(self, samples: List[Tuple[str, int, int]]) -> None:
        """Save subtask index to pickle cache file."""
        cache_data = {
            'visual_prompt_dir': str(self.visual_prompt_dir),
            'extracted_frames_dir': str(self.extracted_frames_dir),
            'num_samples': len(samples),
            'samples': samples,  # List of (task_name, episode_idx, frame_idx)
        }
        
        try:
            with open(self.index_cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            print(f"Subtask index saved to cache: {self.index_cache_file}")
        except Exception as e:
            print(f"Warning: Failed to save cache: {e}")
    
    def _build_subtask_index(self) -> List[Tuple[str, int, int]]:
        """
        Build list of (task_name, episode_idx, frame_idx) for all subtask frames.
        This iterates through all episodes, so it's slow. Use cache when possible.
        
        Returns:
            List of tuples (task_name, episode_idx, frame_idx)
        """
        print("Building subtask frame index (this may take a while for large datasets)...")
        samples = []
        
        # Get list of all task directories
        task_dirs = sorted([d for d in self.visual_prompt_dir.iterdir() if d.is_dir()])
        
        # Iterate through all task directories with progress
        for task_dir in task_dirs:
            task_name = task_dir.name
            # Search both flat (Robocasa-style) and chunked (OXE-style) layouts
            npz_files = sorted(task_dir.glob("episode_*.npz"))
            npz_files += sorted(task_dir.glob("chunk-*/episode_*.npz"))
            
            for npz_file in npz_files:
                # Extract episode index from filename
                episode_str = npz_file.stem.replace("episode_", "")
                try:
                    episode_idx = int(episode_str)
                except ValueError:
                    continue
                
                # Load npz to get subtask frames
                vp_data = load_visual_prompt_data(self.visual_prompt_dir, task_name, episode_idx)
                if vp_data is None:
                    continue
                
                subtask_frames = vp_data.get('subtask_frames', {})
                for frame_idx in subtask_frames.keys():
                    # Verify the extracted frame exists
                    frame_path = self._get_frame_path(task_name, episode_idx, frame_idx)
                    if frame_path.exists():
                        samples.append((task_name, episode_idx, frame_idx))
            
            print(f"  Processed {task_name}: {len(npz_files)} episodes")
        
        print(f"Built index with {len(samples)} subtask-change frames")
        return samples
    
    def __len__(self) -> int:
        return len(self.subtask_samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a VP prediction sample.
        
        Returns dict with:
            - image: PIL.Image with visual prompts applied
            - instruction: str, formatted instruction for VP prediction
            - answer: str, formatted JSON answer
        """
        task_name, episode_idx, frame_idx = self.subtask_samples[idx]
        
        # Load visual prompt data
        vp_data = load_visual_prompt_data(self.visual_prompt_dir, task_name, episode_idx)
        if vp_data is None:
            raise RuntimeError(f"Failed to load VP data for {task_name} episode {episode_idx}")
        
        vlm_entry = vp_data['subtask_frames'].get(frame_idx)
        if vlm_entry is None:
            raise RuntimeError(f"Frame {frame_idx} not in subtask_frames for {task_name} episode {episode_idx}")
        
        # Load image from extracted frames
        frame_path = self._get_frame_path(task_name, episode_idx, frame_idx)
        image = np.array(Image.open(frame_path))
        
        # Get target info from vlm_entry
        frame_target_object = vlm_entry.get("target_object")
        frame_target_location = vlm_entry.get("target_location")
        subtask = vlm_entry.get("subtask", "")
        
        # Get masks and boxes for this frame
        target_mask = None
        target_box = None
        
        if frame_target_object is not None and frame_idx < len(vp_data['target_masks']):
            target_mask = vp_data['target_masks'][frame_idx]
            if not is_valid_mask(target_mask):
                target_mask = None
        
        if frame_target_location is not None and frame_idx < len(vp_data['location_boxes']):
            if frame_idx < len(vp_data['location_masks']) and is_valid_mask(vp_data['location_masks'][frame_idx]):
                target_box = vp_data['location_boxes'][frame_idx]
        
        # Apply visual prompts to image
        image_with_vp = apply_visual_prompts(
            image,
            target_object_mask=target_mask,
            target_location_box=target_box,
            target_object_type=self.target_object_prompt_type,
            target_location_type=self.target_location_prompt_type,
        )
        
        # Extract prediction targets (in 0-1000 scale)
        orig_h, orig_w = image.shape[:2]
        image_size = (orig_w, orig_h)
        
        target_object_loc, target_location_bbox = extract_visual_prompt_targets(
            target_mask, target_box, image_size
        )
        
        # Determine what we have
        has_target_object = target_object_loc is not None
        has_target_location = target_location_bbox is not None
        
        # Format instruction and answer
        instruction = format_visual_prompt_instruction(subtask, has_target_object, has_target_location)
        answer = format_visual_prompt_answer(target_object_loc, target_location_bbox)
        
        # Resize image to 224x224
        pil_image_overlayed = Image.fromarray(image_with_vp).resize((224, 224))
        
        return {
            "image": pil_image_overlayed,
            "instruction": instruction,
            "answer": answer,
            "task_name": task_name,
            "episode_idx": episode_idx,
            "frame_idx": frame_idx,
        }


def vp_collate_fn(batch):
    """Collate function for VP prediction dataset."""
    return batch


def get_vp_prediction_dataset(data_cfg: dict) -> VisualPromptPredictionDataset:
    """
    Get a VisualPromptPredictionDataset object.
    
    Args:
        data_cfg: Dataset configuration with:
            - visual_prompt_dir: Directory containing visual prompt .npz files
            - extracted_frames_dir: Directory containing pre-extracted frame images
            - target_object_prompt_type: Type of overlay for target object (default: "crosshair")
            - target_location_prompt_type: Type of overlay for target location (default: "box")
            - index_cache_file: Optional path to cache file for subtask index (default: auto)
            - use_chunked_frames: If True, compute chunk from episode index (default: False)
        
    Returns:
        VisualPromptPredictionDataset object
    """
    visual_prompt_dir = data_cfg.visual_prompt_dir
    extracted_frames_dir = data_cfg.extracted_frames_dir
    target_object_prompt_type = data_cfg.get("target_object_prompt_type", "crosshair")
    target_location_prompt_type = data_cfg.get("target_location_prompt_type", "box")
    index_cache_file = data_cfg.get("index_cache_file", None)
    use_chunked_frames = data_cfg.get("use_chunked_frames", False)
    
    return VisualPromptPredictionDataset(
        visual_prompt_dir=visual_prompt_dir,
        extracted_frames_dir=extracted_frames_dir,
        target_object_prompt_type=target_object_prompt_type,
        target_location_prompt_type=target_location_prompt_type,
        index_cache_file=index_cache_file,
        use_chunked_frames=use_chunked_frames,
    )


# ============================================================================
# Inline VP Dataset
# Extends VisualPromptMixtureDataset to include VP prediction fields inline.
# VP samples are drawn from randomly sampled training frames (all frames),
# removing the need for a separate VP prediction dataset/dataloader.
# ============================================================================

class VisualPromptMixtureDatasetWithInlineVP(VisualPromptMixtureDataset):
    """
    Dataset that extends VisualPromptMixtureDataset to also return VP prediction
    fields inline in each sample.
    
    When valid visual prompt data exists for a randomly sampled frame, the returned
    dict includes VP prediction targets (has_vp, vp_instruction, vp_answer,
    vp_image_overlayed, vp_image_original). This removes the need for a separate
    VP prediction dataset/dataloader and pre-extracted frames.
    
    Samples without valid VP data simply have has_vp=False and no VP fields.
    The training loop can filter samples with has_vp=True for VP training.
    """
    
    def __init__(
        self,
        *args,
        **kwargs
    ):
        """
        Initialize the dataset.
        
        Args:
            *args: Arguments passed to VisualPromptMixtureDataset
            **kwargs: Keyword arguments passed to VisualPromptMixtureDataset
        """
        super().__init__(*args, **kwargs)
    
    def __getitem__(self, index: int) -> dict:
        """
        Get data for a single trajectory step, with inline VP prediction fields.
        
        Returns dict with standard VLA fields:
            - image: List[PIL.Image]
            - lang: str
            - action: np.ndarray
        Plus inline VP prediction fields (when valid VP data exists):
            - has_vp: bool, always present
            - vp_instruction: str (only if has_vp=True)
            - vp_answer: str (only if has_vp=True)
            - vp_image_overlayed: PIL.Image (only if has_vp=True)
            - vp_image_original: PIL.Image (only if has_vp=True)
        """
        max_retries = 10
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                # Sample a step with limited retries to avoid infinite loop
                video_search_retries = 0
                max_video_search = 100
                while video_search_retries < max_video_search:
                    dataset, trajectory_id, step = self.sample_step(index)
                    key = dataset.modality_keys["video"][0].replace("video.", "")
                    video_path = dataset.get_video_path(trajectory_id, key)
                    if os.path.exists(video_path):
                        break
                    index = random.randint(0, len(self) - 1)
                    video_search_retries += 1
                
                if video_search_retries >= max_video_search:
                    raise RuntimeError(f"Could not find valid video file after {max_video_search} attempts")
                
                # Get raw data and apply transforms
                raw_data = dataset.get_step_data(trajectory_id, step)
                data = dataset.transforms(raw_data)
                
                # Get task name for visual prompt lookup
                task_name = self._get_task_name_from_dataset(dataset)
                
                # Load visual prompt data directly (no cache)
                vp_data = load_visual_prompt_data(self.visual_prompt_dir, task_name, trajectory_id)
                
                # Find closest subtask for visual prompt overlay
                current_vlm_entry = None
                
                if vp_data is not None:
                    subtask_frames = vp_data.get('subtask_frames', {})
                    # Find closest subtask frame <= current step
                    current_vlm_entry = self._find_closest_subtask_frame(subtask_frames, step)
                
                # Always use original task description as language prompt
                original_language = data[dataset.modality_keys["language"][0]][0]
                language = original_language
                
                # Add two-image semantic context to language prompt when feeding both images
                if self.feed_both_images:
                    language = (
                        "You are given two images: the first is the original robot observation, "
                        "and the second has visual prompts overlaid highlighting the target object "
                        "and target location. " + language
                    )
                
                # Process video frames -- also track VP-relevant data from primary view
                prim_images = []
                wrist_views = []
                
                # VP tracking variables (from first primary view only)
                _vp_target_mask = None      # valid target mask used for overlay
                _vp_target_box = None       # valid target box used for overlay
                _vp_original_np = None      # original primary image (numpy, original resolution)
                _vp_overlayed_pil = None    # overlayed primary image (PIL 224x224)
                _vp_original_pil = None     # original primary image (PIL 224x224)
                
                for video_key in dataset.modality_keys["video"]:
                    image = data[video_key][0]  # Shape: (H, W, C)
                    image_original = image.copy()  # Keep original for dual image mode
                    
                    # Apply visual prompts to primary view (not wrist)
                    # Use CURRENT frame (step) for masks/boxes
                    if "wrist" not in video_key and vp_data is not None and current_vlm_entry is not None:
                        # Get target_object and target_location from current subtask's vlm_entry
                        frame_target_object = current_vlm_entry.get("target_object")
                        frame_target_location = current_vlm_entry.get("target_location")
                        
                        # Get masks and boxes for the CURRENT frame (step)
                        target_mask = None
                        target_box = None
                        
                        if frame_target_object is not None and step < len(vp_data['target_masks']):
                            target_mask = vp_data['target_masks'][step]
                        
                        if frame_target_location is not None and step < len(vp_data['location_boxes']):
                            # Check if location mask is valid for current frame
                            if step < len(vp_data['location_masks']) and is_valid_mask(vp_data['location_masks'][step]):
                                target_box = vp_data['location_boxes'][step]
                        
                        # Determine valid mask for overlay
                        valid_mask = target_mask if (target_mask is not None and is_valid_mask(target_mask)) else None
                        
                        # Apply visual prompts to image
                        image = apply_visual_prompts(
                            image,
                            target_object_mask=valid_mask,
                            target_location_box=target_box,
                            target_object_type=self.target_object_prompt_type,
                            target_location_type=self.target_location_prompt_type,
                        )
                        
                        # Save VP data from first primary view for inline VP prediction
                        if _vp_original_np is None:
                            _vp_target_mask = valid_mask
                            _vp_target_box = target_box
                            _vp_original_np = image_original
                    
                    # Resize to 224x224
                    if "wrist" not in video_key:
                        if self.feed_both_images:
                            # Always provide 2 images: original + overlayed (or original twice if no overlay)
                            pil_image_original = Image.fromarray(image_original).resize((224, 224))
                            prim_images.append(pil_image_original)
                        pil_image = Image.fromarray(image).resize((224, 224))
                        prim_images.append(pil_image)
                        
                        # Save first primary view PIL images for VP prediction
                        if _vp_overlayed_pil is None:
                            _vp_overlayed_pil = pil_image
                            _vp_original_pil = Image.fromarray(image_original).resize((224, 224))
                    else:
                        # Wrist views: always use original (no overlays applied)
                        pil_image = Image.fromarray(image).resize((224, 224))
                        wrist_views.append(pil_image)
                
                all_images = prim_images + wrist_views
                
                # Get action data
                action = []
                for action_key in dataset.modality_keys["action"]:
                    action.append(data[action_key])
                action = np.concatenate(action, axis=1).astype(np.float16)
                
                # Build result dict with standard VLA fields
                result = {
                    "action": action,
                    "image": all_images,
                    "lang": language,
                }
                
                # Add state if configured
                if self.data_cfg is not None and self.data_cfg.get("include_state", False) not in ["False", False]:
                    state = []
                    for state_key in dataset.modality_keys["state"]:
                        state.append(data[state_key])
                    state = np.concatenate(state, axis=1).astype(np.float16)
                    result["state"] = state
                
                # === Inline VP prediction fields ===
                has_vp = False
                
                if _vp_original_np is not None and (_vp_target_mask is not None or _vp_target_box is not None):
                    # Extract VP prediction targets (in 0-1000 scale)
                    orig_h, orig_w = _vp_original_np.shape[:2]
                    image_size = (orig_w, orig_h)
                    
                    target_object_loc, target_location_bbox = extract_visual_prompt_targets(
                        _vp_target_mask, _vp_target_box, image_size
                    )
                    
                    has_target_object = target_object_loc is not None
                    has_target_location = target_location_bbox is not None
                    
                    if has_target_object or has_target_location:
                        has_vp = True
                        
                        # Use raw subtask text for VP instruction (not the prefixed VLA language)
                        vp_subtask_text = None
                        if current_vlm_entry is not None:
                            vp_subtask_text = current_vlm_entry.get("subtask", None)
                        if vp_subtask_text is None:
                            vp_subtask_text = original_language
                        
                        result["vp_instruction"] = format_visual_prompt_instruction(
                            vp_subtask_text, has_target_object, has_target_location
                        )
                        result["vp_answer"] = format_visual_prompt_answer(
                            target_object_loc, target_location_bbox
                        )
                        result["vp_image_overlayed"] = _vp_overlayed_pil
                        result["vp_image_original"] = _vp_original_pil
                
                result["has_vp"] = has_vp
                
                return result
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    print(f"Attempt {attempt + 1}/{max_retries} failed for index {index}: {e}")
                    print(f"Retrying with new sample...")
                    index = random.randint(0, len(self) - 1)
                else:
                    print(f"All {max_retries} attempts failed for index {index}")
                    print(f"Last error: {last_exception}")
                    raise last_exception


def get_vla_dataset_with_inline_vp(
    data_cfg: dict,
    mode: str = "train",
    balance_dataset_weights: bool = False,
    balance_trajectory_weights: bool = False,
    seed: int = 42,
    **kwargs: dict,
) -> VisualPromptMixtureDatasetWithInlineVP:
    """
    Get a VisualPromptMixtureDatasetWithInlineVP object.
    
    Same as get_vla_dataset but creates the inline-VP variant that includes
    VP prediction fields in each sample (when valid VP data exists).
    
    Args:
        data_cfg: Dataset configuration (same as get_vla_dataset)
        mode: "train" or "val"
        balance_dataset_weights: Whether to balance dataset weights
        balance_trajectory_weights: Whether to balance trajectory weights
        seed: Random seed
        
    Returns:
        VisualPromptMixtureDatasetWithInlineVP object
    """
    data_root_dir = data_cfg.data_root_dir
    data_mix = data_cfg.data_mix
    delete_pause_frame = data_cfg.get("delete_pause_frame", False)
    
    # Visual prompt configuration
    visual_prompt_dir = data_cfg.get("visual_prompt_dir", None)
    target_object_prompt_type = data_cfg.get("target_object_prompt_type", "crosshair")
    target_location_prompt_type = data_cfg.get("target_location_prompt_type", "box")
    feed_both_images = data_cfg.get("feed_both_images", False)
    
    mixture_spec = DATASET_NAMED_MIXTURES[data_mix]
    included_datasets, filtered_mixture_spec = set(), []
    
    for d_name, d_weight, robot_type in mixture_spec:
        dataset_key = (d_name, robot_type)
        if dataset_key in included_datasets:
            print(f"Skipping Duplicate Dataset: `{(d_name, d_weight, robot_type)}`")
            continue
        
        included_datasets.add(dataset_key)
        filtered_mixture_spec.append((d_name, d_weight, robot_type))
    
    dataset_mixture = []
    for d_name, d_weight, robot_type in filtered_mixture_spec:
        dataset_mixture.append((
            make_VisualPromptSingleDataset(
                Path(data_root_dir), 
                d_name, 
                robot_type, 
                delete_pause_frame=delete_pause_frame, 
                data_cfg=data_cfg
            ), 
            d_weight
        ))
    
    return VisualPromptMixtureDatasetWithInlineVP(
        dataset_mixture,
        mode=mode,
        balance_dataset_weights=balance_dataset_weights,
        balance_trajectory_weights=balance_trajectory_weights,
        seed=seed,
        data_cfg=data_cfg,
        visual_prompt_dir=visual_prompt_dir,
        target_object_prompt_type=target_object_prompt_type,
        target_location_prompt_type=target_location_prompt_type,
        feed_both_images=feed_both_images,
        **kwargs,
    )


if __name__ == "__main__":
    # Test the visual prompt dataset
    import argparse
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_yaml", type=str, 
                       default="./examples/Robocasa_tabletop/train_files/starvla_cotrain_robocasa_gr1.yaml",
                       help="Path to YAML config")
    args = parser.parse_args()
    
    cfg = OmegaConf.load(args.config_yaml)
    vla_dataset_cfg = cfg.datasets.vla_data
    
    # Set visual prompt directory
    vla_dataset_cfg.visual_prompt_dir = "./playground/Datasets/visual_prompt_robocasa_by_frames"
    
    print("Creating VisualPromptMixtureDataset...")
    dataset = get_vla_dataset(data_cfg=vla_dataset_cfg)
    
    print(f"Dataset length: {len(dataset)}")
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=0,
        collate_fn=collate_fn,
    )
    
    print("Testing data loading...")
    for i, batch in enumerate(tqdm(train_dataloader, desc="Loading batches")):
        if i >= 10:
            break
        
        for sample in batch:
            print(f"  Sample keys: {sample.keys()}")
            print(f"    lang: {sample['lang'][:50] if len(sample['lang']) > 50 else sample['lang']}...")
            print(f"    action shape: {sample['action'].shape}")
            print(f"    num images: {len(sample['image'])}")
            break
    
    print("Test completed!")
