# VLM Subtask Detection Server
# Provides subtask detection via websocket using Qwen3-VL
# Uses the golden logic from combined_vlm_sam.py

import asyncio
import logging
import traceback
import time
import argparse
import socket
import json
import re
import os
import sys
from typing import Tuple, Optional, Dict, List

import numpy as np
import torch
import websockets.asyncio.server
import websockets.frames

# Add the utility directory to path for msgpack_utils
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UTILITY_DIR = os.path.join(os.path.dirname(SCRIPT_DIR), "examples", "Robocasa_tabletop", "visual_prompt_utility")
sys.path.insert(0, UTILITY_DIR)

from msgpack_utils import packb, unpackb


# ===============================
# VLM SYSTEM MESSAGE AND PROMPT (from combined_vlm_sam.py)
# ===============================

VLM_SYSTEM_MESSAGE = (
    "You are a robotic manipulation reasoning assistant. "
    "Follow all rules strictly. Output VALID JSON ONLY."
)


def build_subtask_detection_prompt(
    task_description: str,
    current_subtask: str,
    next_subtask: Optional[str],
    gripper_state: Dict[str, str],
    gripper_state_changed: bool = False,
    previous_gripper_state: Optional[Dict[str, str]] = None,
) -> str:
    """
    Build the subtask detection prompt (golden logic from combined_vlm_sam.py).
    """
    gripper_text = (
        f"LEFT_GRIPPER = {gripper_state['left']}\n"
        f"RIGHT_GRIPPER = {gripper_state['right']}"
    )
    
    # Add gripper state change notification if it changed
    gripper_change_notification = ""
    if gripper_state_changed and previous_gripper_state is not None:
        prev_left = previous_gripper_state.get('left', 'UNKNOWN')
        prev_right = previous_gripper_state.get('right', 'UNKNOWN')
        curr_left = gripper_state['left']
        curr_right = gripper_state['right']
        
        changes = []
        if prev_left != curr_left:
            changes.append(f"LEFT_GRIPPER: {prev_left} → {curr_left}")
        if prev_right != curr_right:
            changes.append(f"RIGHT_GRIPPER: {prev_right} → {curr_right}")
        
        if changes:
            change_desc = "\n".join(changes)
            gripper_change_notification = f"""

⚠️ GRIPPER STATE CHANGE DETECTED:
{change_desc}

This gripper state change may indicate that the current subtask has been completed. 
Consider both the visual evidence and this gripper state change when deciding whether to proceed to the next subtask.
"""

    next_subtask_text = f"\nNEXT SUBTASK: {next_subtask}" if next_subtask else "\nNEXT SUBTASK: None (this is the last subtask)"

    return f"""
TASK:
{task_description}

CURRENT SUBTASK:
{current_subtask}{next_subtask_text}

VISUAL CONTEXT:
- Image A (if provided): the frame when the CURRENT subtask started
- Image B: the current frame

Use visual differences between Image A and Image B to judge whether the current subtask has been completed.

GRIPPER STATE (GROUND TRUTH):
{gripper_text}{gripper_change_notification}

YOUR RESPONSIBILITY:
Decide whether to:
- CONTINUE with the current subtask "{current_subtask}" (if it's still in progress)
- PROCEED to the next subtask (if the current subtask is completed)

IMPORTANT:
- Only proceed to the next subtask if the current subtask appears to be COMPLETED based on visual evidence.
- If the current subtask is still in progress, continue with it.
- Consider the gripper state change (if any) as additional evidence.

OUTPUT FORMAT RULES:
- target_object: Must be a NOUN only (e.g., "bottle", "drawer", "door", "cabinet")
- target_location: NOUN or NOUN PHRASE with spatial descriptors if applicable (e.g., "drawer interior", "cabinet shelf", "countertop surface")
- For PICKING tasks: Only specify target_object (the object being picked)
- For PLACING tasks: Specify both target_object (the object being placed) and target_location (where it's being placed). 
- For OTHER tasks (e.g., close, open, push): Only specify target_object (the object being manipulated)

OUTPUT JSON ONLY:
{{
  "reasoning": "explain your judgment based on visual evidence",
  "decision": "continue" or "proceed",
  "target_object": "<noun only>",
  "target_location": "<location for placing tasks, or null>"
}}
""".strip()


def build_task_decomposition_prompt(task_description: str) -> str:
    """
    Build a prompt for decomposing a task description into sequential subtasks.
    Text-only (no images needed).
    """
    return f"""TASK DESCRIPTION:
{task_description}

Decompose this robotic manipulation task into sequential atomic subtasks.
Each subtask should be one of:
- A "pick" action (e.g., "pick up the cup")
- A "place" action (e.g., "place the cup on the table")
- A manipulation action (e.g., "close the drawer", "open the microwave")

Rules:
- Keep subtask descriptions short and clear.
- Preserve the object and location names from the original task description.
- Output 2-4 subtasks at most.

OUTPUT JSON ONLY:
{{
  "subtasks": ["subtask 1", "subtask 2", ...]
}}""".strip()


def parse_task_decomposition_response(response: str) -> List[str]:
    """Parse VLM response to extract subtask list."""
    try:
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = response
        data = json.loads(json_str)
        subtasks = data.get("subtasks", [])
        if isinstance(subtasks, list) and len(subtasks) > 0:
            return [str(s) for s in subtasks]
    except (json.JSONDecodeError, AttributeError) as e:
        logging.warning(f"Failed to parse task decomposition response: {e}, response: {response}")
    return []


def parse_subtask_detection_response(response: str) -> Dict:
    """
    Parse VLM JSON response to extract subtask detection results.
    
    Args:
        response: VLM output text (should be JSON)
        
    Returns:
        Dict with reasoning, decision, target_object, target_location
    """
    try:
        # Find JSON object in response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
        else:
            json_str = response
        
        data = json.loads(json_str)
        
        return {
            "reasoning": data.get("reasoning", ""),
            "decision": data.get("decision", "continue"),
            "target_object": data.get("target_object", ""),
            "target_location": data.get("target_location"),
        }
        
    except (json.JSONDecodeError, AttributeError) as e:
        logging.warning(f"Failed to parse VLM response: {e}, response: {response}")
        return {
            "reasoning": f"Parse error: {e}",
            "decision": "continue",
            "target_object": "",
            "target_location": None,
        }


# ===============================
# VLM MODEL WRAPPER
# ===============================

class VLMModel:
    """Wrapper for Qwen3-VL model for subtask detection."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
        from PIL import Image
        
        self.device = device
        self.Image = Image
        
        logging.info(f"Loading Qwen3-VL model from {model_path}")
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model.eval()
        logging.info("Qwen3-VL model loaded successfully")
    
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
        max_new_tokens: int = 256,
    ) -> dict:
        """
        Detect subtask progress using VLM (golden logic from combined_vlm_sam.py).
        
        Args:
            reference_image: np.ndarray (H, W, 3), uint8 RGB - frame when subtask started (optional)
            current_image: np.ndarray (H, W, 3), uint8 RGB - current frame
            task_description: str - full task description
            current_subtask: str - current subtask being executed
            next_subtask: str or None - next subtask in sequence
            gripper_state: dict with left/right states
            gripper_state_changed: bool - whether gripper state changed
            previous_gripper_state: dict with previous left/right states (if changed)
            max_new_tokens: int - max tokens for generation
            
        Returns:
            dict with:
                - reasoning: str - VLM's reasoning
                - decision: str - "continue" or "proceed"
                - target_object: str - object to segment
                - target_location: str or None - location to segment
                - raw_response: str - full VLM response for debugging
        """
        # Build image content (reference + current, or just current)
        image_content = []
        if reference_image is not None:
            ref_pil = self.Image.fromarray(reference_image).convert("RGB")
            image_content.append({"type": "image", "image": ref_pil})
        
        curr_pil = self.Image.fromarray(current_image).convert("RGB")
        image_content.append({"type": "image", "image": curr_pil})
        
        # Build prompt
        prompt = build_subtask_detection_prompt(
            task_description=task_description,
            current_subtask=current_subtask,
            next_subtask=next_subtask,
            gripper_state=gripper_state,
            gripper_state_changed=gripper_state_changed,
            previous_gripper_state=previous_gripper_state,
        )
        
        # Build messages
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": VLM_SYSTEM_MESSAGE}],
            },
            {
                "role": "user",
                "content": image_content + [{"type": "text", "text": prompt}],
            },
        ]
        
        # Process inputs
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        # Parse response
        parsed = parse_subtask_detection_response(output_text)
        parsed["raw_response"] = output_text
        
        return parsed
    
    def decompose_task(
        self,
        task_description: str,
        max_new_tokens: int = 256,
    ) -> dict:
        """
        Decompose a task description into sequential subtasks using VLM.
        Text-only query (no images).
        """
        prompt = build_task_decomposition_prompt(task_description)
        
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": VLM_SYSTEM_MESSAGE}],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt}],
            },
        ]
        
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(self.model.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        subtasks = parse_task_decomposition_response(output_text)
        
        return {
            "subtasks": subtasks,
            "raw_response": output_text,
        }


# ===============================
# VLM SERVER
# ===============================

class VLMServer:
    """Websocket server for VLM subtask detection."""
    
    def __init__(
        self,
        model: VLMModel,
        host: str = "0.0.0.0",
        port: int = 10102,
        idle_timeout: int = -1,
    ):
        self._model = model
        self._host = host
        self._port = port
        self._idle_timeout = idle_timeout
        self._last_active = time.time()
        self._metadata = {"service": "vlm_subtask_detection"}
        logging.getLogger("websockets.server").setLevel(logging.INFO)
    
    def serve_forever(self) -> None:
        asyncio.run(self.run())
    
    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            # VLM inference can take 30-60+ seconds; disable server ping to avoid
            # "keepalive ping timeout" when server is busy with inference
            ping_interval=None,
            ping_timeout=None,
        ) as server:
            logging.info(f"VLM server listening on {self._host}:{self._port}")
            if self._idle_timeout > 0:
                await self._idle_watchdog(server)
            else:
                await server.serve_forever()
    
    async def _idle_watchdog(self, server):
        while True:
            await asyncio.sleep(5)
            if time.time() - self._last_active > self._idle_timeout:
                logging.info(f"Idle timeout ({self._idle_timeout}s) reached, shutting down server.")
                server.close()
                await server.wait_closed()
                break
    
    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        
        # Send metadata on connection
        await websocket.send(packb(self._metadata))
        
        while True:
            try:
                msg = unpackb(await websocket.recv())
                self._last_active = time.time()
                ret = self._route_message(msg)
                await websocket.send(packb(ret))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error.",
                )
                raise
    
    def _route_message(self, msg: dict) -> dict:
        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "subtask_detection")
        
        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id}
        
        elif mtype == "subtask_detection":
            try:
                reference_image = msg.get("reference_image")  # np.ndarray or None
                current_image = msg.get("current_image")  # np.ndarray (H, W, 3), uint8
                task_description = msg.get("task_description", "")
                current_subtask = msg.get("current_subtask", "")
                next_subtask = msg.get("next_subtask")
                gripper_state = msg.get("gripper_state", {"left": "OPEN", "right": "OPEN"})
                gripper_state_changed = msg.get("gripper_state_changed", False)
                previous_gripper_state = msg.get("previous_gripper_state")
                max_new_tokens = msg.get("max_new_tokens", 256)
                
                if current_image is None:
                    return {
                        "status": "error",
                        "ok": False,
                        "request_id": req_id,
                        "error": {"message": "Missing 'current_image' in request"},
                    }
                
                result = self._model.detect_subtask(
                    reference_image=reference_image,
                    current_image=current_image,
                    task_description=task_description,
                    current_subtask=current_subtask,
                    next_subtask=next_subtask,
                    gripper_state=gripper_state,
                    gripper_state_changed=gripper_state_changed,
                    previous_gripper_state=previous_gripper_state,
                    max_new_tokens=max_new_tokens,
                )
                
                return {
                    "status": "ok",
                    "ok": True,
                    "type": "subtask_detection_result",
                    "request_id": req_id,
                    "data": result,
                }
            except Exception as e:
                logging.exception("VLM subtask detection error (request_id=%s)", req_id)
                return {
                    "status": "error",
                    "ok": False,
                    "type": "subtask_detection_result",
                    "request_id": req_id,
                    "error": {"message": str(e)},
                }
        
        elif mtype == "task_decomposition":
            try:
                task_description = msg.get("task_description", "")
                max_new_tokens = msg.get("max_new_tokens", 256)
                
                if not task_description:
                    return {
                        "status": "error",
                        "ok": False,
                        "request_id": req_id,
                        "error": {"message": "Missing 'task_description' in request"},
                    }
                
                result = self._model.decompose_task(
                    task_description=task_description,
                    max_new_tokens=max_new_tokens,
                )
                
                return {
                    "status": "ok",
                    "ok": True,
                    "type": "task_decomposition_result",
                    "request_id": req_id,
                    "data": result,
                }
            except Exception as e:
                logging.exception("VLM task decomposition error (request_id=%s)", req_id)
                return {
                    "status": "error",
                    "ok": False,
                    "type": "task_decomposition_result",
                    "request_id": req_id,
                    "error": {"message": str(e)},
                }
        
        else:
            return {
                "status": "error",
                "ok": False,
                "type": "unknown",
                "request_id": req_id,
                "error": {"message": f"Unsupported message type '{mtype}'"},
            }


def build_argparser():
    parser = argparse.ArgumentParser(description="VLM Subtask Detection Server")
    parser.add_argument(
        "--model-path",
        type=str,
        default="./playground/Pretrained_models/Qwen3-VL-4B-Instruct",
        help="Path to Qwen3-VL model checkpoint"
    )
    parser.add_argument("--port", type=int, default=10102, help="Server port")
    parser.add_argument("--idle-timeout", type=int, default=-1, help="Idle timeout in seconds, -1 means never close")
    return parser


def main():
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info(f"Starting VLM server (host: {hostname}, ip: {local_ip})")
    
    # Load model
    model = VLMModel(
        model_path=args.model_path,
        device=device,
    )
    
    # Start server
    server = VLMServer(
        model=model,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
