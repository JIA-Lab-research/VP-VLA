# VLM Subtask Detection Server (OXE single-gripper variant)
# Adapted from test/vlm_server_subtask.py for OXE datasets (bridge, fractal)
# that have a single gripper instead of dual left/right grippers.

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
from typing import Optional, Dict

import numpy as np
import torch
import websockets.asyncio.server
import websockets.frames

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(os.path.dirname(SCRIPT_DIR), "..")
UTILITY_DIR = os.path.join(PROJECT_ROOT, "examples", "Robocasa_tabletop", "visual_prompt_utility")
sys.path.insert(0, UTILITY_DIR)

from msgpack_utils import packb, unpackb


# ===============================
# VLM SYSTEM MESSAGE AND PROMPT
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
    gripper_text = f"GRIPPER = {gripper_state['gripper']}"

    gripper_change_notification = ""
    if gripper_state_changed and previous_gripper_state is not None:
        prev = previous_gripper_state.get("gripper", "UNKNOWN")
        curr = gripper_state["gripper"]

        if prev != curr:
            gripper_change_notification = f"""

⚠️ GRIPPER STATE CHANGE DETECTED:
GRIPPER: {prev} → {curr}

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


def parse_subtask_detection_response(response: str) -> Dict:
    try:
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
        image_content = []
        if reference_image is not None:
            ref_pil = self.Image.fromarray(reference_image).convert("RGB")
            image_content.append({"type": "image", "image": ref_pil})

        curr_pil = self.Image.fromarray(current_image).convert("RGB")
        image_content.append({"type": "image", "image": curr_pil})

        prompt = build_subtask_detection_prompt(
            task_description=task_description,
            current_subtask=current_subtask,
            next_subtask=next_subtask,
            gripper_state=gripper_state,
            gripper_state_changed=gripper_state_changed,
            previous_gripper_state=previous_gripper_state,
        )

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

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
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

        parsed = parse_subtask_detection_response(output_text)
        parsed["raw_response"] = output_text

        return parsed


# ===============================
# VLM SERVER
# ===============================

class VLMServer:
    """Websocket server for VLM subtask detection (single-gripper OXE variant)."""

    def __init__(self, model: VLMModel, host: str = "0.0.0.0", port: int = 10102, idle_timeout: int = -1):
        self._model = model
        self._host = host
        self._port = port
        self._idle_timeout = idle_timeout
        self._last_active = time.time()
        self._metadata = {"service": "vlm_subtask_detection_oxe"}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler, self._host, self._port,
            compression=None, max_size=None,
        ) as server:
            logging.info(f"VLM OXE server listening on {self._host}:{self._port}")
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
                reference_image = msg.get("reference_image")
                current_image = msg.get("current_image")
                task_description = msg.get("task_description", "")
                current_subtask = msg.get("current_subtask", "")
                next_subtask = msg.get("next_subtask")
                gripper_state = msg.get("gripper_state", {"gripper": "OPEN"})
                gripper_state_changed = msg.get("gripper_state_changed", False)
                previous_gripper_state = msg.get("previous_gripper_state")
                max_new_tokens = msg.get("max_new_tokens", 256)

                if current_image is None:
                    return {
                        "status": "error", "ok": False, "request_id": req_id,
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
                    "status": "ok", "ok": True,
                    "type": "subtask_detection_result",
                    "request_id": req_id,
                    "data": result,
                }
            except Exception as e:
                logging.exception("VLM subtask detection error (request_id=%s)", req_id)
                return {
                    "status": "error", "ok": False,
                    "type": "subtask_detection_result",
                    "request_id": req_id,
                    "error": {"message": str(e)},
                }

        else:
            return {
                "status": "error", "ok": False, "type": "unknown",
                "request_id": req_id,
                "error": {"message": f"Unsupported message type '{mtype}'"},
            }


def build_argparser():
    parser = argparse.ArgumentParser(description="VLM Subtask Detection Server (OXE single-gripper)")
    parser.add_argument(
        "--model-path", type=str,
        required=True,
        help="Path to Qwen3-VL model checkpoint",
    )
    parser.add_argument("--port", type=int, default=10200, help="Server port")
    parser.add_argument("--idle-timeout", type=int, default=-1, help="Idle timeout in seconds, -1 means never close")
    return parser


def main():
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info(f"Starting VLM OXE server (host: {hostname}, ip: {local_ip})")

    model = VLMModel(model_path=args.model_path, device=device)

    server = VLMServer(
        model=model,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
