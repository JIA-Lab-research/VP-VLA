# SAM3 Segmentation Server
# Provides segmentation masks via websocket for the evaluation pipeline

import asyncio
import logging
import traceback
import time
import argparse
import socket

import numpy as np
import torch
import websockets.asyncio.server
import websockets.frames

from msgpack_utils import packb, unpackb


class SAM3Model:
    """Wrapper for SAM3 model to handle segmentation requests."""
    
    def __init__(self, model_path: str, device: str = "cuda"):
        from transformers import Sam3Processor, Sam3Model
        from PIL import Image
        
        self.device = device
        
        logging.info(f"Loading SAM3 model from {model_path}")
        self.model = Sam3Model.from_pretrained(model_path).to(device)
        self.processor = Sam3Processor.from_pretrained(model_path)
        self.model.eval()
        logging.info("SAM3 model loaded successfully")
    
    def segment(
        self, 
        image: np.ndarray, 
        text_prompt: str,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> dict:
        """
        Perform segmentation on the image using the text prompt.
        
        Args:
            image: np.ndarray of shape (H, W, 3), uint8 RGB image
            text_prompt: str, text description of object to segment
            threshold: float, confidence threshold for instance detection
            mask_threshold: float, threshold for mask binarization
            
        Returns:
            dict with:
                - masks: np.ndarray of shape (N, H, W), bool masks
                - scores: np.ndarray of shape (N,), confidence scores
                - num_masks: int, number of detected masks
        """
        from PIL import Image
        
        # Convert numpy to PIL
        pil_image = Image.fromarray(image).convert("RGB")
        original_size = pil_image.size  # (W, H)
        
        # Process inputs
        inputs = self.processor(
            images=pil_image, 
            text=text_prompt, 
            return_tensors="pt"
        ).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process results
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        
        masks = results.get("masks", torch.zeros((0, image.shape[0], image.shape[1]), dtype=torch.bool))
        scores = results.get("scores", torch.zeros((0,)))
        boxes = results.get("boxes", torch.zeros((0, 4), dtype=torch.float32))
        
        # Convert to numpy
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        
        return {
            "masks": masks.astype(np.uint8),  # (N, H, W)
            "boxes": boxes.astype(np.float32),  # (N, 4) in format (x1, y1, x2, y2)
            "scores": scores.astype(np.float32),  # (N,)
            "num_masks": len(masks),
        }


class SAM3Server:
    """Websocket server for SAM3 segmentation."""
    
    def __init__(
        self,
        model: SAM3Model,
        host: str = "0.0.0.0",
        port: int = 10094,
        idle_timeout: int = -1,
    ):
        self._model = model
        self._host = host
        self._port = port
        self._idle_timeout = idle_timeout
        self._last_active = time.time()
        self._metadata = {"service": "sam3_segmentation"}
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
        ) as server:
            logging.info(f"SAM3 server listening on {self._host}:{self._port}")
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
        mtype = msg.get("type", "segment")
        
        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id}
        
        elif mtype == "segment":
            try:
                image = msg.get("image")  # np.ndarray (H, W, 3), uint8
                text_prompt = msg.get("text_prompt", "")
                threshold = msg.get("threshold", 0.5)
                mask_threshold = msg.get("mask_threshold", 0.5)
                
                if image is None:
                    return {
                        "status": "error",
                        "ok": False,
                        "request_id": req_id,
                        "error": {"message": "Missing 'image' in request"},
                    }
                
                result = self._model.segment(
                    image, 
                    text_prompt, 
                    threshold=threshold, 
                    mask_threshold=mask_threshold,
                )
                
                return {
                    "status": "ok",
                    "ok": True,
                    "type": "segment_result",
                    "request_id": req_id,
                    "data": result,
                }
            except Exception as e:
                logging.exception("SAM3 segmentation error (request_id=%s)", req_id)
                return {
                    "status": "error",
                    "ok": False,
                    "type": "segment_result",
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
    parser = argparse.ArgumentParser(description="SAM3 Segmentation Server")
    parser.add_argument(
        "--model-path", 
        type=str, 
        default="playground/Pretrained_models/sam3",
        help="Path to SAM3 model checkpoint"
    )
    parser.add_argument("--port", type=int, default=10094, help="Server port")
    parser.add_argument("--idle-timeout", type=int, default=1800, help="Idle timeout in seconds, -1 means never close")
    return parser


def main():
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info(f"Starting SAM3 server (host: {hostname}, ip: {local_ip})")
    
    # Load model
    model = SAM3Model(
        model_path=args.model_path,
        device=device,
    )
    
    # Start server
    server = SAM3Server(
        model=model,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()


