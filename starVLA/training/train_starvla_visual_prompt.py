"""
StarVLA Visual Prompt Training Script
Extends the co-training script with visual prompt location prediction.
"""

# Standard Library
import argparse
import json
import os
import random
from pathlib import Path
from typing import Tuple, List, Optional
from torch.utils.data import DataLoader
import numpy as np
import time

# Third-Party Libraries
import torch
import torch.distributed as dist
import wandb
import yaml
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoProcessor, get_scheduler

# Local Modules
from starVLA.dataloader import build_dataloader
from starVLA.training.trainer_utils.trainer_tools import normalize_dotlist_args
from starVLA.model.framework import build_framework
from starVLA.training.trainer_utils.trainer_tools import TrainerUtils
from starVLA.training.trainer_utils.trainer_tools import build_param_lr_groups
from starVLA.training.trainer_utils.config_tracker import wrap_config, AccessTrackedConfig

deepspeed_plugin = DeepSpeedPlugin()
accelerator = Accelerator(deepspeed_plugin=deepspeed_plugin)
accelerator.print(accelerator.state)

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Overwatch =>> Wraps `logging.Logger`
logger = get_logger(__name__)


def setup_directories(cfg) -> Path:
    """create output directory and save config"""
    cfg.output_dir = os.path.join(cfg.run_root_dir, cfg.run_id)
    output_dir = Path(cfg.output_dir)

    if not dist.is_initialized() or dist.get_rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir / "checkpoints", exist_ok=True)

    # Ensure directories exist before other ranks proceed
    if dist.is_initialized():
        dist.barrier()

    return output_dir


def prepare_data(cfg, accelerator, output_dir) -> Tuple[DataLoader, Optional[DataLoader]]:
    """prepare training data - both VLA and VP prediction dataloaders"""
    logger.info(f"Creating VLA Dataset with Mixture `{cfg.datasets.vla_data.data_mix}`")
    vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)

    vp_loss_scale = getattr(cfg.trainer.loss_scale, 'visual_prompt', 0.1)
    if vp_loss_scale > 0:
        logger.info(f"Creating VP Prediction Dataset (loss_scale={vp_loss_scale})")
        vp_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vp_data.dataset_py)
    else:
        logger.info("VP training disabled (trainer.loss_scale.visual_prompt = 0), skipping VP data loading")
        vp_train_dataloader = None

    accelerator.dataloader_config.dispatch_batches = False
    dist.barrier()

    return vla_train_dataloader, vp_train_dataloader


def setup_optimizer_and_scheduler(model, cfg) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """set optimizer and learning rate scheduler"""
    param_groups = build_param_lr_groups(model=model, cfg=cfg)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg.trainer.learning_rate.base,
        betas=tuple(cfg.trainer.optimizer.betas),
        weight_decay=cfg.trainer.optimizer.weight_decay,
        eps=cfg.trainer.optimizer.eps,
    )

    if dist.is_initialized() and dist.get_rank() == 0:
        for i, group in enumerate(optimizer.param_groups):
            logger.info(f"LR Group {group['name']}: lr={group['lr']}, num_params={len(group['params'])}")

    lr_scheduler = get_scheduler(
        name=cfg.trainer.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=cfg.trainer.num_warmup_steps,
        num_training_steps=cfg.trainer.max_train_steps,
        scheduler_specific_kwargs=cfg.trainer.scheduler_specific_kwargs,
    )

    return optimizer, lr_scheduler


class VisualPromptTrainer(TrainerUtils):
    """
    Trainer for visual prompt training with VLA.
    Uses two dataloaders: VLA (action prediction) and VP (visual prompt prediction).
    """
    
    def __init__(self, cfg, model, vla_train_dataloader, vp_train_dataloader, optimizer, lr_scheduler, accelerator):
        self.config = cfg
        self.model = model
        self.vla_train_dataloader = vla_train_dataloader
        self.vp_train_dataloader = vp_train_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.accelerator = accelerator

        self.completed_steps = 0
        self.total_batch_size = self._calculate_total_batch_size()
        
        # Visual prompt loss scale
        self.vp_loss_scale = getattr(cfg.trainer.loss_scale, 'visual_prompt', 0.1)

    def prepare_training(self):
        rank = dist.get_rank() if dist.is_initialized() else 0
        seed = self.config.seed + rank if hasattr(self.config, "seed") else rank + 3047
        set_seed(seed)

        if hasattr(self.config.trainer, "pretrained_checkpoint") and self.config.trainer.pretrained_checkpoint:
            pretrained_checkpoint = self.config.trainer.pretrained_checkpoint
            reload_modules = (
                self.config.trainer.reload_modules if hasattr(self.config.trainer, "reload_modules") else None
            )
            self.model = self.load_pretrained_backbones(self.model, pretrained_checkpoint, reload_modules=reload_modules)

        freeze_modules = (
            self.config.trainer.freeze_modules
            if (self.config and hasattr(self.config.trainer, "freeze_modules"))
            else None
        )
        self.model = self.freeze_backbones(self.model, freeze_modules=freeze_modules)

        self.print_trainable_parameters(self.model)

        # Prepare dataloaders (skip VP if disabled)
        if self.vp_train_dataloader is not None:
            self.model, self.optimizer, self.vla_train_dataloader, self.vp_train_dataloader = (
                self.setup_distributed_training(
                    self.accelerator, self.model, self.optimizer, self.vla_train_dataloader, self.vp_train_dataloader
                )
            )
        else:
            self.model, self.optimizer, self.vla_train_dataloader = (
                self.setup_distributed_training(
                    self.accelerator, self.model, self.optimizer, self.vla_train_dataloader
                )
            )

        self._init_wandb()
        self._init_checkpointing()

    def _calculate_total_batch_size(self):
        return (
            self.config.datasets.vla_data.per_device_batch_size
            * self.accelerator.num_processes
            * self.accelerator.gradient_accumulation_steps
        )

    def _init_wandb(self):
        if self.accelerator.is_main_process:
            wandb.init(
                name=self.config.run_id,
                dir=os.path.join(self.config.output_dir, "wandb"),
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                group="vla-visual-prompt-train",
            )
        # Ensure wandb is initialized before training starts
        self.accelerator.wait_for_everyone()

    def _init_checkpointing(self):
        self.checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        # Only rank 0 creates directories to avoid race conditions
        if self.accelerator.is_main_process:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        # Wait for directory to be created before other ranks proceed
        self.accelerator.wait_for_everyone()

        pretrained_checkpoint = getattr(self.config.trainer, "pretrained_checkpoint", None)
        is_resume = getattr(self.config.trainer, "is_resume", False)

        if pretrained_checkpoint and is_resume:
            self._load_checkpoint(self.config.resume_from_checkpoint)

    def _load_checkpoint(self, checkpoint_path):
        self.accelerator.load_state(checkpoint_path)
        self.accelerator.print(f"Resumed from checkpoint: {checkpoint_path}")

    def _save_checkpoint(self):
        try:
            # All ranks must participate in get_state_dict for DeepSpeed ZeRO
            # This gathers sharded parameters from all processes
            state_dict = self.accelerator.get_state_dict(self.model)
            
            # Only rank 0 writes to disk
            if self.accelerator.is_main_process:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"steps_{self.completed_steps}")
                torch.save(state_dict, checkpoint_path + "_pytorch_model.pt")

                summary_data = {
                    "steps": self.completed_steps,
                }
                with open(os.path.join(self.config.output_dir, "summary.jsonl"), "a") as f:
                    f.write(json.dumps(summary_data) + "\n")
                self.accelerator.print(f"Checkpoint saved at {checkpoint_path}")

                if isinstance(self.config, AccessTrackedConfig):
                    logger.info("Saving accessed configuration...")
                    output_dir = Path(self.config.output_dir)
                    self.config.save_accessed_config(
                        output_dir / "config.yaml", 
                        use_original_values=False 
                    )
                    logger.info("Configuration files saved")
        except Exception as e:
            logger.error(f"Checkpoint saving failed on rank {dist.get_rank()}: {e}")
        finally:
            # Critical: ensure all ranks synchronize even on failure
            self.accelerator.wait_for_everyone()

    def _log_metrics(self, metrics):
        if self.completed_steps % self.config.trainer.logging_frequency == 0:
            if dist.get_rank() == 0:
                metrics["learning_rate"] = self.lr_scheduler.get_last_lr()[0]
                metrics["epoch"] = round(self.completed_steps / len(self.vla_train_dataloader), 2)
                wandb.log(metrics, step=self.completed_steps)
                logger.info(f"Step {self.completed_steps}, Metrics: {metrics}")

    def _create_data_iterators(self):
        """Create iterators for both VLA and VP dataloaders."""
        self.vla_iter = iter(self.vla_train_dataloader)
        self.vp_iter = iter(self.vp_train_dataloader) if self.vp_train_dataloader is not None else None

    def _get_next_batch(self):
        """Get next batch from both VLA and VP dataloaders."""
        # Get VLA batch
        try:
            batch_vla = next(self.vla_iter)
        except StopIteration:
            if not hasattr(self, "vla_epoch_count"):
                self.vla_epoch_count = 0
            self.vla_iter, self.vla_epoch_count = TrainerUtils._reset_dataloader(
                self.vla_train_dataloader, self.vla_epoch_count
            )
            batch_vla = next(self.vla_iter)

        # Get VP batch (empty list if VP training is disabled)
        if self.vp_iter is None:
            batch_vp = []
        else:
            try:
                batch_vp = next(self.vp_iter)
            except StopIteration:
                if not hasattr(self, "vp_epoch_count"):
                    self.vp_epoch_count = 0
                self.vp_iter, self.vp_epoch_count = TrainerUtils._reset_dataloader(
                    self.vp_train_dataloader, self.vp_epoch_count
                )
                batch_vp = next(self.vp_iter)

        return batch_vla, batch_vp

    def train(self):
        self._log_training_config()
        self._create_data_iterators()

        progress_bar = tqdm(
            range(self.config.trainer.max_train_steps), disable=not self.accelerator.is_local_main_process
        )

        while self.completed_steps < self.config.trainer.max_train_steps:
            t_start_data = time.perf_counter()
            batch_vla, batch_vp = self._get_next_batch()
            t_end_data = time.perf_counter()
            
            t_start_model = time.perf_counter()
            step_metrics = self._train_step(batch_vla, batch_vp)
            t_end_model = time.perf_counter()
            
            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.completed_steps += 1
            
            if self.accelerator.is_local_main_process:
                progress_bar.set_postfix(
                    {
                        "data_times": f"{t_end_data - t_start_data:.3f}",
                        "model_times": f"{t_end_model - t_start_model:.3f}",
                    }
                )

            if self.completed_steps % self.config.trainer.eval_interval == 0 and self.completed_steps > 0:
                step_metrics = self.eval_action_model(step_metrics)

            step_metrics["data_time"] = t_end_data - t_start_data
            step_metrics["model_time"] = t_end_model - t_start_model
            self._log_metrics(step_metrics)

            if self.completed_steps % self.config.trainer.save_interval == 0 and self.completed_steps > 0:
                self._save_checkpoint()
                # Note: barrier removed - _save_checkpoint already calls wait_for_everyone()

            if self.completed_steps >= self.config.trainer.max_train_steps:
                break

        self._finalize_training()

    def eval_action_model(self, step_metrics: dict = None) -> float:
        """
        Evaluate the model on both VLA (action prediction) and VLM (VP prediction).
        
        VLA: MSE between predicted and ground truth actions
        VP: Cross-entropy loss on VP location prediction (lower is better)
        
        NOTE: With DeepSpeed ZeRO, all ranks must participate in forward passes
        because model parameters are sharded. Only rank 0 computes/logs metrics.
        """
        if step_metrics is None:
            step_metrics = {}
        
        try:
            # All ranks must call _get_next_batch to keep iterators synchronized
            batch_vla, batch_vp = self._get_next_batch()
            
            # ============ VLA Evaluation ============
            # All ranks must participate in model forward pass for DeepSpeed ZeRO
            try:
                examples = batch_vla
                actions = [example["action"] for example in examples]

                # All ranks run predict_action (required for DeepSpeed ZeRO)
                output_dict = self.model.predict_action(examples=examples)
                normalized_actions = output_dict["normalized_actions"]

                # Only rank 0 computes and logs metrics
                if self.accelerator.is_main_process:
                    actions = np.array(actions)
                    num_pots = np.prod(actions.shape)
                    score = TrainerUtils.euclidean_distance(normalized_actions, actions)
                    average_score = score / num_pots
                    step_metrics["eval_vla_mse"] = average_score
            except Exception as e:
                logger.warning(f"VLA evaluation failed on rank {dist.get_rank()}: {e}")
                if self.accelerator.is_main_process:
                    step_metrics["eval_vla_mse"] = -1.0
            
            # ============ VP Evaluation ============
            # All ranks must participate in model forward pass for DeepSpeed ZeRO
            if len(batch_vp) > 0:
                try:
                    with torch.no_grad():
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            vp_images = [[sample["image"]] for sample in batch_vp]
                            vp_instructions = [f"{sample['instruction']}\nAnswer: {sample['answer']}" for sample in batch_vp]
                            
                            vp_inputs = self.model.qwen_vl_interface.build_qwenvl_inputs(
                                images=vp_images,
                                instructions=vp_instructions
                            )
                            
                            if 'labels' not in vp_inputs:
                                vp_inputs['labels'] = vp_inputs['input_ids'].clone()
                            
                            # All ranks run forward pass (required for DeepSpeed ZeRO)
                            vp_output = self.model.qwen_vl_interface(**vp_inputs)
                            
                            # Only rank 0 logs metrics
                            if self.accelerator.is_main_process:
                                if vp_output.loss is not None:
                                    step_metrics["eval_vp_loss"] = vp_output.loss.item()
                                else:
                                    step_metrics["eval_vp_loss"] = -1.0
                                step_metrics["eval_vp_samples"] = len(batch_vp)
                    
                except Exception as e:
                    logger.warning(f"VP evaluation failed on rank {dist.get_rank()}: {e}")
                    if self.accelerator.is_main_process:
                        step_metrics["eval_vp_loss"] = -1.0
                        step_metrics["eval_vp_samples"] = 0
            else:
                if self.accelerator.is_main_process:
                    step_metrics["eval_vp_loss"] = -1.0
                    step_metrics["eval_vp_samples"] = 0
                    
        except Exception as e:
            logger.error(f"Evaluation failed on rank {dist.get_rank()}: {e}")
            if self.accelerator.is_main_process:
                step_metrics["eval_vla_mse"] = -1.0
                step_metrics["eval_vp_loss"] = -1.0
                step_metrics["eval_vp_samples"] = 0
        finally:
            # Critical: ensure all ranks synchronize even on failure
            self.accelerator.wait_for_everyone()
        
        return step_metrics

    def _log_training_config(self):
        if self.accelerator.is_main_process:
            logger.info("***** Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  VLA per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            if hasattr(self.config.datasets, 'vp_data'):
                logger.info(f"  VP per device batch size = {self.config.datasets.vp_data.per_device_batch_size}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  VLA total batch size = {self.total_batch_size}")
            logger.info(f"  Visual prompt loss scale = {self.vp_loss_scale}")

    def _train_step(self, batch_vla, batch_vp):
        """Execute single training step with visual prompt support.
        
        Args:
            batch_vla: Batch from VLA dataloader (for action prediction)
            batch_vp: Batch from VP dataloader (for visual prompt prediction)
        """
        log_dict = {}
        
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            # VLA task forward propagation
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss * self.config.trainer.loss_scale.vla
            
            self.accelerator.backward(total_loss)

            # VP task forward propagation using separate VP batch
            vp_loss = torch.tensor(0.0, device=self.accelerator.device)
            
            if len(batch_vp) > 0:
                # Build inputs for VLM with visual prompt instructions and answers
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    vp_images = [[sample["image"]] for sample in batch_vp]
                    vp_instructions = [f"{sample['instruction']}\nAnswer: {sample['answer']}" for sample in batch_vp]
                    
                    vp_inputs = self.model.qwen_vl_interface.build_qwenvl_inputs(
                        images=vp_images, 
                        instructions=vp_instructions
                    )
                    
                    # Set up labels for language modeling
                    if 'labels' not in vp_inputs:
                        vp_inputs['labels'] = vp_inputs['input_ids'].clone()
                    
                    vp_output = self.model.qwen_vl_interface(**vp_inputs)
                    if vp_output.loss is not None:
                        vp_loss = vp_output.loss * self.vp_loss_scale
                
                if vp_loss.requires_grad:
                    self.accelerator.backward(vp_loss)

            # Gradient clipping
            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            # Optimizer step
            self.optimizer.step()
            self.lr_scheduler.step()

            log_dict.update(
                {
                    "action_dit_loss": action_loss.item(),
                    "vp_loss": vp_loss.item() if isinstance(vp_loss, torch.Tensor) else vp_loss,
                    "num_vp_samples": len(batch_vp),
                }
            )
        
        return log_dict

    def _finalize_training(self):
        # All ranks must participate in get_state_dict for DeepSpeed ZeRO
        state_dict = self.accelerator.get_state_dict(self.model)
        
        if self.accelerator.is_main_process:
            final_checkpoint = os.path.join(self.config.output_dir, "final_model")
            os.makedirs(final_checkpoint, exist_ok=True)
            torch.save(state_dict, os.path.join(final_checkpoint, "pytorch_model.pt"))
            logger.info(f"Training complete. Final model saved at {final_checkpoint}")
            wandb.finish()

        self.accelerator.wait_for_everyone()


class InlineVPTrainer(VisualPromptTrainer):
    """
    Trainer for inline VP training with VLA.
    
    Unlike VisualPromptTrainer which uses two separate dataloaders (VLA + VP),
    this trainer extracts VP prediction samples directly from the VLA batch.
    Each VLA sample may include inline VP fields (has_vp, vp_instruction, etc.)
    produced by VisualPromptMixtureDatasetWithInlineVP.
    
    Benefits:
    - VP samples are drawn from randomly sampled training frames (all frames)
    - No dependency on pre-extracted frames or separate VP dataset
    - VP data distribution naturally matches VLA training distribution
    
    DDP safety:
    - Uses dist.all_reduce(MIN) to check if all ranks have VP samples before
      the VP forward pass, preventing DeepSpeed ZeRO hangs.
    """
    
    def __init__(self, cfg, model, vla_train_dataloader, optimizer, lr_scheduler, accelerator):
        # Pass vp_train_dataloader=None to parent
        super().__init__(cfg, model, vla_train_dataloader, None, optimizer, lr_scheduler, accelerator)
        
        # Max VP samples to use per batch (caps compute cost)
        self.vp_max_samples_per_batch = int(getattr(cfg.datasets.vla_data, 'vp_max_samples_per_batch', 8))
    
    def _log_training_config(self):
        if self.accelerator.is_main_process:
            logger.info("***** Inline VP Training Configuration *****")
            logger.info(f"  Total optimization steps = {self.config.trainer.max_train_steps}")
            logger.info(f"  VLA per device batch size = {self.config.datasets.vla_data.per_device_batch_size}")
            logger.info(f"  VP max samples per batch = {self.vp_max_samples_per_batch}")
            logger.info(f"  Gradient accumulation steps = {self.config.trainer.gradient_accumulation_steps}")
            logger.info(f"  VLA total batch size = {self.total_batch_size}")
            logger.info(f"  Visual prompt loss scale = {self.vp_loss_scale}")
    
    def _extract_vp_samples(self, batch_vla):
        """
        Extract VP prediction samples from a VLA batch.
        
        Filters samples with has_vp=True and caps to vp_max_samples_per_batch.
        
        Args:
            batch_vla: List of dicts from VLA dataloader
            
        Returns:
            List of dicts with VP fields (vp_instruction, vp_answer, etc.)
        """
        vp_samples = [s for s in batch_vla if s.get('has_vp', False)]
        if len(vp_samples) > self.vp_max_samples_per_batch:
            vp_samples = random.sample(vp_samples, self.vp_max_samples_per_batch)
        return vp_samples
    
    def _build_vp_inputs(self, vp_samples):
        """
        Build VLM inputs from VP samples for forward pass.
        
        Args:
            vp_samples: List of dicts with VP fields
            
        Returns:
            Tuple of (vp_images, vp_instructions) ready for build_qwenvl_inputs
        """
        vp_images = [[s["vp_image_overlayed"]] for s in vp_samples]
        
        vp_instructions = [f"{s['vp_instruction']}\nAnswer: {s['vp_answer']}" for s in vp_samples]
        
        return vp_images, vp_instructions
    
    def _ddp_sync_vp_count(self, vp_samples):
        """
        Synchronize VP sample count across all DDP ranks.
        
        Returns the minimum VP sample count across all ranks.
        This ensures all ranks either all do VP forward or all skip it.
        
        Args:
            vp_samples: List of VP samples on this rank
            
        Returns:
            int: Minimum VP sample count across all ranks
        """
        has_vp_count = torch.tensor(
            [len(vp_samples)], dtype=torch.int64, device=self.accelerator.device
        )
        if dist.is_initialized():
            dist.all_reduce(has_vp_count, op=dist.ReduceOp.MIN)
        return has_vp_count.item()
    
    def _train_step(self, batch_vla, batch_vp):
        """Execute single training step with inline VP from VLA batch.
        
        Args:
            batch_vla: Batch from VLA dataloader (includes inline VP fields)
            batch_vp: Ignored (always empty for InlineVPTrainer)
        """
        log_dict = {}
        
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()

            # VLA task forward propagation (same as parent)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                total_loss = action_loss * self.config.trainer.loss_scale.vla
            
            self.accelerator.backward(total_loss)

            # VP task: extract from VLA batch instead of separate dataloader
            vp_samples = self._extract_vp_samples(batch_vla)
            vp_loss = torch.tensor(0.0, device=self.accelerator.device)
            
            # DDP sync: ensure all ranks agree on whether to do VP forward
            min_vp_count = self._ddp_sync_vp_count(vp_samples)
            
            if min_vp_count > 0:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    vp_images, vp_instructions = self._build_vp_inputs(vp_samples)
                    
                    vp_inputs = self.model.qwen_vl_interface.build_qwenvl_inputs(
                        images=vp_images,
                        instructions=vp_instructions
                    )
                    
                    if 'labels' not in vp_inputs:
                        vp_inputs['labels'] = vp_inputs['input_ids'].clone()
                    
                    vp_output = self.model.qwen_vl_interface(**vp_inputs)
                    if vp_output.loss is not None:
                        vp_loss = vp_output.loss * self.vp_loss_scale
                
                if vp_loss.requires_grad:
                    self.accelerator.backward(vp_loss)

            # Gradient clipping
            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)

            # Optimizer step
            self.optimizer.step()
            self.lr_scheduler.step()

            log_dict.update(
                {
                    "action_dit_loss": action_loss.item(),
                    "vp_loss": vp_loss.item() if isinstance(vp_loss, torch.Tensor) else vp_loss,
                    "num_vp_samples": len(vp_samples),
                    "min_vp_across_ranks": min_vp_count,
                }
            )
        
        return log_dict
    
    def eval_action_model(self, step_metrics: dict = None) -> float:
        """
        Evaluate with VP extracted from VLA batch.
        
        Same as parent for VLA evaluation. VP evaluation uses inline VP samples
        from the VLA batch instead of a separate VP dataloader.
        """
        if step_metrics is None:
            step_metrics = {}
        
        try:
            # Get next batch (batch_vp will be empty for InlineVPTrainer)
            batch_vla, _ = self._get_next_batch()
            
            # ============ VLA Evaluation ============
            try:
                examples = batch_vla
                actions = [example["action"] for example in examples]

                output_dict = self.model.predict_action(examples=examples)
                normalized_actions = output_dict["normalized_actions"]

                if self.accelerator.is_main_process:
                    actions = np.array(actions)
                    num_pots = np.prod(actions.shape)
                    score = TrainerUtils.euclidean_distance(normalized_actions, actions)
                    average_score = score / num_pots
                    step_metrics["eval_vla_mse"] = average_score
            except Exception as e:
                logger.warning(f"VLA evaluation failed on rank {dist.get_rank()}: {e}")
                if self.accelerator.is_main_process:
                    step_metrics["eval_vla_mse"] = -1.0
            
            # ============ VP Evaluation (from VLA batch) ============
            vp_samples = self._extract_vp_samples(batch_vla)
            min_vp_count = self._ddp_sync_vp_count(vp_samples)
            
            if min_vp_count > 0:
                try:
                    with torch.no_grad():
                        with torch.autocast("cuda", dtype=torch.bfloat16):
                            vp_images, vp_instructions = self._build_vp_inputs(vp_samples)
                            
                            vp_inputs = self.model.qwen_vl_interface.build_qwenvl_inputs(
                                images=vp_images,
                                instructions=vp_instructions
                            )
                            
                            if 'labels' not in vp_inputs:
                                vp_inputs['labels'] = vp_inputs['input_ids'].clone()
                            
                            vp_output = self.model.qwen_vl_interface(**vp_inputs)
                            
                            if self.accelerator.is_main_process:
                                if vp_output.loss is not None:
                                    step_metrics["eval_vp_loss"] = vp_output.loss.item()
                                else:
                                    step_metrics["eval_vp_loss"] = -1.0
                                step_metrics["eval_vp_samples"] = len(vp_samples)
                    
                except Exception as e:
                    logger.warning(f"VP evaluation failed on rank {dist.get_rank()}: {e}")
                    if self.accelerator.is_main_process:
                        step_metrics["eval_vp_loss"] = -1.0
                        step_metrics["eval_vp_samples"] = 0
            else:
                if self.accelerator.is_main_process:
                    step_metrics["eval_vp_loss"] = -1.0
                    step_metrics["eval_vp_samples"] = 0
                    
        except Exception as e:
            logger.error(f"Evaluation failed on rank {dist.get_rank()}: {e}")
            if self.accelerator.is_main_process:
                step_metrics["eval_vla_mse"] = -1.0
                step_metrics["eval_vp_loss"] = -1.0
                step_metrics["eval_vp_samples"] = 0
        finally:
            self.accelerator.wait_for_everyone()
        
        return step_metrics


def main(cfg) -> None:
    logger.info("Visual Prompt VLA Training :: Warming Up")

    cfg = wrap_config(cfg)
    logger.info("Configuration wrapped for access tracking")

    output_dir = setup_directories(cfg=cfg)

    vla = build_framework(cfg)

    use_inline_vp = getattr(cfg, 'use_inline_vp', False)

    if use_inline_vp:
        # Inline VP mode: only VLA dataloader needed (VP extracted from VLA batch)
        logger.info("Using Inline VP mode: VP samples extracted from VLA batch")
        logger.info(f"Creating VLA Dataset (inline VP) with Mixture `{cfg.datasets.vla_data.data_mix}`")
        vla_train_dataloader = build_dataloader(cfg=cfg, dataset_py=cfg.datasets.vla_data.dataset_py)
        accelerator.dataloader_config.dispatch_batches = False
        dist.barrier()

        optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

        trainer = InlineVPTrainer(
            cfg=cfg,
            model=vla,
            vla_train_dataloader=vla_train_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
        )
    else:
        # Original mode: separate VLA and VP dataloaders
        vla_train_dataloader, vp_train_dataloader = prepare_data(cfg=cfg, accelerator=accelerator, output_dir=output_dir)
        optimizer, lr_scheduler = setup_optimizer_and_scheduler(model=vla, cfg=cfg)

        trainer = VisualPromptTrainer(
            cfg=cfg,
            model=vla,
            vla_train_dataloader=vla_train_dataloader,
            vp_train_dataloader=vp_train_dataloader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            accelerator=accelerator,
        )

    trainer.prepare_training()
    trainer.train()

    logger.info("Training complete!")
    # Safe cleanup - handle case where some ranks may have failed
    try:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()
    except Exception as e:
        logger.warning(f"Cleanup failed on rank {dist.get_rank() if dist.is_initialized() else 'unknown'}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_yaml", 
        type=str, 
        default="examples/Robocasa_tabletop/train_files/starvla_cotrain_robocasa_gr1_visual_prompt.yaml", 
        help="Path to YAML config"
    )
    args, clipargs = parser.parse_known_args()

    cfg = OmegaConf.load(args.config_yaml)
    dotlist = normalize_dotlist_args(clipargs)
    cli_cfg = OmegaConf.from_dotlist(dotlist)
    cfg = OmegaConf.merge(cfg, cli_cfg)

    if cfg.is_debug and dist.is_initialized() and dist.get_rank() == 0:
        import debugpy
        debugpy.listen(("0.0.0.0", 10092))
        print("Rank 0 waiting for debugger attach on port 10092...")
        debugpy.wait_for_client()

    main(cfg)
