import argparse
import os
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

# Accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed

# Diffusers & Transformers
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model


# ==============================
# Dataset (Unchanged - it was good)
# ==============================
class DreamBoothDataset(Dataset):
    def __init__(self, instance_data_root, tokenizer, size=512):
        self.instance_data_root = Path(instance_data_root)
        self.tokenizer = tokenizer
        self.size = size

        if not self.instance_data_root.exists():
            raise ValueError("Instance data directory does not exist.")

        self.image_paths = [
            p for p in self.instance_data_root.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
        ]

        if len(self.image_paths) == 0:
            raise ValueError("No images found in directory.")

        # Load metadata.json if exists
        metadata_path = self.instance_data_root / "metadata.json"
        self.prompts = {}

        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
                for item in metadata:
                    self.prompts[item["file_name"]] = item["prompt"]

        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index % len(self.image_paths)]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        filename = image_path.name
        prompt = self.prompts.get(
            filename,
            image_path.stem.replace("_", " ") # Better fallback
        )

        tokenized = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        )

        return {
            "pixel_values": image,
            "input_ids": tokenized.input_ids[0],
        }

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}


# ==============================
# Argument Parser
# ==============================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--instance_data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./lora-output")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--max_train_steps", type=int, default=1000)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"]) # Added
    return parser.parse_args()


# ==============================
# Main Training
# ==============================
def main():
    args = parse_args()
    
    # [OPTIMIZATION] Enable TF32 for faster math on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision
    )

    if accelerator.is_local_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load Models
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # Freeze VAE & text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # [OPTIMIZATION] Gradient Checkpointing (Saves VRAM -> allows larger batch size)
    unet.enable_gradient_checkpointing()

    # LoRA Configuration
    lora_config = LoraConfig(
        r=4,
        lora_alpha=4, # Alpha=Rank is usually better for stability
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.0,
        bias="none"
    )
    unet = get_peft_model(unet, lora_config)

    # [OPTIMIZATION] Cast frozen models to half-precision to save memory/speed
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move frozen components to device and correct dtype
    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)

    # Optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-2
    )

    # Dataset & Loader
    dataset = DreamBoothDataset(
        args.instance_data_dir,
        tokenizer,
        size=args.resolution
    )

    train_dataloader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2, # [OPTIMIZATION] Parallel data loading
    )

    # [ADDED] LR Scheduler
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with Accelerator
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    # Recalculate epochs based on steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Tracker
    global_step = 0
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Training")

    # ==============================
    # Training Loop
    # ==============================
    for epoch in range(num_train_epochs):
        unet.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # 1. Convert images to latents
                # We do this inside the loop but without grads. 
                # Note: We must cast pixel_values to weight_dtype (fp16) manually here
                latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample()
                latents = latents * vae.config.scaling_factor

                # 2. Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device).long()

                # 3. Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # 4. Get text embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # 5. Predict noise
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # 6. Loss
                # Get target based on prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                # 7. Backward
                accelerator.backward(loss)
                
                # [ADDED] Gradient Clipping prevents exploding gradients
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), 1.0)
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                progress_bar.set_postfix({"loss": loss.detach().item()})

            if global_step >= args.max_train_steps:
                break
        
        if global_step >= args.max_train_steps:
            break

    # ==============================
    # Saving
    # ==============================
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Unwrap and save only LoRA weights
        unet = accelerator.unwrap_model(unet)
        unet.save_pretrained(args.output_dir)
        print(f"Model saved to {args.output_dir}")

    accelerator.end_training()

if __name__ == "__main__":
    main()