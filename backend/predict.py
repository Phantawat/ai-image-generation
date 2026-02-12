import argparse
import os
import torch
from diffusers import StableDiffusionPipeline

# We don't need 'peft' import anymore for inference!

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--lora_model_path", type=str, default="./lora_output")  # Default to your output folder
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="generated_image.png")
    return parser.parse_args()

def main():
    args = parse_args()

    # 1. Detect Device
    # If in Colab with T4, this will be "cuda". If local without GPU, "cpu".
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 2. Load Base Model
    # We use float16 for GPU (faster) and float32 for CPU (compatibility)
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    pipe = StableDiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=dtype,
        safety_checker=None  # Optional: disables safety checker to save memory
    )
    pipe = pipe.to(device)

    # 3. Load LoRA (The Correct Way)
    if args.lora_model_path and os.path.exists(args.lora_model_path):
        print(f"Loading LoRA from {args.lora_model_path}...")
        try:
            # This is the modern Diffusers method. It handles merging automatically.
            pipe.load_lora_weights(args.lora_model_path)
            print("LoRA loaded successfully.")
        except Exception as e:
            print(f"Error loading LoRA: {e}")
            return
    else:
        print("No LoRA path provided or path does not exist. Using base model.")

    # 4. Get Prompt
    if not args.prompt:
        args.prompt = input("Enter prompt: ")

    print(f"Generating for prompt: '{args.prompt}'")

    # 5. Run Inference
    # Using 30 steps is a good balance of speed/quality
    with torch.no_grad():
        image = pipe(args.prompt, num_inference_steps=30, guidance_scale=7.5).images[0]

    image.save(args.output_path)
    print(f"Success! Image saved to {args.output_path}")

if __name__ == "__main__":
    main()