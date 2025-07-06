import argparse
import os
from diffusers import StableDiffusionPipeline
import torch
import random

def main(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # 初始化 pipeline
    print(f"[GPU {args.gpu_id}] Loading model...")
    
    pipeline = StableDiffusionPipeline.from_pretrained(
        args.model_path, 
        torch_dtype=torch.float16, 
        use_safetensors=True, 
        safety_checker = None
    ).to(f"cuda:{args.gpu_id}")
    
    pipeline.set_progress_bar_config(disable=True)
    
    print(f"[GPU {args.gpu_id}] Start generating images with seed {args.seed}. ages from {args.label_start} to {args.label_end}.")

    batch_size = args.batch_size
    image_count_per_label = args.image_count_per_label

    for age in range(args.label_start, args.label_end+1):
        save_path = os.path.join(args.output_dir, f"{age}")
        
        os.makedirs(save_path, exist_ok=True)
        prompt = [f"a portrait of the face of a {age}-year-old"] * args.batch_size
        
        for i in range(0, image_count_per_label, batch_size):
            images = pipeline(
                prompt,
                height=args.height,    
                width=args.width,
                num_inference_steps=args.steps
            ).images
            
            for batch_i, image in enumerate(images):
                file_name = f"{age}_{i + batch_i}.png"
                image.save(os.path.join(save_path, file_name))
                
                if (i + batch_i + 1) % 16 == 0:
                    print(f"[GPU {args.gpu_id}] Generated {i + batch_i + 1} images with label {age}...")
            
    print(f"[GPU {args.gpu_id}] Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Stable Diffusion model directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save images")
    parser.add_argument("--gpu_id", type=int, required=True, help="GPU index to use")
    parser.add_argument("--label_start", type=int, required=True, help="ages start(inclusive)")
    parser.add_argument("--label_end", type=int, required=True, help="ages end(inclusive)")
    parser.add_argument("--image_count_per_label", type=int, default=1000, help="how many images will be generated for each label")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size, no more than 8.")
    
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--steps", type=int, default=50, help="Sampling steps")
    parser.add_argument("--seed", type=int, default=42, help="seed")
    
    args = parser.parse_args()

    main(args)
