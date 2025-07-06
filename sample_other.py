from diffusers import StableDiffusionPipeline
import torch
import os
import random

root_path = "result/"
model_path = "/root/workspace/finetuned_sd"

    
for age in range(65, 105, 5):
    os.makedirs(os.path.join(root_path, f"{age}"), exist_ok=True)

def main():
    random.seed(42)
    torch.manual_seed(42)
 
    pipeline = StableDiffusionPipeline.from_pretrained(
        model_path, 
        torch_dtype=torch.float16, 
        use_safetensors=True, 
        safety_checker = None
    ).to(f"cuda")
    
    pipeline.set_progress_bar_config(disable=True)
    
    batch_size = 4
    image_count_per_label = 100

    for age in range(65, 105, 5):
        save_path = os.path.join(root_path, f"{age}")
        
        os.makedirs(save_path, exist_ok=True)
        prompt = [f"a portrait of the face of a {age}-year-old"] * batch_size
        
        for i in range(0, image_count_per_label, batch_size):
            images = pipeline(
                prompt,
                height=512,    
                width=512,
                num_inference_steps=50
            ).images
            
            for batch_i, image in enumerate(images):
                file_name = f"{age}_{i + batch_i}.png"
                image.save(os.path.join(save_path, file_name))
                
                if (i + batch_i + 1) % 4 == 0:
                    print(f"Generated {i + batch_i + 1} images with label {age}...")
            
    print(f"Finished!")
    
main()