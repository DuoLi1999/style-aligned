from diffusers import StableDiffusionXLPipeline, DDIMScheduler
import torch
import matplotlib.pyplot as plt
import sa_handler
import os
import safetensors.torch
from pipelines import LayerDiffusionPipeline, LayerDiffusionXLPipeline
from pipelines.models import LatentTransparencyOffsetEncoder, UNet1024
from PIL import Image
import warnings
import peft
warnings.simplefilter(action='ignore', category=FutureWarning)

# from diffusers.models.unets.unet_2d_condition import
# Initialize the DDIMScheduler
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)

# pipeline = StableDiffusionXLPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
#     scheduler=scheduler
# ).to("cuda:3")

transparent_vae_decoder = UNet1024(in_channels=3, out_channels=4)
ckpt_path = "/home/ld/Project/layerdiffuse_ours/ckpts/vae_transparent_decoder.safetensors"
ckpt = safetensors.torch.load_file(ckpt_path, device="cpu")
transparent_vae_decoder.load_state_dict(ckpt, strict=True)
transparent_vae_decoder = transparent_vae_decoder.half()

pipeline = LayerDiffusionXLPipeline.from_pretrained(
	"stabilityai/stable-diffusion-xl-base-1.0",
    transparent_vae_decoder=transparent_vae_decoder,
    torch_dtype=torch.float16,
).to("cuda:1")
pipeline.load_lora_weights(
    pretrained_model_name_or_path_or_dict="layer_xl_transparent_attn_DiffusersStyle_LoRA.safetensors",
    weight_name="layer_xl_transparent_attn_DiffusersStyle_LoRA.safetensors",
    adapter_name="transparent"
)
pipeline.set_adapters(["transparent"], adapter_weights=[1.0])
# Configure the pipeline for CPU offloading and VAE slicing
pipeline.enable_model_cpu_offload() 
pipeline.enable_vae_slicing()

# Initialize the style-aligned handler
handler = sa_handler.Handler(pipeline)
sa_args = sa_handler.StyleAlignedArgs(share_group_norm=False,
                                      share_layer_norm=False,
                                      share_attention=True,
                                      adain_queries=True,
                                      adain_keys=True,
                                      adain_values=False)

handler.register(sa_args)

# Function to generate style-aligned images
def style_aligned_sdxl(initial_prompts, style_prompt, seed=None):
    try:
        # Combine the style prompt with each initial prompt
        gen = None if seed is None else torch.manual_seed(int(seed))
        sets_of_prompts = [prompt + " in the style of " + style_prompt for prompt in initial_prompts if prompt]
        # Generate images using the pipeline
        # images = pipeline(sets_of_prompts, generator=gen)
        # return images.images
        images, layerdiff_out, png_list = pipeline(sets_of_prompts, generator=gen)
        return images.images, png_list
    except Exception as e:
        print(f"Error in generating images: {e}")


# Main function to run the script
def save_images(images, initial_prompts, style_prompt):
    save_path = 'generated_images'
    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
    
    for i, img in enumerate(images, 1):
        if i <= len(initial_prompts):
            filename = f"{initial_prompts[i-1]}_styled_{style_prompt}_{i}.png".replace(" ", "_").replace(",", "").replace("__", "_")
        else:
            filename = f"image_{i}.png"
        img_path = os.path.join(save_path, filename)
        img.save(img_path)
        print(f"Image saved: {img_path}")
def save_images_png(images, initial_prompts, style_prompt):
    save_path = 'generated_images'
    os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists
    
    for i, img in enumerate(images, 1):
        if i <= len(initial_prompts):
            filename = f"{initial_prompts[i-1]}_styled_{style_prompt}_{i}.png".replace(" ", "_").replace(",", "").replace("__", "_")
        else:
            filename = f"image_{i}.png"
        img_path = os.path.join(save_path, filename)
        Image.fromarray(img).save(img_path)
        print(f"Image saved: {img_path}")
# Replace the display_images call in the main function with save_images
# Main function to run the script
def main():
    # initial_prompts = []
    # for i in range(1, 6): # Collect up to 5 prompts
    #     prompt = input(f"Enter initial prompt {i} (or press enter to continue): ")
    #     if prompt:
    #         initial_prompts.append(prompt)
    #     else:
    #         break
    # initial_prompts = ["a toy train", "a toy airplane", "a toy bicycle", "a toy car", "a toy boat"]
    initial_prompts = ["Violin",]
                    # ["a firewoman", "a Gardner", "a scientist", "a policewoman", "a saxophone player", "made of claymation, stop motion animation."],

    style_prompt = "logo, minimal flat design illustration."
    seed = '12345'
    seed = int(seed) if seed.isdigit() else None

    images, png_list = style_aligned_sdxl(initial_prompts, style_prompt, seed)
    # images = style_aligned_sdxl(initial_prompts, style_prompt, seed)

    if images:
        save_images(images, initial_prompts, style_prompt)
        save_images_png(png_list, initial_prompts, 'rgba')

    else:
        print("No images were generated.")

if __name__ == "__main__":
    main()
