import rembg
import torch
import numpy as np
import argparse
from PIL import Image, ImageOps
from diffusers.utils import load_image
from diffusers import StableDiffusionInpaintPipeline

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate an image using Stable Diffusion inpainting.")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image")
    parser.add_argument('--text-prompt', type=str, required=True, help="Text prompt to guide image generation")
    parser.add_argument('--output', type=str, required=True, help="Path to save the output image")
    args = parser.parse_args()

    # Load the input image
    init_image = load_image(args.image).resize((512, 768))

    # Create a larger white background
    background_size = (1024, 1536)  # Bigger white background (you can adjust the size further)
    background_color = (255, 255, 255)  # White background
    background_image = Image.new("RGB", background_size, background_color)

    # Center the original image on the white background
    x_offset = (background_size[0] - init_image.size[0]) // 2
    y_offset = (background_size[1] - init_image.size[1]) // 2
    background_image.paste(init_image, (x_offset, y_offset))

    # Convert the combined image to a numpy array
    input_array = np.array(background_image)

    # Extract mask using rembg
    mask_array = rembg.remove(input_array, only_mask=True)

    # Create a PIL Image from the mask array
    mask_image = Image.fromarray(mask_array)

    # Invert the mask image
    mask_image_inverted = ImageOps.invert(mask_image)

    # Load the inpainting model
    model_id = "stabilityai/stable-diffusion-2-inpainting"
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id)
    pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    # Perform the inpainting with the text prompt and the image/mask
    result_image = pipe(
        prompt=args.text_prompt,
        negative_prompt="tail, deformed, mutated, ugly, disfigured, low quality, distorted, out of place, unnatural lighting",
        image=background_image,
        mask_image=mask_image_inverted,
    ).images[0]

    # Save the final image
    result_image.save(args.output)
    print(f"Image saved at: {args.output}")

if __name__ == "__main__":
    main()
