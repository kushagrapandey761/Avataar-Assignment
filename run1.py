import rembg
import torch
import numpy as np
import argparse
from PIL import Image, ImageOps
from diffusers.utils import load_image
from diffusers import StableDiffusionInpaintPipeline
import os
import imageio

# Load the inpainting model
model_id = "stabilityai/stable-diffusion-2-inpainting"
pipe = StableDiffusionInpaintPipeline.from_pretrained(model_id)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Define the prompt and negative prompt
prompt = """Product in a kitchen used in meal preparation, placed naturally on the countertop, high-quality lighting"""
negative = """tail, deformed, mutated, ugly, disfigured, low quality, distorted, out of place, unnatural lighting"""

# Function to generate a series of frames
def generate_frames(init_image_path, output_dir, num_frames=30, zoom_out=True):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the initial image
    init_image = load_image(init_image_path).resize((1024, 1536))

    for i in range(num_frames):
        # Adjust the image for zoom out or translation
        scale_factor = 1.0 + (i * 0.05 if zoom_out else 0)  # Gradual zoom out

        # Resize image for zoom effect
        new_size = (int(init_image.size[0] * scale_factor), int(init_image.size[1] * scale_factor))
        resized_image = init_image.resize(new_size)

        # Create a larger background and paste resized image centered
        background_size = (2048, 3072)  # High-res background
        background_color = (255, 255, 255)  # White background
        background_image = Image.new("RGB", background_size, background_color)

        # Center the resized image
        x_offset = (background_size[0] - resized_image.size[0]) // 2
        y_offset = (background_size[1] - resized_image.size[1]) // 2
        background_image.paste(resized_image, (x_offset, y_offset))

        # Convert to numpy array for rembg processing
        input_array = np.array(background_image)

        # Extract mask using rembg
        mask_array = rembg.remove(input_array, only_mask=True)

        # Create a PIL Image from the mask array
        mask_image = Image.fromarray(mask_array)

        # Invert the mask image
        mask_image_inverted = ImageOps.invert(mask_image)

        # Generate inpainted frame
        result_image = pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=background_image,
            mask_image=mask_image_inverted,
            width=1024,  # Set width for the frame
            height=1536  # Set height for the frame
        ).images[0]

        # Save the frame
        frame_path = os.path.join(output_dir, f"frame_{i:03d}.png")
        result_image.save(frame_path)
        print(f"Saved frame {i + 1}/{num_frames}")

# Compile frames into a video
def create_video_from_frames(frame_dir, output_video):
    # Get list of frame files
    frame_files = sorted([os.path.join(frame_dir, file) for file in os.listdir(frame_dir) if file.endswith('.png')])
    
    # Use imageio with ffmpeg to create a video from images
    with imageio.get_writer(output_video, fps=10, codec='libx264', format='mp4') as video_writer:
        for frame_file in frame_files:
            frame = imageio.imread(frame_file)
            video_writer.append_data(frame)
    print(f"Video saved as {output_video}")

# Main function to handle arguments and run the script
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate zoom-out video from an image using Stable Diffusion inpainting.")
    parser.add_argument('--image', type=str, required=True, help="Path to the input image")
    parser.add_argument('--output-dir', type=str, default="./frames", help="Directory to save the generated frames")
    parser.add_argument('--output-video', type=str, default="generated_zoom_out.mp4", help="Path to save the output video")
    parser.add_argument('--num-frames', type=int, default=30, help="Number of frames to generate")
    parser.add_argument('--zoom-out', action='store_true', help="Enable zoom-out effect")

    args = parser.parse_args()

    # Generate frames
    generate_frames(args.image, args.output_dir, num_frames=args.num_frames, zoom_out=args.zoom_out)

    # Compile frames into video
    create_video_from_frames(args.output_dir, args.output_video)

if __name__ == "__main__":
    main()
