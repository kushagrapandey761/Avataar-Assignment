# Image Generation Using Stable Diffusion Inpainting

This repository contains the code for generating an image based on a given input image and text prompt using the **Stable Diffusion** inpainting model. The user can provide the path to the image and a text prompt via the command line, and the model will produce an enhanced version of the input image.

## Requirements

Ensure that the following Python libraries are installed:

- `torch`
- `diffusers`
- `rembg`
- `Pillow`
- `numpy`
- `argparse`

You can install the required libraries using the following commands:

```bash
pip install torch diffusers rembg pillow numpy argparse
```
Additionally, if you have GPU support, you can enable CUDA for faster performance:

```bash
pip install torch --upgrade --extra-index-url https://download.pytorch.org/whl/cu117
```

## Usage
You can run the script from the command line with the following parameters:

```bash
python run.py --image <image_path> --text-prompt "<your_text_prompt>" --output <output_path>
```

## Example Command:

```bash
python run.py --image ./example.jpg --text-prompt "A product in a kitchen used in meal preparation" --output ./generated.png
```
## Arguments

-`--image`: Path to the input image. (e.g., `./example.jpg`)

-`--text-prompt`: Text prompt that describes the scene to guide the inpainting model. (e.g., `"Product in a kitchen used in meal preparation"`)

-`--output`: Path to save the generated output image. (e.g., `./generated.png`)

## How it Works
1. Image Resizing and Background Creation: The input image is resized to fit a 512x768 resolution and placed in the center of a larger white background to create a 1024x1536 canvas.

2. Mask Creation: Using the rembg library, a mask is created to detect the object in the image, and the mask is then inverted for inpainting.

3. Inpainting with Stable Diffusion: The Stable Diffusion Inpainting model is applied to the image, using the provided text prompt to guide the generation process.

4. Result: The final image is saved at the specified location.

## Example Results

Input Image:

(./example1.jpg)
