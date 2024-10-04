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

You can run the script from the command line with the following parameters:

```bash
python run.py --image <image_path> --text-prompt "<your_text_prompt>" --output <output_path>
```

Example Command:

```bash
python run.py --image ./example.jpg --text-prompt "A product in a kitchen used in meal preparation" --output ./generated.png
```

