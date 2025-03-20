import argparse
import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import yaml
from easydict import EasyDict
from model.network.hf import HierarchyFlow  # Import your model
from model.losses import VGGLoss  # Required for model loading

def load_model(model_path, device, config):
    """Load the trained model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)
    
    # Remove 'module.' prefix if present (from DistributedDataParallel)
    state_dict = checkpoint['state_dict']
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            new_state_dict[key[7:]] = value  # Remove the "module." prefix
        else:
            new_state_dict[key] = value
    
    model = HierarchyFlow(
        pad_size=config.network.pad_size,  # Use config value for padding size
        in_channel=config.network.in_channel,
        out_channels=config.network.out_channels,  # Use config value for output channels
        weight_type=config.network.weight_type  # Use config value for weight type
    )
    
    # Load the modified state_dict into the model
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model


def process_image(model: HierarchyFlow, content_path, style_path, output_path, device, config):
    """Run the model on an image and save the output"""
    transform = transforms.Compose([
        transforms.Resize((1080, 1920 )),  # Use config values
        transforms.ToTensor()
    ])

    content_img = transform(Image.open(content_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(content_img, content_img)
        output = torch.clamp(output, 0, 1)

    save_image(output, output_path)
    print(f"Converted image saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Style Transfer using HierarchyFlow")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file (.yaml)")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model checkpoint (.pth.tar)")
    parser.add_argument("--content", type=str, required=True, help="Path to the content image (.png)")
    parser.add_argument("--output", type=str, required=True, help="Path to save the converted image (.png)")
    
    args = parser.parse_args()

    # Load the configuration file
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = EasyDict(cfg)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = load_model(args.model, device, cfg)

    # Process the image using the loaded model
    process_image(model, args.content, args.style, args.output, device, cfg)

if __name__ == "__main__":
    main()
