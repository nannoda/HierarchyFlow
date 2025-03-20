import argparse
import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from model.network.hf import HierarchyFlow  # Import your model
from model.losses import VGGLoss  # Required for model loading

def load_model(model_path, device):
    """Load the trained model from checkpoint"""
    checkpoint = torch.load(model_path, map_location=device)
    model = HierarchyFlow(
        pad_size=10,  # Adjust based on your config
        in_channel=3,
        out_channels=[30, 120],  # Adjust based on your config
        weight_type='default'  # Adjust as needed
    )
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    model.eval()
    return model

def process_image(model: HierarchyFlow, content_path, style_path, output_path, device):
    """Run the model on an image and save the output"""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Adjust based on model input size
        transforms.ToTensor()
    ])

    content_img = transform(Image.open(content_path).convert("RGB")).unsqueeze(0).to(device)
    style_img = transform(Image.open(style_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(content_img, style_img)
        output = torch.clamp(output, 0, 1)

    save_image(output, output_path)
    print(f"Converted image saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Style Transfer using HierarchyFlow")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model checkpoint (.pth.tar)")
    parser.add_argument("--content", type=str, required=True, help="Path to the content image (.png)")
    parser.add_argument("--style", type=str, required=True, help="Path to the style image (.png)")
    parser.add_argument("--output", type=str, required=True, help="Path to save the converted image (.png)")
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)
    process_image(model, args.content, args.style, args.output, device)

if __name__ == "__main__":
    main()
