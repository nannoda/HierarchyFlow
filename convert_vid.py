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
import subprocess
import cv2


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


def process_image(model: HierarchyFlow, content_img, device, config):
    """Apply style transfer to an image tensor"""
    with torch.no_grad():
        output = model(content_img, content_img)
        output = torch.clamp(output, 0, 1)
    return output


def process_video(model, video_path, output_video_path, device, config):
    """Process a video by applying style transfer to each frame"""
    
    # Set up the video capture and writer
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create temporary directory for frames
    temp_dir = "temp_frames"
    os.makedirs(temp_dir, exist_ok=True)
    
    frame_files = []
    frame_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame to PIL image and apply transformations
        frame_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        transform = transforms.Compose([
            transforms.Resize((1080, 1920)),  # You can adjust the size if necessary
            transforms.ToTensor()
        ])
        
        content_img = transform(frame_img).unsqueeze(0).to(device)
        
        # Process the frame with style transfer
        output = process_image(model, content_img, device, config)
        
        # Convert the tensor back to image format (CPU and from Tensor to PIL)
        output_img = output.squeeze().cpu().permute(1, 2, 0).numpy()
        output_img = (output_img * 255).astype('uint8')
        
        # Save the processed frame
        frame_filename = os.path.join(temp_dir, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(frame_filename, cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))
        frame_files.append(frame_filename)
        frame_idx += 1
    
    cap.release()

    # Now create the output video from processed frames using ffmpeg (VP9 codec)
    frame_pattern = os.path.join(temp_dir, "frame_%04d.png")
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-framerate", str(frame_rate), "-i", frame_pattern, 
        "-c:v", "libopenh264", "-pix_fmt", "yuv420p", 
        output_video_path
    ]
    
    subprocess.run(ffmpeg_cmd)
    
    # Clean up the temporary frames
    for frame_file in frame_files:
        os.remove(frame_file)
    os.rmdir(temp_dir)
    print(f"Video saved to {output_video_path}")


def main():
    parser = argparse.ArgumentParser(description="Style Transfer on Video using HierarchyFlow")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file (.yaml)")
    parser.add_argument("--model", type=str, required=True, help="Path to the trained model checkpoint (.pth.tar)")
    parser.add_argument("--video", type=str, required=True, help="Path to the input video file (.mp4 or .avi)")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output video (.webm or .mp4)")
    
    args = parser.parse_args()

    # Load the configuration file
    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    cfg = EasyDict(cfg)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model
    model = load_model(args.model, device, cfg)

    # Process the video using the loaded model
    process_video(model, args.video, args.output, device, cfg)


if __name__ == "__main__":
    main()
