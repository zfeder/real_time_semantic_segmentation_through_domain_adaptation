import os
import torch
import numpy as np
import time
from fvcore.nn import FlopCountAnalysis, flop_count_table
from config import config
from models.bisenet import BiSeNet
from models.deeplab import get_deeplab_v2

def load_model(config):
    device = config["device"]
    model_name = config["model_name"]
    num_classes = config["num_classes"]

    if model_name == "DeepLabV2":
        model = get_deeplab_v2(num_classes=num_classes, pretrain=False).to(device)
    elif model_name == "BiSeNet":
        model = BiSeNet(num_classes=num_classes, context_path="resnet18").to(device)
    else:
        raise ValueError("Only 'DeepLabV2' and 'BiSeNet' are supported.")

    checkpoint_dir = config["checkpoint_dir"]
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith(".pth")]
    if not checkpoint_files:
        raise FileNotFoundError("No checkpoint found for evaluation.")

    latest_checkpoint = sorted(checkpoint_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))[-1]
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    return model

def compute_flops(model: torch.nn.Module, height: int, width: int):
    image = torch.zeros((1, 3, height, width))
    flops = FlopCountAnalysis(model.cpu(), image)
    table = flop_count_table(flops)
    
    n_param_table = table.split('\n')[2].split('|')[2].strip()
    flops_table = table.split('\n')[2].split('|')[3].strip()

    return {'Parameters': n_param_table, 'FLOPS': flops_table}

def compute_latency_and_fps(model: torch.nn.Module, height: int, width: int, iterations: int = 100, device: str = 'cuda'):
    latencies = []
    fps_records = []
    
    model.to(device)
    
    with torch.no_grad():
        for _ in range(iterations):
            image = torch.zeros((1, 3, height, width)).to(device)
            start_time = time.time()
            model(image)
            end_time = time.time() 
            
            latency = end_time - start_time
            latencies.append(latency)
            fps_records.append(1 / latency)

    return {
        'mean_latency': np.mean(latencies),
        'std_latency': np.std(latencies),
        'mean_fps': np.mean(fps_records),
        'std_fps': np.std(fps_records)
    }

if __name__ == "__main__":
    model = load_model(config)
    
    flop_results = compute_flops(model, height=config["image_height"], width=config["image_width"])
    print("\nFLOP Results:")
    print(f"Parameters: {flop_results['Parameters']}")
    print(f"FLOPS: {flop_results['FLOPS']}")

    latency_fps_results = compute_latency_and_fps(model, height=config["image_height"], width=config["image_width"], iterations=100, device=config["device"])
    print("\nLatency and FPS Results:")
    print(f"Mean Latency: {latency_fps_results['mean_latency']:.6f} seconds")
    print(f"Std Latency: {latency_fps_results['std_latency']:.6f} seconds")
    print(f"Mean FPS: {latency_fps_results['mean_fps']:.2f}")
    print(f"Std FPS: {latency_fps_results['std_fps']:.2f}")
