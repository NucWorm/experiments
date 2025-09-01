#!/usr/bin/env python
import os
import glob
import argparse
import numpy as np
import tifffile
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import tempfile
import shutil
import subprocess

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

#############################################
# Centroid Extraction Function
#############################################
def run_centroid_extraction(heatmap_path, output_dir, gt_folder=None):
    """
    Run centroid extraction on a heatmap file.
    
    Args:
        heatmap_path (str): Path to heatmap TIFF file
        output_dir (str): Output directory for centroids
        gt_folder (str): Optional ground truth folder for evaluation
    
    Returns:
        str: Path to output centroids file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get base name for output
    base_name = os.path.splitext(os.path.basename(heatmap_path))[0]
    centroids_output = os.path.join(output_dir, f"{base_name}_points.npy")
    
    # The postprocessing script expects input_folder and output_folder
    # We need to create a temporary folder with just this heatmap file
    temp_input_dir = tempfile.mkdtemp()
    temp_heatmap_path = os.path.join(temp_input_dir, os.path.basename(heatmap_path))
    shutil.copy2(heatmap_path, temp_heatmap_path)
    
    # Build postprocessing command
    cmd = [
        "python", "src/postprocess.py",
        "--input_folder", temp_input_dir,
        "--output_folder", output_dir,
        "--min_distance", "5",
        "--threshold_abs", "0.1"
    ]
    
    if gt_folder:
        cmd.extend(["--gt_folder", gt_folder])
    
    print(f"Running centroid extraction: {' '.join(cmd)}")
    
    # Run postprocessing
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Centroid extraction completed successfully")
        print(f"STDOUT: {result.stdout}")
        if result.stderr:
            print(f"STDERR: {result.stderr}")
        
        # Clean up temporary directory
        shutil.rmtree(temp_input_dir)
        
        return centroids_output
    except subprocess.CalledProcessError as e:
        print(f"Centroid extraction failed with error: {e}")
        print(f"STDOUT: {e.stdout}")
        print(f"STDERR: {e.stderr}")
        
        # Clean up temporary directory
        shutil.rmtree(temp_input_dir)
        
        return None

#############################################
# Sliding-window Inference Function
#############################################
def sliding_window_inference(model, image, patch_size, stride, device):
    """
    Perform sliding-window inference on a 3D volume.
    
    Args:
        model: The segmentation model.
        image: Input image tensor of shape (C, D, H, W) on CPU.
        patch_size (tuple): The patch size (pD, pH, pW).
        stride (tuple): The stride (sD, sH, sW) for sliding windows.
        device: Torch device.
    
    Returns:
        output_tensor: Full-volume prediction tensor of shape (n_classes, D, H, W).
    """
    model.eval()
    # Record original image dimensions
    original_shape = image.shape  # (C, orig_D, orig_H, orig_W)
    C, orig_D, orig_H, orig_W = original_shape
    pD, pH, pW = patch_size

    # Pad the image if any dimension is smaller than the patch size
    pad_d = max(0, pD - orig_D)
    pad_h = max(0, pH - orig_H)
    pad_w = max(0, pW - orig_W)
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        # Padding format: (w_left, w_right, h_left, h_right, d_left, d_right)
        pad = (pad_w // 2, pad_w - pad_w // 2,
               pad_h // 2, pad_h - pad_h // 2,
               pad_d // 2, pad_d - pad_d // 2)
        image = F.pad(image, pad, mode='constant', value=0)
    else:
        pad = (0, 0, 0, 0, 0, 0)
    
    # Update dimensions after padding
    C, D, H, W = image.shape
    output_tensor = torch.zeros((1, D, H, W), dtype=torch.float32)
    count_tensor = torch.zeros((1, D, H, W), dtype=torch.float32)
    
    sD, sH, sW = stride

    # Calculate sliding window starting indices for each dimension
    d_starts = list(range(0, D - pD + 1, sD)) if D >= pD else [0]
    if D >= pD and (len(d_starts) == 0 or d_starts[-1] != D - pD):
        d_starts.append(D - pD)
        
    h_starts = list(range(0, H - pH + 1, sH)) if H >= pH else [0]
    if H >= pH and (len(h_starts) == 0 or h_starts[-1] != H - pH):
        h_starts.append(H - pH)
        
    w_starts = list(range(0, W - pW + 1, sW)) if W >= pW else [0]
    if W >= pW and (len(w_starts) == 0 or w_starts[-1] != W - pW):
        w_starts.append(W - pW)
    
    # Inference using sliding windows
    total_patches = len(d_starts) * len(h_starts) * len(w_starts)
    print(f"Total patches to process: {total_patches}")
    
    with torch.no_grad():
        patch_count = 0
        for d in tqdm(d_starts, desc="Processing D dimension", leave=False):
            for h in tqdm(h_starts, desc="Processing H dimension", leave=False):
                for w in tqdm(w_starts, desc="Processing W dimension", leave=False):
                    patch_count += 1
                    # Extract patch from image
                    patch = image[:, d:d+pD, h:h+pH, w:w+pW]  # shape: (C, pD, pH, pW)
                    patch = patch.unsqueeze(0).to(device)      # shape: (1, C, pD, pH, pW)
                    pred = model(patch)
                    # Apply Sigmoid to obtain probabilities
                    pred = torch.sigmoid(pred)  # shape: (1, n_classes, pD, pH, pW)
                    pred = pred.squeeze(0)      # shape: (n_classes, pD, pH, pW)
                    output_tensor[:, d:d+pD, h:h+pH, w:w+pW] += pred.cpu()
                    count_tensor[:, d:d+pD, h:h+pH, w:w+pW] += 1
                    
    # Average overlapping patches
    output_tensor /= count_tensor

    # Crop the output back to the original image size if padding was applied
    if any(pad):
        # pad order: (w_left, w_right, h_left, h_right, d_left, d_right)
        w_left, w_right, h_left, h_right, d_left, d_right = pad
        output_tensor = output_tensor[:, 
                                      d_left:d_left+orig_D, 
                                      h_left:h_left+orig_H, 
                                      w_left:w_left+orig_W]
    
    return output_tensor

def build_nnUNet_model():
    
    model = get_network_from_plans(
        arch_class_name="dynamic_network_architectures.architectures.unet.PlainConvUNet",
        arch_kwargs= {
                    "n_stages": 6,
                    "features_per_stage": [
                        32,
                        64,
                        128,
                        256,
                        320,
                        320
                    ],
                    "conv_op": "torch.nn.modules.conv.Conv3d",
                    "kernel_sizes": [
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ],
                        [
                            3,
                            3,
                            3
                        ]
                    ],
                    "strides": [
                        [
                            1,
                            1,
                            1
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            2,
                            2,
                            2
                        ],
                        [
                            1,
                            2,
                            2
                        ]
                    ],
                    "n_conv_per_stage": [
                        2,
                        2,
                        2,
                        2,
                        2,
                        2
                    ],
                    "n_conv_per_stage_decoder": [
                        2,
                        2,
                        2,
                        2,
                        2
                    ],
                    "conv_bias": True,
                    "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
                    "norm_op_kwargs": {
                        "eps": 1e-05,
                        "affine": True
                    },
                    "dropout_op": None,
                    "dropout_op_kwargs": None,
                    "nonlin": "torch.nn.LeakyReLU",
                    "nonlin_kwargs": {
                        "inplace": True
                    }
                },
        arch_kwargs_req_import=["conv_op", "norm_op", "dropout_op", "nonlin"],
        input_channels=3,
        output_channels=1,
        allow_init=True,
        deep_supervision=False,
    )
    return model
#############################################
# Main prediction function using argparse
#############################################
def main(args):
    print(f"=== Starting main function ===")
    print(f"Input path: {args.input}")
    print(f"Output directory: {args.output_dir}")
    print(f"Model path: {args.model_path}")
    print(f"Patch size: {args.patch_size}")
    print(f"Stride: {args.stride}")
    print(f"Device: {args.device}")
    
    # Parse patch size and stride from strings (e.g., "96,96,192")
    patch_size = tuple(map(int, args.patch_size.split(',')))
    stride = tuple(map(int, args.stride.split(',')))
    print(f"Parsed patch_size: {patch_size}")
    print(f"Parsed stride: {stride}")

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() or args.device=='cpu' else 'cpu')
    print(f"Using device: {device}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Create model and load saved weights.
    print("Building model...")
    model = build_nnUNet_model().to(device)
    print("Model built successfully")
    
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    print(f"Model file exists: {args.model_path}")
    
    print("Loading model weights...")
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print("Loaded model weights.")

    # Determine if input is a file or directory
    print(f"Checking input path: {args.input}")
    if os.path.isdir(args.input):
        print(f"Input is a directory")
        # Look for both .tiff and .tif files recursively in subdirectories
        tiff_files = glob.glob(os.path.join(args.input, '**/*.tiff'), recursive=True)
        tif_files = glob.glob(os.path.join(args.input, '**/*.tif'), recursive=True)
        file_list = sorted(tiff_files + tif_files)
        print(f"Found {len(tiff_files)} .tiff files and {len(tif_files)} .tif files")
        print(f"Total files: {len(file_list)}")
        print(f"File list: {file_list}")
    elif os.path.isfile(args.input):
        print(f"Input is a single file")
        file_list = [args.input]
        print(f"File list: {file_list}")
    else:
        raise ValueError("Input path is neither a file nor a directory.")

    print(f"Creating output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory created/verified")

    print(f"=== Starting file processing loop ===")
    print(f"Total files to process: {len(file_list)}")
    
    # Loop over each file and perform prediction.
    for file_path in tqdm(file_list, desc="Processing files", unit="file"):
        print(f"")
        print(f"--- Processing file ---")
        print(f"File: {file_path}")
        print(f"Processing: {file_path}")
        # Read the image using tifffile.
        print(f"Reading image with tifffile...")
        img = tifffile.imread(file_path)
        print(f"Image loaded, shape: {img.shape}, dtype: {img.dtype}")
        
        # If image is 3D (D, H, W), add a channel dimension.
        if img.ndim == 3:
            print(f"Adding channel dimension to 3D image")
            img = np.expand_dims(img, axis=0)
            print(f"Image shape after adding channel: {img.shape}")
        
        # Handle channel duplication if requested
        if args.duplicate_channels and img.shape[0] == 1:
            print(f"Duplicating single channel to {args.input_channels} channels")
            img = np.repeat(img, args.input_channels, axis=0)
            print(f"Image shape after channel duplication: {img.shape}")
        
        # Convert image to float32.
        print(f"Converting to float32...")
        img = img.astype(np.float32)
        print(f"Image dtype after conversion: {img.dtype}")
        
        if img.shape[1] == 3:
            print(f"Transposing image from (D, C, H, W) to (C, D, H, W)")
            img = np.transpose(img, (1, 0, 2, 3))
            print(f"Image shape after transpose: {img.shape}")
        
        # Convert to torch tensor.
        print(f"Converting to torch tensor...")
        img_tensor = torch.from_numpy(img)  # shape: (C, D, H, W)
        print(f"Tensor shape: {img_tensor.shape}, dtype: {img_tensor.dtype}")
        
        # clip img.
        print(f"Clipping image values to [0, 60000]...")
        img_tensor = torch.clamp(img_tensor, 0, 60000)
        print(f"Image range after clipping: [{img_tensor.min():.2f}, {img_tensor.max():.2f}]")
        
        # z-score
        print(f"Applying z-score normalization...")
        img_tensor = (img_tensor - img_tensor.mean()) / img_tensor.std()
        print(f"Image stats after normalization: mean={img_tensor.mean():.2f}, std={img_tensor.std():.2f}")

        # Check if sliding-window inference is needed.
        C, D, H, W = img_tensor.shape
        print(f"Image dimensions: C={C}, D={D}, H={H}, W={W}")
        print(f"Patch size: {patch_size}")
        
        if (D > patch_size[0]) or (H > patch_size[1]) or (W > patch_size[2]):
            print("Using sliding-window inference.")
            # Make sure the tensor is on CPU for sliding-window function.
            print(f"Moving tensor to CPU for sliding-window inference...")
            img_tensor_cpu = img_tensor.to('cpu')
            print(f"Starting sliding-window inference...")
            pred = sliding_window_inference(model, img_tensor_cpu, patch_size, stride, device)
            print(f"Sliding-window inference completed")
        else:
            # Direct inference
            print("Using direct inference (no sliding-window needed)")
            img_tensor = img_tensor.unsqueeze(0).to(device)  # (1, C, D, H, W)
            print(f"Input tensor shape for model: {img_tensor.shape}")
            with torch.no_grad():
                print(f"Running model inference...")
                pred = model(img_tensor).squeeze(0)  # (n_classes, D, H, W)
                print(f"Model inference completed")

        print(f"Prediction tensor shape: {pred.shape}, dtype: {pred.dtype}")
        
        # Convert prediction to numpy array.
        print(f"Converting prediction to numpy...")
        pred_np = pred.cpu().numpy()
        print(f"Prediction numpy shape: {pred_np.shape}, dtype: {pred_np.dtype}")
        print(f"Prediction max: {pred_np.max()}, min: {pred_np.min()}")
        
        # Normalize prediction between 0 and 1.
        print(f"Normalizing prediction...")
        pred_np = (pred_np - pred_np.min()) / (pred_np.max() - pred_np.min())
        pred_np = pred_np * 255
        print(f"Prediction after normalization: max={pred_np.max()}, min={pred_np.min()}")
        
        # If prediction has 1 channel, remove channel dimension.
        if pred_np.shape[0] == 1:
            print(f"Removing single channel dimension")
            pred_np = pred_np[0]
            print(f"Final prediction shape: {pred_np.shape}")
        
        # Create output filename.
        base_name = os.path.basename(file_path)
        out_path = os.path.join(args.output_dir, base_name)
        print(f"Output path: {out_path}")
        
        # Save prediction as tif.
        print(f"Saving prediction to TIFF...")
        tifffile.imwrite(out_path, pred_np.astype(np.uint8))
        print(f"Successfully saved prediction to: {out_path}")
        
        # Verify file was created
        if os.path.exists(out_path):
            file_size = os.path.getsize(out_path)
            print(f"File verification: {out_path} exists, size: {file_size} bytes")
        else:
            print(f"ERROR: File was not created: {out_path}")
            continue
        
        # Run centroid extraction if requested
        if args.extract_centroids:
            print(f"Running centroid extraction on: {os.path.basename(out_path)}")
            centroids_dir = os.path.join(args.output_dir, "centroids")
            centroids_path = run_centroid_extraction(
                out_path,
                centroids_dir,
                args.gt_folder
            )
            if centroids_path:
                print(f"Centroids saved to: {centroids_path}")
            else:
                print(f"Centroid extraction failed for: {os.path.basename(out_path)}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="3D TIF Prediction with UNet3DSmall")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to input file or directory containing tif files.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the prediction results.")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the saved model state dictionary (e.g., unet3d_small_model.pth).")
    parser.add_argument("--patch_size", type=str, default="32,96,64",
                        help="Patch size (D,H,W) for inference (default: 96,96,192).")
    parser.add_argument("--stride", type=str, default="16,48,32",
                        help="Stride (D,H,W) for sliding window inference (default: 48,48,96).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for inference ('cuda' or 'cpu').")
    parser.add_argument("--input_channels", type=int, default=3,
                        help="Number of input channels for the model.")
    parser.add_argument("--duplicate_channels", action="store_true",
                        help="Duplicate single channel data to match model input channels.")
    parser.add_argument("--extract_centroids", action="store_true",
                        help="Extract centroids from heatmaps using postprocessing.")
    parser.add_argument("--gt_folder", type=str, default=None,
                        help="Ground truth folder for evaluation (optional).")
    args = parser.parse_args()
    main(args)
