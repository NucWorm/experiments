#!/usr/bin/env python3
"""
Cellpose3 inference script for NucWorm benchmark.

This script performs 3D neuron segmentation using Cellpose3 and generates
instance masks for centroid extraction.
"""

import os
import sys
import argparse
import yaml
import numpy as np
import tifffile as tiff
from pathlib import Path
from tqdm import tqdm
import logging

# Add cellpose to path
try:
    from cellpose import models, io, utils, core
    from cellpose.io import imread
except ImportError:
    print("Error: Cellpose3 not installed. Please install with: pip install cellpose>=3.0.0")
    sys.exit(1)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_output_directories(config):
    """Create output directories for results."""
    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (output_dir / 'masks').mkdir(exist_ok=True)
    (output_dir / 'flows').mkdir(exist_ok=True)
    (output_dir / 'outlines').mkdir(exist_ok=True)
    
    return output_dir


def load_volume(volume_path):
    """Load 3D volume from TIFF file."""
    try:
        volume = tiff.imread(volume_path)
        logger.info(f"Loaded volume: {volume.shape}, dtype: {volume.dtype}")
        
        # Handle different volume formats
        if volume.ndim == 4:
            if volume.shape[0] == 3:
                # Volume is (C=3, Z, Y, X) - need to transpose to (Z, C, Y, X)
                volume = np.transpose(volume, (1, 0, 2, 3))  # (C, Z, Y, X) -> (Z, C, Y, X)
                logger.info(f"Transposed volume from (C, Z, Y, X) to (Z, C, Y, X): {volume.shape}")
            elif volume.shape[1] == 3:
                # Volume is already (Z, C, Y, X) with C=3 RGB channels
                logger.info(f"Volume has RGB channels: {volume.shape}")
            else:
                logger.warning(f"Unexpected 4D volume shape: {volume.shape}")
                return None
            return volume
        elif volume.ndim == 3:
            # Volume is already (Z, Y, X) - add channel dimension
            volume = np.expand_dims(volume, axis=1)  # Add channel dimension
            logger.info(f"Added channel dimension: {volume.shape}")
            return volume
        else:
            logger.warning(f"Unexpected volume shape: {volume.shape}")
            return None
        
    except Exception as e:
        logger.error(f"Error loading volume {volume_path}: {e}")
        return None


def preprocess_volume(volume, config, dataset_name=None):
    """Preprocess volume for Cellpose3 using same approach as nnUNet."""
    if volume is None:
        logger.error("Volume is None, cannot preprocess")
        return None
        
    logger.info(f"Original volume stats: shape={volume.shape}, dtype={volume.dtype}, min={volume.min()}, max={volume.max()}")
    
    # Convert to float32 (same as nnUNet)
    volume = volume.astype(np.float32)
    
    # Apply same clipping as nnUNet: [0, 60000]
    volume = np.clip(volume, 0, 60000)
    logger.info(f"After clipping: min={volume.min()}, max={volume.max()}")
    
    # For RGB volumes, convert to grayscale for Cellpose3
    # Cellpose3 works best with single-channel input
    if volume.ndim == 4 and volume.shape[1] == 3:
        # Convert RGB to grayscale using standard weights: 0.299*R + 0.587*G + 0.114*B
        rgb_weights = np.array([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1)
        volume = np.sum(volume * rgb_weights, axis=1, keepdims=False)  # Remove keepdims to get (Z, Y, X)
        logger.info(f"Converted RGB to grayscale: {volume.shape}")
    
    # Apply z-score normalization (same as nnUNet)
    volume_mean = volume.mean()
    volume_std = volume.std()
    volume = (volume - volume_mean) / volume_std
    logger.info(f"After z-score normalization: mean={volume.mean():.2f}, std={volume.std():.2f}")
    
    # Rescale if specified
    rescale = config['processing']['rescale']
    if not all(sf == 1.0 for sf in rescale):
        from skimage.transform import resize
        # After RGB conversion, volume should be (Z, Y, X)
        new_shape = (
            int(volume.shape[0] * rescale[0]),
            int(volume.shape[1] * rescale[1]),
            int(volume.shape[2] * rescale[2])
        )
        volume = resize(volume, new_shape, order=1, preserve_range=True, anti_aliasing=True)
        logger.info(f"Rescaled volume to: {volume.shape}")
    
    return volume


def segment_volume(model, volume, config, dataset_name=None):
    """Perform 3D segmentation using Cellpose3 with dataset-specific parameters."""
    seg_config = config['segmentation']
    proc_config = config['processing']
    
    # Get dataset-specific diameter
    diameter = get_dataset_diameter(dataset_name, seg_config)
    logger.info(f"Using diameter: {diameter} for dataset: {dataset_name}")
    
    # Set up segmentation parameters
    flow_threshold = seg_config['flow_threshold']
    cellprob_threshold = seg_config['cellprob_threshold']
    min_size = seg_config['min_size']
    stitch_threshold = seg_config['stitch_threshold']
    
    # Get dataset-specific anisotropy
    anisotropy = get_dataset_anisotropy(dataset_name, proc_config['anisotropy'])
    
    # Perform 3D segmentation
    logger.info("Starting Cellpose3 3D segmentation...")
    logger.info(f"Volume shape: {volume.shape}")
    logger.info(f"Anisotropy: {anisotropy}")
    
    # After preprocessing, volume should be grayscale (Z, Y, X)
    # Cellpose3 expects grayscale input, so we use channels=[0,0]
    channels = [0, 0]  # Grayscale, no nuclear channel
    logger.info("Using grayscale volume for segmentation (RGB converted to grayscale)")
    
    # Cellpose3 returns: masks, flows, styles, diams
    # We only need the segmentation masks
    masks, flows, styles, diams = model.eval(
        volume,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
        stitch_threshold=stitch_threshold,
        do_3D=True,  # Enable 3D segmentation
        anisotropy=anisotropy,  # Handle Z vs XY sampling differences
        channels=channels  # Grayscale channels
    )
    
    logger.info(f"3D segmentation complete. Found {masks.max()} cells.")
    return masks, flows, styles


def get_dataset_diameter(dataset_name, seg_config):
    """Get dataset-specific diameter based on the 3μm neuron size."""
    # Default diameter from config
    default_diameter = seg_config['diameter']
    
    if dataset_name is None:
        return default_diameter
    
    # Dataset-specific diameters based on 3μm neuron size
    dataset_diameters = {
        'yemini21': 15,      # 15 pixels in X,Y directions
        'nejatbakhsh20': 15, # 15 pixels in X,Y directions  
        'wen20': 10          # 10 pixels in X,Y directions
    }
    
    return dataset_diameters.get(dataset_name, default_diameter)


def get_dataset_anisotropy(dataset_name, default_anisotropy):
    """Get dataset-specific anisotropy based on resolution data."""
    if dataset_name is None:
        return default_anisotropy
    
    # Dataset-specific anisotropy based on resolution data from your screenshot
    # anisotropy = Z_resolution / XY_resolution
    dataset_anisotropy = {
        'yemini21': 0.75 / 0.21,      # 0.75μm Z / 0.21μm XY ≈ 3.57
        'nejatbakhsh20': 1.5 / 0.21,  # 1.5μm Z / 0.21μm XY ≈ 7.14
        'wen20': 1.5 / 0.32           # 1.5μm Z / 0.32μm XY ≈ 4.69
    }
    
    return dataset_anisotropy.get(dataset_name, default_anisotropy)


def save_results(masks, flows, styles, volume_name, output_dir, config):
    """Save segmentation results."""
    # Create output directories if they don't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save masks
    if config['output']['save_masks']:
        masks_dir = output_dir / 'masks'
        masks_dir.mkdir(parents=True, exist_ok=True)
        mask_path = masks_dir / f"{volume_name}_masks.tiff"
        tiff.imwrite(mask_path, masks.astype(np.uint16), compression='zlib')
        logger.info(f"Saved masks: {mask_path}")
    
    # Save flows
    if config['output']['save_flows']:
        flows_dir = output_dir / 'flows'
        flows_dir.mkdir(parents=True, exist_ok=True)
        flow_path = flows_dir / f"{volume_name}_flows.tiff"
        # Handle flows as list (Cellpose3 returns flows as list)
        if isinstance(flows, list):
            logger.warning("Flows returned as list, skipping flow saving")
        else:
            tiff.imwrite(flow_path, flows.astype(np.float32), compression='zlib')
            logger.info(f"Saved flows: {flow_path}")
    
    # Save outlines
    if config['output']['save_outlines']:
        outlines_dir = output_dir / 'outlines'
        outlines_dir.mkdir(parents=True, exist_ok=True)
        from cellpose.utils import outlines_list
        outlines = outlines_list(masks)
        outline_path = outlines_dir / f"{volume_name}_outlines.npy"
        np.save(outline_path, outlines)
        logger.info(f"Saved outlines: {outline_path}")
    
    return masks


def process_dataset(input_dir, output_dir, config):
    """Process all volumes in a dataset."""
    input_path = Path(input_dir)
    dataset_name = input_path.name
    
    logger.info(f"Processing dataset: {dataset_name}")
    
    # Initialize Cellpose3 model
    model_config = config['model']
    
    # Check GPU availability
    use_GPU = core.use_gpu()
    logger.info(f"GPU activated? {use_GPU}")
    
    # Initialize Cellpose3 model for segmentation only (no denoising)
    model = models.Cellpose(
        gpu=model_config['gpu'],
        model_type=model_config['model_type']
    )
    
    # Create dataset output directory
    dataset_output_dir = output_dir / dataset_name
    dataset_output_dir.mkdir(exist_ok=True)
    
    # Process each case directory
    case_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    logger.info(f"Found {len(case_dirs)} case directories")
    
    for case_dir in tqdm(case_dirs, desc=f"Processing {dataset_name}"):
        case_name = case_dir.name
        logger.info(f"Processing case: {case_name}")
        
        # Find TIFF files in case directory
        tiff_files = list(case_dir.glob("*.tiff")) + list(case_dir.glob("*.tif"))
        
        if not tiff_files:
            logger.warning(f"No TIFF files found in {case_dir}")
            continue
        
        # Process each TIFF file
        for tiff_file in tiff_files:
            # Use the original filename without duplicating case_name
            volume_name = tiff_file.stem
            logger.info(f"Processing volume: {volume_name}")
            
            # Load and preprocess volume
            volume = load_volume(tiff_file)
            if volume is None:
                logger.error(f"Failed to load volume: {tiff_file}")
                continue
            
            volume = preprocess_volume(volume, config, dataset_name)
            if volume is None:
                logger.error(f"Failed to preprocess volume: {tiff_file}")
                continue
            
            # Perform segmentation
            masks, flows, styles = segment_volume(model, volume, config, dataset_name)
            
            # Save results
            case_output_dir = dataset_output_dir / case_name
            case_output_dir.mkdir(exist_ok=True)
            
            save_results(masks, flows, styles, volume_name, case_output_dir, config)
    
    logger.info(f"Completed processing dataset: {dataset_name}")


def main():
    parser = argparse.ArgumentParser(description="Cellpose3 inference for NucWorm benchmark")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--input_dir', type=str,
                       help='Input directory (overrides config)')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory (overrides config)')
    parser.add_argument('--dataset', type=str, choices=['nejatbakhsh20', 'wen20', 'yemini21'],
                       help='Specific dataset to process')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.input_dir:
        config['data']['input_dir'] = args.input_dir
    if args.output_dir:
        config['data']['output_dir'] = args.output_dir
    
    # Setup output directories
    output_dir = setup_output_directories(config)
    
    # Process datasets
    input_dir = Path(config['data']['input_dir'])
    
    if args.dataset:
        # Process specific dataset
        dataset_path = input_dir / args.dataset
        if dataset_path.exists():
            process_dataset(dataset_path, output_dir, config)
        else:
            logger.error(f"Dataset {args.dataset} not found at {dataset_path}")
    else:
        # Process all datasets
        datasets = ['nejatbakhsh20', 'wen20', 'yemini21']
        for dataset in datasets:
            dataset_path = input_dir / dataset
            if dataset_path.exists():
                process_dataset(dataset_path, output_dir, config)
            else:
                logger.warning(f"Dataset {dataset} not found at {dataset_path}")
    
    logger.info("Cellpose3 inference completed!")


if __name__ == "__main__":
    main()
