#!/usr/bin/env python3
"""
Cellpose-SAM post-processing script for NucWorm benchmark.

This script extracts centroid coordinates from Cellpose-SAM instance segmentation masks
and saves them in the format expected by the NucWorm evaluation framework.
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
from scipy import ndimage

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_output_directories(config):
    """Create output directories for centroids."""
    output_dir = Path(config['data']['output_dir'])
    center_point_dir = output_dir / 'center_point'
    center_point_dir.mkdir(parents=True, exist_ok=True)
    
    return center_point_dir


def load_mask(mask_path):
    """Load instance segmentation mask."""
    try:
        mask = tiff.imread(mask_path)
        logger.info(f"Loaded mask: {mask.shape}, dtype: {mask.dtype}, max_label: {mask.max()}")
        return mask
    except Exception as e:
        logger.error(f"Error loading mask {mask_path}: {e}")
        return None


def extract_centroids(mask):
    """Extract centroid coordinates from instance segmentation mask using scipy."""
    centroids = []
    
    # Get unique labels (excluding background)
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]  # Remove background
    
    logger.info(f"Found {len(unique_labels)} instances in mask")
    
    # Extract centroids using scipy center of mass
    for label in unique_labels:
        # Create binary mask for this instance
        binary_mask = (mask == label)
        
        # Calculate center of mass (returns Z, Y, X coordinates)
        center_of_mass = ndimage.center_of_mass(binary_mask)
        centroids.append(center_of_mass)
    
    if centroids:
        centroids = np.array(centroids)
        logger.info(f"Extracted {len(centroids)} centroids using center of mass")
    else:
        logger.warning("No centroids found in mask")
        centroids = np.empty((0, 3))
    
    return centroids


def extract_centroids_alternative(mask):
    """Alternative centroid extraction method using center of mass."""
    centroids = []
    
    # Get unique labels (excluding background)
    unique_labels = np.unique(mask)
    unique_labels = unique_labels[unique_labels > 0]
    
    for label in unique_labels:
        # Create binary mask for this instance
        binary_mask = (mask == label)
        
        # Calculate center of mass
        center_of_mass = ndimage.center_of_mass(binary_mask)
        centroids.append(center_of_mass)
    
    if centroids:
        centroids = np.array(centroids)
        logger.info(f"Extracted {len(centroids)} centroids using center of mass")
    else:
        logger.warning("No centroids found in mask")
        centroids = np.empty((0, 3))
    
    return centroids


def save_centroids(centroids, output_path):
    """Save centroids to NPY file in NucWorm format."""
    try:
        # Convert to float32 for consistency
        centroids = centroids.astype(np.float32)
        
        # Save as NPY file
        np.save(output_path, centroids)
        logger.info(f"Saved {len(centroids)} centroids to {output_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error saving centroids to {output_path}: {e}")
        return False


def process_mask_file(mask_path, output_path, method='regionprops'):
    """Process a single mask file and extract centroids."""
    logger.info(f"Processing mask: {mask_path}")
    
    # Load mask
    mask = load_mask(mask_path)
    if mask is None:
        return False
    
    # Extract centroids
    if method == 'regionprops':
        centroids = extract_centroids(mask)
    elif method == 'center_of_mass':
        centroids = extract_centroids_alternative(mask)
    else:
        logger.error(f"Unknown centroid extraction method: {method}")
        return False
    
    # Save centroids
    success = save_centroids(centroids, output_path)
    
    return success


def process_dataset(input_dir, output_dir, config):
    """Process all masks in a dataset."""
    input_path = Path(input_dir)
    dataset_name = input_path.name
    
    logger.info(f"Processing dataset: {dataset_name}")
    
    # Create dataset output directory
    dataset_output_dir = output_dir / dataset_name
    dataset_output_dir.mkdir(exist_ok=True)
    
    # Process each case directory
    case_dirs = [d for d in input_path.iterdir() if d.is_dir()]
    logger.info(f"Found {len(case_dirs)} case directories")
    
    total_masks = 0
    successful_masks = 0
    
    for case_dir in tqdm(case_dirs, desc=f"Processing {dataset_name}"):
        case_name = case_dir.name
        logger.info(f"Processing case: {case_name}")
        
        # Find mask files in case directory
        mask_files = list(case_dir.glob("*_masks.tiff")) + list(case_dir.glob("*_masks.tif"))
        
        if not mask_files:
            logger.warning(f"No mask files found in {case_dir}")
            continue
        
        # Process each mask file
        for mask_file in mask_files:
            total_masks += 1
            
            # Create output filename (following NucWorm naming convention)
            # Convert from: case_name_volume_name_masks.tiff
            # To: case_name_volume_name_im_points.npy
            base_name = mask_file.stem.replace('_masks', '')
            output_filename = f"{base_name}_im_points.npy"
            output_path = dataset_output_dir / output_filename
            
            # Process mask
            success = process_mask_file(mask_file, output_path)
            if success:
                successful_masks += 1
    
    logger.info(f"Completed processing dataset: {dataset_name}")
    logger.info(f"Successfully processed {successful_masks}/{total_masks} masks")
    
    return successful_masks, total_masks


def main():
    parser = argparse.ArgumentParser(description="Cellpose-SAM post-processing for NucWorm benchmark")
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--input_dir', type=str,
                       help='Input directory containing masks (overrides config)')
    parser.add_argument('--output_dir', type=str,
                       help='Output directory for centroids (overrides config)')
    parser.add_argument('--dataset', type=str, choices=['nejatbakhsh20', 'wen20', 'yemini21'],
                       help='Specific dataset to process')
    parser.add_argument('--method', type=str, choices=['regionprops', 'center_of_mass'], 
                       default='regionprops',
                       help='Centroid extraction method')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.input_dir:
        config['data']['output_dir'] = args.input_dir  # Input is the output from inference
    if args.output_dir:
        config['data']['output_dir'] = args.output_dir
    
    # Setup output directories
    center_point_dir = setup_output_directories(config)
    
    # Process datasets
    # Input directory is the output from inference (contains masks)
    input_dir = Path(config['data']['output_dir'])
    
    total_successful = 0
    total_masks = 0
    
    if args.dataset:
        # Process specific dataset
        dataset_path = input_dir / args.dataset
        if dataset_path.exists():
            successful, masks = process_dataset(dataset_path, center_point_dir, config)
            total_successful += successful
            total_masks += masks
        else:
            logger.error(f"Dataset {args.dataset} not found at {dataset_path}")
    else:
        # Process all datasets
        datasets = ['nejatbakhsh20', 'wen20', 'yemini21']
        for dataset in datasets:
            dataset_path = input_dir / dataset
            if dataset_path.exists():
                successful, masks = process_dataset(dataset_path, center_point_dir, config)
                total_successful += successful
                total_masks += masks
            else:
                logger.warning(f"Dataset {dataset} not found at {dataset_path}")
    
    logger.info("Cellpose-SAM post-processing completed!")
    logger.info(f"Overall: Successfully processed {total_successful}/{total_masks} masks")
    
    if total_successful == 0:
        logger.error("No masks were successfully processed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
