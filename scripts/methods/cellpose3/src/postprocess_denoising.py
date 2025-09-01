#!/usr/bin/env python3
"""
Post-processing script for Cellpose3 denoising results.

This script extracts centroids from segmentation masks and organizes them
for evaluation in the NucWorm benchmark.
"""

import argparse
import logging
import sys
from pathlib import Path
import yaml
import numpy as np
from tqdm import tqdm
from scipy import ndimage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


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


def process_mask_file(mask_file, output_path):
    """Process a single mask file and extract centroids."""
    try:
        # Load mask
        mask = np.load(mask_file) if mask_file.suffix == '.npy' else None
        if mask is None:
            # Try loading as TIFF
            import tifffile as tiff
            mask = tiff.imread(mask_file)
        
        logger.info(f"Loaded mask: {mask.shape}, dtype: {mask.dtype}")
        
        # Extract centroids
        centroids = extract_centroids(mask)
        
        # Save centroids
        np.save(output_path, centroids)
        logger.info(f"Saved centroids: {output_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing {mask_file}: {e}")
        return False


def postprocess_dataset(output_dir, config):
    """Post-process all datasets."""
    output_path = Path(output_dir)
    
    # Create center_point directory
    center_point_dir = output_path / 'center_point'
    center_point_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each dataset
    for dataset_dir in sorted(output_path.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name == 'center_point':
            continue
            
        dataset_name = dataset_dir.name
        logger.info(f"Post-processing dataset: {dataset_name}")
        
        # Create dataset output directory
        dataset_output_dir = center_point_dir / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find case directories
        case_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
        logger.info(f"Found {len(case_dirs)} case directories")
        
        # Process each case
        for case_dir in tqdm(case_dirs, desc=f"Post-processing {dataset_name}"):
            case_name = case_dir.name
            logger.info(f"Post-processing case: {case_name}")
            
            # Find mask files
            mask_files = list(case_dir.glob("**/*_masks.tiff"))
            
            if not mask_files:
                logger.warning(f"No mask files found in {case_dir}")
                continue
            
            # Process each mask file
            for mask_file in mask_files:
                # Create output filename (following NucWorm naming convention)
                # Convert from: case_name_volume_name_masks.tiff
                # To: case_name_volume_name_im_points.npy
                base_name = mask_file.stem.replace('_masks', '')
                output_filename = f"{base_name}_im_points.npy"
                output_path = dataset_output_dir / output_filename
                
                # Process mask
                success = process_mask_file(mask_file, output_path)
                if success:
                    logger.info(f"✅ Processed: {mask_file.name} -> {output_filename}")
                else:
                    logger.error(f"❌ Failed to process: {mask_file.name}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Post-process Cellpose3 denoising results")
    parser.add_argument("--config", type=str, default="config_denoising.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--output_dir", type=str,
                       help="Override output directory from config")
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override output directory if provided
    if args.output_dir:
        config['data']['output_dir'] = args.output_dir
    
    logger.info("Starting Cellpose3 denoising post-processing...")
    logger.info(f"Output directory: {config['data']['output_dir']}")
    
    # Post-process datasets
    postprocess_dataset(
        config['data']['output_dir'],
        config
    )
    
    logger.info("Cellpose3 denoising post-processing completed!")


if __name__ == "__main__":
    main()
