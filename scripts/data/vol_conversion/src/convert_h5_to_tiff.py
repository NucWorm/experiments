#!/usr/bin/env python3
"""
Simple H5 to TIFF conversion script.
Converts neuropal H5 volumes to TIFF stacks based on actual data structure.
"""

import os
import glob
import argparse
import h5py
import numpy as np
import tifffile
from pathlib import Path

def find_neuropal_h5_files(dataset=None):
    """Find .h5 files in the neuropal subdirectories."""
    neuropal_base = "/projects/weilab/dataset/NucWorm/neuropal"
    h5_files = []
    
    if dataset:
        # Process only the specified dataset
        subdirs = [dataset]
    else:
        # Process all datasets
        subdirs = ['nejatbakhsh20', 'yemini21', 'wen20']
    
    for subdir in subdirs:
        subdir_path = os.path.join(neuropal_base, subdir)
        if os.path.exists(subdir_path):
            pattern = os.path.join(subdir_path, "*.h5")
            files = glob.glob(pattern)
            for file_path in files:
                filename = os.path.basename(file_path)
                case_group = filename.split('_')[0]
                h5_files.append({
                    'file_path': file_path,
                    'subdir': subdir,
                    'case_group': case_group,
                    'filename': filename
                })
    
    return h5_files

def convert_h5_to_tiff(h5_file_path, output_dir):
    """
    Convert a single H5 file to TIFF stack.
    Based on actual data structure: (Z, C, Y, X) -> multi-page TIFF
    """
    print(f"Converting: {os.path.basename(h5_file_path)}")
    
    try:
        with h5py.File(h5_file_path, 'r') as f:
            # Get the main dataset
            if 'main' not in f:
                print(f"  Error: 'main' dataset not found in {h5_file_path}")
                return False
            
            data = f['main'][:]  # Shape: (Z, C, Y, X)
            print(f"  Data shape: {data.shape}")
            print(f"  Data type: {data.dtype}")
            print(f"  Data range: [{data.min()}, {data.max()}]")
            
            # Data is already uint8, no normalization needed
            if data.dtype != np.uint8:
                print(f"  Warning: Expected uint8, got {data.dtype}")
            
            # Transpose from (Z, C, Y, X) to (C, Z, Y, X) for nnUNet compatibility
            data = np.transpose(data, (1, 0, 2, 3))
            print(f"  Transposed shape: {data.shape} (C, Z, Y, X)")
            
            # Create output filename
            base_name = os.path.splitext(os.path.basename(h5_file_path))[0]
            output_file = os.path.join(output_dir, f"{base_name}.tiff")
            
            # Save as multi-page TIFF
            # Now in (C, Z, Y, X) format that nnUNet expects
            tifffile.imwrite(output_file, data, photometric='minisblack')
            print(f"  Saved: {output_file}")
            
            return True
            
    except Exception as e:
        print(f"  Error converting {h5_file_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert H5 volumes to TIFF stacks")
    parser.add_argument("--output_dir", default="/projects/weilab/gohaina/nucworm/outputs/data/neuropal_as_tiff",
                       help="Output directory for TIFF files")
    parser.add_argument("--dataset", choices=['nejatbakhsh20', 'yemini21', 'wen20'],
                       help="Process only the specified dataset (default: all datasets)")
    parser.add_argument("--dry_run", action="store_true",
                       help="Show what would be converted without actually converting")
    
    args = parser.parse_args()
    
    print("=== H5 to TIFF Conversion ===")
    print(f"Output directory: {args.output_dir}")
    if args.dataset:
        print(f"Processing dataset: {args.dataset}")
    else:
        print("Processing all datasets")
    
    # Find H5 files
    h5_files = find_neuropal_h5_files(args.dataset)
    print(f"Found {len(h5_files)} .h5 files to convert")
    
    if not h5_files:
        print("No .h5 files found!")
        return
    
    if args.dry_run:
        print("\n=== DRY RUN - No files will be converted ===")
        for file_info in h5_files:
            print(f"Would convert: {file_info['filename']}")
            print(f"  From: {file_info['file_path']}")
            print(f"  To: {args.output_dir}/{file_info['subdir']}/{file_info['case_group']}/")
        return
    
    # Process each file
    successful_conversions = 0
    for i, file_info in enumerate(h5_files):
        print(f"\n--- Converting file {i+1}/{len(h5_files)} ---")
        print(f"File: {file_info['filename']}")
        print(f"Subdirectory: {file_info['subdir']}")
        print(f"Case group: {file_info['case_group']}")
        
        # Create output directory
        case_output_dir = os.path.join(args.output_dir, file_info['subdir'], file_info['case_group'])
        os.makedirs(case_output_dir, exist_ok=True)
        
        # Convert H5 to TIFF
        success = convert_h5_to_tiff(file_info['file_path'], case_output_dir)
        if success:
            successful_conversions += 1
        
        print(f"  {'✓ Success' if success else '✗ Failed'}")
    
    print(f"\n=== Conversion Complete ===")
    print(f"Successfully converted: {successful_conversions}/{len(h5_files)} files")
    print(f"Results saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
