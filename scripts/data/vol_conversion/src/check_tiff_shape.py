#!/usr/bin/env python3
"""
Quick script to check the shape of a TIFF file
Usage: python check_tiff_shape.py <path_to_tiff_file>
"""

import tifffile
import numpy as np
import sys
import os

def check_tiff_shape(tiff_path):
    """Check the shape and properties of the specified TIFF file."""
    
    if not os.path.exists(tiff_path):
        print(f"Error: File not found: {tiff_path}")
        return
    
    try:
        # Read the TIFF file
        tiff_data = tifffile.imread(tiff_path)
        
        print("=== TIFF File Analysis ===")
        print(f"File: {tiff_path}")
        print(f"Shape: {tiff_data.shape}")
        print(f"Data type: {tiff_data.dtype}")
        print(f"Data range: [{tiff_data.min()}, {tiff_data.max()}]")
        print(f"Number of dimensions: {tiff_data.ndim}")
        
        # Check if it's a multi-page TIFF
        if tiff_data.ndim == 4:
            print(f"Multi-page TIFF with shape (C, Z, Y, X)")
            print(f"  - Channels: {tiff_data.shape[0]}")
            print(f"  - Z slices: {tiff_data.shape[1]}")
            print(f"  - Height: {tiff_data.shape[2]}")
            print(f"  - Width: {tiff_data.shape[3]}")
        elif tiff_data.ndim == 3:
            print(f"3D TIFF with shape (Z, Y, X) or (C, Y, X)")
            print(f"  - First dimension: {tiff_data.shape[0]}")
            print(f"  - Height: {tiff_data.shape[1]}")
            print(f"  - Width: {tiff_data.shape[2]}")
        
        # Show a small sample
        if tiff_data.ndim >= 3:
            sample = tiff_data[:min(2, tiff_data.shape[0]), 
                              :min(2, tiff_data.shape[1]), 
                              :min(2, tiff_data.shape[2])]
            print(f"Sample data (2x2x2):\n{sample}")
            
    except Exception as e:
        print(f"Error reading TIFF file: {e}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python check_tiff_shape.py <path_to_tiff_file>")
        print("Example: python check_tiff_shape.py /path/to/your/file.tiff")
        sys.exit(1)
    
    tiff_path = sys.argv[1]
    check_tiff_shape(tiff_path)

if __name__ == "__main__":
    main()
