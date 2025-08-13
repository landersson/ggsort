#!/usr/bin/env python3
"""
Render script for GGSort
Renders detection boxes onto exported images using metadata.json for verification
"""

import os
import json
import argparse
import sys
import cv2
import numpy as np
from pathlib import Path

# Color mapping for different categories (in BGR format for OpenCV)
COLORS = {
    1: (0, 0, 255),    # Red for GangGangs
    2: (255, 0, 0),    # Blue for persons
    3: (0, 255, 0),    # Green for vehicles
    4: (0, 255, 255),  # Yellow for possums
    5: (255, 255, 0)   # Cyan for other
}

# Category names for labels
CATEGORY_NAMES = {
    1: 'GangGang',
    2: 'Person',
    3: 'Vehicle',
    4: 'Possum',
    5: 'Other'
}

def render_detections_on_image(image_path: str, detections: list, output_path: str):
    """Render detection boxes on a single image"""
    
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return False
    
    height, width = img.shape[:2]
    
    # Draw each detection
    for detection in detections:
        category = detection['category']
        x = detection['x']
        y = detection['y']
        box_width = detection['width']
        box_height = detection['height']
        hard = detection.get('hard', False)
        
        # Convert normalized coordinates to pixel coordinates
        x1 = int(x * width)
        y1 = int(y * height)
        x2 = int((x + box_width) * width)
        y2 = int((y + box_height) * height)
        
        # Get color for this category
        color = COLORS.get(category, (128, 128, 128))  # Default to gray
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Create label
        category_name = CATEGORY_NAMES.get(category, f'Cat{category}')
        label = category_name
        if hard:
            label += " (HARD)"
        
        # Draw label background and text
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
        cv2.putText(img, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Save the annotated image
    try:
        cv2.imwrite(output_path, img)
        return True
    except Exception as e:
        print(f"Error saving image {output_path}: {e}")
        return False

def render_all_images(input_dir: str, output_dir: str):
    """Render detection boxes on all images using metadata.json"""
    
    # Load metadata
    metadata_path = os.path.join(input_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        print(f"Error: metadata.json not found in {input_dir}")
        return 1
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error reading metadata.json: {e}")
        return 1
    
    print(f"Loaded metadata for {len(metadata)} images")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each image
    processed_count = 0
    total_detections = 0
    
    for image_meta in metadata:
        file_path = image_meta['file_path']
        detections = image_meta['detections']
        
        # Construct input and output paths
        input_image_path = os.path.join(input_dir, file_path)
        
        # Create corresponding output subdirectory structure
        output_subdir = os.path.dirname(file_path)
        if output_subdir:
            output_subdir_path = os.path.join(output_dir, output_subdir)
            os.makedirs(output_subdir_path, exist_ok=True)
        
        output_image_path = os.path.join(output_dir, file_path)
        
        # Check if input image exists
        if not os.path.exists(input_image_path):
            print(f"Warning: Input image not found, skipping: {input_image_path}")
            continue
        
        # Render detections on the image
        success = render_detections_on_image(input_image_path, detections, output_image_path)
        
        if success:
            processed_count += 1
            total_detections += len(detections)
            
            if processed_count % 100 == 0:
                print(f"Processed {processed_count}/{len(metadata)} images...")
        else:
            print(f"Failed to process {input_image_path}")
    
    print(f"Rendering completed:")
    print(f"  - Processed {processed_count} images")
    print(f"  - Rendered {total_detections} total detections")
    print(f"  - Output saved to: {output_dir}")
    
    # Print detection category statistics
    category_counts = {}
    for image_meta in metadata:
        for detection in image_meta['detections']:
            category = detection['category']
            category_counts[category] = category_counts.get(category, 0) + 1
    
    if category_counts:
        print(f"Detection categories:")
        for category, count in sorted(category_counts.items()):
            category_name = CATEGORY_NAMES.get(category, f'Category {category}')
            color_bgr = COLORS.get(category, (128, 128, 128))
            # Convert BGR to RGB for display
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
            print(f"  - {category_name}: {count} detections (Color: RGB{color_rgb})")
    
    return 0

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Render detection boxes on exported images')
    parser.add_argument('--input-dir', required=True,
                        help='Input directory containing metadata.json and image subdirectories')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for rendered images with detection boxes')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return 1
    
    if not os.path.exists(os.path.join(args.input_dir, 'metadata.json')):
        print(f"Error: metadata.json not found in {args.input_dir}")
        return 1
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Rendering detection boxes on images...")
    
    return render_all_images(args.input_dir, args.output_dir)

if __name__ == "__main__":
    sys.exit(main())