#!/usr/bin/env python3
"""
Convert custom JSON metadata format to MSCOCO annotation format
"""

import json
import argparse
import sys
import os
from datetime import datetime
from PIL import Image


def get_image_dimensions(image_dir, file_path):
    """Get image dimensions from file"""
    try:
        full_path = os.path.join(image_dir, file_path)
        with Image.open(full_path) as img:
            return img.width, img.height
    except Exception as e:
        print(f"Warning: Could not get dimensions for {file_path}: {e}")
        # Return default dimensions if image not found
        return 1920, 1080


def convert_to_coco(input_file: str, output_file: str, image_dir: str = None):
    """
    Convert custom JSON format to MSCOCO format
    
    Args:
        input_file: Path to input JSON file in custom format
        output_file: Path to output MSCOCO JSON file
        image_dir: Directory containing images (for getting dimensions)
    """
    
    # Load custom format data
    with open(input_file, 'r') as f:
        custom_data = json.load(f)
    
    # Define category mapping
    # Based on the detection categories from the original code:
    # Gang-gang (1), Person (2), Vehicle (3), Possum (4), Other (5)
    categories = [
        {"id": 1, "name": "gang_gang", "supercategory": "bird"},
        {"id": 2, "name": "person", "supercategory": "person"},
        {"id": 3, "name": "vehicle", "supercategory": "vehicle"},
        {"id": 4, "name": "possum", "supercategory": "animal"},
        {"id": 5, "name": "other", "supercategory": "object"}
    ]
    
    # Initialize COCO structure
    coco_data = {
        "info": {
            "description": "GGSort Dataset converted to COCO format",
            "url": "",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "GGSort",
            "date_created": datetime.now().isoformat()
        },
        "licenses": [
            {
                "id": 1,
                "name": "Unknown",
                "url": ""
            }
        ],
        "images": [],
        "annotations": [],
        "categories": categories
    }
    
    # Process each image
    image_id = 1
    annotation_id = 1
    
    for item in custom_data:
        file_path = item["file_path"]
        detections = item["detections"]
        
        # Get image dimensions
        if image_dir:
            width, height = get_image_dimensions(image_dir, file_path)
        else:
            # Default dimensions if no image directory provided
            width, height = 1920, 1080
            print(f"Warning: Using default dimensions for {file_path}")
        
        # Create image entry
        image_entry = {
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": file_path,
            "license": 1,
            "flickr_url": "",
            "coco_url": "",
            "date_captured": ""
        }
        coco_data["images"].append(image_entry)
        
        # Process detections for this image
        for detection in detections:
            # Convert normalized coordinates to absolute coordinates
            # Custom format: x, y, width, height (normalized 0-1)
            # COCO format: x, y, width, height (absolute pixels)
            x_norm = detection["x"]
            y_norm = detection["y"]
            width_norm = detection["width"] 
            height_norm = detection["height"]
            
            # Convert to absolute coordinates
            x_abs = x_norm * width
            y_abs = y_norm * height
            width_abs = width_norm * width
            height_abs = height_norm * height
            
            # Calculate area
            area = width_abs * height_abs
            
            # Create annotation entry
            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": detection["category"],
                "segmentation": [],  # Empty for bounding box only
                "area": area,
                "bbox": [x_abs, y_abs, width_abs, height_abs],  # COCO format: [x, y, width, height]
                "iscrowd": 0
            }
            
            coco_data["annotations"].append(annotation)
            annotation_id += 1
        
        image_id += 1
    
    # Save COCO format data
    with open(output_file, 'w') as f:
        json.dump(coco_data, f, indent=2)
    
    print(f"Successfully converted {len(custom_data)} images to COCO format")
    print(f"Total annotations: {len(coco_data['annotations'])}")
    print(f"Categories: {len(categories)}")
    print(f"Output saved to: {output_file}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Convert custom JSON metadata to MSCOCO format')
    parser.add_argument('--input', '-i', required=True, 
                        help='Input JSON file in custom format')
    parser.add_argument('--output', '-o', required=True,
                        help='Output JSON file in MSCOCO format')
    parser.add_argument('--image-dir', '-d',
                        help='Directory containing images (for getting dimensions)')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return 1
    
    # Validate image directory if provided
    if args.image_dir and not os.path.exists(args.image_dir):
        print(f"Error: Image directory not found: {args.image_dir}")
        return 1
    
    print(f"Converting {args.input} to MSCOCO format...")
    print(f"Output: {args.output}")
    if args.image_dir:
        print(f"Image directory: {args.image_dir}")
    
    try:
        convert_to_coco(args.input, args.output, args.image_dir)
        return 0
    except Exception as e:
        print(f"Error during conversion: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())