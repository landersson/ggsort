#!/usr/bin/env python3
"""
Quick script to analyze metadata.json and report detection counts by category
"""

import json
import argparse
import sys
import os

# Category names for display
CATEGORY_NAMES = {
    1: 'GangGang',
    2: 'Person',
    3: 'Vehicle',
    4: 'Possum',
    5: 'Other'
}

def analyze_metadata(metadata_file: str):
    """Analyze metadata.json and report detection statistics"""
    
    # Load metadata
    if not os.path.exists(metadata_file):
        print(f"Error: File not found: {metadata_file}")
        return 1
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
    except Exception as e:
        print(f"Error reading {metadata_file}: {e}")
        return 1
    
    # Count detections by category
    category_counts = {}
    hard_counts = {}
    total_detections = 0
    total_images = len(metadata)
    images_with_detections = 0
    
    for image_meta in metadata:
        detections = image_meta.get('detections', [])
        if detections:
            images_with_detections += 1
        
        for detection in detections:
            category = detection.get('category')
            hard = detection.get('hard', False)
            
            if category is not None:
                category_counts[category] = category_counts.get(category, 0) + 1
                if hard:
                    hard_counts[category] = hard_counts.get(category, 0) + 1
                total_detections += 1
    
    # Print results
    print(f"Metadata Analysis for: {metadata_file}")
    print("=" * 50)
    print(f"Total images: {total_images}")
    print(f"Images with detections: {images_with_detections}")
    print(f"Total detections: {total_detections}")
    print()
    
    if category_counts:
        print("Detection counts by category:")
        print("-" * 30)
        for category in sorted(category_counts.keys()):
            count = category_counts[category]
            hard_count = hard_counts.get(category, 0)
            category_name = CATEGORY_NAMES.get(category, f'Category {category}')
            percentage = (count / total_detections) * 100 if total_detections > 0 else 0
            
            hard_info = f" ({hard_count} hard)" if hard_count > 0 else ""
            print(f"  {category}: {category_name:<10} - {count:>6} detections ({percentage:5.1f}%){hard_info}")
        
        print()
        print(f"Average detections per image: {total_detections / images_with_detections:.2f}")
    else:
        print("No detections found in metadata.")
    
    return 0

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Analyze metadata.json detection counts')
    parser.add_argument('metadata_file', help='Path to metadata.json file')
    
    args = parser.parse_args()
    
    return analyze_metadata(args.metadata_file)

if __name__ == "__main__":
    sys.exit(main())