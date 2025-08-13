#!/usr/bin/env python3
"""
Export script for GGSort
Exports images from database to organized output directory structure
"""

import os
import sqlite3
import argparse
import shutil
import sys
import json
from pathlib import Path

def export_images(db_file: str, images_dir: str, output_dir: str, max_images: int = None):
    """Export images from database to output directory with subdirectory organization"""
    
    # Connect to database
    if not os.path.exists(db_file):
        print(f"Error: Database file not found: {db_file}")
        return 1
    
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # Get all images from database
        cursor.execute("SELECT id, file_path FROM images ORDER BY file_path")
        images = cursor.fetchall()
        
        total_images = len(images)
        print(f"Found {total_images} images in database")
        
        if max_images:
            images = images[:max_images]
            print(f"Limiting export to {len(images)} images due to --max-images setting")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize metadata array
        metadata = []
        
        # Process images
        exported_count = 0
        current_subdir = 0
        current_subdir_count = 0
        image_counter = 0  # Sequential counter for filename numbering
        
        for image in images:
            image_id = image['id']
            file_path = image['file_path']
            
            # Get detections for this image
            cursor.execute("""
                SELECT category, x, y, width, height, hard
                FROM detections 
                WHERE image_id = ? AND deleted = 0
                ORDER BY confidence DESC
            """, (image_id,))
            detections_rows = cursor.fetchall()
            
            # Convert detections to list of dictionaries
            detections = []
            for det in detections_rows:
                detections.append({
                    'category': det['category'],
                    'x': det['x'],
                    'y': det['y'],
                    'width': det['width'],
                    'height': det['height'],
                    'hard': bool(det['hard']) if det['hard'] is not None else False
                })
            
            # Skip images with no valid detections
            if not detections:
                continue
            
            # Construct full input path
            if os.path.isabs(file_path):
                input_path = file_path
            else:
                input_path = os.path.join(images_dir, file_path)
            
            # Check if input file exists
            if not os.path.exists(input_path):
                print(f"Warning: Input file not found, skipping: {input_path}")
                continue
            
            # Create new subdirectory if current one has 1000 images
            if current_subdir_count >= 1000:
                current_subdir += 1
                current_subdir_count = 0
            
            # Create subdirectory path
            subdir_name = f"batch_{current_subdir:04d}"
            subdir_path = os.path.join(output_dir, subdir_name)
            os.makedirs(subdir_path, exist_ok=True)
            
            # Generate sequential filename
            final_filename = f"img_{image_counter:04d}.jpg"
            output_path = os.path.join(subdir_path, final_filename)
            
            try:
                # Copy the file
                shutil.copy2(input_path, output_path)
                
                # Add metadata entry (relative path from output_dir)
                relative_output_path = os.path.join(subdir_name, final_filename)
                metadata.append({
                    'file_path': relative_output_path,
                    'detections': detections
                })
                
                exported_count += 1
                current_subdir_count += 1
                image_counter += 1  # Increment counter for next image
                
                if exported_count % 100 == 0:
                    print(f"Exported {exported_count}/{len(images)} images...")
                
            except Exception as e:
                print(f"Error copying {input_path} to {output_path}: {e}")
                continue
        
        # Write metadata to JSON file
        metadata_path = os.path.join(output_dir, 'metadata.json')
        try:
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Metadata saved to: {metadata_path}")
        except Exception as e:
            print(f"Error writing metadata file: {e}")
        
        print(f"Export completed: {exported_count} images exported to {output_dir}")
        print(f"Images organized into {current_subdir + 1} subdirectories")
        print(f"Total detections exported: {sum(len(img['detections']) for img in metadata)}")
        
        return 0
        
    except Exception as e:
        print(f"Database error: {e}")
        return 1
    
    finally:
        conn.close()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Export images from GGSort database')
    parser.add_argument('--db-file', required=True, help='SQLite database file')
    parser.add_argument('--images-dir', required=True, 
                        help='Base directory containing input image files')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for exported images')
    parser.add_argument('--max-images', type=int,
                        help='Maximum number of images to export')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.db_file):
        print(f"Error: Database file not found: {args.db_file}")
        return 1
    
    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory not found: {args.images_dir}")
        return 1
    
    if args.max_images and args.max_images <= 0:
        print(f"Error: --max-images must be a positive integer")
        return 1
    
    print(f"Exporting from database: {args.db_file}")
    print(f"Input images directory: {args.images_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.max_images:
        print(f"Maximum images: {args.max_images}")
    
    return export_images(args.db_file, args.images_dir, args.output_dir, args.max_images)

if __name__ == "__main__":
    sys.exit(main())