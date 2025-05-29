#!/usr/bin/env python3

import json
import argparse
import os
import sqlite3
import sys
import exifread
from tqdm import tqdm

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Filter detection data from results.json and create SQLite database')
    parser.add_argument('--base-dir', required=True,
                        help='Filter images by first component of the file path')
    parser.add_argument('--image-dir', default='images',
                        help='Directory containing wildlife images (default: images)')
    parser.add_argument('--output',
                        help='Output SQLite database file (default: {base-dir}.db)')
    parser.add_argument('--conf-threshold', type=float, default=0.0,
                        help='Confidence threshold for including detections (default: 0.0)')
    parser.add_argument('--include-categories', default='1',
                        help='Comma-separated list of category IDs to include (default: 1)')
    return parser.parse_args()

def create_database(db_path):
    """Create SQLite database with required schema"""
    # Remove existing file if it exists
    if os.path.exists(db_path):
        os.remove(db_path)
    
    # Connect to database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create images table
    cursor.execute('''
    CREATE TABLE images (
        id INTEGER PRIMARY KEY,
        file_path TEXT UNIQUE NOT NULL,
        width INTEGER,
        height INTEGER,
        datetime_original TEXT,
        manually_processed BOOLEAN DEFAULT 0
    )
    ''')
    
    # Create detections table
    cursor.execute('''
    CREATE TABLE detections (
        id INTEGER PRIMARY KEY,
        image_id INTEGER NOT NULL,
        category INTEGER,
        confidence REAL,
        x REAL,
        y REAL,
        width REAL,
        height REAL,
        deleted BOOLEAN DEFAULT 0,
        subcategory INTEGER,
        hard BOOLEAN DEFAULT 0,
        FOREIGN KEY (image_id) REFERENCES images(id)
    )
    ''')
    
    # Create indices for better performance
    cursor.execute('CREATE INDEX idx_image_id ON detections(image_id)')
    cursor.execute('CREATE INDEX idx_file_path ON images(file_path)')
    
    conn.commit()
    return conn

def get_datetime_original(image_path):
    """Extract DateTimeOriginal from image EXIF data"""
    try:
        with open(image_path, 'rb') as f:
            tags = exifread.process_file(f)
            if 'EXIF DateTimeOriginal' in tags:
                return str(tags['EXIF DateTimeOriginal'])
    except Exception as e:
        # Silently handle any errors (corrupted files, no EXIF, etc.)
        pass
    return None

def main():
    args = parse_args()
    base_dir = args.base_dir
    image_dir = args.image_dir
    conf_threshold = args.conf_threshold
    
    # Parse include categories
    try:
        include_categories = [int(cat.strip()) for cat in args.include_categories.split(',')]
    except ValueError:
        print("Error: --include-categories must be a comma-separated list of integers")
        return 1
    
    # Determine output database path
    if args.output:
        db_path = args.output
    else:
        db_path = f"{base_dir}.db"
    
    print(f"Using image directory: {image_dir}")
    print(f"Using confidence threshold: {conf_threshold}")
    print(f"Including categories: {include_categories}")
    print(f"Output database: {db_path}")
    
    # Load JSON data
    try:
        with open('data/results.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: data/results.json file not found")
        return 1
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in results.json")
        return 1
    
    # Get all images
    all_images = data.get('images', [])
    
    # Filter images if base_dir is specified
    filtered_images = all_images
    if base_dir:
        filtered_images = []
        for img in all_images:
            file_path = img.get('file', '')
            # Split the path and get the first component
            path_components = file_path.split('/')
            if path_components and path_components[0] == base_dir:
                filtered_images.append(img)
    
    print(f"Found {len(filtered_images)} matching images based on directory filter")
    
    if not filtered_images:
        print("No images to process, exiting.")
        return 1
    
    # Create database
    print(f"Creating SQLite database: {db_path}")
    conn = create_database(db_path)
    cursor = conn.cursor()
    
    # Process each image and its detections
    total_detections = 0
    filtered_detections = 0
    images_with_valid_detections = 0
    missing_files = 0
    images_with_exif = 0
    
    try:
        # Start a transaction
        cursor.execute("BEGIN TRANSACTION")
        
        for img in tqdm(filtered_images, desc="Processing images", unit="image"):
            file_path = img.get('file', '')
            width = img.get('width', 0)
            height = img.get('height', 0)
            
            # Construct the full image path
            # If file_path is relative, join with image_dir
            full_path = file_path
            if not os.path.isabs(file_path):
                full_path = os.path.join(image_dir, file_path)
            
            # Check if the image file actually exists
            if not os.path.exists(full_path):
                print(f"Warning: Image file not found: {full_path}")
                missing_files += 1
                continue
            
            
            # Filter out detections with confidence below threshold
            all_detections = img.get('detections', [])
            if all_detections is None:
                print(f"Warning: No detections found for image: {full_path}")
                continue
            
            valid_detections = []
            for detection in all_detections:
                confidence = detection.get('conf', 0.0)
                category = int(detection.get('category', 0))
                # print(f"Category: {category}")
                
                if confidence >= conf_threshold and category in include_categories:
                    valid_detections.append(detection)
                else:
                    filtered_detections += 1
            
            # Skip images with no valid detections
            if not valid_detections:
                continue

            # Extract DateTimeOriginal from EXIF data
            print(f"Processing image: {full_path}")
            datetime_original = get_datetime_original(full_path)
            if datetime_original:
                images_with_exif += 1
                # print(f"Extracted EXIF timestamp: {datetime_original}")
                
            # Insert image record
            cursor.execute(
                "INSERT INTO images (file_path, width, height, datetime_original, manually_processed) VALUES (?, ?, ?, ?, 0)",
                (file_path, width, height, datetime_original)
            )
            
            # Get the image ID for the newly inserted record
            image_id = cursor.lastrowid
            images_with_valid_detections += 1
            
            # Process valid detections for this image
            for detection in valid_detections:
                category = detection.get('category', 0)
                confidence = detection.get('conf', 0.0)
                bbox = detection.get('bbox', [0, 0, 0, 0])
                
                # Insert detection record with additional fields
                cursor.execute(
                    "INSERT INTO detections (image_id, category, confidence, x, y, width, height, deleted, subcategory, hard) VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL, 0)",
                    (image_id, category, confidence, bbox[0], bbox[1], bbox[2], bbox[3])
                )
                
                total_detections += 1
        
        # Commit the transaction
        conn.commit()
        print(f"Database created successfully with {images_with_valid_detections} images and {total_detections} detections")
        print(f"Extracted EXIF timestamps from {images_with_exif} of {images_with_valid_detections} images")
        print(f"Skipped {len(filtered_images) - images_with_valid_detections} images with no detections above threshold")
        print(f"Filtered out {filtered_detections} detections (confidence < {conf_threshold} or category not in {include_categories})")
        print(f"Skipped {missing_files} images with missing files")
        
    except sqlite3.Error as e:
        conn.rollback()
        print(f"SQLite error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        conn.rollback()
        print(f"Unexpected error: {e}", file=sys.stderr)
        raise

    finally:
        conn.close()
    
    # Return success code
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 