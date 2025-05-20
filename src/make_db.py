#!/usr/bin/env python3

import json
import argparse
import os
import sqlite3
import sys

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Filter detection data from results.json and create SQLite database')
    parser.add_argument('--base-dir', 
                        help='Filter images by first component of the file path')
    parser.add_argument('--output', default='wildlife.db',
                        help='Output SQLite database file (default: wildlife.db)')
    parser.add_argument('--conf-threshold', type=float, default=0.0,
                        help='Confidence threshold for including detections (default: 0.0)')
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
        subcategory TEXT,
        FOREIGN KEY (image_id) REFERENCES images(id)
    )
    ''')
    
    # Create indices for better performance
    cursor.execute('CREATE INDEX idx_image_id ON detections(image_id)')
    cursor.execute('CREATE INDEX idx_file_path ON images(file_path)')
    
    conn.commit()
    return conn

def main():
    args = parse_args()
    base_dir = args.base_dir
    db_path = args.output
    conf_threshold = args.conf_threshold
    
    print(f"Using confidence threshold: {conf_threshold}")
    
    # Load JSON data
    try:
        with open('results.json', 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Error: results.json file not found")
        return
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in results.json")
        return
    
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
        return
    
    # Create database
    print(f"Creating SQLite database: {db_path}")
    conn = create_database(db_path)
    cursor = conn.cursor()
    
    # Process each image and its detections
    total_detections = 0
    filtered_detections = 0
    images_with_valid_detections = 0
    
    try:
        # Start a transaction
        cursor.execute("BEGIN TRANSACTION")
        
        for img in filtered_images:
            file_path = img.get('file', '')
            width = img.get('width', 0)
            height = img.get('height', 0)
            
            # Filter out detections with confidence below threshold
            all_detections = img.get('detections', [])
            valid_detections = []
            for detection in all_detections:
                confidence = detection.get('conf', 0.0)
                if confidence >= conf_threshold:
                    valid_detections.append(detection)
                else:
                    filtered_detections += 1
            
            # Skip images with no valid detections
            if not valid_detections:
                continue
                
            # Insert image record
            cursor.execute(
                "INSERT INTO images (file_path, width, height, manually_processed) VALUES (?, ?, ?, 0)",
                (file_path, width, height)
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
                    "INSERT INTO detections (image_id, category, confidence, x, y, width, height, deleted, subcategory) VALUES (?, ?, ?, ?, ?, ?, ?, 0, NULL)",
                    (image_id, category, confidence, bbox[0], bbox[1], bbox[2], bbox[3])
                )
                
                total_detections += 1
        
        # Commit the transaction
        conn.commit()
        print(f"Database created successfully with {images_with_valid_detections} images and {total_detections} detections")
        print(f"Skipped {len(filtered_images) - images_with_valid_detections} images with no detections above threshold")
        print(f"Filtered out {filtered_detections} detections with confidence below {conf_threshold}")
        
    except sqlite3.Error as e:
        conn.rollback()
        print(f"SQLite error: {e}", file=sys.stderr)
    finally:
        conn.close()

if __name__ == "__main__":
    main() 