#!/usr/bin/env python3

import os
import sqlite3
import cv2
import argparse
import sys
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class Detection:
    """Represents a single detection with its properties"""
    id: int
    category: int
    confidence: float
    x: float  # normalized coordinates
    y: float
    width: float
    height: float
    deleted: bool = False
    subcategory: Optional[int] = None
    hard: bool = False

class DatabaseManager:
    """Handles all database operations"""
    def __init__(self, db_file: str):
        self.db_file = db_file
        self.conn = self._connect()

    def _connect(self) -> sqlite3.Connection:
        """Connect to the SQLite database"""
        if not os.path.exists(self.db_file):
            raise FileNotFoundError(f"Database file not found: {self.db_file}")
        
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row
        return conn

    def get_images_from_database(self) -> List[sqlite3.Row]:
        """Get all images from the database"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, file_path, width, height FROM images ORDER BY file_path")
        return cursor.fetchall()

    def get_ganggang_detections_for_image(self, image_id: int) -> List[Detection]:
        """Get GangGang detections for a specific image that are not deleted and not hard"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, category, confidence, x, y, width, height, deleted, subcategory, hard
            FROM detections 
            WHERE image_id = ? AND category = 1 AND (deleted = 0 OR deleted IS NULL) AND (hard = 0 OR hard IS NULL)
            ORDER BY confidence DESC
        """, (image_id,))
        
        detections = []
        for row in cursor.fetchall():
            detection = Detection(
                id=row['id'],
                category=row['category'],
                confidence=row['confidence'],
                x=row['x'],
                y=row['y'],
                width=row['width'],
                height=row['height'],
                deleted=bool(row['deleted']) if row['deleted'] is not None else False,
                subcategory=row['subcategory'],
                hard=bool(row['hard']) if row['hard'] is not None else False
            )
            detections.append(detection)
        return detections

    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()

class PatchExtractor:
    """Handles extraction of detection patches from images"""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self._ensure_output_dir()

    def _ensure_output_dir(self):
        """Create output directory if it doesn't exist"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Created output directory: {self.output_dir}")

    def extract_detection_patch(self, image, detection: Detection, image_width: int, image_height: int) -> Optional[any]:
        """Extract a detection patch from the image"""
        # Convert normalized coordinates to pixel coordinates
        x_min = int(detection.x * image_width)
        y_min = int(detection.y * image_height)
        patch_width = int(detection.width * image_width)
        patch_height = int(detection.height * image_height)
        
        x_max = x_min + patch_width
        y_max = y_min + patch_height
        
        # Ensure coordinates are within image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image_width, x_max)
        y_max = min(image_height, y_max)
        
        # Extract the patch
        if x_max > x_min and y_max > y_min:
            patch = image[y_min:y_max, x_min:x_max]
            return patch
        else:
            print(f"Warning: Invalid coordinates for detection {detection.id}")
            return None

    def save_patch(self, patch, detection: Detection, image_name: str) -> bool:
        """Save a detection patch as a JPEG file"""
        if patch is None or patch.size == 0:
            return False
        
        # Create filename: imagename_detectionid_confidence.jpg
        base_name = os.path.splitext(image_name)[0]
        filename = f"{base_name}_det{detection.id}_conf{detection.confidence:.3f}.jpg"
        output_path = os.path.join(self.output_dir, filename)
        
        # Save the patch
        try:
            cv2.imwrite(output_path, patch)
            return True
        except Exception as e:
            print(f"Error saving patch {output_path}: {e}")
            return False

class ExtractPatchesApplication:
    """Main application class for extracting detection patches"""
    
    def __init__(self, db_file: str, image_dir: str, output_dir: str):
        self.db_manager = DatabaseManager(db_file)
        self.patch_extractor = PatchExtractor(output_dir)
        self.image_dir = image_dir
        
        print(f"Database: {db_file}")
        print(f"Image directory: {image_dir}")
        print(f"Output directory: {output_dir}")

    def run(self) -> int:
        """Run the main application"""
        try:
            # Get all images from database
            images = self.db_manager.get_images_from_database()
            total_images = len(images)
            print(f"Found {total_images} images in the database")
            
            if total_images == 0:
                print("No images found in the database")
                return 1

            total_patches_extracted = 0
            images_processed = 0
            
            # Process each image
            for i, image_row in enumerate(images):
                image_id = image_row['id']
                file_path = image_row['file_path']
                db_width = image_row['width']
                db_height = image_row['height']
                
                print(f"Processing image {i+1}/{total_images}: {file_path}")
                
                # Get GangGang detections for this image
                detections = self.db_manager.get_ganggang_detections_for_image(image_id)
                
                if not detections:
                    print(f"  No valid GangGang detections found")
                    continue
                
                print(f"  Found {len(detections)} valid GangGang detection(s)")
                
                # Construct full image path
                full_path = file_path
                if not os.path.isabs(file_path):
                    full_path = os.path.join(self.image_dir, file_path)
                
                # Load the image
                image = cv2.imread(full_path)
                if image is None:
                    print(f"  Error: Could not read image: {full_path}")
                    continue
                
                # Get actual image dimensions
                actual_height, actual_width = image.shape[:2]
                
                # Use actual dimensions for patch extraction
                image_width = actual_width
                image_height = actual_height
                
                if db_width and db_height:
                    if abs(actual_width - db_width) > 1 or abs(actual_height - db_height) > 1:
                        print(f"  Warning: Image dimensions mismatch. DB: {db_width}x{db_height}, Actual: {actual_width}x{actual_height}")
                
                # Extract patches for each detection
                patches_extracted_for_image = 0
                for detection in detections:
                    patch = self.patch_extractor.extract_detection_patch(
                        image, detection, image_width, image_height)
                    
                    if patch is not None:
                        image_name = os.path.basename(file_path)
                        success = self.patch_extractor.save_patch(patch, detection, image_name)
                        if success:
                            patches_extracted_for_image += 1
                            total_patches_extracted += 1
                            print(f"    Extracted patch for detection {detection.id} (confidence: {detection.confidence:.3f})")
                        else:
                            print(f"    Failed to save patch for detection {detection.id}")
                    else:
                        print(f"    Failed to extract patch for detection {detection.id}")
                
                print(f"  Extracted {patches_extracted_for_image} patch(es) from this image")
                images_processed += 1
            
            print(f"\nCompleted processing {images_processed} images")
            print(f"Total patches extracted: {total_patches_extracted}")
            
            return 0
            
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1
        finally:
            self.db_manager.close()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Extract GangGang detection patches from images')
    parser.add_argument('--db-file', required=True, help='SQLite database file')
    parser.add_argument('--image-dir', default='images', 
                        help='Directory containing wildlife images (default: images)')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for extracted patches')
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    
    print(f"Extract Patches - GangGang Detection Extractor")
    print(f"=" * 50)
    
    app = ExtractPatchesApplication(
        db_file=args.db_file,
        image_dir=args.image_dir,
        output_dir=args.output_dir
    )
    
    return app.run()

if __name__ == "__main__":
    sys.exit(main()) 