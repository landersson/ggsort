
#!/usr/bin/env python3

import json
import argparse
import os
import sys
import exifread
from tqdm import tqdm
from datetime import datetime, time

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Filter detection data from results.json and create SQLite database')
    parser.add_argument('--input-json', required=True,
                        help='Input results.json file')
    parser.add_argument('--output-json',
                        help='Output JSON file (default: {input-json}.filtered.json)')
    parser.add_argument('--image-dir', default='images',
                        help='Directory containing wildlife images (default: images)')
    parser.add_argument('--time-range',
                        help='Time range in format HH:MM-HH:MM (24-hour format, e.g., 18:00-06:00)')
    return parser.parse_args()

def parse_time_range(time_range_str):
    """Parse time range string in format HH:MM-HH:MM"""
    if not time_range_str:
        return None, None
    
    try:
        start_str, end_str = time_range_str.split('-')
        start_time = datetime.strptime(start_str.strip(), '%H:%M').time()
        end_time = datetime.strptime(end_str.strip(), '%H:%M').time()
        return start_time, end_time
    except ValueError as e:
        raise ValueError(f"Invalid time range format. Expected HH:MM-HH:MM, got: {time_range_str}") from e

def is_time_in_range(timestamp_str, start_time, end_time):
    """Check if timestamp falls within the specified time range"""
    if not start_time or not end_time:
        return True  # No time range specified, include all
    
    try:
        # Parse the EXIF timestamp (format: YYYY:MM:DD HH:MM:SS)
        timestamp = datetime.strptime(timestamp_str, '%Y:%m:%d %H:%M:%S')
        image_time = timestamp.time()
        
        # Handle time ranges that cross midnight (e.g., 18:00-06:00)
        if start_time > end_time:
            # Range crosses midnight: start_time to 23:59:59 OR 00:00:00 to end_time
            return image_time >= start_time or image_time <= end_time
        else:
            # Normal range within same day
            return start_time <= image_time <= end_time
    except ValueError as e:
        print(f"Warning: Could not parse timestamp '{timestamp_str}': {e}")
        return False

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
    image_dir = args.image_dir
    include_categories = [1]
    conf_threshold = 0.0
    missing_files = 0
    filtered_detections = 0
    images_without_exif = 0
    images_outside_time_range = 0
    images_without_valid_detections = 0
    # Parse time range if provided
    start_time, end_time = None, None
    if args.time_range:
        try:
            start_time, end_time = parse_time_range(args.time_range)
            print(f"Time range: {start_time} to {end_time}")
        except ValueError as e:
            print(f"Error: {e}")
            return 1
    
    print(f"Using image directory: {image_dir}")
    print(f"Including categories: {include_categories}")
    
    # Load JSON data
    try:
        with open(args.input_json, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {args.input_json} file not found")
        return 1
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {args.input_json}")
        return 1
    
    images = data.get('images', [])
    filtered_images = []
    
    try:
        
        for img in tqdm(images, desc="Processing images", unit="image"):
        # for img in images:
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
                images_without_valid_detections += 1
                # print(f"No valid detections for image: {full_path}")
                continue

            # Extract DateTimeOriginal from EXIF data
            # print(f"Processing image: {full_path}")
            datetime_original = get_datetime_original(full_path)
            if datetime_original:
                # print(f"Extracted EXIF timestamp: {datetime_original}")
                pass
            else:
                images_without_exif += 1
                print(f"No EXIF timestamp found for image: {full_path}")
                continue
            
            # Check if the image is within the time range
            if not is_time_in_range(datetime_original, start_time, end_time):
                images_outside_time_range += 1
                continue
            
            # Update the image with filtered detections and add to filtered list
            img['detections'] = valid_detections
            filtered_images.append(img)
            
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        raise

    # Save filtered results
    output_file = args.output_json
    if not output_file:
        base_name = os.path.splitext(args.input_json)[0]
        output_file = f"{base_name}.filtered.json"
    
    filtered_data = {
        'images': filtered_images,
        'info': data.get('info', {}),
        'version': data.get('version', '')
    }
    
    try:
        with open(output_file, 'w') as f:
            json.dump(filtered_data, f, indent=2)
        print(f"Filtered results saved to: {output_file}")
    except Exception as e:
        print(f"Error saving output file: {e}")
        return 1

    print(f"Images without EXIF: {images_without_exif}")
    print(f"Missing files: {missing_files}")
    print(f"Filtered detections: {filtered_detections}")
    print(f"Images outside time range: {images_outside_time_range}")
    print(f"Images without valid detections: {images_without_valid_detections}")
    print(f"Total images processed: {len(images)}")
    print(f"Images in output: {len(filtered_images)}")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 