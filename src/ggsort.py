#!/usr/bin/env python3

import os
import json
import cv2
import glob
import argparse

# Configuration
JSON_FILE = 'results_filtered_waynes.json'  # Default JSON file
CONFIDENCE_THRESHOLD = 0.0  # Only display detections with confidence above this threshold

# Color mapping for different categories (in BGR format)
COLORS = {
    '1': (0, 255, 0),    # Green for animals
    '2': (255, 0, 0),    # Blue for persons
    '3': (0, 0, 255)     # Red for vehicles
}

# Variables to track mouse position and bounding boxes
mouse_x, mouse_y = 0, 0
current_boxes = []
current_image = None
current_metadata = None
current_categories = None
current_confidence_threshold = 0.0
need_redraw = False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='View wildlife images with detection boxes')
    parser.add_argument('--image-dir', default='images', 
                        help='Directory containing wildlife images (default: images)')
    parser.add_argument('--confidence', type=float, default=CONFIDENCE_THRESHOLD,
                        help='Confidence threshold for displaying detections (default: 0.0)')
    parser.add_argument('--json-file', default=JSON_FILE,
                        help=f'JSON metadata file (default: {JSON_FILE})')
    return parser.parse_args()

def load_metadata(json_file):
    """Load the JSON metadata file"""
    print(f"Loading metadata from {json_file}...")
    with open(json_file, 'r') as f:
        return json.load(f)

def find_all_images(images_dir):
    """Find all JPG images in the images directory recursively"""
    image_pattern = os.path.join(images_dir, "**", "*.JPG")
    return glob.glob(image_pattern, recursive=True)

def create_image_to_metadata_map(metadata):
    """Create a dictionary mapping image paths to their metadata"""
    image_metadata = {}
    for img_data in metadata.get('images', []):
        file_path = img_data['file']
        # Store using the original path from JSON
        image_metadata[file_path] = img_data
    return image_metadata

def mouse_callback(event, x, y, flags, param):
    """Mouse callback function to track mouse position"""
    global mouse_x, mouse_y, need_redraw
    
    # Only respond to mouse movement
    if event == cv2.EVENT_MOUSEMOVE:
        # Update the mouse coordinates
        mouse_x, mouse_y = x, y
        need_redraw = True  # Flag to indicate we need to redraw the image

def draw_image_with_boxes():
    """Draw the image with bounding boxes based on current state"""
    global current_image, current_metadata, current_categories, current_confidence_threshold, current_boxes, need_redraw
    
    if current_image is None or current_metadata is None:
        return None
    
    # Create a copy of the original image to draw on
    img_copy = current_image.copy()
    
    height, width = img_copy.shape[:2]
    
    # Clear the current boxes list
    current_boxes = []
    
    # Get valid detections
    valid_detections = [d for d in current_metadata.get('detections', []) 
                      if d.get('conf', 0) >= current_confidence_threshold]
    
    # Draw bounding boxes
    for detection in valid_detections:
        confidence = detection.get('conf', 0)
        category = detection.get('category', '1')
        bbox = detection.get('bbox', [0, 0, 0, 0])
        
        # Convert from [x, y, w, h] normalized to pixel coordinates
        x, y, w, h = bbox
        x_min = int(x * width)
        y_min = int(y * height)
        box_width = int(w * width)
        box_height = int(h * height)
        
        # Store box coordinates for mouse interaction
        box_info = {
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_min + box_width,
            'y_max': y_min + box_height,
            'category': category,
            'confidence': confidence
        }
        current_boxes.append(box_info)
        
        # Check if mouse pointer is inside this box
        is_mouse_inside = (x_min <= mouse_x <= x_min + box_width and 
                           y_min <= mouse_y <= y_min + box_height)
        
        # Draw the rectangle
        if is_mouse_inside:
            # Highlight with white if mouse is inside
            cv2.rectangle(img_copy, (x_min-0, y_min-0), (x_min + box_width+0, y_min + box_height+0), (255, 255, 255), 10)
        else:
            # Normal black outline
            cv2.rectangle(img_copy, (x_min-0, y_min-0), (x_min + box_width+0, y_min + box_height+0), (0, 0, 0), 10)
        
        # Then draw the colored rectangle on top
        color = COLORS.get(category, (255, 255, 255))
        cv2.rectangle(img_copy, (x_min, y_min), (x_min + box_width, y_min + box_height), color, 4)
        
        # Add label with category and confidence
        category_name = current_categories.get(category, 'unknown')
        label = f"{category_name}: {confidence:.2f}"
        # Draw black background for text to improve visibility
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(img_copy, (x_min, y_min - text_height - 10), (x_min + text_width, y_min), (0, 0, 0), -1)
        # Draw the text in the color of the category
        cv2.putText(img_copy, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return img_copy

def display_image_with_boxes(image_path, metadata_entry, categories, confidence_threshold, current_index, total_images):
    """Display an image with its bounding boxes"""
    global current_image, current_metadata, current_categories, current_confidence_threshold, need_redraw
    
    # Check if we have metadata for this image
    if not metadata_entry:
        print(f"No metadata found for {image_path}")
        return 1  # Continue to next image
    
    # Check if there are any detections above the confidence threshold
    valid_detections = [d for d in metadata_entry.get('detections', []) 
                         if d.get('conf', 0) >= confidence_threshold]
    
    if not valid_detections:
        print(f"Skipping {image_path} - no detections above confidence threshold {confidence_threshold}")
        return 1  # Continue to next image
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not read image: {image_path}")
        return 1  # Continue to next image
    
    # Update the global variables for mouse interaction
    current_image = img
    current_metadata = metadata_entry
    current_categories = categories
    current_confidence_threshold = confidence_threshold
    
    # Create the window
    window_name = "Wildlife Detector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Set up the mouse callback
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Draw the initial image with boxes
    display_img = draw_image_with_boxes()
    
    # Display the filename with index information
    basename = os.path.basename(image_path)
    display_text = f"{current_index+1} / {total_images} : {basename}"
    cv2.putText(display_img, display_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display the image
    cv2.imshow(window_name, display_img)
    
    print(f"Showing {image_path} with {len(valid_detections)} detection(s)")
    print(f"Controls: SPACE = next image, BACKSPACE = previous image, ESC = exit")
    
    # Main event loop
    while True:
        # Check if we need to redraw due to mouse movement
        if need_redraw:
            # Redraw the image with boxes
            display_img = draw_image_with_boxes()
            # Re-add the filename text
            cv2.putText(display_img, display_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # Show the updated image
            cv2.imshow(window_name, display_img)
            need_redraw = False
        
        # Wait for key press with a short timeout to allow mouse movement updates
        key = cv2.waitKey(50) & 0xFF
        
        # If no key was pressed, continue the loop
        if key == 255:
            continue
            
        print(f"Key pressed: {key}")
        if key == 27 or key == 113:  # ESC key or q
            return 0  # Exit the viewer
        elif key == 32:  # SPACE key
            return 1  # Continue to next image
        elif key == 8 or key == 127:  # BACKSPACE key
            return -1  # Go to previous image

def main():
    # Parse command line arguments
    args = parse_args()
    images_dir = args.image_dir
    confidence_threshold = args.confidence
    json_file = args.json_file
    
    print(f"Using confidence threshold: {confidence_threshold}")
    
    # Load the JSON metadata
    metadata = load_metadata(json_file)
    categories = metadata.get('detection_categories', {})
    print(f"Categories in metadata: {categories}")
    
    # Create a lookup table for faster image-to-metadata mapping
    image_to_metadata = create_image_to_metadata_map(metadata)
    print(f"Loaded metadata for {len(image_to_metadata)} images")
    
    # Find all images
    all_images = find_all_images(images_dir)
    total_images = len(all_images)
    print(f"Found {total_images} images in {images_dir} directory")
    
    # Display images one by one
    i = 0
    while 0 <= i < total_images:
        image_path = all_images[i]
        
        # Get the path relative to the image directory
        # First convert to absolute path to handle cases where images_dir is relative
        abs_image_path = os.path.abspath(image_path)
        abs_images_dir = os.path.abspath(images_dir)
        
        # Extract the part of the path that's relative to the image directory
        if abs_image_path.startswith(abs_images_dir):
            rel_path = abs_image_path[len(abs_images_dir):].lstrip(os.path.sep)
            # Convert to the format used in the JSON (forward slashes)
            normalized_path = rel_path.replace(os.path.sep, '/')
            
            # Find metadata for this image
            metadata_entry = image_to_metadata.get(normalized_path)
            
            # Display the image with boxes
            result = display_image_with_boxes(image_path, metadata_entry, categories, 
                                             confidence_threshold, i, total_images)
            
            if result == 0:
                print("Exiting viewer...")
                break
            elif result == 1:
                # Move to next image
                i += 1
            elif result == -1:
                # Move to previous image
                i = max(0, i - 1)
        else:
            print(f"Warning: Image {image_path} is not within the specified image directory {images_dir}")
            i += 1
    
    cv2.destroyAllWindows()
    print("Done!")

if __name__ == "__main__":
    main() 