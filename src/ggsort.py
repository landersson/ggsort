#!/usr/bin/env python3

import os
import sqlite3
import cv2
import argparse
import sys

# Configuration
DEFAULT_DB_FILE = 'wildlife.db'  # Default SQLite database file
CONFIDENCE_THRESHOLD = 0.0  # Only display detections with confidence above this threshold

# Color mapping for different categories (in BGR format)
COLORS = {
    '1': (0, 0, 255),    # Red for GangGangs
    '2': (255, 0, 0),    # Blue for persons
    '3': (0, 255, 0),     # Green for vehicles
    '4': (0, 255, 255),   # Yellow for possums
    '5': (255, 255, 0)    # Purple for other
}

# Grey color for deleted detections
DELETED_COLOR = (128, 128, 128)  # Grey

# Variables to track mouse position and bounding boxes
mouse_x, mouse_y = 0, 0
current_boxes = []
current_image = None
current_metadata = None
current_categories = None
current_confidence_threshold = 0.0
need_redraw = False
selected_detection_id = None

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='View wildlife images with detection boxes')
    parser.add_argument('--image-dir', default='images', 
                        help='Directory containing wildlife images (default: images)')
    parser.add_argument('--confidence', type=float, default=CONFIDENCE_THRESHOLD,
                        help='Confidence threshold for displaying detections (default: 0.0)')
    parser.add_argument('--db-file', default=DEFAULT_DB_FILE,
                        help=f'SQLite database file (default: {DEFAULT_DB_FILE})')
    parser.add_argument('--start-index', type=int, help='Start from specific image index (overrides saved position)')
    return parser.parse_args()

def connect_to_database(db_file):
    """Connect to the SQLite database"""
    if not os.path.exists(db_file):
        raise FileNotFoundError(f"Database file not found: {db_file}")
    
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row  # Use row factory for named columns
    return conn

def ensure_app_state_table(conn):
    """Ensure app_state table exists in the database"""
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS app_state (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.commit()

def save_last_image_index(conn, index):
    """Save the last viewed image index to the database"""
    ensure_app_state_table(conn)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO app_state (key, value) VALUES ('last_image_index', ?)
    """, (str(index),))
    conn.commit()

def get_last_image_index(conn, default=0):
    """Get the last viewed image index from the database"""
    ensure_app_state_table(conn)
    cursor = conn.cursor()
    cursor.execute("SELECT value FROM app_state WHERE key = 'last_image_index'")
    result = cursor.fetchone()
    
    if result:
        try:
            return int(result['value'])
        except (ValueError, TypeError):
            return default
    return default

def get_detection_categories(conn):
    """Get detection categories from database if available, or use defaults"""
    # In the current database, we don't store categories, so we'll use the defaults
    # This can be expanded if categories are added to the database later
    return {
        '1': 'GangGang',
        '2': 'Person',
        '3': 'Vehicle',
        '4': 'Possum',
        '5': 'Other'
    }

def get_images_from_database(conn):
    """Get all images from the database"""
    cursor = conn.cursor()
    cursor.execute("SELECT id, file_path, width, height FROM images ORDER BY file_path")
    return cursor.fetchall()

def get_detections_for_image(conn, image_id, confidence_threshold):
    """Get detections for a specific image from the database"""
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, category, confidence, x, y, width, height, deleted, subcategory 
        FROM detections 
        WHERE image_id = ? AND confidence >= ?
        ORDER BY confidence DESC
    """, (image_id, confidence_threshold))
    return cursor.fetchall()

def toggle_detection_deleted_status(conn, detection_id):
    """Toggle the deleted status of a detection in the database"""
    cursor = conn.cursor()
    
    # First, get the current deleted status
    cursor.execute("SELECT deleted FROM detections WHERE id = ?", (detection_id,))
    result = cursor.fetchone()
    
    if result is None:
        return False  # Detection ID not found
    
    # Toggle the deleted status (0->1, 1->0)
    current_status = result['deleted']
    new_status = 1 if current_status == 0 else 0
    
    # Update the detection with the new status
    cursor.execute("UPDATE detections SET deleted = ? WHERE id = ?", (new_status, detection_id))
    conn.commit()
    
    return cursor.rowcount > 0

def update_detection_category(conn, detection_id, category_id):
    """Update the category of a detection in the database"""
    cursor = conn.cursor()
    cursor.execute("UPDATE detections SET category = ? WHERE id = ?", (category_id, detection_id))
    conn.commit()
    return cursor.rowcount > 0

def mouse_callback(event, x, y, flags, param):
    """Mouse callback function to track mouse position"""
    global mouse_x, mouse_y, need_redraw, selected_detection_id
    
    # Only respond to mouse movement
    if event == cv2.EVENT_MOUSEMOVE:
        # Update the mouse coordinates
        mouse_x, mouse_y = x, y
        need_redraw = True  # Flag to indicate we need to redraw the image

def draw_image_with_boxes(img_path, detections, categories, confidence_threshold):
    """Draw the image with bounding boxes based on current state"""
    global current_boxes, need_redraw, selected_detection_id
    
    # Read the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Could not read image: {img_path}")
        return None
    
    # Create a copy of the original image to draw on
    img_copy = img.copy()
    
    height, width = img_copy.shape[:2]
    
    # Clear the current boxes list
    current_boxes = []
    selected_detection_id = None
    
    # Draw bounding boxes
    for detection in detections:
        category = str(detection['category'])
        confidence = detection['confidence']
        is_deleted = detection['deleted'] == 1
        detection_id = detection['id']
        
        # Convert from normalized coordinates to pixel coordinates
        x_min = int(detection['x'] * width)
        y_min = int(detection['y'] * height)
        box_width = int(detection['width'] * width)
        box_height = int(detection['height'] * height)
        
        # Store box coordinates for mouse interaction
        box_info = {
            'x_min': x_min,
            'y_min': y_min,
            'x_max': x_min + box_width,
            'y_max': y_min + box_height,
            'category': category,
            'confidence': confidence,
            'detection_id': detection_id,
            'is_deleted': is_deleted
        }
        current_boxes.append(box_info)
        
        # Check if mouse pointer is inside this box
        is_mouse_inside = (x_min <= mouse_x <= x_min + box_width and 
                          y_min <= mouse_y <= y_min + box_height)
        
        if is_mouse_inside:
            # Set the selected detection ID if mouse is inside
            selected_detection_id = detection_id
        
        # Draw the rectangle
        if is_mouse_inside:
            # Highlight with white if mouse is inside
            cv2.rectangle(img_copy, (x_min-0, y_min-0), (x_min + box_width+0, y_min + box_height+0), (255, 255, 255), 10)
        else:
            # Normal black outline
            cv2.rectangle(img_copy, (x_min-0, y_min-0), (x_min + box_width+0, y_min + box_height+0), (0, 0, 0), 10)
        
        # Then draw the colored rectangle on top
        if is_deleted:
            color = DELETED_COLOR  # Grey for deleted detections
        else:
            color = COLORS.get(category, (255, 255, 255))
        cv2.rectangle(img_copy, (x_min, y_min), (x_min + box_width, y_min + box_height), color, 4)
        
        # Add label with category and confidence
        category_name = categories.get(category, 'unknown')
        label = f"{category_name}: {confidence:.2f}"
        if is_deleted:
            label += " (DELETED)"
        
        # Draw black background for text to improve visibility
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img_copy, (x_min - 2, y_min - text_height - 19), (x_min + text_width + 2, y_min - 6), (0, 0, 0), -1)
        # Draw the text in the color of the category
        cv2.putText(img_copy, label, (x_min, y_min - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return img, img_copy

def display_image_with_boxes(conn, image_id, image_path, confidence_threshold, categories, current_index, total_images):
    """Display an image with its bounding boxes"""
    global need_redraw, selected_detection_id
    
    # Get detections for this image
    detections = get_detections_for_image(conn, image_id, confidence_threshold)
    
    if not detections:
        print(f"Skipping {image_path} - no detections above confidence threshold {confidence_threshold}")
        return 1  # Continue to next image
    
    # Draw the image with boxes
    result = draw_image_with_boxes(image_path, detections, categories, confidence_threshold)
    if result is None:
        return 1  # Continue to next image
    
    original_img, display_img = result
    
    # Create the window
    window_name = "Wildlife Detector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Set up the mouse callback
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Display the filename with index information
    basename = os.path.basename(image_path)
    display_text = f"{current_index+1} / {total_images} : {basename}"
    # cv2.putText(display_img, display_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # cv2.putText(display_img, display_text, (100, display_img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    # Display the image
    # cv2.imshow(window_name, display_img)
    
    print(f"Showing {image_path} with {len(detections)} detection(s)")
    print(f"Controls: SPACE/RIGHT ARROW = next, LEFT ARROW = previous, BACKSPACE/x = toggle delete, p = mark as possum, o = mark as other, ESC = exit")
    
    # Save current position to database
    save_last_image_index(conn, current_index)
    
    need_redraw = True
    # Main event loop
    while True:
        # Check if we need to redraw due to mouse movement
        if need_redraw:
            # Refresh detections from the database in case they've been modified
            detections = get_detections_for_image(conn, image_id, confidence_threshold)
            
            # Redraw the image with boxes
            _, updated_img = draw_image_with_boxes(image_path, detections, categories, confidence_threshold)
            # Re-add the filename text at the bottom of the image
            text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
            text_height = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[1]
            text_x = int(updated_img.shape[1] / 2 - text_size[0] / 2)
            text_y = int(updated_img.shape[0] - text_height - 0)  # 10px padding from bottom
            cv2.putText(updated_img, display_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            # Show the updated image
            cv2.imshow(window_name, updated_img)
            need_redraw = False
        
        # Wait for key press with a short timeout to allow mouse movement updates
        key = cv2.waitKey(50) & 0xFF
        
        # If no key was pressed, continue the loop
        if key == 255:
            continue
            
        print(f"Key pressed: {key}")
        if key == 27 or key == 113:  # ESC key or q
            # Save the current index before exiting
            save_last_image_index(conn, current_index)
            return 0  # Exit the viewer
        elif key == 32 or key == 3:  # SPACE or right arrow
            # Save position before moving
            save_last_image_index(conn, current_index + 1)
            return 1  # Continue to next image
        elif key == 2:  # LEFT ARROW key
            # Save position before moving
            save_last_image_index(conn, max(0, current_index - 1))
            return -1  # Go to previous image
        elif key == 8 or key == 127 or key == 120:  # BACKSPACE or 'x' key
            # Toggle the deleted status of the currently selected detection
            if selected_detection_id is not None:
                success = toggle_detection_deleted_status(conn, selected_detection_id)
                if success:
                    print(f"Toggled deleted status for detection {selected_detection_id}")
                    need_redraw = True
                else:
                    print(f"Failed to toggle deleted status for detection {selected_detection_id}")
            else:
                print("No detection selected. Hover over a detection to select it.")
        elif key == 112:  # 'p' key
            # Set category to possum (category ID 4)
            if selected_detection_id is not None:
                success = update_detection_category(conn, selected_detection_id, 4)
                if success:
                    print(f"Changed detection {selected_detection_id} category to Possum")
                    need_redraw = True
                else:
                    print(f"Failed to change detection {selected_detection_id} category")
            else:
                print("No detection selected. Hover over a detection to select it.")
        elif key == 111:  # 'o' key
            # Set category to other (category ID 5)
            if selected_detection_id is not None:
                success = update_detection_category(conn, selected_detection_id, 5)
                if success:
                    print(f"Changed detection {selected_detection_id} category to Other")
                    need_redraw = True
                else:
                    print(f"Failed to change detection {selected_detection_id} category")
            else:
                print("No detection selected. Hover over a detection to select it.")

def main():
    # Parse command line arguments
    args = parse_args()
    image_dir = args.image_dir
    confidence_threshold = args.confidence
    db_file = args.db_file
    start_index = args.start_index
    
    print(f"Using database: {db_file}")
    print(f"Using confidence threshold: {confidence_threshold}")
    
    try:
        # Connect to the database
        conn = connect_to_database(db_file)
        
        # Get detection categories
        categories = get_detection_categories(conn)
        print(f"Categories: {categories}")
        
        # Get all images from the database
        images = get_images_from_database(conn)
        total_images = len(images)
        print(f"Found {total_images} images in the database")
        
        if total_images == 0:
            print("No images found in the database")
            return
        
        # Determine starting index
        i = 0
        if start_index is not None:
            # Use the specified index from command line if provided
            i = max(0, min(start_index, total_images - 1))
            print(f"Starting from specified index: {i}")
        else:
            # Otherwise use the last saved position
            i = get_last_image_index(conn)
            # Ensure the index is valid
            i = max(0, min(i, total_images - 1))
            print(f"Resuming from last position: {i}")
        
        # Display images one by one
        while 0 <= i < total_images:
            image = images[i]
            image_id = image['id']
            file_path = image['file_path']
            
            # Construct the full image path
            # If file_path is relative, join with image_dir
            full_path = file_path
            if not os.path.isabs(file_path):
                full_path = os.path.join(image_dir, file_path)
            
            # Display the image with boxes
            result = display_image_with_boxes(conn, image_id, full_path, confidence_threshold, 
                                             categories, i, total_images)
            
            if result == 0:
                print("Exiting viewer...")
                break
            elif result == 1:
                # Move to next image
                i += 1
            elif result == -1:
                # Move to previous image
                i = max(0, i - 1)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # Close the database connection
        if 'conn' in locals():
            conn.close()
        
        cv2.destroyAllWindows()
        print("Done!")

if __name__ == "__main__":
    main() 