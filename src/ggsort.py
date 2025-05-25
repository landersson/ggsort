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

# Category key mappings
CATEGORY_KEYS = {
    112: {'id': 4, 'name': 'Possum'},  # 'p' key
    111: {'id': 5, 'name': 'Other'},   # 'o' key
    103: {'id': 1, 'name': 'GangGang'} # 'g' key
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

# Variables for relocation mode
relocation_mode = False
relocation_detection_id = None
relocation_clicks = []
relocation_conn = None
relocation_original_img = None

# Variables for handling overlapping detections
overlapping_detections = []
current_overlap_index = 0

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

def update_detection_coordinates(conn, detection_id, x, y, width, height):
    """Update the coordinates of a detection in the database"""
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE detections 
        SET x = ?, y = ?, width = ?, height = ?
        WHERE id = ?
    """, (x, y, width, height, detection_id))
    conn.commit()
    return cursor.rowcount > 0

def handle_category_change(conn, detection_id, category_info):
    """Handle changing a detection's category"""
    if detection_id is None:
        print("No detection selected. Hover over a detection to select it.")
        return False
    
    success = update_detection_category(conn, detection_id, category_info['id'])
    if success:
        print(f"Changed detection {detection_id} category to {category_info['name']}")
        return True
    else:
        print(f"Failed to change detection {detection_id} category")
        return False

def cycle_overlapping_detections():
    """Cycle through overlapping detections under the mouse cursor"""
    global overlapping_detections, current_overlap_index, selected_detection_id, need_redraw
    
    if len(overlapping_detections) <= 1:
        print("No overlapping detections to cycle through")
        return
    
    # Move to the next detection in the overlapping list
    current_overlap_index = (current_overlap_index + 1) % len(overlapping_detections)
    selected_detection_id = overlapping_detections[current_overlap_index]['detection_id']
    
    print(f"Cycled to detection {selected_detection_id} ({current_overlap_index + 1}/{len(overlapping_detections)})")
    need_redraw = True

def handle_relocation_mode(conn, image_id, original_img):
    """Handle relocation mode toggle and processing"""
    global relocation_mode, relocation_detection_id, relocation_clicks, selected_detection_id, need_redraw
    global relocation_conn, relocation_original_img
    
    if not relocation_mode:
        # Entering relocation mode
        if selected_detection_id is None:
            print("No detection selected. Hover over a detection to select it first.")
            return
        
        print(f"Entering relocation mode for detection {selected_detection_id}")
        print("Click twice to define new bounding box: first click = top-left, second click = bottom-right")
        print("Press 'c' again to cancel relocation")
        
        relocation_mode = True
        relocation_detection_id = selected_detection_id
        relocation_clicks = []
        relocation_conn = conn
        relocation_original_img = original_img
        need_redraw = True
        
    else:
        # Already in relocation mode - cancel it
        print("Cancelled relocation mode")
        relocation_mode = False
        relocation_detection_id = None
        relocation_clicks = []
        relocation_conn = None
        relocation_original_img = None
        need_redraw = True

def apply_relocation(conn, image_id, original_img):
    """Apply the relocation changes to the database"""
    global relocation_mode, relocation_detection_id, relocation_clicks, need_redraw
    
    if len(relocation_clicks) != 2:
        print("Error: Need exactly 2 clicks to apply relocation")
        return
    
    # Get image dimensions
    height, width = original_img.shape[:2]
    
    # Calculate new bounding box from clicks
    x1, y1 = relocation_clicks[0]
    x2, y2 = relocation_clicks[1]
    
    # Calculate proper top-left and bottom-right coordinates
    new_x_min = min(x1, x2)
    new_y_min = min(y1, y2)
    new_x_max = max(x1, x2)
    new_y_max = max(y1, y2)
    
    # Convert to normalized coordinates (0-1 range)
    new_x = new_x_min / width
    new_y = new_y_min / height
    new_width = (new_x_max - new_x_min) / width
    new_height = (new_y_max - new_y_min) / height
    
    # Ensure coordinates are within valid range
    new_x = max(0, min(1, new_x))
    new_y = max(0, min(1, new_y))
    new_width = max(0, min(1 - new_x, new_width))
    new_height = max(0, min(1 - new_y, new_height))
    
    # Update the database
    success = update_detection_coordinates(conn, relocation_detection_id, new_x, new_y, new_width, new_height)
    
    if success:
        print(f"Successfully relocated detection {relocation_detection_id}")
        print(f"New coordinates: x={new_x:.3f}, y={new_y:.3f}, width={new_width:.3f}, height={new_height:.3f}")
    else:
        print(f"Failed to relocate detection {relocation_detection_id}")
    
    # Exit relocation mode
    relocation_mode = False
    relocation_detection_id = None
    relocation_clicks = []
    need_redraw = True

def apply_relocation_from_callback():
    """Apply relocation from mouse callback using global variables"""
    global relocation_conn, relocation_original_img
    
    if relocation_conn is None or relocation_original_img is None:
        print("Error: Relocation context not available")
        return
    
    apply_relocation(relocation_conn, None, relocation_original_img)

def mouse_callback(event, x, y, flags, param):
    """Mouse callback function to track mouse position and handle relocation clicks"""
    global mouse_x, mouse_y, need_redraw, selected_detection_id, relocation_mode, relocation_clicks
    
    # Always update mouse coordinates for hover detection
    if event == cv2.EVENT_MOUSEMOVE:
        # Update the mouse coordinates
        mouse_x, mouse_y = x, y
        need_redraw = True  # Flag to indicate we need to redraw the image
    
    # Handle mouse clicks during relocation mode
    elif event == cv2.EVENT_LBUTTONDOWN and relocation_mode:
        # Add the click coordinates to the list
        relocation_clicks.append((x, y))
        print(f"Click {len(relocation_clicks)}: ({x}, {y})")
        
        if len(relocation_clicks) == 2:
            print("Two clicks recorded. Applying new coordinates...")
            # Automatically apply the relocation after second click
            apply_relocation_from_callback()
        
        need_redraw = True

def draw_boxes_on_image(original_img, detections, categories):
    """Draw bounding boxes on a copy of the original image"""
    global current_boxes, selected_detection_id, relocation_mode, relocation_detection_id, relocation_clicks
    global overlapping_detections, current_overlap_index
    
    # Create a copy of the original image to draw on
    img_copy = original_img.copy()
    
    height, width = img_copy.shape[:2]
    
    # Clear the current boxes list
    current_boxes = []
    
    # First pass: collect all detection boxes and find overlapping ones
    detection_boxes = []
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
            'is_deleted': is_deleted,
            'detection': detection
        }
        detection_boxes.append(box_info)
        current_boxes.append(box_info)
    
    # Find all detections under the mouse cursor (only if not in relocation mode)
    if not relocation_mode:
        # Find current overlapping detections
        current_overlapping = []
        for box_info in detection_boxes:
            x_min, y_min = box_info['x_min'], box_info['y_min']
            x_max, y_max = box_info['x_max'], box_info['y_max']
            
            if x_min <= mouse_x <= x_max and y_min <= mouse_y <= y_max:
                current_overlapping.append(box_info)
        
        # Check if the overlapping detections have changed (mouse moved to different area)
        current_ids = [box['detection_id'] for box in current_overlapping]
        previous_ids = [box['detection_id'] for box in overlapping_detections]
        
        if current_ids != previous_ids:
            # Mouse moved to a different set of detections, reset the cycling
            overlapping_detections = current_overlapping
            current_overlap_index = 0
        
        # Set the selected detection
        if overlapping_detections:
            current_overlap_index = current_overlap_index % len(overlapping_detections)
            selected_detection_id = overlapping_detections[current_overlap_index]['detection_id']
        else:
            current_overlap_index = 0
            selected_detection_id = None
    
    # Second pass: draw all detection boxes
    for box_info in detection_boxes:
        detection = box_info['detection']
        category = box_info['category']
        confidence = box_info['confidence']
        is_deleted = box_info['is_deleted']
        detection_id = box_info['detection_id']
        x_min, y_min = box_info['x_min'], box_info['y_min']
        x_max, y_max = box_info['x_max'], box_info['y_max']
        box_width = x_max - x_min
        box_height = y_max - y_min
        
        # Check if this detection is currently selected
        is_selected = (detection_id == selected_detection_id)
        
        # Check if this is the detection being relocated
        is_being_relocated = (relocation_mode and detection_id == relocation_detection_id)
        
        # Draw the rectangle outline
        if is_selected and not relocation_mode:
            # Highlight with white if this is the selected detection
            cv2.rectangle(img_copy, (x_min-0, y_min-0), (x_max+0, y_max+0), (255, 255, 255), 10)
        else:
            # Normal black outline
            cv2.rectangle(img_copy, (x_min-0, y_min-0), (x_max+0, y_max+0), (0, 0, 0), 10)
        
        # Then draw the colored rectangle on top
        if is_being_relocated:
            color = (0, 0, 0)  # Black for detection being relocated
        elif is_deleted:
            color = DELETED_COLOR  # Grey for deleted detections
        else:
            color = COLORS.get(category, (255, 255, 255))
        cv2.rectangle(img_copy, (x_min, y_min), (x_max, y_max), color, 4)
        
        # Add label with category and confidence
        category_name = categories.get(category, 'unknown')
        label = f"{category_name}: {confidence:.2f}"
        if is_deleted:
            label += " (DELETED)"
        
        # If this is the selected detection and there are overlapping detections, show the index
        if is_selected and len(overlapping_detections) > 1:
            label += f" [{current_overlap_index + 1}/{len(overlapping_detections)}]"
        
        # Draw black background for text to improve visibility
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img_copy, (x_min - 2, y_min - text_height - 19), (x_min + text_width + 2, y_min - 6), (0, 0, 0), -1)
        # Draw the text in the color of the category
        cv2.putText(img_copy, label, (x_min, y_min - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Draw relocation clicks if in relocation mode
    if relocation_mode and relocation_clicks:
        for i, (click_x, click_y) in enumerate(relocation_clicks):
            # Draw a small circle at each click position
            cv2.circle(img_copy, (click_x, click_y), 8, (0, 255, 255), -1)  # Yellow circle
            # Add number label
            cv2.putText(img_copy, str(i + 1), (click_x - 5, click_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # If we have two clicks, draw the new bounding box
        if len(relocation_clicks) == 2:
            x1, y1 = relocation_clicks[0]
            x2, y2 = relocation_clicks[1]
            # Calculate proper top-left and bottom-right coordinates
            new_x_min = min(x1, x2)
            new_y_min = min(y1, y2)
            new_x_max = max(x1, x2)
            new_y_max = max(y1, y2)
            # Draw the new bounding box in bright green
            cv2.rectangle(img_copy, (new_x_min, new_y_min), (new_x_max, new_y_max), (0, 255, 0), 3)
    
    return img_copy

def display_image_with_boxes(conn, image_id, image_path, confidence_threshold, categories, current_index, total_images):
    """Display an image with its bounding boxes"""
    global need_redraw, selected_detection_id, relocation_mode, relocation_detection_id, relocation_clicks
    global relocation_conn, relocation_original_img, overlapping_detections, current_overlap_index
    
    # Reset relocation mode when switching images
    if relocation_mode:
        print("Exiting relocation mode due to image change")
        relocation_mode = False
        relocation_detection_id = None
        relocation_clicks = []
        relocation_conn = None
        relocation_original_img = None
    
    # Reset overlapping detection state when switching images
    overlapping_detections = []
    current_overlap_index = 0
    
    # Get detections for this image
    detections = get_detections_for_image(conn, image_id, confidence_threshold)
    
    if not detections:
        print(f"Skipping {image_path} - no detections above confidence threshold {confidence_threshold}")
        return 1  # Continue to next image
    
    # Load the image from disk - only do this once
    original_img = cv2.imread(image_path)
    if original_img is None:
        print(f"Could not read image: {image_path}")
        return 1  # Continue to next image
    
    # Draw the initial boxes
    display_img = draw_boxes_on_image(original_img, detections, categories)
    
    # Create the window
    window_name = "GGSort"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Set up the mouse callback
    cv2.setMouseCallback(window_name, mouse_callback)
    
    # Display the filename with index information
    basename = os.path.basename(image_path)
    display_text = f"{current_index+1} / {total_images} : {basename}"
    
    print(f"Showing {image_path} with {len(detections)} detection(s)")
    print(f"Controls: SPACE/RIGHT ARROW = next, LEFT ARROW = previous, BACKSPACE/x = toggle delete, p = mark as possum, o = mark as other, c = relocate detection, TAB = cycle overlapping detections, ESC = exit")
    
    # Save current position to database
    save_last_image_index(conn, current_index)
    
    need_redraw = True
    # Main event loop
    while True:
        # Check if we need to redraw due to mouse movement
        if need_redraw:
            # Refresh detections from the database in case they've been modified
            detections = get_detections_for_image(conn, image_id, confidence_threshold)
            
            # Redraw using the cached original image
            updated_img = draw_boxes_on_image(original_img, detections, categories)
            
            # Add the filename text at the bottom of the image
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
        elif key == 99:  # 'c' key for relocation mode
            handle_relocation_mode(conn, image_id, original_img)
        elif key == 9:  # TAB key for cycling through overlapping detections
            cycle_overlapping_detections()
        elif key in CATEGORY_KEYS:  # Category change keys (p, o, etc.)
            if handle_category_change(conn, selected_detection_id, CATEGORY_KEYS[key]):
                need_redraw = True

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