#!/usr/bin/env python3

import os
import sqlite3
import cv2
import argparse
import sys
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

# Configuration
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

@dataclass
class BoundingBox:
    """Represents a bounding box in pixel coordinates"""
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    detection_id: int
    category: str
    confidence: float
    is_deleted: bool
    detection: Detection

class AppState:
    """Manages the application state"""
    def __init__(self):
        self.mouse_x: int = 0
        self.mouse_y: int = 0
        self.current_boxes: List[BoundingBox] = []
        self.need_redraw: bool = False
        self.selected_detection_id: Optional[int] = None
        
        # Relocation mode state
        self.relocation_mode: bool = False
        self.relocation_detection_id: Optional[int] = None
        self.relocation_clicks: List[Tuple[int, int]] = []
        
        # Overlapping detections state
        self.overlapping_detections: List[BoundingBox] = []
        self.current_overlap_index: int = 0

    def reset_relocation_mode(self):
        """Reset relocation mode state"""
        self.relocation_mode = False
        self.relocation_detection_id = None
        self.relocation_clicks = []

    def reset_overlapping_detections(self):
        """Reset overlapping detection state"""
        self.overlapping_detections = []
        self.current_overlap_index = 0

    def update_mouse_position(self, x: int, y: int):
        """Update mouse position and trigger redraw"""
        self.mouse_x = x
        self.mouse_y = y
        self.need_redraw = True

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

    def ensure_app_state_table(self):
        """Ensure app_state table exists in the database"""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS app_state (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)
        self.conn.commit()

    def save_last_image_index(self, index: int):
        """Save the last viewed image index to the database"""
        self.ensure_app_state_table()
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO app_state (key, value) VALUES ('last_image_index', ?)
        """, (str(index),))
        self.conn.commit()

    def get_last_image_index(self, default: int = 0) -> int:
        """Get the last viewed image index from the database"""
        self.ensure_app_state_table()
        cursor = self.conn.cursor()
        cursor.execute("SELECT value FROM app_state WHERE key = 'last_image_index'")
        result = cursor.fetchone()
        
        if result:
            try:
                return int(result['value'])
            except (ValueError, TypeError):
                return default
        return default

    def get_detection_categories(self) -> Dict[str, str]:
        """Get detection categories from database if available, or use defaults"""
        return {
            '1': 'GangGang',
            '2': 'Person',
            '3': 'Vehicle',
            '4': 'Possum',
            '5': 'Other'
        }

    def get_images_from_database(self) -> List[sqlite3.Row]:
        """Get all images from the database"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, file_path, width, height FROM images ORDER BY file_path")
        return cursor.fetchall()

    def get_detections_for_image(self, image_id: int, confidence_threshold: float) -> List[Detection]:
        """Get detections for a specific image from the database"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, category, confidence, x, y, width, height, deleted, subcategory 
            FROM detections 
            WHERE image_id = ? AND confidence >= ?
            ORDER BY confidence DESC
        """, (image_id, confidence_threshold))
        
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
                deleted=bool(row['deleted']),
                subcategory=row['subcategory']
            )
            detections.append(detection)
        return detections

    def toggle_detection_deleted_status(self, detection_id: int) -> bool:
        """Toggle the deleted status of a detection in the database"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT deleted FROM detections WHERE id = ?", (detection_id,))
        result = cursor.fetchone()
        
        if result is None:
            return False
        
        current_status = result['deleted']
        new_status = 1 if current_status == 0 else 0
        
        cursor.execute("UPDATE detections SET deleted = ? WHERE id = ?", (new_status, detection_id))
        self.conn.commit()
        
        return cursor.rowcount > 0

    def update_detection_category(self, detection_id: int, category_id: int) -> bool:
        """Update the category of a detection in the database"""
        cursor = self.conn.cursor()
        cursor.execute("UPDATE detections SET category = ? WHERE id = ?", (category_id, detection_id))
        self.conn.commit()
        return cursor.rowcount > 0

    def update_detection_coordinates(self, detection_id: int, x: float, y: float, width: float, height: float) -> bool:
        """Update the coordinates of a detection in the database"""
        cursor = self.conn.cursor()
        cursor.execute("""
            UPDATE detections 
            SET x = ?, y = ?, width = ?, height = ?
            WHERE id = ?
        """, (x, y, width, height, detection_id))
        self.conn.commit()
        return cursor.rowcount > 0

    def mark_all_detections_deleted(self, image_id: int) -> int:
        """Mark all detections on a specific image as deleted. Returns number of detections affected."""
        cursor = self.conn.cursor()
        cursor.execute("UPDATE detections SET deleted = 1 WHERE image_id = ?", (image_id,))
        self.conn.commit()
        return cursor.rowcount

    def close(self):
        """Close the database connection"""
        if self.conn:
            self.conn.close()

class DetectionRenderer:
    """Handles rendering of detection boxes on images"""
    def __init__(self, categories: Dict[str, str]):
        self.categories = categories

    def convert_to_pixel_coordinates(self, detection: Detection, img_width: int, img_height: int) -> BoundingBox:
        """Convert normalized detection coordinates to pixel coordinates"""
        x_min = int(detection.x * img_width)
        y_min = int(detection.y * img_height)
        box_width = int(detection.width * img_width)
        box_height = int(detection.height * img_height)
        
        return BoundingBox(
            x_min=x_min,
            y_min=y_min,
            x_max=x_min + box_width,
            y_max=y_min + box_height,
            detection_id=detection.id,
            category=str(detection.category),
            confidence=detection.confidence,
            is_deleted=detection.deleted,
            detection=detection
        )

    def find_overlapping_detections(self, boxes: List[BoundingBox], mouse_x: int, mouse_y: int) -> List[BoundingBox]:
        """Find all detection boxes that contain the mouse position"""
        overlapping = []
        for box in boxes:
            if box.x_min <= mouse_x <= box.x_max and box.y_min <= mouse_y <= box.y_max:
                overlapping.append(box)
        return overlapping

    def draw_boxes_on_image(self, original_img, detections: List[Detection], state: AppState) -> Any:
        """Draw bounding boxes on a copy of the original image"""
        img_copy = original_img.copy()
        height, width = img_copy.shape[:2]
        
        # Convert detections to pixel coordinates
        detection_boxes = []
        for detection in detections:
            box = self.convert_to_pixel_coordinates(detection, width, height)
            detection_boxes.append(box)
        
        state.current_boxes = detection_boxes
        
        # Handle overlapping detections (only if not in relocation mode)
        if not state.relocation_mode:
            current_overlapping = self.find_overlapping_detections(detection_boxes, state.mouse_x, state.mouse_y)
            
            # Check if overlapping detections changed
            current_ids = [box.detection_id for box in current_overlapping]
            previous_ids = [box.detection_id for box in state.overlapping_detections]
            
            if current_ids != previous_ids:
                state.overlapping_detections = current_overlapping
                state.current_overlap_index = 0
            
            # Set selected detection
            if state.overlapping_detections:
                state.current_overlap_index = state.current_overlap_index % len(state.overlapping_detections)
                state.selected_detection_id = state.overlapping_detections[state.current_overlap_index].detection_id
            else:
                state.current_overlap_index = 0
                state.selected_detection_id = None
        
        # Draw all detection boxes
        for box in detection_boxes:
            self._draw_single_box(img_copy, box, state)
        
        # Draw relocation UI elements
        if state.relocation_mode:
            self._draw_relocation_ui(img_copy, state)
        
        return img_copy

    def _draw_single_box(self, img_copy, box: BoundingBox, state: AppState):
        """Draw a single detection box"""
        is_selected = (box.detection_id == state.selected_detection_id)
        is_being_relocated = (state.relocation_mode and box.detection_id == state.relocation_detection_id)
        
        # Draw outline
        if is_selected and not state.relocation_mode:
            cv2.rectangle(img_copy, (box.x_min, box.y_min), (box.x_max, box.y_max), (255, 255, 255), 10)
        else:
            cv2.rectangle(img_copy, (box.x_min, box.y_min), (box.x_max, box.y_max), (0, 0, 0), 10)
        
        # Draw colored rectangle
        if is_being_relocated:
            color = (0, 0, 0)  # Black for detection being relocated
        elif box.is_deleted:
            color = DELETED_COLOR
        else:
            color = COLORS.get(box.category, (255, 255, 255))
        
        cv2.rectangle(img_copy, (box.x_min, box.y_min), (box.x_max, box.y_max), color, 4)
        
        # Draw label
        self._draw_label(img_copy, box, state, color, is_selected)

    def _draw_label(self, img_copy, box: BoundingBox, state: AppState, color: Tuple[int, int, int], is_selected: bool):
        """Draw the detection label"""
        category_name = self.categories.get(box.category, 'unknown')
        label = f"{category_name}: {box.confidence:.2f}"
        
        if box.is_deleted:
            label += " (DELETED)"
        
        if is_selected and len(state.overlapping_detections) > 1:
            label += f" [{state.current_overlap_index + 1}/{len(state.overlapping_detections)}]"
        
        # Draw background and text
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img_copy, (box.x_min - 2, box.y_min - text_height - 19), 
                     (box.x_min + text_width + 2, box.y_min - 6), (0, 0, 0), -1)
        cv2.putText(img_copy, label, (box.x_min, box.y_min - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    def _draw_relocation_ui(self, img_copy, state: AppState):
        """Draw relocation mode UI elements"""
        for i, (click_x, click_y) in enumerate(state.relocation_clicks):
            cv2.circle(img_copy, (click_x, click_y), 8, (0, 255, 255), -1)
            cv2.putText(img_copy, str(i + 1), (click_x - 5, click_y + 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        if len(state.relocation_clicks) == 2:
            x1, y1 = state.relocation_clicks[0]
            x2, y2 = state.relocation_clicks[1]
            new_x_min = min(x1, x2)
            new_y_min = min(y1, y2)
            new_x_max = max(x1, x2)
            new_y_max = max(y1, y2)
            cv2.rectangle(img_copy, (new_x_min, new_y_min), (new_x_max, new_y_max), (0, 255, 0), 3)

class DetectionController:
    """Handles detection-related operations and business logic"""
    def __init__(self, db_manager: DatabaseManager, renderer: DetectionRenderer):
        self.db_manager = db_manager
        self.renderer = renderer

    def handle_category_change(self, detection_id: Optional[int], category_info: Dict[str, Any]) -> bool:
        """Handle changing a detection's category"""
        if detection_id is None:
            print("No detection selected. Hover over a detection to select it.")
            return False
        
        success = self.db_manager.update_detection_category(detection_id, category_info['id'])
        if success:
            print(f"Changed detection {detection_id} category to {category_info['name']}")
            return True
        else:
            print(f"Failed to change detection {detection_id} category")
            return False

    def cycle_overlapping_detections(self, state: AppState) -> bool:
        """Cycle through overlapping detections under the mouse cursor"""
        if len(state.overlapping_detections) <= 1:
            print("No overlapping detections to cycle through")
            return False
        
        state.current_overlap_index = (state.current_overlap_index + 1) % len(state.overlapping_detections)
        state.selected_detection_id = state.overlapping_detections[state.current_overlap_index].detection_id
        
        print(f"Cycled to detection {state.selected_detection_id} "
              f"({state.current_overlap_index + 1}/{len(state.overlapping_detections)})")
        state.need_redraw = True
        return True

    def handle_relocation_mode(self, state: AppState) -> bool:
        """Handle relocation mode toggle and processing"""
        if not state.relocation_mode:
            if state.selected_detection_id is None:
                print("No detection selected. Hover over a detection to select it first.")
                return False
            
            print(f"Entering relocation mode for detection {state.selected_detection_id}")
            print("Click twice to define new bounding box: first click = top-left, second click = bottom-right")
            print("Press 'c' again to cancel relocation")
            
            state.relocation_mode = True
            state.relocation_detection_id = state.selected_detection_id
            state.relocation_clicks = []
            state.need_redraw = True
            return True
        else:
            print("Cancelled relocation mode")
            state.reset_relocation_mode()
            state.need_redraw = True
            return True

    def apply_relocation(self, state: AppState, original_img) -> bool:
        """Apply the relocation changes to the database"""
        if len(state.relocation_clicks) != 2:
            print("Error: Need exactly 2 clicks to apply relocation")
            return False
        
        height, width = original_img.shape[:2]
        
        x1, y1 = state.relocation_clicks[0]
        x2, y2 = state.relocation_clicks[1]
        
        new_x_min = min(x1, x2)
        new_y_min = min(y1, y2)
        new_x_max = max(x1, x2)
        new_y_max = max(y1, y2)
        
        # Convert to normalized coordinates
        new_x = new_x_min / width
        new_y = new_y_min / height
        new_width = (new_x_max - new_x_min) / width
        new_height = (new_y_max - new_y_min) / height
        
        # Ensure coordinates are within valid range
        new_x = max(0, min(1, new_x))
        new_y = max(0, min(1, new_y))
        new_width = max(0, min(1 - new_x, new_width))
        new_height = max(0, min(1 - new_y, new_height))
        
        success = self.db_manager.update_detection_coordinates(
            state.relocation_detection_id, new_x, new_y, new_width, new_height)
        
        if success:
            print(f"Successfully relocated detection {state.relocation_detection_id}")
            print(f"New coordinates: x={new_x:.3f}, y={new_y:.3f}, width={new_width:.3f}, height={new_height:.3f}")
        else:
            print(f"Failed to relocate detection {state.relocation_detection_id}")
        
        state.reset_relocation_mode()
        state.need_redraw = True
        return success

    def mark_all_detections_deleted(self, image_id: int) -> bool:
        """Mark all detections on the current image as deleted"""
        affected_count = self.db_manager.mark_all_detections_deleted(image_id)
        if affected_count > 0:
            print(f"Marked {affected_count} detection(s) as deleted on current image")
            return True
        else:
            print("No detections found to mark as deleted")
            return False

class UserInterface:
    """Handles user interface operations"""
    def __init__(self, controller: DetectionController, renderer: DetectionRenderer, state: AppState):
        self.controller = controller
        self.renderer = renderer
        self.state = state
        self.window_name = "GGSort"
        self.current_original_img = None  # Store current image for relocation

    def mouse_callback(self, event, x: int, y: int, flags, param):
        """Handle mouse events"""
        if event == cv2.EVENT_MOUSEMOVE:
            self.state.update_mouse_position(x, y)
        elif event == cv2.EVENT_LBUTTONDOWN and self.state.relocation_mode:
            self.state.relocation_clicks.append((x, y))
            print(f"Click {len(self.state.relocation_clicks)}: ({x}, {y})")
            
            if len(self.state.relocation_clicks) == 2:
                print("Two clicks recorded. Applying new coordinates...")
                # Apply relocation immediately when second click is made
                if self.current_original_img is not None:
                    self.controller.apply_relocation(self.state, self.current_original_img)
                else:
                    print("Error: No image available for relocation")
                    self.state.reset_relocation_mode()
            
            self.state.need_redraw = True

    def handle_key_press(self, key: int, original_img, image_id: int = None) -> Optional[int]:
        """Handle key press events. Returns navigation command or None"""
        # print(f"Key pressed: {key}")
        
        if key == 27 or key == 113:  # ESC key or q
            return 0  # Exit
        elif key == 32 or key == 3:  # SPACE or right arrow
            return 1  # Next image
        elif key == 2:  # LEFT ARROW key
            return -1  # Previous image
        elif key == 8 or key == 127 or key == 120:  # BACKSPACE or 'x' key
            if self.state.selected_detection_id is not None:
                success = self.controller.db_manager.toggle_detection_deleted_status(self.state.selected_detection_id)
                if success:
                    print(f"Toggled deleted status for detection {self.state.selected_detection_id}")
                    self.state.need_redraw = True
                else:
                    print(f"Failed to toggle deleted status for detection {self.state.selected_detection_id}")
            else:
                print("No detection selected. Hover over a detection to select it.")
        elif key == 122:  # 'z' key for marking all detections as deleted
            if image_id is not None:
                success = self.controller.mark_all_detections_deleted(image_id)
                if success:
                    self.state.need_redraw = True
            else:
                print("Error: No image ID available for marking detections as deleted")
        elif key == 99:  # 'c' key for relocation mode
            self.controller.handle_relocation_mode(self.state)
        elif key == 9:  # TAB key for cycling through overlapping detections
            self.controller.cycle_overlapping_detections(self.state)
        elif key in CATEGORY_KEYS:  # Category change keys (p, o, etc.)
            if self.controller.handle_category_change(self.state.selected_detection_id, CATEGORY_KEYS[key]):
                self.state.need_redraw = True
        
        # Note: Relocation completion is now handled in mouse_callback when second click is made
        
        return None

    def display_image_with_boxes(self, db_manager: DatabaseManager, image_id: int, image_path: str, 
                                confidence_threshold: float, categories: Dict[str, str], 
                                current_index: int, total_images: int) -> int:
        """Display an image with its bounding boxes"""
        # Reset state when switching images
        if self.state.relocation_mode:
            print("Exiting relocation mode due to image change")
            self.state.reset_relocation_mode()
        
        self.state.reset_overlapping_detections()
        
        # Get detections for this image
        detections = db_manager.get_detections_for_image(image_id, confidence_threshold)
        
        if not detections:
            print(f"Skipping {image_path} - no detections above confidence threshold {confidence_threshold}")
            return 1
        
        # Load the image
        original_img = cv2.imread(image_path)
        if original_img is None:
            print(f"Could not read image: {image_path}")
            return 1
        
        # Store the original image for relocation operations
        self.current_original_img = original_img
        
        # Setup window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Display info
        basename = os.path.basename(image_path)
        display_text = f"{current_index+1} / {total_images} : {basename}"
        
        print(f"Showing {image_path} with {len(detections)} detection(s)")
        print(f"Controls: SPACE/RIGHT ARROW = next, LEFT ARROW = previous, BACKSPACE/x = toggle delete, "
              f"z = delete all detections, p = mark as possum, o = mark as other, c = relocate detection, TAB = cycle overlapping detections, ESC = exit")
        
        # Save position
        db_manager.save_last_image_index(current_index)
        
        self.state.need_redraw = True
        
        # Main event loop
        while True:
            if self.state.need_redraw:
                # Refresh detections and redraw
                detections = db_manager.get_detections_for_image(image_id, confidence_threshold)
                updated_img = self.renderer.draw_boxes_on_image(original_img, detections, self.state)
                
                # Add filename text
                text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                text_height = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[1]
                text_x = int(updated_img.shape[1] / 2 - text_size[0] / 2)
                text_y = int(updated_img.shape[0] - text_height - 0)
                cv2.putText(updated_img, display_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                cv2.imshow(self.window_name, updated_img)
                self.state.need_redraw = False
            
            # Handle key press
            key = cv2.waitKey(50) & 0xFF
            if key == 255:
                continue
            
            result = self.handle_key_press(key, original_img, image_id)
            if result is not None:
                if result == 0:
                    db_manager.save_last_image_index(current_index)
                elif result == 1:
                    db_manager.save_last_image_index(current_index + 1)
                elif result == -1:
                    db_manager.save_last_image_index(max(0, current_index - 1))
                return result

class GGSortApplication:
    """Main application class that coordinates all components"""
    def __init__(self, db_file: str, image_dir: str, confidence_threshold: float, start_index: Optional[int] = None):
        self.db_manager = DatabaseManager(db_file)
        self.categories = self.db_manager.get_detection_categories()
        self.renderer = DetectionRenderer(self.categories)
        self.state = AppState()
        self.controller = DetectionController(self.db_manager, self.renderer)
        self.ui = UserInterface(self.controller, self.renderer, self.state)
        
        self.image_dir = image_dir
        self.confidence_threshold = confidence_threshold
        self.start_index = start_index

    def run(self) -> int:
        """Run the main application"""
        try:
            print(f"Using confidence threshold: {self.confidence_threshold}")
            print(f"Categories: {self.categories}")
            
            # Get all images
            images = self.db_manager.get_images_from_database()
            total_images = len(images)
            print(f"Found {total_images} images in the database")
            
            if total_images == 0:
                print("No images found in the database")
                return 1
            
            # Determine starting index
            if self.start_index is not None:
                i = max(0, min(self.start_index, total_images - 1))
                print(f"Starting from specified index: {i}")
            else:
                i = self.db_manager.get_last_image_index()
                i = max(0, min(i, total_images - 1))
                print(f"Resuming from last position: {i}")
            
            # Main image display loop
            while True:
                # Ensure index is within bounds
                i = max(0, min(i, total_images - 1))
                
                image = images[i]
                image_id = image['id']
                file_path = image['file_path']
                
                # Construct full path
                full_path = file_path
                if not os.path.isabs(file_path):
                    full_path = os.path.join(self.image_dir, file_path)
                
                # Display image
                result = self.ui.display_image_with_boxes(
                    self.db_manager, image_id, full_path, self.confidence_threshold,
                    self.categories, i, total_images)
                
                if result == 0:
                    print("Exiting viewer...")
                    break
                elif result == 1:
                    # Move to next image, but clamp to last image
                    new_i = i + 1
                    if new_i >= total_images:
                        print(f"Already at last image ({i + 1}/{total_images})")
                        # Stay at current image instead of exiting
                        continue
                    i = new_i
                elif result == -1:
                    # Move to previous image, but clamp to first image
                    new_i = i - 1
                    if new_i < 0:
                        print(f"Already at first image ({i + 1}/{total_images})")
                        # Stay at current image instead of exiting
                        continue
                    i = new_i
            
            return 0
            
        except Exception as e:
            print(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return 1
        finally:
            self.db_manager.close()
            cv2.destroyAllWindows()

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='View wildlife images with detection boxes')
    parser.add_argument('--image-dir', default='images', 
                        help='Directory containing wildlife images (default: images)')
    parser.add_argument('--confidence', type=float, default=CONFIDENCE_THRESHOLD,
                        help='Confidence threshold for displaying detections (default: 0.0)')
    parser.add_argument('--db-file', help='SQLite database file')
    parser.add_argument('--start-index', type=int, help='Start from specific image index (overrides saved position)')
    return parser.parse_args()

def main():
    """Main entry point"""
    args = parse_args()
    
    if not args.db_file:
        print("Error: --db-file is required")
        return 1
    
    print(f"Using database: {args.db_file}")
    
    app = GGSortApplication(
        db_file=args.db_file,
        image_dir=args.image_dir,
        confidence_threshold=args.confidence,
        start_index=args.start_index
    )
    
    return app.run()

if __name__ == "__main__":
    sys.exit(main()) 