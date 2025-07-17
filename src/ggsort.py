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
AUTODELETE_PIXEL_THRESHOLD = 20  # Pixel threshold for autodelete rectangle matching

# Corner knob configuration
CORNER_PROXIMITY_THRESHOLD = 60  # Pixels - show corner knobs when mouse is within this distance
CORNER_KNOB_RADIUS = 20  # Radius of corner knobs in pixels
CORNER_KNOB_COLOR = (180, 180, 180)  # Light grey  color for corner knobs

# Color mapping for different categories (in BGR format)
COLORS = {
    '1': (0, 0, 255),    # Red for GangGangs
    '2': (255, 0, 0),    # Blue for persons
    '3': (0, 255, 0),     # Green for vehicles
    '4': (66,200,94),   # Yellow for possums
    '5': (255, 255, 0)    # Purple for other
}

# Category key mappings
CATEGORY_KEYS = {
    112: {'id': 4, 'name': 'Possum'},  # 'p' key
    111: {'id': 5, 'name': 'Other'},   # 'o' key
    103: {'id': 1, 'name': 'GangGang'} # 'g' key
}

# Category key mappings for ALL detections in image (uppercase versions)
CATEGORY_KEYS_ALL = {
    80: {'id': 4, 'name': 'Possum'},   # 'P' key (Shift+p)
    79: {'id': 5, 'name': 'Other'},    # 'O' key (Shift+o)
    71: {'id': 1, 'name': 'GangGang'}  # 'G' key (Shift+g)
}

# Grey color for deleted detections
DELETED_COLOR = (128, 128, 128)  # Grey

# Color for autodelete template rectangles
TEMPLATE_COLOR = (128, 128, 128)  # Grey
TEMPLATE_LINE_WIDTH = 2

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
class AutodeleteTemplate:
    """Represents an autodelete template rectangle"""
    id: int
    x: float  # normalized coordinates
    y: float
    width: float
    height: float
    image_width: int  # original image dimensions for pixel matching
    image_height: int
    description: str = ""

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
        
        # Corner dragging state
        self.corner_dragging_mode: bool = False
        self.dragged_corner_detection_id: Optional[int] = None
        self.dragged_corner_type: Optional[str] = None  # 'top-left', 'top-right', 'bottom-left', 'bottom-right'
        self.original_detection_coords: Optional[Tuple[float, float, float, float]] = None  # x, y, width, height
        self.drag_start_mouse_pos: Optional[Tuple[int, int]] = None
        
        # Performance optimization for dragging
        self.last_drag_redraw_pos: Optional[Tuple[int, int]] = None

    def reset_relocation_mode(self):
        """Reset relocation mode state"""
        self.relocation_mode = False
        self.relocation_detection_id = None
        self.relocation_clicks = []

    def reset_overlapping_detections(self):
        """Reset overlapping detection state"""
        self.overlapping_detections = []
        self.current_overlap_index = 0

    def reset_corner_dragging_mode(self):
        """Reset corner dragging mode state"""
        self.corner_dragging_mode = False
        self.dragged_corner_detection_id = None
        self.dragged_corner_type = None
        self.original_detection_coords = None
        self.drag_start_mouse_pos = None
        self.last_drag_redraw_pos = None

    def get_corner_type(self, box: BoundingBox, corner_x: int, corner_y: int) -> str:
        """Determine which corner of the box the coordinates represent"""
        if corner_x == box.x_min and corner_y == box.y_min:
            return 'top-left'
        elif corner_x == box.x_max and corner_y == box.y_min:
            return 'top-right'
        elif corner_x == box.x_min and corner_y == box.y_max:
            return 'bottom-left'
        elif corner_x == box.x_max and corner_y == box.y_max:
            return 'bottom-right'
        else:
            return 'unknown'

    def update_mouse_position(self, x: int, y: int):
        """Update mouse position and trigger redraw"""
        self.mouse_x = x
        self.mouse_y = y
        self.need_redraw = True

    def should_redraw_for_drag(self, x: int, y: int, threshold: int = 3) -> bool:
        """Check if we should redraw during corner dragging based on mouse movement threshold"""
        if not self.corner_dragging_mode:
            return True
        
        if self.last_drag_redraw_pos is None:
            self.last_drag_redraw_pos = (x, y)
            return True
        
        last_x, last_y = self.last_drag_redraw_pos
        distance = ((x - last_x) ** 2 + (y - last_y) ** 2) ** 0.5
        
        if distance >= threshold:
            self.last_drag_redraw_pos = (x, y)
            return True
        
        return False

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

    def ensure_autodelete_table(self):
        """Ensure autodelete_templates table exists in the database"""
        cursor = self.conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS autodelete_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                x REAL NOT NULL,
                y REAL NOT NULL,
                width REAL NOT NULL,
                height REAL NOT NULL,
                image_width INTEGER NOT NULL,
                image_height INTEGER NOT NULL,
                description TEXT DEFAULT '',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
            SELECT id, category, confidence, x, y, width, height, deleted, subcategory, hard
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
                subcategory=row['subcategory'],
                hard=bool(row['hard']) if row['hard'] is not None else False
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

    def toggle_detection_hard_flag(self, detection_id: int) -> bool:
        """Toggle the hard flag of a detection in the database"""
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT hard FROM detections WHERE id = ?", (detection_id,))
        result = cursor.fetchone()
        
        if result is None:
            return False
        
        current_status = result['hard']
        new_status = 1 if current_status == 0 else 0
        
        cursor.execute("UPDATE detections SET hard = ? WHERE id = ?", (new_status, detection_id))
        self.conn.commit()
        
        return cursor.rowcount > 0

    def update_detection_category(self, detection_id: int, category_id: int, image_id: Optional[int] = None) -> bool:
        """Update the category of a detection in the database.
        If detection_id is negative and image_id is provided, updates all detections in the image."""
        cursor = self.conn.cursor()
        
        if detection_id < 0 and image_id is not None:
            # Update all detections in the image
            cursor.execute("UPDATE detections SET category = ? WHERE image_id = ?", (category_id, image_id))
        else:
            # Update specific detection
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

    def add_autodelete_template(self, detection: Detection, image_width: int, image_height: int, description: str = "") -> bool:
        """Add a new autodelete template based on a detection"""
        self.ensure_autodelete_table()
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT INTO autodelete_templates (x, y, width, height, image_width, image_height, description)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (detection.x, detection.y, detection.width, detection.height, image_width, image_height, description))
        self.conn.commit()
        return cursor.rowcount > 0

    def get_autodelete_templates(self) -> List[AutodeleteTemplate]:
        """Get all autodelete templates from the database"""
        self.ensure_autodelete_table()
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT id, x, y, width, height, image_width, image_height, description
            FROM autodelete_templates
            ORDER BY id
        """)
        
        templates = []
        for row in cursor.fetchall():
            template = AutodeleteTemplate(
                id=row['id'],
                x=row['x'],
                y=row['y'],
                width=row['width'],
                height=row['height'],
                image_width=row['image_width'],
                image_height=row['image_height'],
                description=row['description'] or ""
            )
            templates.append(template)
        return templates

    def is_detection_similar_to_template(self, detection: Detection, template: AutodeleteTemplate, image_width: int, image_height: int) -> bool:
        """Check if a detection is similar to an autodelete template (within pixel threshold)"""
        # Convert detection to pixel coordinates
        det_x1 = int(detection.x * image_width)
        det_y1 = int(detection.y * image_height)
        det_x2 = int((detection.x + detection.width) * image_width)
        det_y2 = int((detection.y + detection.height) * image_height)
        
        # Convert template to pixel coordinates (scale from original image size)
        scale_x = image_width / template.image_width
        scale_y = image_height / template.image_height
        
        temp_x1 = int(template.x * template.image_width * scale_x)
        temp_y1 = int(template.y * template.image_height * scale_y)
        temp_x2 = int((template.x + template.width) * template.image_width * scale_x)
        temp_y2 = int((template.y + template.height) * template.image_height * scale_y)
        
        # Check if all corners are within threshold
        threshold = AUTODELETE_PIXEL_THRESHOLD
        return (abs(det_x1 - temp_x1) <= threshold and
                abs(det_y1 - temp_y1) <= threshold and
                abs(det_x2 - temp_x2) <= threshold and
                abs(det_y2 - temp_y2) <= threshold)

    def auto_delete_similar_detections(self, image_id: int, image_width: int, image_height: int) -> int:
        """Automatically delete detections similar to autodelete templates. Returns count of deleted detections."""
        templates = self.get_autodelete_templates()
        if not templates:
            return 0
        
        # Get all non-deleted detections for this image
        detections = self.get_detections_for_image(image_id, 0.0)  # Get all detections regardless of confidence
        non_deleted_detections = [d for d in detections if not d.deleted]
        
        deleted_count = 0
        for detection in non_deleted_detections:
            for template in templates:
                if self.is_detection_similar_to_template(detection, template, image_width, image_height):
                    # Mark this detection as deleted
                    cursor = self.conn.cursor()
                    cursor.execute("UPDATE detections SET deleted = 1 WHERE id = ?", (detection.id,))
                    deleted_count += 1
                    break  # Only need to match one template
        
        if deleted_count > 0:
            self.conn.commit()
        
        return deleted_count

    def clear_all_autodelete_templates(self) -> int:
        """Clear all autodelete templates from the database. Returns count of deleted templates."""
        self.ensure_autodelete_table()
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM autodelete_templates")
        count = cursor.fetchone()[0]
        
        if count > 0:
            cursor.execute("DELETE FROM autodelete_templates")
            self.conn.commit()
        
        return count

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

    def find_closest_corner(self, boxes: List[BoundingBox], mouse_x: int, mouse_y: int, threshold: int = CORNER_PROXIMITY_THRESHOLD) -> Optional[Tuple[int, int, int]]:
        """Find the closest corner across all non-deleted detection boxes. Returns (corner_x, corner_y, detection_id) or None."""
        closest_corner = None
        closest_distance = float('inf')
        closest_detection_id = None
        
        for box in boxes:
            # Skip deleted detections
            if box.is_deleted:
                continue
                
            corners = [
                (box.x_min, box.y_min),  # top-left
                (box.x_max, box.y_min),  # top-right
                (box.x_min, box.y_max),  # bottom-left
                (box.x_max, box.y_max)   # bottom-right
            ]
            
            for corner_x, corner_y in corners:
                distance = ((mouse_x - corner_x) ** 2 + (mouse_y - corner_y) ** 2) ** 0.5
                if distance <= threshold and distance < closest_distance:
                    closest_distance = distance
                    closest_corner = (corner_x, corner_y)
                    closest_detection_id = box.detection_id
        
        if closest_corner is not None:
            return (closest_corner[0], closest_corner[1], closest_detection_id)
        return None

    def is_mouse_near_corner(self, box: BoundingBox, mouse_x: int, mouse_y: int, threshold: int = CORNER_PROXIMITY_THRESHOLD) -> List[Tuple[int, int]]:
        """Check if mouse is near any corner of the bounding box. Returns list of corners that are close."""
        corners = [
            (box.x_min, box.y_min),  # top-left
            (box.x_max, box.y_min),  # top-right
            (box.x_min, box.y_max),  # bottom-left
            (box.x_max, box.y_max)   # bottom-right
        ]
        
        close_corners = []
        for corner_x, corner_y in corners:
            distance = ((mouse_x - corner_x) ** 2 + (mouse_y - corner_y) ** 2) ** 0.5
            if distance <= threshold:
                close_corners.append((corner_x, corner_y))
        
        return close_corners

    def draw_corner_knobs(self, img_copy, corners: List[Tuple[int, int]], color: Tuple[int, int, int] = CORNER_KNOB_COLOR):
        """Draw circular knobs at the specified corner positions"""
        for corner_x, corner_y in corners:
            # Draw filled circle
            cv2.circle(img_copy, (corner_x, corner_y), CORNER_KNOB_RADIUS, color, -1)
            # Draw black outline
            cv2.circle(img_copy, (corner_x, corner_y), CORNER_KNOB_RADIUS, (0, 0, 0), 2)

    def draw_boxes_on_image(self, original_img, detections: List[Detection], state: AppState, db_manager=None) -> Any:
        """Draw bounding boxes on a copy of the original image"""
        img_copy = original_img.copy()
        height, width = img_copy.shape[:2]
        
        # Convert detections to pixel coordinates
        detection_boxes = []
        for detection in detections:
            box = self.convert_to_pixel_coordinates(detection, width, height)
            detection_boxes.append(box)
        
        state.current_boxes = detection_boxes
        
        # Find the closest corner across all non-deleted detections
        closest_corner_info = self.find_closest_corner(detection_boxes, state.mouse_x, state.mouse_y)
        
        # Handle overlapping detections (only if not in relocation mode and not dragging)
        if not state.relocation_mode and not state.corner_dragging_mode:
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
        
        # Draw autodelete template rectangles (skip during corner dragging for performance)
        if db_manager is not None and not state.corner_dragging_mode:
            self._draw_autodelete_templates(img_copy, db_manager, width, height)
        
        # Draw all detection boxes
        for box in detection_boxes:
            self._draw_single_box(img_copy, box, state, closest_corner_info)
        
        # Draw relocation UI elements (skip during corner dragging)
        if state.relocation_mode and not state.corner_dragging_mode:
            self._draw_relocation_ui(img_copy, state)
        
        return img_copy

    def _draw_single_box(self, img_copy, box: BoundingBox, state: AppState, closest_corner_info: Optional[Tuple[int, int, int]]):
        """Draw a single detection box"""
        is_selected = (box.detection_id == state.selected_detection_id)
        is_being_relocated = (state.relocation_mode and box.detection_id == state.relocation_detection_id)
        is_being_dragged = (state.corner_dragging_mode and box.detection_id == state.dragged_corner_detection_id)
        
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
        
        # Draw corner knob if this detection has the closest corner and is not in relocation mode
        if not state.relocation_mode:
            # Handle dragging mode separately
            if (state.corner_dragging_mode and 
                state.dragged_corner_detection_id == box.detection_id):
                # Always show white knob at mouse position when dragging this detection
                corner_x, corner_y = state.mouse_x, state.mouse_y
                knob_color = (255, 255, 255)  # White for dragged corner
                self.draw_corner_knobs(img_copy, [(corner_x, corner_y)], knob_color)
            elif (closest_corner_info is not None and 
                  closest_corner_info[2] == box.detection_id):
                # Show knob at corner position when not dragging
                corner_x, corner_y = closest_corner_info[0], closest_corner_info[1]
                
                # Check if mouse is within the knob radius for white color
                distance = ((state.mouse_x - corner_x) ** 2 + (state.mouse_y - corner_y) ** 2) ** 0.5
                if distance <= CORNER_KNOB_RADIUS:
                    knob_color = (255, 255, 255)  # White when mouse is inside knob
                else:
                    knob_color = CORNER_KNOB_COLOR  # Default grey when nearby but outside knob
                
                self.draw_corner_knobs(img_copy, [(corner_x, corner_y)], knob_color)
        
        # Draw label (skip for non-dragged boxes during corner dragging for performance)
        if not state.corner_dragging_mode or is_being_dragged:
            self._draw_label(img_copy, box, state, color, is_selected)

    def _draw_label(self, img_copy, box: BoundingBox, state: AppState, color: Tuple[int, int, int], is_selected: bool):
        """Draw the detection label"""
        category_name = self.categories.get(box.category, 'unknown')
        label = f"{category_name}: {box.confidence:.2f}"
        
        if box.detection.hard:
            label += " (HARD!)"
        
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

    def _draw_autodelete_templates(self, img_copy, db_manager, img_width: int, img_height: int):
        """Draw autodelete template rectangles for debugging"""
        templates = db_manager.get_autodelete_templates()
        
        for template in templates:
            # Scale template coordinates to current image size
            scale_x = img_width / template.image_width
            scale_y = img_height / template.image_height
            
            # Convert normalized coordinates to pixel coordinates with scaling
            x1 = int(template.x * template.image_width * scale_x)
            y1 = int(template.y * template.image_height * scale_y)
            x2 = int((template.x + template.width) * template.image_width * scale_x)
            y2 = int((template.y + template.height) * template.image_height * scale_y)
            
            # Draw template rectangle
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), TEMPLATE_COLOR, TEMPLATE_LINE_WIDTH)
            
            # Add small label to identify it as a template
            label = f"T{template.id}"
            (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_copy, (x1 - 1, y1 - text_height - 5), 
                         (x1 + text_width + 1, y1 - 2), TEMPLATE_COLOR, -1)
            cv2.putText(img_copy, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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

    def handle_category_change(self, detection_id: Optional[int], category_info: Dict[str, Any], image_id: Optional[int] = None) -> bool:
        """Handle changing a detection's category.
        If detection_id is negative and image_id is provided, changes all detections in the image."""
        if detection_id is None:
            print("No detection selected. Hover over a detection to select it.")
            return False
        
        if detection_id < 0 and image_id is not None:
            # Update all detections in the image
            success = self.db_manager.update_detection_category(detection_id, category_info['id'], image_id)
            if success:
                print(f"Changed all detections in image {image_id} to category {category_info['name']}")
                return True
            else:
                print(f"Failed to change all detections in image {image_id} to category {category_info['name']}")
                return False
        else:
            # Update specific detection
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

    def add_autodelete_template(self, detection_id: int, original_img, state: AppState) -> bool:
        """Add a detection as an autodelete template"""
        # Find the detection in current boxes
        detection = None
        for box in state.current_boxes:
            if box.detection_id == detection_id:
                detection = box.detection
                break
        
        if detection is None:
            print(f"Could not find detection {detection_id} to add as autodelete template")
            return False
        
        height, width = original_img.shape[:2]
        description = f"Template from detection {detection_id}"
        
        success = self.db_manager.add_autodelete_template(detection, width, height, description)
        if success:
            print(f"Added detection {detection_id} as autodelete template")
            print(f"Template location: x={detection.x:.3f}, y={detection.y:.3f}, w={detection.width:.3f}, h={detection.height:.3f}")
            return True
        else:
            print(f"Failed to add detection {detection_id} as autodelete template")
            return False

    def clear_all_autodelete_templates(self) -> bool:
        """Clear all autodelete templates from the database"""
        count = self.db_manager.clear_all_autodelete_templates()
        if count > 0:
            print(f"Cleared {count} autodelete template(s) from database")
            return True
        else:
            print("No autodelete templates found to clear")
            return False

    def start_corner_dragging(self, state: AppState, closest_corner_info: Tuple[int, int, int]) -> bool:
        """Start corner dragging mode"""
        corner_x, corner_y, detection_id = closest_corner_info
        
        # Find the detection box
        target_box = None
        for box in state.current_boxes:
            if box.detection_id == detection_id:
                target_box = box
                break
        
        if target_box is None:
            return False
        
        # Store original coordinates
        detection = target_box.detection
        state.original_detection_coords = (detection.x, detection.y, detection.width, detection.height)
        
        # Determine corner type
        corner_type = state.get_corner_type(target_box, corner_x, corner_y)
        
        # Set dragging state
        state.corner_dragging_mode = True
        state.dragged_corner_detection_id = detection_id
        state.dragged_corner_type = corner_type
        state.drag_start_mouse_pos = (state.mouse_x, state.mouse_y)
        
        print(f"Started dragging {corner_type} corner of detection {detection_id}")
        return True

    def update_detection_during_drag(self, state: AppState, img_width: int, img_height: int) -> bool:
        """Update detection coordinates during corner dragging"""
        if not state.corner_dragging_mode or state.dragged_corner_detection_id is None:
            return False
        
        # Find the detection being dragged
        target_box = None
        for box in state.current_boxes:
            if box.detection_id == state.dragged_corner_detection_id:
                target_box = box
                break
        
        if target_box is None:
            return False
        
        # Get current mouse position in normalized coordinates
        mouse_norm_x = max(0, min(1, state.mouse_x / img_width))
        mouse_norm_y = max(0, min(1, state.mouse_y / img_height))
        
        # Get original coordinates
        orig_x, orig_y, orig_width, orig_height = state.original_detection_coords
        
        # Calculate new coordinates based on corner type
        if state.dragged_corner_type == 'top-left':
            new_x = mouse_norm_x
            new_y = mouse_norm_y
            new_width = max(0.01, orig_x + orig_width - new_x)  # Ensure minimum width
            new_height = max(0.01, orig_y + orig_height - new_y)  # Ensure minimum height
        elif state.dragged_corner_type == 'top-right':
            new_x = orig_x
            new_y = mouse_norm_y
            new_width = max(0.01, mouse_norm_x - orig_x)
            new_height = max(0.01, orig_y + orig_height - new_y)
        elif state.dragged_corner_type == 'bottom-left':
            new_x = mouse_norm_x
            new_y = orig_y
            new_width = max(0.01, orig_x + orig_width - new_x)
            new_height = max(0.01, mouse_norm_y - orig_y)
        elif state.dragged_corner_type == 'bottom-right':
            new_x = orig_x
            new_y = orig_y
            new_width = max(0.01, mouse_norm_x - orig_x)
            new_height = max(0.01, mouse_norm_y - orig_y)
        else:
            return False
        
        # Ensure coordinates are within bounds
        new_x = max(0, min(1 - new_width, new_x))
        new_y = max(0, min(1 - new_height, new_y))
        new_width = max(0.01, min(1 - new_x, new_width))
        new_height = max(0.01, min(1 - new_y, new_height))
        
        # Update the detection object
        target_box.detection.x = new_x
        target_box.detection.y = new_y
        target_box.detection.width = new_width
        target_box.detection.height = new_height
        
        # Update the bounding box pixel coordinates
        target_box.x_min = int(new_x * img_width)
        target_box.y_min = int(new_y * img_height)
        target_box.x_max = int((new_x + new_width) * img_width)
        target_box.y_max = int((new_y + new_height) * img_height)
        
        return True

    def finish_corner_dragging(self, state: AppState) -> bool:
        """Finish corner dragging and save to database"""
        if not state.corner_dragging_mode or state.dragged_corner_detection_id is None:
            return False
        
        # Find the detection
        target_box = None
        for box in state.current_boxes:
            if box.detection_id == state.dragged_corner_detection_id:
                target_box = box
                break
        
        if target_box is None:
            state.reset_corner_dragging_mode()
            return False
        
        # Save to database
        detection = target_box.detection
        success = self.db_manager.update_detection_coordinates(
            state.dragged_corner_detection_id,
            detection.x, detection.y, detection.width, detection.height
        )
        
        if success:
            print(f"Updated detection {state.dragged_corner_detection_id} coordinates: "
                  f"x={detection.x:.3f}, y={detection.y:.3f}, w={detection.width:.3f}, h={detection.height:.3f}")
        else:
            print(f"Failed to update detection {state.dragged_corner_detection_id} coordinates")
        
        state.reset_corner_dragging_mode()
        # Force redraw to refresh detections from database
        state.need_redraw = True
        return success

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
            # Always update mouse position
            self.state.mouse_x = x
            self.state.mouse_y = y
            
            # During corner dragging, use optimized redraw logic
            if self.state.corner_dragging_mode:
                # On macOS trackpads, mouse events are less frequent during button press
                # So we need to be more aggressive about updates
                if self.current_original_img is not None:
                    height, width = self.current_original_img.shape[:2]
                    success = self.controller.update_detection_during_drag(self.state, width, height)
                    if success:
                        self.state.need_redraw = True
            else:
                # Normal mouse movement - always trigger redraw for corner knob updates
                self.state.need_redraw = True
                
        elif event == cv2.EVENT_LBUTTONDOWN:
            if self.state.relocation_mode:
                # Handle relocation clicks
                self.state.relocation_clicks.append((x, y))
                print(f"Click {len(self.state.relocation_clicks)}: ({x}, {y})")
                
                if len(self.state.relocation_clicks) == 2:
                    print("Two clicks recorded. Applying new coordinates...")
                    if self.current_original_img is not None:
                        self.controller.apply_relocation(self.state, self.current_original_img)
                    else:
                        print("Error: No image available for relocation")
                        self.state.reset_relocation_mode()
                
                self.state.need_redraw = True
            elif not self.state.corner_dragging_mode:
                # Try to start corner dragging if mouse is within a corner knob (same as 'd' key)
                if hasattr(self, '_last_closest_corner_info') and self._last_closest_corner_info is not None:
                    corner_x, corner_y, detection_id = self._last_closest_corner_info
                    # Check if mouse is actually within the corner knob radius
                    distance = ((x - corner_x) ** 2 + (y - corner_y) ** 2) ** 0.5
                    if distance <= CORNER_KNOB_RADIUS:
                        success = self.controller.start_corner_dragging(self.state, self._last_closest_corner_info)
                        if success:
                            print(f"Started dragging {self.state.dragged_corner_type} corner of detection {detection_id}")
                            print("Click again or press 'd' to finish dragging, or ESC to cancel")
                            self.state.need_redraw = True
                        else:
                            print("Failed to start corner dragging")
                    else:
                        print(f"Mouse must be within the corner knob (white circle) to start dragging")
                # If no corner is available, do nothing (no error message for cleaner UX)
            else:
                # Finish corner dragging (same as 'd' key)
                print("Finishing corner drag")
                success = self.controller.finish_corner_dragging(self.state)
                if success:
                    self.state.need_redraw = True

    def handle_key_press(self, key: int, original_img, image_id: int = None) -> Optional[int]:
        """Handle key press events. Returns navigation command or None"""
        # print(f"Key pressed: {key}")
        
        # Handle escape key specially to cancel corner dragging
        if key == 27:  # ESC key
            if self.state.corner_dragging_mode:
                print("Cancelled corner dragging - returning to original position")
                self.state.reset_corner_dragging_mode()
                self.state.need_redraw = True
                return None
            else:
                return 0  # Exit
        
        # Prevent most actions during corner dragging
        if self.state.corner_dragging_mode:
            if key == 113:  # 'q' key - allow exit
                return 0
            elif key == 100:  # 'd' key - allow finishing drag
                pass  # Will be handled below
            else:
                print("Corner dragging in progress. Press 'd' or click to finish, or ESC to cancel.")
                return None
        
        if key == 113:  # q
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
        elif key == 104:  # 'h' key for toggling hard flag
            if self.state.selected_detection_id is not None:
                success = self.controller.db_manager.toggle_detection_hard_flag(self.state.selected_detection_id)
                if success:
                    print(f"Toggled hard flag for detection {self.state.selected_detection_id}")
                    self.state.need_redraw = True
                else:
                    print(f"Failed to toggle hard flag for detection {self.state.selected_detection_id}")
            else:
                print("No detection selected. Hover over a detection to select it.")
        elif key == 75:  # 'K' key (Shift+k) for adding autodelete template
            if self.state.selected_detection_id is not None and original_img is not None:
                success = self.controller.add_autodelete_template(self.state.selected_detection_id, original_img, self.state)
                if success:
                    self.state.need_redraw = True
            else:
                print("No detection selected or no image available. Hover over a detection to select it.")
        elif key == 82:  # 'R' key (Shift+r) for clearing all autodelete templates
            success = self.controller.clear_all_autodelete_templates()
            if success:
                self.state.need_redraw = True
        elif key == 99:  # 'c' key for relocation mode
            self.controller.handle_relocation_mode(self.state)
        elif key == 100:  # 'd' key for corner dragging mode
            if not self.state.corner_dragging_mode:
                # Try to start corner dragging if mouse is within a corner knob
                if hasattr(self, '_last_closest_corner_info') and self._last_closest_corner_info is not None:
                    corner_x, corner_y, detection_id = self._last_closest_corner_info
                    # Check if mouse is actually within the corner knob radius (not just proximity threshold)
                    distance = ((self.state.mouse_x - corner_x) ** 2 + (self.state.mouse_y - corner_y) ** 2) ** 0.5
                    if distance <= CORNER_KNOB_RADIUS:
                        success = self.controller.start_corner_dragging(self.state, self._last_closest_corner_info)
                        if success:
                            print(f"Started dragging {self.state.dragged_corner_type} corner of detection {detection_id}")
                            print("Press 'd' again or click to finish dragging, or ESC to cancel")
                            self.state.need_redraw = True
                        else:
                            print("Failed to start corner dragging")
                    else:
                        print(f"Mouse must be within the corner knob (white/grey circle) to start dragging")
                        print(f"Current distance: {distance:.1f} pixels, required: ≤{CORNER_KNOB_RADIUS} pixels")
                else:
                    print("No corner highlighted. Move mouse near a corner first, then press 'd' or click")
            else:
                # Finish corner dragging
                print("Finishing corner drag")
                success = self.controller.finish_corner_dragging(self.state)
                if success:
                    self.state.need_redraw = True
        elif key == 9:  # TAB key for cycling through overlapping detections
            self.controller.cycle_overlapping_detections(self.state)
        elif key in CATEGORY_KEYS:  # Category change keys (p, o, etc.)
            if self.controller.handle_category_change(self.state.selected_detection_id, CATEGORY_KEYS[key], image_id):
                self.state.need_redraw = True
        elif key in CATEGORY_KEYS_ALL:  # Category change keys for ALL detections in image (uppercase versions)
            if self.controller.handle_category_change(-1, CATEGORY_KEYS_ALL[key], image_id):
                self.state.need_redraw = True
        elif key == 118:  # 'v' key for changing ALL detections to 'Other' (convenient shortcut)
            category_info = {'id': 5, 'name': 'Other'}
            if self.controller.handle_category_change(-1, category_info, image_id):
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
        
        if self.state.corner_dragging_mode:
            print("Exiting corner dragging mode due to image change")
            self.state.reset_corner_dragging_mode()
        
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
        
        # Auto-delete similar detections before displaying
        auto_deleted_count = db_manager.auto_delete_similar_detections(image_id, original_img.shape[1], original_img.shape[0])
        if auto_deleted_count > 0:
            print(f"Auto-deleted {auto_deleted_count} detection(s) matching autodelete templates")
            # Refresh detections after auto-deletion
            detections = db_manager.get_detections_for_image(image_id, confidence_threshold)
        
        # Display info
        basename = os.path.basename(image_path)
        display_text = f"{current_index+1} / {total_images} : {basename}"
        
        print(f"Showing {image_path} with {len(detections)} detection(s)")
        # print(f"Controls: SPACE/RIGHT ARROW = next, LEFT ARROW = previous, BACKSPACE/x = toggle delete, "
        #       f"z = delete all detections, h = toggle hard flag, p = mark as possum, o = mark as other, g = mark as ganggang, "
        #       f"P = mark ALL as possum, O = mark ALL as other, G = mark ALL as ganggang, v = mark ALL as other (quick), "
        #       f"c = relocate detection, CLICK or d = drag corner (inside white knob), Shift+K = add autodelete template, "
        #       f"Shift+R = clear all templates, TAB = cycle overlapping detections, ESC = exit")
        
        # Save position
        db_manager.save_last_image_index(current_index)
        
        # Initialize detections
        detections = db_manager.get_detections_for_image(image_id, confidence_threshold)
        
        self.state.need_redraw = True
        
        # Main event loop
        while True:
            if self.state.need_redraw:
                # Refresh detections and redraw - but NOT during corner dragging
                if not self.state.corner_dragging_mode:
                    detections = db_manager.get_detections_for_image(image_id, confidence_threshold)
                # During corner dragging, keep using the existing detections with modified coordinates
                updated_img = self.renderer.draw_boxes_on_image(original_img, detections, self.state, db_manager)
                
                # Store the closest corner info for click detection
                # Use the current_boxes that were just calculated in draw_boxes_on_image
                closest_corner_info = self.renderer.find_closest_corner(self.state.current_boxes, self.state.mouse_x, self.state.mouse_y)
                self._last_closest_corner_info = closest_corner_info
                
                # Add filename text
                text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                text_height = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[1]
                text_x = int(updated_img.shape[1] / 2 - text_size[0] / 2)
                text_y = int(updated_img.shape[0] - text_height - 0)
                cv2.putText(updated_img, display_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                cv2.imshow(self.window_name, updated_img)
                self.state.need_redraw = False
            
            # Handle key press
            wait_time = 30 if self.state.corner_dragging_mode else 50  # Optimized for keyboard-based dragging
            key = cv2.waitKey(wait_time) & 0xFF
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
                        # If we can't read the last image, exit cleanly to avoid infinite loop
                        print("Cannot read the last image file. Exiting...")
                        break
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