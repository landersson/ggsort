#!/usr/bin/env python3

import unittest
import tempfile
import os
import sys
import numpy as np
from unittest.mock import MagicMock, patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ggsort import (
    DetectionRenderer, AppState, Detection, BoundingBox, COLORS, DELETED_COLOR
)

class TestDetectionRenderer(unittest.TestCase):
    """Test cases for DetectionRenderer class"""
    
    def setUp(self):
        """Set up test environment for each test"""
        self.categories = {
            '1': 'GangGang',
            '2': 'Person',
            '3': 'Vehicle',
            '4': 'Possum',
            '5': 'Other'
        }
        self.renderer = DetectionRenderer(self.categories)
        self.state = AppState()
        
        # Create test detections
        self.test_detection1 = Detection(
            id=1, category=1, confidence=0.95, x=0.1, y=0.2, width=0.3, height=0.4
        )
        self.test_detection2 = Detection(
            id=2, category=2, confidence=0.85, x=0.5, y=0.6, width=0.2, height=0.3, deleted=True
        )
        self.test_detection3 = Detection(
            id=3, category=4, confidence=0.75, x=0.7, y=0.1, width=0.15, height=0.25
        )
    
    def test_convert_to_pixel_coordinates(self):
        """Test conversion from normalized to pixel coordinates"""
        img_width, img_height = 1920, 1080
        
        # Test normal detection
        box = self.renderer.convert_to_pixel_coordinates(self.test_detection1, img_width, img_height)
        
        # Expected pixel coordinates
        expected_x_min = int(0.1 * 1920)  # 192
        expected_y_min = int(0.2 * 1080)  # 216
        expected_x_max = expected_x_min + int(0.3 * 1920)  # 192 + 576 = 768
        expected_y_max = expected_y_min + int(0.4 * 1080)  # 216 + 432 = 648
        
        self.assertEqual(box.x_min, expected_x_min)
        self.assertEqual(box.y_min, expected_y_min)
        self.assertEqual(box.x_max, expected_x_max)
        self.assertEqual(box.y_max, expected_y_max)
        self.assertEqual(box.detection_id, 1)
        self.assertEqual(box.category, '1')
        self.assertEqual(box.confidence, 0.95)
        self.assertFalse(box.is_deleted)
        self.assertEqual(box.detection, self.test_detection1)
    
    def test_convert_to_pixel_coordinates_deleted_detection(self):
        """Test conversion for deleted detection"""
        img_width, img_height = 1280, 720
        
        box = self.renderer.convert_to_pixel_coordinates(self.test_detection2, img_width, img_height)
        
        # Expected pixel coordinates for deleted detection
        expected_x_min = int(0.5 * 1280)  # 640
        expected_y_min = int(0.6 * 720)   # 432
        expected_x_max = expected_x_min + int(0.2 * 1280)  # 640 + 256 = 896
        expected_y_max = expected_y_min + int(0.3 * 720)   # 432 + 216 = 648
        
        self.assertEqual(box.x_min, expected_x_min)
        self.assertEqual(box.y_min, expected_y_min)
        self.assertEqual(box.x_max, expected_x_max)
        self.assertEqual(box.y_max, expected_y_max)
        self.assertTrue(box.is_deleted)
    
    def test_find_overlapping_detections(self):
        """Test finding detections that overlap with mouse position"""
        # Create bounding boxes
        box1 = BoundingBox(
            x_min=100, y_min=200, x_max=400, y_max=600,
            detection_id=1, category='1', confidence=0.95, is_deleted=False,
            detection=self.test_detection1
        )
        box2 = BoundingBox(
            x_min=300, y_min=400, x_max=600, y_max=700,
            detection_id=2, category='2', confidence=0.85, is_deleted=False,
            detection=self.test_detection2
        )
        box3 = BoundingBox(
            x_min=700, y_min=100, x_max=900, y_max=300,
            detection_id=3, category='4', confidence=0.75, is_deleted=False,
            detection=self.test_detection3
        )
        
        boxes = [box1, box2, box3]
        
        # Test mouse position that overlaps with box1 only
        overlapping = self.renderer.find_overlapping_detections(boxes, 250, 350)
        self.assertEqual(len(overlapping), 1)
        self.assertEqual(overlapping[0].detection_id, 1)
        
        # Test mouse position that overlaps with both box1 and box2
        overlapping = self.renderer.find_overlapping_detections(boxes, 350, 450)
        self.assertEqual(len(overlapping), 2)
        detection_ids = [box.detection_id for box in overlapping]
        self.assertIn(1, detection_ids)
        self.assertIn(2, detection_ids)
        
        # Test mouse position that doesn't overlap with any box
        overlapping = self.renderer.find_overlapping_detections(boxes, 50, 50)
        self.assertEqual(len(overlapping), 0)
        
        # Test mouse position that overlaps with box3 only
        overlapping = self.renderer.find_overlapping_detections(boxes, 800, 200)
        self.assertEqual(len(overlapping), 1)
        self.assertEqual(overlapping[0].detection_id, 3)
    
    def test_find_overlapping_detections_edge_cases(self):
        """Test edge cases for overlapping detection finding"""
        box = BoundingBox(
            x_min=100, y_min=200, x_max=400, y_max=600,
            detection_id=1, category='1', confidence=0.95, is_deleted=False,
            detection=self.test_detection1
        )
        
        # Test exact boundary positions
        overlapping = self.renderer.find_overlapping_detections([box], 100, 200)  # Top-left corner
        self.assertEqual(len(overlapping), 1)
        
        overlapping = self.renderer.find_overlapping_detections([box], 400, 600)  # Bottom-right corner
        self.assertEqual(len(overlapping), 1)
        
        # Test just outside boundaries
        overlapping = self.renderer.find_overlapping_detections([box], 99, 200)   # Just left
        self.assertEqual(len(overlapping), 0)
        
        overlapping = self.renderer.find_overlapping_detections([box], 401, 600)  # Just right
        self.assertEqual(len(overlapping), 0)
    
    @patch('cv2.rectangle')
    @patch('cv2.putText')
    @patch('cv2.getTextSize')
    def test_draw_boxes_on_image(self, mock_getTextSize, mock_putText, mock_rectangle):
        """Test drawing boxes on image"""
        # Setup mocks
        mock_getTextSize.return_value = ((100, 20), 5)  # (text_size, baseline)
        
        # Create test image
        original_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        detections = [self.test_detection1, self.test_detection2, self.test_detection3]
        
        # Test drawing
        result_img = self.renderer.draw_boxes_on_image(original_img, detections, self.state)
        
        # Verify image was copied (not modified in place)
        self.assertIsNot(result_img, original_img)
        
        # Verify current_boxes was populated
        self.assertEqual(len(self.state.current_boxes), 3)
        
        # Verify cv2.rectangle was called for each detection
        # The exact number may vary based on rendering implementation
        self.assertGreater(mock_rectangle.call_count, 0)
        self.assertLessEqual(mock_rectangle.call_count, 15)  # Reasonable upper bound
    
    @patch('cv2.rectangle')
    @patch('cv2.putText')
    @patch('cv2.getTextSize')
    def test_draw_boxes_with_selection(self, mock_getTextSize, mock_putText, mock_rectangle):
        """Test drawing boxes with a selected detection"""
        mock_getTextSize.return_value = ((100, 20), 5)
        
        # Setup state with mouse position over detection 1
        self.state.update_mouse_position(300, 300)  # Should be over detection 1
        
        original_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        detections = [self.test_detection1, self.test_detection2]
        
        # Draw boxes (this should populate overlapping detections and select one)
        result_img = self.renderer.draw_boxes_on_image(original_img, detections, self.state)
        
        # Verify a detection was selected
        self.assertIsNotNone(self.state.selected_detection_id)
        
        # Verify overlapping detections were found
        self.assertGreater(len(self.state.overlapping_detections), 0)
    
    @patch('cv2.rectangle')
    @patch('cv2.putText')
    @patch('cv2.getTextSize')
    def test_draw_boxes_with_relocation_mode(self, mock_getTextSize, mock_putText, mock_rectangle):
        """Test drawing boxes in relocation mode"""
        mock_getTextSize.return_value = ((100, 20), 5)
        
        # Setup relocation mode
        self.state.relocation_mode = True
        self.state.relocation_detection_id = 1
        self.state.relocation_clicks = [(100, 200), (400, 600)]
        
        original_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        detections = [self.test_detection1]
        
        # Mock cv2.circle for relocation UI
        with patch('cv2.circle') as mock_circle:
            result_img = self.renderer.draw_boxes_on_image(original_img, detections, self.state)
        
        # Verify relocation UI was drawn (circles for clicks)
        self.assertEqual(mock_circle.call_count, 2)  # Two clicks = two circles
        
        # Verify additional rectangle for new bounding box preview
        # Should have: outline + colored rectangle for detection + preview rectangle
        self.assertGreaterEqual(mock_rectangle.call_count, 3)
    
    @patch('cv2.rectangle')
    @patch('cv2.putText')
    @patch('cv2.getTextSize')
    def test_draw_single_box_colors(self, mock_getTextSize, mock_putText, mock_rectangle):
        """Test that correct colors are used for different detection states"""
        mock_getTextSize.return_value = ((100, 20), 5)
        
        original_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Test normal detection color
        detections = [self.test_detection1]
        self.renderer.draw_boxes_on_image(original_img, detections, self.state)
        
        # Check that the correct color was used (category 1 = GangGang = red)
        expected_color = COLORS['1']  # (0, 0, 255) - Red
        
        # Find the colored rectangle call (second rectangle call for each detection)
        colored_rect_calls = [call for call in mock_rectangle.call_args_list if call[0][3] == expected_color]
        self.assertGreater(len(colored_rect_calls), 0)
        
        # Reset mock
        mock_rectangle.reset_mock()
        
        # Test deleted detection color
        detections = [self.test_detection2]  # This one is deleted
        self.renderer.draw_boxes_on_image(original_img, detections, self.state)
        
        # Check that deleted color was used
        deleted_rect_calls = [call for call in mock_rectangle.call_args_list if call[0][3] == DELETED_COLOR]
        self.assertGreater(len(deleted_rect_calls), 0)
    
    @patch('cv2.putText')
    @patch('cv2.getTextSize')
    @patch('cv2.rectangle')
    def test_draw_label_content(self, mock_rectangle, mock_getTextSize, mock_putText):
        """Test that labels contain correct information"""
        mock_getTextSize.return_value = ((100, 20), 5)
        
        original_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Test normal detection label
        detections = [self.test_detection1]
        self.renderer.draw_boxes_on_image(original_img, detections, self.state)
        
        # Check that putText was called with correct label
        label_calls = mock_putText.call_args_list
        self.assertGreater(len(label_calls), 0)
        
        # Find the label text (first argument of putText)
        label_text = label_calls[0][0][1]  # Second argument of first call
        self.assertIn('GangGang', label_text)
        self.assertIn('0.95', label_text)
        
        # Reset mock
        mock_putText.reset_mock()
        
        # Test deleted detection label
        detections = [self.test_detection2]
        self.renderer.draw_boxes_on_image(original_img, detections, self.state)
        
        label_calls = mock_putText.call_args_list
        label_text = label_calls[0][0][1]
        self.assertIn('DELETED', label_text)
    
    def test_overlapping_detection_state_management(self):
        """Test that overlapping detection state is properly managed"""
        original_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        detections = [self.test_detection1, self.test_detection2, self.test_detection3]
        
        # Position mouse where no detections overlap
        self.state.update_mouse_position(50, 50)
        self.renderer.draw_boxes_on_image(original_img, detections, self.state)
        
        # Should have no overlapping detections
        self.assertEqual(len(self.state.overlapping_detections), 0)
        self.assertIsNone(self.state.selected_detection_id)
        
        # Position mouse over detection area
        self.state.update_mouse_position(300, 300)
        self.renderer.draw_boxes_on_image(original_img, detections, self.state)
        
        # Should have overlapping detections and selection
        if len(self.state.overlapping_detections) > 0:
            self.assertIsNotNone(self.state.selected_detection_id)
            self.assertEqual(self.state.current_overlap_index, 0)
    
    def test_relocation_mode_state_isolation(self):
        """Test that relocation mode doesn't interfere with normal detection selection"""
        original_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        detections = [self.test_detection1, self.test_detection2]
        
        # First, establish normal selection
        self.state.update_mouse_position(300, 300)
        self.renderer.draw_boxes_on_image(original_img, detections, self.state)
        normal_selection = self.state.selected_detection_id
        
        # Enter relocation mode
        self.state.relocation_mode = True
        self.state.relocation_detection_id = 1
        
        # Move mouse to different position
        self.state.update_mouse_position(800, 200)
        self.renderer.draw_boxes_on_image(original_img, detections, self.state)
        
        # In relocation mode, overlapping detection logic should be skipped
        # So selection should remain the same as before entering relocation mode
        self.assertEqual(self.state.selected_detection_id, normal_selection)
    
    def test_coordinate_boundary_handling(self):
        """Test handling of detections at image boundaries"""
        # Create detection at image boundary
        boundary_detection = Detection(
            id=99, category=1, confidence=0.9, x=0.0, y=0.0, width=1.0, height=1.0
        )
        
        img_width, img_height = 1920, 1080
        box = self.renderer.convert_to_pixel_coordinates(boundary_detection, img_width, img_height)
        
        # Should handle full-image detection correctly
        self.assertEqual(box.x_min, 0)
        self.assertEqual(box.y_min, 0)
        self.assertEqual(box.x_max, img_width)
        self.assertEqual(box.y_max, img_height)
        
        # Test very small detection
        tiny_detection = Detection(
            id=100, category=1, confidence=0.9, x=0.5, y=0.5, width=0.001, height=0.001
        )
        
        tiny_box = self.renderer.convert_to_pixel_coordinates(tiny_detection, img_width, img_height)
        
        # Should handle tiny detection without errors
        self.assertGreaterEqual(tiny_box.x_max, tiny_box.x_min)
        self.assertGreaterEqual(tiny_box.y_max, tiny_box.y_min)


if __name__ == '__main__':
    unittest.main() 