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
    DetectionController, DatabaseManager, DetectionRenderer, AppState, 
    Detection, BoundingBox, CATEGORY_KEYS
)

class TestDetectionController(unittest.TestCase):
    """Test cases for DetectionController class"""
    
    def setUp(self):
        """Set up test environment for each test"""
        # Create mock database manager
        self.mock_db_manager = MagicMock(spec=DatabaseManager)
        
        # Create mock renderer
        self.mock_renderer = MagicMock(spec=DetectionRenderer)
        
        # Create controller
        self.controller = DetectionController(self.mock_db_manager, self.mock_renderer)
        
        # Create test state
        self.state = AppState()
        
        # Create test detections
        self.test_detection1 = Detection(
            id=1, category=1, confidence=0.95, x=0.1, y=0.2, width=0.3, height=0.4
        )
        self.test_detection2 = Detection(
            id=2, category=2, confidence=0.85, x=0.5, y=0.6, width=0.2, height=0.3
        )
        
        # Create test bounding boxes
        self.test_box1 = BoundingBox(
            x_min=100, y_min=200, x_max=400, y_max=600,
            detection_id=1, category='1', confidence=0.95, is_deleted=False,
            detection=self.test_detection1
        )
        self.test_box2 = BoundingBox(
            x_min=500, y_min=600, x_max=700, y_max=900,
            detection_id=2, category='2', confidence=0.85, is_deleted=False,
            detection=self.test_detection2
        )
    
    def test_handle_category_change_success(self):
        """Test successful category change"""
        # Setup
        self.state.selected_detection_id = 1
        self.mock_db_manager.update_detection_category.return_value = True
        category_info = CATEGORY_KEYS[112]  # Possum
        
        # Execute
        result = self.controller.handle_category_change(1, category_info)
        
        # Verify
        self.assertTrue(result)
        self.mock_db_manager.update_detection_category.assert_called_once_with(1, 4)
    
    def test_handle_category_change_no_selection(self):
        """Test category change with no detection selected"""
        # Setup
        category_info = CATEGORY_KEYS[112]  # Possum
        
        # Execute
        with patch('builtins.print') as mock_print:
            result = self.controller.handle_category_change(None, category_info)
        
        # Verify
        self.assertFalse(result)
        mock_print.assert_called_with("No detection selected. Hover over a detection to select it.")
        self.mock_db_manager.update_detection_category.assert_not_called()
    
    def test_handle_category_change_database_failure(self):
        """Test category change when database update fails"""
        # Setup
        self.mock_db_manager.update_detection_category.return_value = False
        category_info = CATEGORY_KEYS[112]  # Possum
        
        # Execute
        with patch('builtins.print') as mock_print:
            result = self.controller.handle_category_change(1, category_info)
        
        # Verify
        self.assertFalse(result)
        mock_print.assert_called_with("Failed to change detection 1 category")
    
    def test_cycle_overlapping_detections_success(self):
        """Test successful cycling through overlapping detections"""
        # Setup
        self.state.overlapping_detections = [self.test_box1, self.test_box2]
        self.state.current_overlap_index = 0
        self.state.selected_detection_id = 1
        
        # Execute
        with patch('builtins.print') as mock_print:
            result = self.controller.cycle_overlapping_detections(self.state)
        
        # Verify
        self.assertTrue(result)
        self.assertEqual(self.state.current_overlap_index, 1)
        self.assertEqual(self.state.selected_detection_id, 2)
        self.assertTrue(self.state.need_redraw)
        mock_print.assert_called_with("Cycled to detection 2 (2/2)")
    
    def test_cycle_overlapping_detections_wraparound(self):
        """Test cycling wraps around to first detection"""
        # Setup
        self.state.overlapping_detections = [self.test_box1, self.test_box2]
        self.state.current_overlap_index = 1  # Start at last detection
        self.state.selected_detection_id = 2
        
        # Execute
        with patch('builtins.print') as mock_print:
            result = self.controller.cycle_overlapping_detections(self.state)
        
        # Verify
        self.assertTrue(result)
        self.assertEqual(self.state.current_overlap_index, 0)  # Wrapped around
        self.assertEqual(self.state.selected_detection_id, 1)
        mock_print.assert_called_with("Cycled to detection 1 (1/2)")
    
    def test_cycle_overlapping_detections_no_overlaps(self):
        """Test cycling when no overlapping detections"""
        # Setup
        self.state.overlapping_detections = []
        
        # Execute
        with patch('builtins.print') as mock_print:
            result = self.controller.cycle_overlapping_detections(self.state)
        
        # Verify
        self.assertFalse(result)
        mock_print.assert_called_with("No overlapping detections to cycle through")
    
    def test_cycle_overlapping_detections_single_detection(self):
        """Test cycling with only one detection"""
        # Setup
        self.state.overlapping_detections = [self.test_box1]
        
        # Execute
        with patch('builtins.print') as mock_print:
            result = self.controller.cycle_overlapping_detections(self.state)
        
        # Verify
        self.assertFalse(result)
        mock_print.assert_called_with("No overlapping detections to cycle through")
    
    def test_handle_relocation_mode_enter(self):
        """Test entering relocation mode"""
        # Setup
        self.state.selected_detection_id = 1
        self.state.relocation_mode = False
        
        # Execute
        with patch('builtins.print') as mock_print:
            result = self.controller.handle_relocation_mode(self.state)
        
        # Verify
        self.assertTrue(result)
        self.assertTrue(self.state.relocation_mode)
        self.assertEqual(self.state.relocation_detection_id, 1)
        self.assertEqual(self.state.relocation_clicks, [])
        self.assertTrue(self.state.need_redraw)
        
        # Check print calls
        expected_calls = [
            unittest.mock.call("Entering relocation mode for detection 1"),
            unittest.mock.call("Click twice to define new bounding box: first click = top-left, second click = bottom-right"),
            unittest.mock.call("Press 'c' again to cancel relocation")
        ]
        mock_print.assert_has_calls(expected_calls)
    
    def test_handle_relocation_mode_enter_no_selection(self):
        """Test entering relocation mode with no detection selected"""
        # Setup
        self.state.selected_detection_id = None
        self.state.relocation_mode = False
        
        # Execute
        with patch('builtins.print') as mock_print:
            result = self.controller.handle_relocation_mode(self.state)
        
        # Verify
        self.assertFalse(result)
        self.assertFalse(self.state.relocation_mode)
        mock_print.assert_called_with("No detection selected. Hover over a detection to select it first.")
    
    def test_handle_relocation_mode_cancel(self):
        """Test canceling relocation mode"""
        # Setup
        self.state.relocation_mode = True
        self.state.relocation_detection_id = 1
        self.state.relocation_clicks = [(100, 200)]
        
        # Execute
        with patch('builtins.print') as mock_print:
            result = self.controller.handle_relocation_mode(self.state)
        
        # Verify
        self.assertTrue(result)
        self.assertFalse(self.state.relocation_mode)
        self.assertIsNone(self.state.relocation_detection_id)
        self.assertEqual(self.state.relocation_clicks, [])
        self.assertTrue(self.state.need_redraw)
        mock_print.assert_called_with("Cancelled relocation mode")
    
    def test_apply_relocation_success(self):
        """Test successful relocation application"""
        # Setup
        self.state.relocation_detection_id = 1
        self.state.relocation_clicks = [(100, 200), (400, 600)]
        self.mock_db_manager.update_detection_coordinates.return_value = True
        
        # Create mock image (1000x800)
        mock_img = np.zeros((800, 1000, 3), dtype=np.uint8)
        
        # Execute
        with patch('builtins.print') as mock_print:
            result = self.controller.apply_relocation(self.state, mock_img)
        
        # Verify
        self.assertTrue(result)
        self.assertFalse(self.state.relocation_mode)
        self.assertIsNone(self.state.relocation_detection_id)
        self.assertEqual(self.state.relocation_clicks, [])
        self.assertTrue(self.state.need_redraw)
        
        # Check database call with normalized coordinates
        # (100, 200) to (400, 600) on 1000x800 image
        # normalized: x=0.1, y=0.25, width=0.3, height=0.5
        self.mock_db_manager.update_detection_coordinates.assert_called_once_with(
            1, 0.1, 0.25, 0.3, 0.5
        )
        
        mock_print.assert_any_call("Successfully relocated detection 1")
    
    def test_apply_relocation_insufficient_clicks(self):
        """Test relocation with insufficient clicks"""
        # Setup
        self.state.relocation_clicks = [(100, 200)]  # Only one click
        
        # Execute
        with patch('builtins.print') as mock_print:
            result = self.controller.apply_relocation(self.state, None)
        
        # Verify
        self.assertFalse(result)
        mock_print.assert_called_with("Error: Need exactly 2 clicks to apply relocation")
        self.mock_db_manager.update_detection_coordinates.assert_not_called()
    
    def test_apply_relocation_database_failure(self):
        """Test relocation when database update fails"""
        # Setup
        self.state.relocation_detection_id = 1
        self.state.relocation_clicks = [(100, 200), (400, 600)]
        self.mock_db_manager.update_detection_coordinates.return_value = False
        
        # Create mock image
        mock_img = np.zeros((800, 1000, 3), dtype=np.uint8)
        
        # Execute
        with patch('builtins.print') as mock_print:
            result = self.controller.apply_relocation(self.state, mock_img)
        
        # Verify
        self.assertFalse(result)
        mock_print.assert_any_call("Failed to relocate detection 1")
        # State should still be reset even on failure
        self.assertFalse(self.state.relocation_mode)
    
    def test_apply_relocation_coordinate_clamping(self):
        """Test that coordinates are clamped to valid range [0,1]"""
        # Setup - clicks that would result in negative or >1 coordinates
        self.state.relocation_detection_id = 1
        self.state.relocation_clicks = [(0, 0), (1200, 900)]  # Beyond image bounds
        self.mock_db_manager.update_detection_coordinates.return_value = True
        
        # Create mock image (1000x800)
        mock_img = np.zeros((800, 1000, 3), dtype=np.uint8)
        
        # Execute
        result = self.controller.apply_relocation(self.state, mock_img)
        
        # Verify coordinates are clamped
        # x should be clamped to max 1.0, width should be adjusted
        call_args = self.mock_db_manager.update_detection_coordinates.call_args[0]
        x, y, width, height = call_args[1:5]
        
        self.assertGreaterEqual(x, 0)
        self.assertLessEqual(x, 1)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(y, 1)
        self.assertGreaterEqual(width, 0)
        self.assertLessEqual(width, 1 - x)
        self.assertGreaterEqual(height, 0)
        self.assertLessEqual(height, 1 - y)
    
    def test_apply_relocation_reversed_clicks(self):
        """Test relocation with clicks in reverse order (bottom-right first)"""
        # Setup
        self.state.relocation_detection_id = 1
        self.state.relocation_clicks = [(400, 600), (100, 200)]  # Reversed order
        self.mock_db_manager.update_detection_coordinates.return_value = True
        
        # Create mock image (1000x800)
        mock_img = np.zeros((800, 1000, 3), dtype=np.uint8)
        
        # Execute
        result = self.controller.apply_relocation(self.state, mock_img)
        
        # Verify
        self.assertTrue(result)
        
        # Should still result in correct normalized coordinates
        # min(100,400)=100, min(200,600)=200, max(400,100)=400, max(600,200)=600
        # normalized: x=0.1, y=0.25, width=0.3, height=0.5
        self.mock_db_manager.update_detection_coordinates.assert_called_once_with(
            1, 0.1, 0.25, 0.3, 0.5
        )

    def test_mark_all_detections_deleted_success(self):
        """Test successfully marking all detections as deleted"""
        # Setup
        self.mock_db_manager.mark_all_detections_deleted.return_value = 3
        
        # Execute
        with patch('builtins.print') as mock_print:
            result = self.controller.mark_all_detections_deleted(1)
        
        # Verify
        self.assertTrue(result)
        self.mock_db_manager.mark_all_detections_deleted.assert_called_once_with(1)
        mock_print.assert_called_with("Marked 3 detection(s) as deleted on current image")
    
    def test_mark_all_detections_deleted_no_detections(self):
        """Test marking all detections as deleted when no detections exist"""
        # Setup
        self.mock_db_manager.mark_all_detections_deleted.return_value = 0
        
        # Execute
        with patch('builtins.print') as mock_print:
            result = self.controller.mark_all_detections_deleted(1)
        
        # Verify
        self.assertFalse(result)
        self.mock_db_manager.mark_all_detections_deleted.assert_called_once_with(1)
        mock_print.assert_called_with("No detections found to mark as deleted")

if __name__ == '__main__':
    unittest.main() 