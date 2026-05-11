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
    
    # ------------------------------------------------------------------
    # Relocation now flows through insert mode (target_detection_id set).
    # The 'c' key calls enter_insert_mode(state, target_detection_id=…) in
    # the UI layer; the controller treats relocation as a flavor of insert.
    # ------------------------------------------------------------------

    def test_enter_insert_mode_with_target_sets_target_id(self):
        """Entering insert mode with a target marks the state as a relocation."""
        with patch('builtins.print') as mock_print:
            result = self.controller.enter_insert_mode(self.state, target_detection_id=7)

        self.assertTrue(result)
        self.assertTrue(self.state.insert_mode)
        self.assertEqual(self.state.insert_stage, 'awaiting_top_left')
        self.assertEqual(self.state.insert_target_detection_id, 7)
        # Re-entering toggles off and clears target.
        with patch('builtins.print'):
            self.controller.enter_insert_mode(self.state)
        self.assertFalse(self.state.insert_mode)
        self.assertIsNone(self.state.insert_target_detection_id)

    def test_confirm_insert_corner_relocates_when_target_set(self):
        """When insert_target_detection_id is set, the second confirm calls update_detection_coordinates instead of add_detection."""
        self.state.insert_mode = True
        self.state.insert_stage = 'awaiting_bottom_right'
        self.state.insert_top_left = (100, 200)
        self.state.insert_target_detection_id = 42
        self.state.mouse_x, self.state.mouse_y = 400, 600
        mock_img = np.zeros((800, 1000, 3), dtype=np.uint8)
        self.mock_db_manager.update_detection_coordinates.return_value = True

        with patch('builtins.print') as mock_print:
            result = self.controller.confirm_insert_corner(self.state, mock_img, image_id=7)

        self.assertTrue(result)
        # Relocates the existing row, doesn't insert a new one.
        self.mock_db_manager.update_detection_coordinates.assert_called_once_with(
            42, 0.1, 0.25, 0.3, 0.5
        )
        self.mock_db_manager.add_detection.assert_not_called()
        # Mode cleared after success.
        self.assertFalse(self.state.insert_mode)
        self.assertIsNone(self.state.insert_target_detection_id)
        mock_print.assert_any_call(
            'Relocated detection 42: x=0.100, y=0.250, w=0.300, h=0.500'
        )

    def test_confirm_insert_corner_relocate_reversed_corners(self):
        """Relocation accepts the second corner above-and-left of the locked one (min/max)."""
        self.state.insert_mode = True
        self.state.insert_stage = 'awaiting_bottom_right'
        self.state.insert_top_left = (400, 600)
        self.state.insert_target_detection_id = 9
        self.state.mouse_x, self.state.mouse_y = 100, 200
        mock_img = np.zeros((800, 1000, 3), dtype=np.uint8)
        self.mock_db_manager.update_detection_coordinates.return_value = True

        with patch('builtins.print'):
            result = self.controller.confirm_insert_corner(self.state, mock_img, image_id=7)

        self.assertTrue(result)
        self.mock_db_manager.update_detection_coordinates.assert_called_once_with(
            9, 0.1, 0.25, 0.3, 0.5
        )

    def test_confirm_insert_corner_relocate_db_failure(self):
        """If update_detection_coordinates fails, mode is reset and result is False."""
        self.state.insert_mode = True
        self.state.insert_stage = 'awaiting_bottom_right'
        self.state.insert_top_left = (0, 0)
        self.state.insert_target_detection_id = 1
        self.state.mouse_x, self.state.mouse_y = 100, 100
        mock_img = np.zeros((800, 1000, 3), dtype=np.uint8)
        self.mock_db_manager.update_detection_coordinates.return_value = False

        with patch('builtins.print') as mock_print:
            result = self.controller.confirm_insert_corner(self.state, mock_img, image_id=7)

        self.assertFalse(result)
        self.assertFalse(self.state.insert_mode)
        mock_print.assert_any_call('Failed to relocate detection 1')

    def test_enter_insert_mode_with_target_cancel_prints_relocation_message(self):
        """Cancel message differs slightly when there's a relocation target."""
        with patch('builtins.print'):
            self.controller.enter_insert_mode(self.state, target_detection_id=3)
        with patch('builtins.print') as mock_print:
            self.controller.enter_insert_mode(self.state)
        mock_print.assert_any_call('Cancelled relocation')

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

    # ------------------------------------------------------------------
    # Insert mode (interactive two-step new detection)
    # ------------------------------------------------------------------

    def test_enter_insert_mode_sets_stage(self):
        """Entering insert mode sets the awaiting_top_left stage."""
        with patch('builtins.print'):
            result = self.controller.enter_insert_mode(self.state)

        self.assertTrue(result)
        self.assertTrue(self.state.insert_mode)
        self.assertEqual(self.state.insert_stage, 'awaiting_top_left')
        self.assertIsNone(self.state.insert_top_left)
        self.assertTrue(self.state.need_redraw)

    def test_enter_insert_mode_toggle_off(self):
        """Pressing 'n' while already in insert mode cancels it."""
        self.state.insert_mode = True
        self.state.insert_stage = 'awaiting_top_left'
        self.state.insert_top_left = (10, 20)

        with patch('builtins.print') as mock_print:
            result = self.controller.enter_insert_mode(self.state)

        self.assertTrue(result)
        self.assertFalse(self.state.insert_mode)
        self.assertIsNone(self.state.insert_stage)
        self.assertIsNone(self.state.insert_top_left)
        mock_print.assert_any_call("Cancelled insert mode")

    def test_enter_insert_mode_refused_during_corner_drag(self):
        """Insert mode refuses to start when corner dragging is active."""
        self.state.corner_dragging_mode = True

        with patch('builtins.print'):
            result = self.controller.enter_insert_mode(self.state)

        self.assertFalse(result)
        self.assertFalse(self.state.insert_mode)

    def test_confirm_insert_corner_first_locks_top_left(self):
        """First confirm locks top-left and advances to stage 2."""
        self.state.insert_mode = True
        self.state.insert_stage = 'awaiting_top_left'
        self.state.mouse_x = 150
        self.state.mouse_y = 240
        mock_img = np.zeros((800, 1000, 3), dtype=np.uint8)

        with patch('builtins.print'):
            result = self.controller.confirm_insert_corner(self.state, mock_img, image_id=42)

        self.assertTrue(result)
        self.assertEqual(self.state.insert_stage, 'awaiting_bottom_right')
        self.assertEqual(self.state.insert_top_left, (150, 240))
        # No DB insert yet
        self.mock_db_manager.add_detection.assert_not_called()

    def test_confirm_insert_corner_full_flow_inserts_with_majority_category(self):
        """Two SPACE/click presses insert a detection with normalized coords and majority category."""
        # Existing detections: two cat=1, one cat=4 → majority is 1.
        det_a = Detection(id=10, category=1, confidence=0.9, x=0.1, y=0.1, width=0.1, height=0.1)
        det_b = Detection(id=11, category=1, confidence=0.9, x=0.2, y=0.2, width=0.1, height=0.1)
        det_c = Detection(id=12, category=4, confidence=0.9, x=0.3, y=0.3, width=0.1, height=0.1)
        self.state.current_boxes = [
            BoundingBox(0, 0, 1, 1, 10, '1', 0.9, False, det_a),
            BoundingBox(0, 0, 1, 1, 11, '1', 0.9, False, det_b),
            BoundingBox(0, 0, 1, 1, 12, '4', 0.9, False, det_c),
        ]
        self.state.insert_mode = True
        self.state.insert_stage = 'awaiting_top_left'
        mock_img = np.zeros((800, 1000, 3), dtype=np.uint8)
        self.mock_db_manager.add_detection.return_value = 99

        # First confirm at (100, 200) → top-left locked.
        self.state.mouse_x, self.state.mouse_y = 100, 200
        with patch('builtins.print'):
            self.controller.confirm_insert_corner(self.state, mock_img, image_id=7)

        # Second confirm at (400, 600) → DB insert with majority category 1.
        self.state.mouse_x, self.state.mouse_y = 400, 600
        with patch('builtins.print'):
            result = self.controller.confirm_insert_corner(self.state, mock_img, image_id=7)

        self.assertTrue(result)
        # 1000x800 → (100,200)→(400,600) is normalized x=0.1, y=0.25, w=0.3, h=0.5.
        self.mock_db_manager.add_detection.assert_called_once_with(
            image_id=7, category=1, confidence=1.0,
            x=0.1, y=0.25, width=0.3, height=0.5,
        )
        # Mode cleared on success.
        self.assertFalse(self.state.insert_mode)
        self.assertIsNone(self.state.insert_stage)
        self.assertIsNone(self.state.insert_top_left)

    def test_confirm_insert_corner_falls_back_to_other_with_no_existing(self):
        """With no non-deleted existing detections, category falls back to 5 ('Other')."""
        self.state.current_boxes = []
        self.state.insert_mode = True
        self.state.insert_stage = 'awaiting_bottom_right'
        self.state.insert_top_left = (0, 0)
        self.state.mouse_x, self.state.mouse_y = 500, 400
        mock_img = np.zeros((800, 1000, 3), dtype=np.uint8)
        self.mock_db_manager.add_detection.return_value = 1

        with patch('builtins.print'):
            self.controller.confirm_insert_corner(self.state, mock_img, image_id=7)

        kwargs = self.mock_db_manager.add_detection.call_args.kwargs
        self.assertEqual(kwargs['category'], 5)

    def test_confirm_insert_corner_reversed_drag(self):
        """A bottom-right confirm above-and-left of the locked top-left is still accepted (min/max)."""
        self.state.insert_mode = True
        self.state.insert_stage = 'awaiting_bottom_right'
        self.state.insert_top_left = (400, 600)
        self.state.mouse_x, self.state.mouse_y = 100, 200
        mock_img = np.zeros((800, 1000, 3), dtype=np.uint8)
        self.mock_db_manager.add_detection.return_value = 1

        with patch('builtins.print'):
            result = self.controller.confirm_insert_corner(self.state, mock_img, image_id=7)

        self.assertTrue(result)
        self.mock_db_manager.add_detection.assert_called_once_with(
            image_id=7, category=5, confidence=1.0,
            x=0.1, y=0.25, width=0.3, height=0.5,
        )

    def test_confirm_insert_corner_zero_size_keeps_mode_active(self):
        """A second corner at the same pixel as the locked top-left is rejected and mode stays active."""
        self.state.insert_mode = True
        self.state.insert_stage = 'awaiting_bottom_right'
        self.state.insert_top_left = (100, 200)
        self.state.mouse_x, self.state.mouse_y = 100, 200
        mock_img = np.zeros((800, 1000, 3), dtype=np.uint8)

        with patch('builtins.print') as mock_print:
            result = self.controller.confirm_insert_corner(self.state, mock_img, image_id=7)

        self.assertFalse(result)
        self.mock_db_manager.add_detection.assert_not_called()
        # Still in stage 2 — user can keep moving.
        self.assertTrue(self.state.insert_mode)
        self.assertEqual(self.state.insert_stage, 'awaiting_bottom_right')
        self.assertEqual(self.state.insert_top_left, (100, 200))
        mock_print.assert_any_call("Zero-sized box — keep moving and confirm again.")

    def test_confirm_insert_corner_db_failure_resets_mode(self):
        """If add_detection returns None, mode is reset and result is False."""
        self.state.insert_mode = True
        self.state.insert_stage = 'awaiting_bottom_right'
        self.state.insert_top_left = (0, 0)
        self.state.mouse_x, self.state.mouse_y = 100, 100
        mock_img = np.zeros((800, 1000, 3), dtype=np.uint8)
        self.mock_db_manager.add_detection.return_value = None

        with patch('builtins.print') as mock_print:
            result = self.controller.confirm_insert_corner(self.state, mock_img, image_id=7)

        self.assertFalse(result)
        self.assertFalse(self.state.insert_mode)
        mock_print.assert_any_call("Failed to add new detection")

    def test_confirm_insert_corner_no_op_when_not_in_mode(self):
        """Confirm is a no-op outside insert mode."""
        self.state.insert_mode = False
        result = self.controller.confirm_insert_corner(
            self.state, np.zeros((100, 100, 3), dtype=np.uint8), image_id=1
        )
        self.assertFalse(result)
        self.mock_db_manager.add_detection.assert_not_called()

    def test_reset_insert_mode_clears_all_fields(self):
        """state.reset_insert_mode() clears all insert-mode fields."""
        self.state.insert_mode = True
        self.state.insert_stage = 'awaiting_bottom_right'
        self.state.insert_top_left = (50, 60)
        self.state.reset_insert_mode()
        self.assertFalse(self.state.insert_mode)
        self.assertIsNone(self.state.insert_stage)
        self.assertIsNone(self.state.insert_top_left)

    def test_start_corner_dragging_refused_during_insert_mode(self):
        """start_corner_dragging refuses while insert mode is active."""
        self.state.insert_mode = True
        self.state.current_boxes = [self.test_box1]
        # Pretend the click is on the top-left corner of box1.
        info = (self.test_box1.x_min, self.test_box1.y_min, self.test_box1.detection_id)

        with patch('builtins.print') as mock_print:
            result = self.controller.start_corner_dragging(self.state, info)

        self.assertFalse(result)
        self.assertFalse(self.state.corner_dragging_mode)
        mock_print.assert_called_with("Finish insert mode before dragging a corner.")


if __name__ == '__main__':
    unittest.main()
