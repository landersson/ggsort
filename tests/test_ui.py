#!/usr/bin/env python3

import unittest
import tempfile
import os
import sys
import numpy as np
from unittest.mock import MagicMock, patch, call

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ggsort import (
    UserInterface, DetectionController, DetectionRenderer, AppState, 
    DatabaseManager, Detection, CATEGORY_KEYS, CATEGORY_KEYS_ALL
)

class TestUserInterface(unittest.TestCase):
    """Test cases for UserInterface class"""
    
    def setUp(self):
        """Set up test environment for each test"""
        # Create mock components
        self.mock_controller = MagicMock(spec=DetectionController)
        self.mock_renderer = MagicMock(spec=DetectionRenderer)
        self.state = AppState()
        
        # Setup mock database manager on controller
        self.mock_db_manager = MagicMock(spec=DatabaseManager)
        self.mock_controller.db_manager = self.mock_db_manager
        
        # Create UI instance
        self.ui = UserInterface(self.mock_controller, self.mock_renderer, self.state)
        
        # Create test image
        self.test_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    def test_mouse_callback_mousemove(self):
        """Test mouse move event handling"""
        # Test mouse move event (cv2.EVENT_MOUSEMOVE = 0)
        self.ui.mouse_callback(0, 100, 200, 0, None)
        
        # Verify state was updated
        self.assertEqual(self.state.mouse_x, 100)
        self.assertEqual(self.state.mouse_y, 200)
        self.assertTrue(self.state.need_redraw)
    
    def test_mouse_callback_left_click_normal_mode(self):
        """Test left click in normal mode (should do nothing)"""
        # Test left click when not in relocation mode (cv2.EVENT_LBUTTONDOWN = 1)
        self.state.relocation_mode = False
        
        self.ui.mouse_callback(1, 100, 200, 0, None)
        
        # Should not add any clicks
        self.assertEqual(len(self.state.relocation_clicks), 0)
    
    def test_mouse_callback_left_click_relocation_mode(self):
        """Test left click in relocation mode"""
        # Setup relocation mode
        self.state.relocation_mode = True
        self.state.relocation_detection_id = 1
        self.ui.current_original_img = self.test_image  # Need image for relocation
        
        # Configure mock to actually reset state when apply_relocation is called
        def mock_apply_relocation(state, img):
            state.reset_relocation_mode()
            state.need_redraw = True
            return True
        
        self.mock_controller.apply_relocation.side_effect = mock_apply_relocation
        
        with patch('builtins.print') as mock_print:
            # First click
            self.ui.mouse_callback(1, 100, 200, 0, None)
            
            # Verify click was recorded but relocation not completed yet
            self.assertEqual(len(self.state.relocation_clicks), 1)
            self.assertEqual(self.state.relocation_clicks[0], (100, 200))
            self.assertTrue(self.state.need_redraw)
            self.assertTrue(self.state.relocation_mode)  # Still in relocation mode
            mock_print.assert_called_with("Click 1: (100, 200)")
            
            # Second click - should complete relocation immediately
            self.ui.mouse_callback(1, 400, 600, 0, None)
            
            # Verify relocation was completed and state reset
            self.assertEqual(len(self.state.relocation_clicks), 0)  # Reset after completion
            self.assertFalse(self.state.relocation_mode)  # Exited relocation mode
            
            # Check print calls
            expected_calls = [
                call("Click 1: (100, 200)"),
                call("Click 2: (400, 600)"),
                call("Two clicks recorded. Applying new coordinates...")
            ]
            mock_print.assert_has_calls(expected_calls)
    
    def test_handle_key_press_exit_keys(self):
        """Test exit key handling"""
        # Test ESC key (27)
        result = self.ui.handle_key_press(27, self.test_image)
        self.assertEqual(result, 0)
        
        # Test 'q' key (113)
        result = self.ui.handle_key_press(113, self.test_image)
        self.assertEqual(result, 0)
    
    def test_handle_key_press_navigation_keys(self):
        """Test navigation key handling"""
        # Test SPACE key (32)
        result = self.ui.handle_key_press(32, self.test_image)
        self.assertEqual(result, 1)
        
        # Test RIGHT ARROW key (3)
        result = self.ui.handle_key_press(3, self.test_image)
        self.assertEqual(result, 1)
        
        # Test LEFT ARROW key (2)
        result = self.ui.handle_key_press(2, self.test_image)
        self.assertEqual(result, -1)
    
    def test_handle_key_press_delete_with_selection(self):
        """Test delete key with detection selected"""
        # Setup: Select a detection
        self.state.selected_detection_id = 1
        self.mock_db_manager.toggle_detection_deleted_status.return_value = True
        
        with patch('builtins.print') as mock_print:
            # Test BACKSPACE key (8)
            result = self.ui.handle_key_press(8, self.test_image)
        
        # Verify behavior
        self.assertIsNone(result)  # Should not navigate
        self.assertTrue(self.state.need_redraw)
        self.mock_db_manager.toggle_detection_deleted_status.assert_called_once_with(1)
        mock_print.assert_called_with("Toggled deleted status for detection 1")
        
        # Test 'x' key (120)
        self.mock_db_manager.toggle_detection_deleted_status.reset_mock()
        result = self.ui.handle_key_press(120, self.test_image)
        self.mock_db_manager.toggle_detection_deleted_status.assert_called_once_with(1)
    
    def test_handle_key_press_delete_without_selection(self):
        """Test delete key without detection selected"""
        # Setup: No selection
        self.state.selected_detection_id = None
        
        with patch('builtins.print') as mock_print:
            result = self.ui.handle_key_press(8, self.test_image)
        
        # Verify behavior
        self.assertIsNone(result)
        self.assertFalse(self.state.need_redraw)
        self.mock_db_manager.toggle_detection_deleted_status.assert_not_called()
        mock_print.assert_called_with("No detection selected. Hover over a detection to select it.")
    
    def test_handle_key_press_delete_database_failure(self):
        """Test delete key when database operation fails"""
        # Setup: Select a detection but database operation fails
        self.state.selected_detection_id = 1
        self.mock_db_manager.toggle_detection_deleted_status.return_value = False
        
        with patch('builtins.print') as mock_print:
            result = self.ui.handle_key_press(8, self.test_image)
        
        # Verify behavior
        self.assertIsNone(result)
        self.assertFalse(self.state.need_redraw)  # Should not redraw on failure
        mock_print.assert_called_with("Failed to toggle deleted status for detection 1")
    
    def test_handle_key_press_relocation_mode(self):
        """Test relocation mode key ('c')"""
        # Test entering relocation mode
        result = self.ui.handle_key_press(99, self.test_image)  # 'c' key
        
        self.assertIsNone(result)
        self.mock_controller.handle_relocation_mode.assert_called_once_with(self.state)
    
    def test_handle_key_press_cycle_overlapping(self):
        """Test cycling through overlapping detections (TAB)"""
        # Test TAB key (9)
        result = self.ui.handle_key_press(9, self.test_image)
        
        self.assertIsNone(result)
        self.mock_controller.cycle_overlapping_detections.assert_called_once_with(self.state)
    
    def test_handle_key_press_category_change_success(self):
        """Test category change keys"""
        # Setup: Select a detection
        self.state.selected_detection_id = 1
        self.mock_controller.handle_category_change.return_value = True
        
        # Test 'p' key for possum (112)
        result = self.ui.handle_key_press(112, self.test_image)
        
        self.assertIsNone(result)
        self.assertTrue(self.state.need_redraw)
        self.mock_controller.handle_category_change.assert_called_once_with(1, CATEGORY_KEYS[112], None)
        
        # Test 'o' key for other (111)
        self.mock_controller.handle_category_change.reset_mock()
        self.state.need_redraw = False
        
        result = self.ui.handle_key_press(111, self.test_image)
        
        self.assertIsNone(result)
        self.assertTrue(self.state.need_redraw)
        self.mock_controller.handle_category_change.assert_called_once_with(1, CATEGORY_KEYS[111], None)
    
    def test_handle_key_press_category_change_failure(self):
        """Test category change when it fails"""
        # Setup: Select a detection but category change fails
        self.state.selected_detection_id = 1
        self.mock_controller.handle_category_change.return_value = False
        
        result = self.ui.handle_key_press(112, self.test_image)
        
        self.assertIsNone(result)
        self.assertFalse(self.state.need_redraw)  # Should not redraw on failure
    
    def test_handle_key_press_relocation_completion(self):
        """Test that relocation completion no longer happens in key press handling"""
        # Setup relocation mode with two clicks
        self.state.relocation_mode = True
        self.state.relocation_clicks = [(100, 200), (400, 600)]
        
        # With the new design, relocation completion happens in mouse_callback
        # when the second click is made, not in handle_key_press
        # So any key press should not trigger relocation completion
        result = self.ui.handle_key_press(65, self.test_image)  # 'A' key (unknown key)
        
        # Should NOT call apply_relocation since completion happens in mouse callback now
        self.mock_controller.apply_relocation.assert_not_called()
        self.assertIsNone(result)  # Unknown key should not navigate
    
    def test_handle_key_press_unknown_key(self):
        """Test handling of unknown keys"""
        # Test a key that's not mapped to any function
        result = self.ui.handle_key_press(65, self.test_image)  # 'A' key
        
        self.assertIsNone(result)
        # Should not call any controller methods
        self.mock_controller.handle_relocation_mode.assert_not_called()
        self.mock_controller.cycle_overlapping_detections.assert_not_called()
        self.mock_controller.handle_category_change.assert_not_called()
    
    def test_handle_key_press_delete_all_detections_success(self):
        """Test 'z' key for marking all detections as deleted"""
        # Setup: Mock the controller method to return success
        self.mock_controller.mark_all_detections_deleted.return_value = True
        
        with patch('builtins.print') as mock_print:
            # Test 'z' key (122)
            result = self.ui.handle_key_press(122, self.test_image, image_id=1)
        
        # Verify behavior
        self.assertIsNone(result)  # Should not navigate
        self.assertTrue(self.state.need_redraw)  # Should trigger redraw
        self.mock_controller.mark_all_detections_deleted.assert_called_once_with(1)
    
    def test_handle_key_press_delete_all_detections_failure(self):
        """Test 'z' key when marking all detections fails"""
        # Setup: Mock the controller method to return failure
        self.mock_controller.mark_all_detections_deleted.return_value = False
        
        with patch('builtins.print') as mock_print:
            result = self.ui.handle_key_press(122, self.test_image, image_id=1)
        
        # Verify behavior
        self.assertIsNone(result)  # Should not navigate
        self.assertFalse(self.state.need_redraw)  # Should not redraw on failure
        self.mock_controller.mark_all_detections_deleted.assert_called_once_with(1)
    
    def test_handle_key_press_delete_all_detections_no_image_id(self):
        """Test 'z' key when no image ID is provided"""
        with patch('builtins.print') as mock_print:
            result = self.ui.handle_key_press(122, self.test_image, image_id=None)
        
        # Verify behavior
        self.assertIsNone(result)  # Should not navigate
        self.assertFalse(self.state.need_redraw)  # Should not redraw
        self.mock_controller.mark_all_detections_deleted.assert_not_called()
        mock_print.assert_called_with("Error: No image ID available for marking detections as deleted")
    
    @patch('cv2.imread')
    @patch('cv2.namedWindow')
    @patch('cv2.setMouseCallback')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('cv2.getTextSize')
    @patch('cv2.putText')
    def test_display_image_with_boxes_basic_flow(self, mock_putText, mock_getTextSize, 
                                                 mock_waitKey, mock_imshow, mock_setMouseCallback, 
                                                 mock_namedWindow, mock_imread):
        """Test basic flow of displaying image with boxes"""
        # Setup mocks
        mock_imread.return_value = self.test_image
        mock_getTextSize.return_value = ((100, 20), 5)
        mock_waitKey.side_effect = [255, 27]  # No key, then ESC
        
        # Mock database manager
        mock_db_manager = MagicMock()
        mock_db_manager.get_detections_for_image.return_value = [
            Detection(id=1, category=1, confidence=0.95, x=0.1, y=0.2, width=0.3, height=0.4)
        ]
        mock_db_manager.auto_delete_similar_detections.return_value = 0
        
        # Mock renderer
        self.mock_renderer.draw_boxes_on_image.return_value = self.test_image
        
        # Test display
        result = self.ui.display_image_with_boxes(
            mock_db_manager, 1, 'test_image.jpg', 0.0, 
            {'1': 'GangGang'}, 0, 1
        )
        
        # Verify result
        self.assertEqual(result, 0)  # Should exit due to ESC
        
        # Verify setup calls - cv2.WINDOW_NORMAL is actually 0, not 1
        mock_namedWindow.assert_called_once_with("GGSort", 0)
        mock_setMouseCallback.assert_called_once()
        mock_db_manager.save_last_image_index.assert_called()
    
    @patch('cv2.imread')
    def test_display_image_with_boxes_no_detections(self, mock_imread):
        """Test displaying image with no detections"""
        mock_imread.return_value = self.test_image
        
        # Mock database manager with no detections
        mock_db_manager = MagicMock()
        mock_db_manager.get_detections_for_image.return_value = []
        
        with patch('builtins.print') as mock_print:
            result = self.ui.display_image_with_boxes(
                mock_db_manager, 1, 'test_image.jpg', 0.5, 
                {'1': 'GangGang'}, 0, 1
            )
        
        # Should skip image and return 1 (next)
        self.assertEqual(result, 1)
        mock_print.assert_called_with("Skipping test_image.jpg - no detections above confidence threshold 0.5")
    
    @patch('cv2.imread')
    def test_display_image_with_boxes_image_load_failure(self, mock_imread):
        """Test handling of image load failure"""
        mock_imread.return_value = None  # Simulate failed image load
        
        mock_db_manager = MagicMock()
        mock_db_manager.get_detections_for_image.return_value = [
            Detection(id=1, category=1, confidence=0.95, x=0.1, y=0.2, width=0.3, height=0.4)
        ]
        
        with patch('builtins.print') as mock_print:
            result = self.ui.display_image_with_boxes(
                mock_db_manager, 1, 'test_image.jpg', 0.0, 
                {'1': 'GangGang'}, 0, 1
            )
        
        # Should skip image and return 1 (next)
        self.assertEqual(result, 1)
        mock_print.assert_called_with("Could not read image: test_image.jpg")
    
    @patch('cv2.imread')
    @patch('cv2.namedWindow')
    @patch('cv2.setMouseCallback')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('cv2.getTextSize')
    @patch('cv2.putText')
    def test_display_image_state_reset_on_image_change(self, mock_putText, mock_getTextSize,
                                                       mock_waitKey, mock_imshow, mock_setMouseCallback,
                                                       mock_namedWindow, mock_imread):
        """Test that state is properly reset when changing images"""
        # Setup mocks
        mock_imread.return_value = self.test_image
        mock_getTextSize.return_value = ((100, 20), 5)
        mock_waitKey.return_value = 27  # ESC to exit immediately
        
        # Setup state with relocation mode and overlapping detections
        self.state.relocation_mode = True
        self.state.relocation_detection_id = 1
        self.state.relocation_clicks = [(100, 200)]
        self.state.overlapping_detections = [MagicMock()]
        self.state.current_overlap_index = 1
        
        mock_db_manager = MagicMock()
        mock_db_manager.get_detections_for_image.return_value = [
            Detection(id=1, category=1, confidence=0.95, x=0.1, y=0.2, width=0.3, height=0.4)
        ]
        mock_db_manager.auto_delete_similar_detections.return_value = 0
        
        self.mock_renderer.draw_boxes_on_image.return_value = self.test_image
        
        with patch('builtins.print') as mock_print:
            result = self.ui.display_image_with_boxes(
                mock_db_manager, 1, 'test_image.jpg', 0.0,
                {'1': 'GangGang'}, 0, 1
            )
        
        # Verify relocation mode was reset
        self.assertFalse(self.state.relocation_mode)
        self.assertIsNone(self.state.relocation_detection_id)
        self.assertEqual(len(self.state.relocation_clicks), 0)
        
        # Verify overlapping detections were reset
        self.assertEqual(len(self.state.overlapping_detections), 0)
        self.assertEqual(self.state.current_overlap_index, 0)
        
        # Verify print message about exiting relocation mode
        mock_print.assert_any_call("Exiting relocation mode due to image change")
    
    @patch('cv2.imread')
    @patch('cv2.namedWindow')
    @patch('cv2.setMouseCallback')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('cv2.getTextSize')
    @patch('cv2.putText')
    def test_display_image_redraw_cycle(self, mock_putText, mock_getTextSize,
                                       mock_waitKey, mock_imshow, mock_setMouseCallback,
                                       mock_namedWindow, mock_imread):
        """Test the redraw cycle in display loop"""
        # Setup mocks
        mock_imread.return_value = self.test_image
        mock_getTextSize.return_value = ((100, 20), 5)
        # First iteration: no key (255), second iteration: ESC (27)
        mock_waitKey.side_effect = [255, 27]
        
        mock_db_manager = MagicMock()
        test_detection = Detection(id=1, category=1, confidence=0.95, x=0.1, y=0.2, width=0.3, height=0.4)
        mock_db_manager.get_detections_for_image.return_value = [test_detection]
        mock_db_manager.auto_delete_similar_detections.return_value = 0
        
        self.mock_renderer.draw_boxes_on_image.return_value = self.test_image
        
        # Test display
        result = self.ui.display_image_with_boxes(
            mock_db_manager, 1, 'test_image.jpg', 0.0,
            {'1': 'GangGang'}, 0, 1
        )
        
        # Verify redraw happened
        # Should call get_detections_for_image at least once (initial load)
        self.assertGreaterEqual(mock_db_manager.get_detections_for_image.call_count, 1)
        
        # Should call draw_boxes_on_image at least once
        self.assertGreaterEqual(self.mock_renderer.draw_boxes_on_image.call_count, 1)
        
        # Should call imshow at least once
        self.assertGreaterEqual(mock_imshow.call_count, 1)
    
    def test_window_name_property(self):
        """Test that window name is set correctly"""
        self.assertEqual(self.ui.window_name, "GGSort")
    
    def test_state_need_redraw_management(self):
        """Test that need_redraw flag is properly managed"""
        # Initially should not need redraw
        self.assertFalse(self.state.need_redraw)
        
        # Mouse movement should trigger redraw
        self.ui.mouse_callback(0, 100, 200, 0, None)
        self.assertTrue(self.state.need_redraw)
        
        # Reset for next test
        self.state.need_redraw = False
        
        # Left click in relocation mode should trigger redraw
        self.state.relocation_mode = True
        self.ui.mouse_callback(1, 100, 200, 0, None)
        self.assertTrue(self.state.need_redraw)

    def test_mouse_callback_relocation_completion(self):
        """Test that relocation completes immediately when second click is made"""
        # Setup relocation mode
        self.state.relocation_mode = True
        self.state.relocation_detection_id = 1
        self.ui.current_original_img = self.test_image
        
        with patch('builtins.print') as mock_print:
            # First click
            self.ui.mouse_callback(1, 100, 200, 0, None)
            
            # Verify first click was recorded but relocation not completed
            self.assertEqual(len(self.state.relocation_clicks), 1)
            self.assertTrue(self.state.relocation_mode)
            self.mock_controller.apply_relocation.assert_not_called()
            
            # Second click - should complete relocation immediately
            self.ui.mouse_callback(1, 400, 600, 0, None)
            
            # Verify relocation was applied
            self.mock_controller.apply_relocation.assert_called_once_with(self.state, self.test_image)
            
            # Check print messages
            expected_calls = [
                call("Click 1: (100, 200)"),
                call("Click 2: (400, 600)"),
                call("Two clicks recorded. Applying new coordinates...")
            ]
            mock_print.assert_has_calls(expected_calls)


class TestUserInterfaceKeyMappings(unittest.TestCase):
    """Test specific key mappings and their ASCII values"""
    
    def setUp(self):
        """Set up for key mapping tests"""
        self.mock_controller = MagicMock(spec=DetectionController)
        self.mock_renderer = MagicMock(spec=DetectionRenderer)
        self.state = AppState()
        
        # Setup mock database manager on controller
        self.mock_db_manager = MagicMock(spec=DatabaseManager)
        self.mock_controller.db_manager = self.mock_db_manager
        
        self.ui = UserInterface(self.mock_controller, self.mock_renderer, self.state)
        self.test_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    
    def test_all_category_keys(self):
        """Test all category change keys"""
        self.state.selected_detection_id = 1
        self.mock_controller.handle_category_change.return_value = True
        
        # Test all keys in CATEGORY_KEYS
        for key_code, category_info in CATEGORY_KEYS.items():
            with self.subTest(key=key_code, category=category_info['name']):
                self.mock_controller.handle_category_change.reset_mock()
                self.state.need_redraw = False
                
                result = self.ui.handle_key_press(key_code, self.test_image)
                
                self.assertIsNone(result)
                self.assertTrue(self.state.need_redraw)
                self.mock_controller.handle_category_change.assert_called_once_with(1, category_info, None)
    
    def test_delete_key_variations(self):
        """Test all delete key variations"""
        self.state.selected_detection_id = 1
        self.mock_db_manager.toggle_detection_deleted_status.return_value = True
        
        delete_keys = [8, 127, 120]  # BACKSPACE, DELETE, 'x'
        
        for key_code in delete_keys:
            with self.subTest(key=key_code):
                self.mock_db_manager.toggle_detection_deleted_status.reset_mock()
                self.state.need_redraw = False
                
                result = self.ui.handle_key_press(key_code, self.test_image)
                
                self.assertIsNone(result)
                self.assertTrue(self.state.need_redraw)
                self.mock_db_manager.toggle_detection_deleted_status.assert_called_once_with(1)
    
    def test_navigation_key_variations(self):
        """Test all navigation key variations"""
        # Test next keys
        next_keys = [32, 3]  # SPACE, RIGHT ARROW
        for key_code in next_keys:
            with self.subTest(key=key_code, direction="next"):
                result = self.ui.handle_key_press(key_code, self.test_image)
                self.assertEqual(result, 1)
        
        # Test previous keys
        prev_keys = [2]  # LEFT ARROW
        for key_code in prev_keys:
            with self.subTest(key=key_code, direction="previous"):
                result = self.ui.handle_key_press(key_code, self.test_image)
                self.assertEqual(result, -1)
        
        # Test exit keys
        exit_keys = [27, 113]  # ESC, 'q'
        for key_code in exit_keys:
            with self.subTest(key=key_code, direction="exit"):
                result = self.ui.handle_key_press(key_code, self.test_image)
                self.assertEqual(result, 0)
    
    def test_uppercase_category_keys_all_detections(self):
        """Test uppercase category change keys that change ALL detections"""
        self.mock_controller.handle_category_change.return_value = True
        
        # Test all keys in CATEGORY_KEYS_ALL
        for key_code, category_info in CATEGORY_KEYS_ALL.items():
            with self.subTest(key=key_code, category=category_info['name']):
                self.mock_controller.handle_category_change.reset_mock()
                self.state.need_redraw = False
                
                result = self.ui.handle_key_press(key_code, self.test_image, image_id=1)
                
                self.assertIsNone(result)
                self.assertTrue(self.state.need_redraw)
                # Should call with -1 for detection_id to indicate all detections
                self.mock_controller.handle_category_change.assert_called_once_with(-1, category_info, 1)

    def test_v_key_all_other_shortcut(self):
        """Test 'v' key for changing ALL detections to 'Other' (quick shortcut)"""
        self.mock_controller.handle_category_change.return_value = True
        
        result = self.ui.handle_key_press(118, self.test_image, image_id=1)  # 'v' key
        
        self.assertIsNone(result)
        self.assertTrue(self.state.need_redraw)
        # Should call with -1 for detection_id and 'Other' category
        expected_category_info = {'id': 5, 'name': 'Other'}
        self.mock_controller.handle_category_change.assert_called_once_with(-1, expected_category_info, 1)


if __name__ == '__main__':
    unittest.main() 