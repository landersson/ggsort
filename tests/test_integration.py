#!/usr/bin/env python3

import unittest
import tempfile
import os
import sys
import sqlite3
import numpy as np
from unittest.mock import MagicMock, patch, call

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ggsort import (
    GGSortApplication, DatabaseManager, DetectionController, DetectionRenderer, 
    UserInterface, AppState, Detection, BoundingBox, CATEGORY_KEYS
)

class TestIntegrationDatabaseUI(unittest.TestCase):
    """Integration tests focusing on database handling and UI logic"""
    
    def setUp(self):
        """Set up test environment with real database and components"""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        # Create temporary image directory
        self.temp_image_dir = tempfile.mkdtemp()
        
        # Create test database with schema and data
        self._create_test_database()
        
        # Create real components (not mocked)
        self.db_manager = DatabaseManager(self.db_path)
        self.categories = self.db_manager.get_detection_categories()
        self.renderer = DetectionRenderer(self.categories)
        self.state = AppState()
        self.controller = DetectionController(self.db_manager, self.renderer)
        self.ui = UserInterface(self.controller, self.renderer, self.state)
        
        # Create test image file
        self.test_image_path = os.path.join(self.temp_image_dir, 'test_image1.jpg')
        self._create_test_image()
    
    def tearDown(self):
        """Clean up after each test"""
        self.db_manager.close()
        os.unlink(self.db_path)
        
        # Clean up image files
        if os.path.exists(self.test_image_path):
            os.unlink(self.test_image_path)
        os.rmdir(self.temp_image_dir)
    
    def _create_test_database(self):
        """Create a test database with realistic schema and data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create images table
        cursor.execute('''
        CREATE TABLE images (
            id INTEGER PRIMARY KEY,
            file_path TEXT UNIQUE NOT NULL,
            width INTEGER,
            height INTEGER,
            datetime_original TEXT,
            manually_processed BOOLEAN DEFAULT 0
        )
        ''')
        
        # Create detections table
        cursor.execute('''
        CREATE TABLE detections (
            id INTEGER PRIMARY KEY,
            image_id INTEGER NOT NULL,
            category INTEGER,
            confidence REAL,
            x REAL,
            y REAL,
            width REAL,
            height REAL,
            deleted BOOLEAN DEFAULT 0,
            subcategory INTEGER,
            hard BOOLEAN DEFAULT 0,
            FOREIGN KEY (image_id) REFERENCES images(id)
        )
        ''')
        
        # Insert test images
        cursor.execute('''
        INSERT INTO images (id, file_path, width, height) 
        VALUES (1, 'test_image1.jpg', 1920, 1080)
        ''')
        
        cursor.execute('''
        INSERT INTO images (id, file_path, width, height) 
        VALUES (2, 'test_image2.jpg', 1280, 720)
        ''')
        
        # Insert test detections with various scenarios
        test_detections = [
            (1, 1, 1, 0.95, 0.1, 0.2, 0.3, 0.4, 0, None, 0),  # GangGang
            (2, 1, 2, 0.85, 0.5, 0.6, 0.2, 0.3, 0, None, 0),  # Person
            (3, 1, 3, 0.75, 0.2, 0.3, 0.15, 0.25, 1, None, 0), # Vehicle (deleted)
            (4, 1, 1, 0.65, 0.7, 0.8, 0.1, 0.15, 0, None, 0),  # Another GangGang
            (5, 2, 4, 0.88, 0.3, 0.4, 0.2, 0.3, 0, None, 0),   # Possum
        ]
        
        for detection in test_detections:
            cursor.execute('''
            INSERT INTO detections (id, image_id, category, confidence, x, y, width, height, deleted, subcategory, hard)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', detection)
        
        conn.commit()
        conn.close()
    
    def _create_test_image(self):
        """Create a simple test image file"""
        # Create a simple 1920x1080 test image using numpy
        import cv2
        test_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        test_img[:] = (100, 150, 200)  # Fill with a color
        cv2.imwrite(self.test_image_path, test_img)
    
    def test_detection_deletion_integration(self):
        """Test that detection deletion works end-to-end from UI to database"""
        # Setup: Select a detection
        self.state.selected_detection_id = 1
        
        # Verify initial state
        detections_before = self.db_manager.get_detections_for_image(1, 0.0)
        detection_before = next(d for d in detections_before if d.id == 1)
        self.assertFalse(detection_before.deleted)
        
        # Simulate user pressing delete key (BACKSPACE = 8)
        with patch('builtins.print') as mock_print:
            result = self.ui.handle_key_press(8, None)
        
        # Verify UI response
        self.assertIsNone(result)  # Should not navigate
        self.assertTrue(self.state.need_redraw)  # Should trigger redraw
        
        # Verify database was updated
        detections_after = self.db_manager.get_detections_for_image(1, 0.0)
        detection_after = next(d for d in detections_after if d.id == 1)
        self.assertTrue(detection_after.deleted)
        
        # Verify print message
        mock_print.assert_called_with("Toggled deleted status for detection 1")
        
        # Test toggling back
        result = self.ui.handle_key_press(8, None)
        detections_final = self.db_manager.get_detections_for_image(1, 0.0)
        detection_final = next(d for d in detections_final if d.id == 1)
        self.assertFalse(detection_final.deleted)
    
    def test_category_change_integration(self):
        """Test that category changes work end-to-end from UI to database"""
        # Setup: Select a detection
        self.state.selected_detection_id = 1
        
        # Verify initial category
        detections_before = self.db_manager.get_detections_for_image(1, 0.0)
        detection_before = next(d for d in detections_before if d.id == 1)
        self.assertEqual(detection_before.category, 1)  # GangGang
        
        # Simulate user pressing 'p' key to mark as possum (112 = 'p')
        with patch('builtins.print') as mock_print:
            result = self.ui.handle_key_press(112, None)
        
        # Verify UI response
        self.assertIsNone(result)  # Should not navigate
        self.assertTrue(self.state.need_redraw)  # Should trigger redraw
        
        # Verify database was updated
        detections_after = self.db_manager.get_detections_for_image(1, 0.0)
        detection_after = next(d for d in detections_after if d.id == 1)
        self.assertEqual(detection_after.category, 4)  # Possum
        
        # Verify print message
        mock_print.assert_called_with("Changed detection 1 category to Possum")
        
        # Test changing to 'other' category (111 = 'o')
        result = self.ui.handle_key_press(111, None)
        detections_final = self.db_manager.get_detections_for_image(1, 0.0)
        detection_final = next(d for d in detections_final if d.id == 1)
        self.assertEqual(detection_final.category, 5)  # Other
    
    def test_relocation_integration(self):
        """Test that detection relocation works end-to-end"""
        # Create a mock image for coordinate calculations
        mock_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Setup: Select a detection and store the image in UI
        self.state.selected_detection_id = 1
        self.ui.current_original_img = mock_img  # Store image for relocation
        
        # Verify initial coordinates
        detections_before = self.db_manager.get_detections_for_image(1, 0.0)
        detection_before = next(d for d in detections_before if d.id == 1)
        self.assertEqual(detection_before.x, 0.1)
        self.assertEqual(detection_before.y, 0.2)
        self.assertEqual(detection_before.width, 0.3)
        self.assertEqual(detection_before.height, 0.4)
        
        # Step 1: Enter relocation mode (99 = 'c')
        with patch('builtins.print') as mock_print:
            result = self.ui.handle_key_press(99, mock_img)
        
        self.assertIsNone(result)
        self.assertTrue(self.state.relocation_mode)
        self.assertEqual(self.state.relocation_detection_id, 1)
        self.assertEqual(len(self.state.relocation_clicks), 0)
        
        # Step 2: Simulate first click (top-left)
        self.ui.mouse_callback(1, 100, 200, 0, None)  # cv2.EVENT_LBUTTONDOWN = 1
        self.assertEqual(len(self.state.relocation_clicks), 1)
        self.assertEqual(self.state.relocation_clicks[0], (100, 200))
        self.assertTrue(self.state.relocation_mode)  # Should still be in relocation mode
        
        # Step 3: Simulate second click (bottom-right) - this should complete relocation immediately
        with patch('builtins.print') as mock_print:
            self.ui.mouse_callback(1, 400, 600, 0, None)
        
        # Verify relocation was applied immediately
        self.assertFalse(self.state.relocation_mode)  # Should exit relocation mode immediately
        self.assertEqual(len(self.state.relocation_clicks), 0)  # Should be reset
        
        # Verify database was updated with new coordinates
        detections_after = self.db_manager.get_detections_for_image(1, 0.0)
        detection_after = next(d for d in detections_after if d.id == 1)
        
        # Expected normalized coordinates: (100/1920, 200/1080, 300/1920, 400/1080)
        expected_x = 100 / 1920  # ≈ 0.052
        expected_y = 200 / 1080  # ≈ 0.185
        expected_width = 300 / 1920  # ≈ 0.156
        expected_height = 400 / 1080  # ≈ 0.370
        
        self.assertAlmostEqual(detection_after.x, expected_x, places=3)
        self.assertAlmostEqual(detection_after.y, expected_y, places=3)
        self.assertAlmostEqual(detection_after.width, expected_width, places=3)
        self.assertAlmostEqual(detection_after.height, expected_height, places=3)
        
        # Verify print messages
        expected_calls = [
            call("Click 2: (400, 600)"),
            call("Two clicks recorded. Applying new coordinates..."),
        ]
        mock_print.assert_has_calls(expected_calls, any_order=False)
    
    def test_overlapping_detection_cycling_integration(self):
        """Test cycling through overlapping detections with database consistency"""
        # Create overlapping detections by positioning mouse over detection area
        # Detection 1 is at (0.1, 0.2, 0.3, 0.4) in normalized coords
        # In 1920x1080 image: (192, 216) to (768, 648)
        
        # Position mouse in the overlapping area
        self.state.update_mouse_position(300, 300)
        
        # Get detections and render to populate overlapping detections
        detections = self.db_manager.get_detections_for_image(1, 0.0)
        mock_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # This should populate overlapping detections
        self.renderer.draw_boxes_on_image(mock_img, detections, self.state)
        
        # Verify we have overlapping detections
        self.assertGreater(len(self.state.overlapping_detections), 0)
        initial_selection = self.state.selected_detection_id
        
        # Test cycling through overlapping detections (TAB = 9)
        with patch('builtins.print') as mock_print:
            result = self.ui.handle_key_press(9, None)
        
        # Verify cycling worked
        if len(self.state.overlapping_detections) > 1:
            self.assertIsNone(result)  # Should not navigate
            self.assertTrue(self.state.need_redraw)
            self.assertNotEqual(self.state.selected_detection_id, initial_selection)
        
        # Verify the selected detection exists in database
        if self.state.selected_detection_id:
            detections_check = self.db_manager.get_detections_for_image(1, 0.0)
            selected_exists = any(d.id == self.state.selected_detection_id for d in detections_check)
            self.assertTrue(selected_exists)
    
    def test_mouse_interaction_with_detection_selection(self):
        """Test that mouse movement properly selects detections"""
        # Get detections and create mock image
        detections = self.db_manager.get_detections_for_image(1, 0.0)
        mock_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        # Test mouse outside any detection
        self.state.update_mouse_position(50, 50)
        self.renderer.draw_boxes_on_image(mock_img, detections, self.state)
        self.assertIsNone(self.state.selected_detection_id)
        
        # Test mouse inside detection 1 area (normalized 0.1,0.2,0.3,0.4 -> pixel ~192,216 to 768,648)
        self.state.update_mouse_position(300, 300)
        self.renderer.draw_boxes_on_image(mock_img, detections, self.state)
        
        # Should have selected a detection
        self.assertIsNotNone(self.state.selected_detection_id)
        
        # Verify the selected detection exists in our test data
        selected_detection = next((d for d in detections if d.id == self.state.selected_detection_id), None)
        self.assertIsNotNone(selected_detection)
    
    def test_navigation_with_state_persistence(self):
        """Test that navigation properly saves state to database"""
        current_index = 0
        
        # Test next navigation (SPACE = 32)
        result = self.ui.handle_key_press(32, None)
        self.assertEqual(result, 1)  # Should return next command
        
        # Test previous navigation (LEFT ARROW = 2)
        result = self.ui.handle_key_press(2, None)
        self.assertEqual(result, -1)  # Should return previous command
        
        # Test exit navigation (ESC = 27)
        result = self.ui.handle_key_press(27, None)
        self.assertEqual(result, 0)  # Should return exit command
    
    def test_error_handling_invalid_detection_operations(self):
        """Test error handling when operations are performed on invalid detections"""
        # Test deletion with no selection
        self.state.selected_detection_id = None
        
        with patch('builtins.print') as mock_print:
            result = self.ui.handle_key_press(8, None)  # BACKSPACE
        
        self.assertIsNone(result)
        self.assertFalse(self.state.need_redraw)
        mock_print.assert_called_with("No detection selected. Hover over a detection to select it.")
        
        # Test category change with no selection
        with patch('builtins.print') as mock_print:
            result = self.ui.handle_key_press(112, None)  # 'p' key
        
        self.assertIsNone(result)
        self.assertFalse(self.state.need_redraw)
        
        # Test relocation with no selection
        with patch('builtins.print') as mock_print:
            result = self.ui.handle_key_press(99, None)  # 'c' key
        
        self.assertIsNone(result)
        self.assertFalse(self.state.relocation_mode)
    
    def test_database_transaction_consistency(self):
        """Test that database operations maintain consistency"""
        # Perform multiple operations and verify database state
        self.state.selected_detection_id = 1
        
        # Change category
        self.ui.handle_key_press(112, None)  # Mark as possum
        
        # Toggle deletion
        self.ui.handle_key_press(8, None)  # Delete
        
        # Verify both changes persisted
        detections = self.db_manager.get_detections_for_image(1, 0.0)
        detection = next(d for d in detections if d.id == 1)
        
        self.assertEqual(detection.category, 4)  # Possum
        self.assertTrue(detection.deleted)  # Deleted
        
        # Verify other detections unchanged
        other_detection = next(d for d in detections if d.id == 2)
        self.assertEqual(other_detection.category, 2)  # Still Person
        self.assertFalse(other_detection.deleted)  # Still not deleted
    
    def test_confidence_threshold_filtering(self):
        """Test that confidence threshold properly filters detections"""
        # Test with high threshold - should filter out low confidence detections
        high_threshold_detections = self.db_manager.get_detections_for_image(1, 0.9)
        self.assertEqual(len(high_threshold_detections), 1)  # Only 0.95 confidence detection
        
        # Test with low threshold - should include all detections
        low_threshold_detections = self.db_manager.get_detections_for_image(1, 0.0)
        self.assertEqual(len(low_threshold_detections), 4)  # All detections for image 1
        
        # Test with medium threshold
        medium_threshold_detections = self.db_manager.get_detections_for_image(1, 0.8)
        self.assertEqual(len(medium_threshold_detections), 2)  # 0.95 and 0.85 confidence
    
    def test_app_state_persistence(self):
        """Test that application state is properly saved and restored"""
        # Test saving last image index
        test_index = 5
        self.db_manager.save_last_image_index(test_index)
        
        # Verify it was saved
        retrieved_index = self.db_manager.get_last_image_index()
        self.assertEqual(retrieved_index, test_index)
        
        # Test with default value
        # Create new database manager to test fresh state
        temp_db2 = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        temp_db2.close()
        
        try:
            # Create minimal database
            conn = sqlite3.connect(temp_db2.name)
            conn.execute("CREATE TABLE images (id INTEGER PRIMARY KEY)")
            conn.close()
            
            db_manager2 = DatabaseManager(temp_db2.name)
            default_index = db_manager2.get_last_image_index(default=10)
            self.assertEqual(default_index, 10)
            db_manager2.close()
        finally:
            os.unlink(temp_db2.name)


class TestFullWorkflowIntegration(unittest.TestCase):
    """Test complete workflows from start to finish"""
    
    def setUp(self):
        """Set up for workflow tests"""
        # Create temporary database
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        # Create temporary image directory
        self.temp_image_dir = tempfile.mkdtemp()
        
        # Create test database
        self._create_test_database()
        
        # Create test image
        self.test_image_path = os.path.join(self.temp_image_dir, 'test_image1.jpg')
        self._create_test_image()
    
    def tearDown(self):
        """Clean up after workflow tests"""
        os.unlink(self.db_path)
        if os.path.exists(self.test_image_path):
            os.unlink(self.test_image_path)
        os.rmdir(self.temp_image_dir)
    
    def _create_test_database(self):
        """Create test database for workflow tests"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
        CREATE TABLE images (
            id INTEGER PRIMARY KEY,
            file_path TEXT UNIQUE NOT NULL,
            width INTEGER,
            height INTEGER
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE detections (
            id INTEGER PRIMARY KEY,
            image_id INTEGER NOT NULL,
            category INTEGER,
            confidence REAL,
            x REAL,
            y REAL,
            width REAL,
            height REAL,
            deleted BOOLEAN DEFAULT 0,
            subcategory INTEGER,
            hard BOOLEAN DEFAULT 0,
            FOREIGN KEY (image_id) REFERENCES images(id)
        )
        ''')
        
        # Insert test data
        cursor.execute("INSERT INTO images (id, file_path, width, height) VALUES (1, 'test_image1.jpg', 1920, 1080)")
        cursor.execute('''
        INSERT INTO detections (id, image_id, category, confidence, x, y, width, height, deleted)
        VALUES (1, 1, 1, 0.95, 0.1, 0.2, 0.3, 0.4, 0)
        ''')
        
        conn.commit()
        conn.close()
    
    def _create_test_image(self):
        """Create test image file"""
        import cv2
        test_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
        test_img[:] = (100, 150, 200)
        cv2.imwrite(self.test_image_path, test_img)
    
    @patch('cv2.imshow')
    @patch('cv2.namedWindow')
    @patch('cv2.setMouseCallback')
    @patch('cv2.waitKey')
    @patch('cv2.destroyAllWindows')
    def test_complete_detection_modification_workflow(self, mock_destroy, mock_waitkey, 
                                                     mock_setmouse, mock_window, mock_imshow):
        """Test a complete workflow of modifying detections"""
        # Setup mocks to simulate user interaction
        # Simulate: mouse move -> select detection -> change category -> delete -> exit
        key_sequence = [
            255,  # No key (mouse move)
            112,  # 'p' key (mark as possum)
            8,    # BACKSPACE (delete)
            27    # ESC (exit)
        ]
        mock_waitkey.side_effect = key_sequence
        
        # Create application
        app = GGSortApplication(
            db_file=self.db_path,
            image_dir=self.temp_image_dir,
            confidence_threshold=0.0
        )
        
        # Mock mouse movement to select detection
        app.state.update_mouse_position(300, 300)  # Position over detection
        
        # Run application (will exit due to ESC key)
        with patch('builtins.print'):
            result = app.run()
        
        # Verify the workflow completed successfully
        self.assertEqual(result, 0)
        
        # Verify database changes were applied
        # Create a new database manager to check final state since app.db_manager is closed
        final_db_manager = DatabaseManager(self.db_path)
        try:
            final_detections = final_db_manager.get_detections_for_image(1, 0.0)
            detection = final_detections[0]
            
            # Should have been changed to possum (category 4) and deleted
            self.assertEqual(detection.category, 4)
            self.assertTrue(detection.deleted)
        finally:
            final_db_manager.close()


if __name__ == '__main__':
    unittest.main() 