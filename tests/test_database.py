#!/usr/bin/env python3

import unittest
import tempfile
import os
import sqlite3
import sys
from unittest.mock import patch, MagicMock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ggsort import DatabaseManager, Detection

class TestDatabaseManager(unittest.TestCase):
    """Test cases for DatabaseManager class"""
    
    def setUp(self):
        """Set up test database for each test"""
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db_path = self.temp_db.name
        
        # Create test database with schema
        self._create_test_database()
        self.db_manager = DatabaseManager(self.db_path)
    
    def tearDown(self):
        """Clean up after each test"""
        self.db_manager.close()
        os.unlink(self.db_path)
    
    def _create_test_database(self):
        """Create a test database with the expected schema"""
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
        
        # Insert test data
        cursor.execute('''
        INSERT INTO images (id, file_path, width, height) 
        VALUES (1, 'test_image1.jpg', 1920, 1080)
        ''')
        
        cursor.execute('''
        INSERT INTO images (id, file_path, width, height) 
        VALUES (2, 'test_image2.jpg', 1280, 720)
        ''')
        
        # Insert test detections
        cursor.execute('''
        INSERT INTO detections (id, image_id, category, confidence, x, y, width, height, deleted, subcategory, hard)
        VALUES (1, 1, 1, 0.95, 0.1, 0.2, 0.3, 0.4, 0, NULL, 0)
        ''')
        
        cursor.execute('''
        INSERT INTO detections (id, image_id, category, confidence, x, y, width, height, deleted, subcategory, hard)
        VALUES (2, 1, 2, 0.85, 0.5, 0.6, 0.2, 0.3, 0, NULL, 0)
        ''')
        
        cursor.execute('''
        INSERT INTO detections (id, image_id, category, confidence, x, y, width, height, deleted, subcategory, hard)
        VALUES (3, 2, 1, 0.75, 0.2, 0.3, 0.4, 0.5, 1, NULL, 0)
        ''')
        
        conn.commit()
        conn.close()
    
    def test_database_connection(self):
        """Test that database connection works"""
        self.assertIsNotNone(self.db_manager.conn)
        self.assertEqual(self.db_manager.db_file, self.db_path)
    
    def test_database_connection_file_not_found(self):
        """Test that FileNotFoundError is raised for non-existent database"""
        with self.assertRaises(FileNotFoundError):
            DatabaseManager('non_existent_file.db')
    
    def test_get_detection_categories(self):
        """Test getting detection categories"""
        categories = self.db_manager.get_detection_categories()
        expected_categories = {
            '1': 'GangGang',
            '2': 'Person',
            '3': 'Vehicle',
            '4': 'Possum',
            '5': 'Other'
        }
        self.assertEqual(categories, expected_categories)
    
    def test_get_images_from_database(self):
        """Test retrieving images from database"""
        images = self.db_manager.get_images_from_database()
        self.assertEqual(len(images), 2)
        
        # Check first image
        self.assertEqual(images[0]['id'], 1)
        self.assertEqual(images[0]['file_path'], 'test_image1.jpg')
        self.assertEqual(images[0]['width'], 1920)
        self.assertEqual(images[0]['height'], 1080)
        
        # Check second image
        self.assertEqual(images[1]['id'], 2)
        self.assertEqual(images[1]['file_path'], 'test_image2.jpg')
        self.assertEqual(images[1]['width'], 1280)
        self.assertEqual(images[1]['height'], 720)
    
    def test_get_detections_for_image(self):
        """Test retrieving detections for a specific image"""
        # Test image 1 (should have 2 detections)
        detections = self.db_manager.get_detections_for_image(1, 0.0)
        self.assertEqual(len(detections), 2)
        
        # Check first detection (higher confidence should come first)
        detection1 = detections[0]
        self.assertEqual(detection1.id, 1)
        self.assertEqual(detection1.category, 1)
        self.assertEqual(detection1.confidence, 0.95)
        self.assertEqual(detection1.x, 0.1)
        self.assertEqual(detection1.y, 0.2)
        self.assertEqual(detection1.width, 0.3)
        self.assertEqual(detection1.height, 0.4)
        self.assertFalse(detection1.deleted)
        
        # Check second detection
        detection2 = detections[1]
        self.assertEqual(detection2.id, 2)
        self.assertEqual(detection2.category, 2)
        self.assertEqual(detection2.confidence, 0.85)
        self.assertFalse(detection2.deleted)
    
    def test_get_detections_with_confidence_threshold(self):
        """Test filtering detections by confidence threshold"""
        # With threshold 0.9, should only get detection with 0.95 confidence
        detections = self.db_manager.get_detections_for_image(1, 0.9)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].confidence, 0.95)
        
        # With threshold 0.8, should get both detections
        detections = self.db_manager.get_detections_for_image(1, 0.8)
        self.assertEqual(len(detections), 2)
    
    def test_toggle_detection_deleted_status(self):
        """Test toggling detection deleted status"""
        # Initially not deleted
        detections = self.db_manager.get_detections_for_image(1, 0.0)
        detection = next(d for d in detections if d.id == 1)
        self.assertFalse(detection.deleted)
        
        # Toggle to deleted
        success = self.db_manager.toggle_detection_deleted_status(1)
        self.assertTrue(success)
        
        # Verify it's now deleted
        detections = self.db_manager.get_detections_for_image(1, 0.0)
        detection = next(d for d in detections if d.id == 1)
        self.assertTrue(detection.deleted)
        
        # Toggle back to not deleted
        success = self.db_manager.toggle_detection_deleted_status(1)
        self.assertTrue(success)
        
        # Verify it's not deleted again
        detections = self.db_manager.get_detections_for_image(1, 0.0)
        detection = next(d for d in detections if d.id == 1)
        self.assertFalse(detection.deleted)
    
    def test_toggle_detection_deleted_status_invalid_id(self):
        """Test toggling deleted status with invalid detection ID"""
        success = self.db_manager.toggle_detection_deleted_status(999)
        self.assertFalse(success)
    
    def test_update_detection_category(self):
        """Test updating detection category"""
        # Update category from 1 to 4
        success = self.db_manager.update_detection_category(1, 4)
        self.assertTrue(success)
        
        # Verify the change
        detections = self.db_manager.get_detections_for_image(1, 0.0)
        detection = next(d for d in detections if d.id == 1)
        self.assertEqual(detection.category, 4)
    
    def test_update_detection_category_invalid_id(self):
        """Test updating category with invalid detection ID"""
        success = self.db_manager.update_detection_category(999, 4)
        self.assertFalse(success)
    
    def test_update_detection_coordinates(self):
        """Test updating detection coordinates"""
        # Update coordinates
        new_x, new_y, new_width, new_height = 0.15, 0.25, 0.35, 0.45
        success = self.db_manager.update_detection_coordinates(1, new_x, new_y, new_width, new_height)
        self.assertTrue(success)
        
        # Verify the change
        detections = self.db_manager.get_detections_for_image(1, 0.0)
        detection = next(d for d in detections if d.id == 1)
        self.assertEqual(detection.x, new_x)
        self.assertEqual(detection.y, new_y)
        self.assertEqual(detection.width, new_width)
        self.assertEqual(detection.height, new_height)
    
    def test_update_detection_coordinates_invalid_id(self):
        """Test updating coordinates with invalid detection ID"""
        success = self.db_manager.update_detection_coordinates(999, 0.1, 0.2, 0.3, 0.4)
        self.assertFalse(success)
    
    def test_app_state_table_creation(self):
        """Test that app_state table is created correctly"""
        self.db_manager.ensure_app_state_table()
        
        # Check that table exists
        cursor = self.db_manager.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='app_state'")
        result = cursor.fetchone()
        self.assertIsNotNone(result)
    
    def test_save_and_get_last_image_index(self):
        """Test saving and retrieving last image index"""
        # Save index
        test_index = 42
        self.db_manager.save_last_image_index(test_index)
        
        # Retrieve index
        retrieved_index = self.db_manager.get_last_image_index()
        self.assertEqual(retrieved_index, test_index)
    
    def test_get_last_image_index_default(self):
        """Test getting last image index when none exists"""
        # Should return default value
        index = self.db_manager.get_last_image_index(default=10)
        self.assertEqual(index, 10)
        
        # Should return 0 when no default specified
        index = self.db_manager.get_last_image_index()
        self.assertEqual(index, 0)
    
    def test_get_last_image_index_invalid_value(self):
        """Test getting last image index with invalid stored value"""
        # Insert invalid value
        self.db_manager.ensure_app_state_table()
        cursor = self.db_manager.conn.cursor()
        cursor.execute("INSERT OR REPLACE INTO app_state (key, value) VALUES ('last_image_index', 'invalid')")
        self.db_manager.conn.commit()
        
        # Should return default
        index = self.db_manager.get_last_image_index(default=5)
        self.assertEqual(index, 5)
    
    def test_database_close(self):
        """Test closing database connection"""
        self.db_manager.close()
        # After closing, connection should be closed
        # Note: We can't easily test this without accessing private attributes

if __name__ == '__main__':
    unittest.main() 