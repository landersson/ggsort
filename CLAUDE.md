# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Running the Application
```bash
./bin/ggsort <LOCATION> [IMAGE_DATA_DIR]
```
Example: `./bin/ggsort Waynes`

### Testing
```bash
# Run all tests
python tests/run_all_tests.py

# Run specific test categories
python tests/run_all_tests.py database
python tests/run_all_tests.py controller
python tests/run_all_tests.py integration
python tests/run_all_tests.py renderer
python tests/run_all_tests.py ui
```

### Database Operations
```bash
# Create new database from detection results
python src/make_db.py --base-dir <LOCATION> --image-dir <IMAGE_PATH> --output <DB_FILE>

# Extract data from database
python src/extract.py

# Merge multiple result files
python src/merge.py
```

### Dependencies
```bash
pip install -r requirements.txt
```

## Architecture Overview

GGSort is a computer vision detection sorting application for wildlife camera images, specifically designed for Gang-gang Cockatoo identification.

### Core Components

1. **Main Application (`src/ggsort.py`)**
   - Interactive OpenCV-based image viewer
   - Real-time detection editing and categorization
   - SQLite database integration for persistence

2. **Database Layer (`src/make_db.py`)**
   - Processes detection results from JSON format
   - Creates SQLite database with images and detections tables
   - Handles EXIF metadata extraction and coordinate normalization

3. **Detection Categories**
   - Gang-gang (1): Red bounding boxes
   - Person (2): Blue bounding boxes  
   - Vehicle (3): Green bounding boxes
   - Possum (4): Yellow bounding boxes
   - Other (5): Turquoise bounding boxes
   - Deleted detections: Grey bounding boxes

### Key Data Flow

1. Raw detection data in `data/results.json` (compressed as .gz)
2. Database creation via `make_db.py` filters and normalizes detection data
3. Main application loads database and provides interactive editing interface
4. User modifications are persisted back to the database in real-time

### Interactive Controls Architecture

The application uses OpenCV mouse callbacks and keyboard handlers:
- Mouse hover for detection selection
- TAB key for cycling through overlapping detections  
- Category assignment keys (p/P, o/O, g/G for individual/all detections)
- Detection relocation with two-click coordinate system
- Real-time visual feedback with color-coded bounding boxes

### Database Schema

**images table**: Stores image metadata and paths
**detections table**: Stores bounding box coordinates, categories, confidence scores, and deletion status

All coordinates are stored in normalized format (0.0-1.0) and converted to pixel coordinates for display.