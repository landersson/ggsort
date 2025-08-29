#!/usr/bin/env python3
"""
Export script for GGSort
Exports images from database to organized output directory structure
"""

import os
import sqlite3
import argparse
import shutil
import sys
import json
from pathlib import Path

def sample_images_by_category(cursor, location_stats, category, target_total, category_name):
    """
    Sample images with detections of a specific category, distributed as evenly as possible across locations
    
    Args:
        cursor: Database cursor
        location_stats: Dictionary of location -> image count
        category: Detection category number (1=Gang Gang, 4=Possum, etc.)
        target_total: Total number of images to sample
        category_name: Human-readable name for the category (for logging)
    
    Returns:
        List of selected image records
    """
    print(f"\n--- {category_name} Sampling ---")
    print(f"Target {category_name} images: {target_total}")
    
    # First pass: get available counts for each location
    location_availability = {}
    total_available = 0
    
    for location in sorted(location_stats.keys()):
        cursor.execute("""
            SELECT DISTINCT i.id, i.file_path
            FROM images i
            JOIN detections d ON i.id = d.image_id
            WHERE d.category = ? AND d.deleted = 0 
            AND (d.hard = 0 OR d.hard IS NULL)
            AND d.confidence >= 0.20
            AND i.file_path LIKE ?
            ORDER BY i.file_path
        """, (category, f"{location}%"))
        
        location_images = cursor.fetchall()
        available_count = len(location_images)
        location_availability[location] = {
            'available': available_count,
            'images': location_images
        }
        total_available += available_count
    
    if total_available == 0:
        print(f"No {category_name} images available")
        return []
    
    if total_available <= target_total:
        print(f"Taking all {total_available} available images (less than target)")
        all_images = []
        for location_data in location_availability.values():
            all_images.extend(location_data['images'])
        return all_images
    
    # Even distribution strategy with scaling to reach target
    # Only count locations that actually have images for this category
    locations_with_images = {loc: data for loc, data in location_availability.items() if data['available'] > 0}
    total_locations_with_images = len(locations_with_images)
    
    if total_locations_with_images == 0:
        print(f"No locations have {category_name} images")
        return []
    
    # Add small buffer to ensure we reach target after redistribution
    buffer_target = target_total + min(10, total_locations_with_images)  # Add up to 10 extra images as buffer
    base_per_location = buffer_target // total_locations_with_images
    
    print(f"Locations with {category_name} images: {total_locations_with_images}")
    print(f"Initial base per location: {base_per_location} (with buffer)")
    
    # First pass: allocate base amount, collect shortfalls
    location_allocations = {}
    total_shortfall = 0
    
    # Initialize all locations to 0
    for location in location_availability.keys():
        location_allocations[location] = 0
    
    # Only allocate to locations that have images
    for location, data in locations_with_images.items():
        available_count = data['available']
        if available_count <= base_per_location:
            # Location has fewer than base - take all
            location_allocations[location] = available_count
            shortfall = base_per_location - available_count
            total_shortfall += shortfall
        else:
            # Location has more than base - allocate base amount
            location_allocations[location] = base_per_location
    
    print(f"Total shortfall from small locations: {total_shortfall}")
    
    # Second pass: redistribute shortfall to locations with more availability
    if total_shortfall > 0:
        print(f"Redistributing {total_shortfall} images to locations with capacity...")
        
        remaining_shortfall = total_shortfall
        while remaining_shortfall > 0:
            # Find locations that can take extra images
            locations_with_capacity = []
            for location, data in location_availability.items():
                available = data['available']
                current_allocation = location_allocations[location]
                if available > current_allocation:
                    capacity = available - current_allocation
                    locations_with_capacity.append((location, capacity))
            
            if not locations_with_capacity:
                print(f"Warning: Cannot redistribute remaining {remaining_shortfall} images - no locations have capacity")
                break
            
            # Sort by capacity (descending) to prioritize locations with most capacity
            locations_with_capacity.sort(key=lambda x: x[1], reverse=True)
            
            # Distribute one image at a time to maintain even distribution
            distributed_this_round = 0
            for location, capacity in locations_with_capacity:
                if remaining_shortfall <= 0:
                    break
                if capacity > 0:
                    # Give one more image to this location
                    location_allocations[location] += 1
                    remaining_shortfall -= 1
                    distributed_this_round += 1
            
            if distributed_this_round == 0:
                print(f"Warning: Could not distribute any more images - stopping with {remaining_shortfall} remaining")
                break
        
        print(f"Successfully redistributed {total_shortfall - remaining_shortfall} images")
    
    # Now sample images according to final allocations
    selected_images = []
    actual_total = 0
    
    for location, target_count in sorted(location_allocations.items()):
        data = location_availability[location]
        available_count = data['available']
        images = data['images']
        
        if target_count > 0:
            if target_count >= available_count:
                # Take all available images
                selected_location_images = images
                selected_count = available_count
            else:
                # Sample evenly across available images
                step = available_count / target_count
                selected_location_images = []
                for j in range(target_count):
                    index = int(j * step)
                    selected_location_images.append(images[index])
                selected_count = len(selected_location_images)
            
            selected_images.extend(selected_location_images)
            actual_total += selected_count
        else:
            selected_count = 0
        
        print(f"{location}: {available_count} available, {selected_count} selected")
    
    print(f"Total selected {category_name} images: {actual_total}")
    
    # Final step: trim to exact target if we oversampled
    if len(selected_images) > target_total:
        print(f"Trimming from {len(selected_images)} to exactly {target_total} images")
        selected_images = selected_images[:target_total]
    
    return selected_images

def export_images(db_file: str, images_dir: str, output_dir: str, max_images: int = None, target_gang_gang: int = None, target_possum: int = None, include_locations: str = None, exclude_locations: str = None):
    """Export images from database to output directory with subdirectory organization"""
    
    # Connect to database
    if not os.path.exists(db_file):
        print(f"Error: Database file not found: {db_file}")
        return 1
    
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    try:
        # Get all images from database
        cursor.execute("SELECT id, file_path FROM images ORDER BY file_path")
        images = cursor.fetchall()
        
        total_images = len(images)
        print(f"Found {total_images} images in database")
        
        # Collect statistics by location
        location_stats = {}
        for image in images:
            file_path = image['file_path']
            # Extract location from first directory component of file path
            path_parts = file_path.split(os.sep)
            location = path_parts[0] if path_parts else file_path
            
            if location in location_stats:
                location_stats[location] += 1
            else:
                location_stats[location] = 1
        
        # Get detection counts by category and location
        def get_detection_stats_by_category(category):
            cursor.execute("""
                SELECT i.file_path, COUNT(*) as detection_count
                FROM images i
                JOIN detections d ON i.id = d.image_id
                WHERE d.category = ? AND d.deleted = 0 
                AND (d.hard = 0 OR d.hard IS NULL)
                AND d.confidence >= 0.20
                GROUP BY i.id, i.file_path
            """, (category,))
            results = cursor.fetchall()
            
            stats = {}
            for result in results:
                file_path = result['file_path']
                path_parts = file_path.split(os.sep)
                location = path_parts[0] if path_parts else file_path
                count = result['detection_count']
                
                if location in stats:
                    stats[location] += count
                else:
                    stats[location] = count
            return stats
        
        gang_gang_stats = get_detection_stats_by_category(1)
        possum_stats = get_detection_stats_by_category(4)
        
        # Print location statistics
        print("\nImage count by location:")
        print("-" * 60)
        for location, count in sorted(location_stats.items()):
            gang_gang_count = gang_gang_stats.get(location, 0)
            possum_count = possum_stats.get(location, 0)
            print(f"{location}: {count} images, {gang_gang_count} Gang Gang detections, {possum_count} Possum detections")
        
        print(f"\nTotal locations: {len(location_stats)}")
        total_gang_gangs = sum(gang_gang_stats.values())
        total_possums = sum(possum_stats.values())
        print(f"Total Gang Gang detections: {total_gang_gangs}")
        print(f"Total Possum detections: {total_possums}")
        
        # Apply location filtering
        filtered_location_stats = location_stats.copy()
        
        if include_locations:
            # Parse include list
            include_list = [loc.strip() for loc in include_locations.split(',')]
            # Only keep locations in the include list
            filtered_location_stats = {loc: count for loc, count in location_stats.items() if loc in include_list}
            print(f"\nIncluding only locations: {', '.join(include_list)}")
            print(f"Filtered to {len(filtered_location_stats)} locations")
            
        if exclude_locations:
            # Parse exclude list
            exclude_list = [loc.strip() for loc in exclude_locations.split(',')]
            # Remove locations in the exclude list
            filtered_location_stats = {loc: count for loc, count in filtered_location_stats.items() if loc not in exclude_list}
            print(f"\nExcluding locations: {', '.join(exclude_list)}")
            print(f"Filtered to {len(filtered_location_stats)} locations")
        
        # Update statistics for filtered locations
        if filtered_location_stats != location_stats:
            print(f"\nFiltered location statistics:")
            for location in sorted(filtered_location_stats.keys()):
                image_count = filtered_location_stats[location]
                gang_gang_count = gang_gang_stats.get(location, 0)
                possum_count = possum_stats.get(location, 0)
                print(f"{location}: {image_count} images, {gang_gang_count} Gang Gang detections, {possum_count} Possum detections")
        
        # Use filtered locations for sampling
        location_stats = filtered_location_stats
        
        # Sample images by category
        all_selected_images = []
        
        # Gang Gang sampling
        if target_gang_gang:
            gang_gang_images = sample_images_by_category(cursor, location_stats, 1, target_gang_gang, "Gang Gang")
            all_selected_images.extend(gang_gang_images)
        
        # Possum sampling
        if target_possum:
            possum_images = sample_images_by_category(cursor, location_stats, 4, target_possum, "Possum")
            all_selected_images.extend(possum_images)
        
        # Remove duplicates based on file_path (some images may have both categories)
        if all_selected_images:
            seen_paths = set()
            unique_images = []
            duplicates_removed = 0
            
            for image in all_selected_images:
                file_path = image['file_path']
                if file_path not in seen_paths:
                    seen_paths.add(file_path)
                    unique_images.append(image)
                else:
                    duplicates_removed += 1
            
            all_selected_images = unique_images
            if duplicates_removed > 0:
                print(f"\nRemoved {duplicates_removed} duplicate images (images with both categories)")
                print(f"Final unique image count: {len(all_selected_images)}")
        
        # Export all selected images if output directory is specified
        if output_dir and all_selected_images:
            os.makedirs(output_dir, exist_ok=True)
            
            print(f"\nExporting {len(all_selected_images)} total images to {output_dir}...")
            
            # Initialize metadata array
            metadata = []
            copied_count = 0
            
            for idx, image in enumerate(all_selected_images):
                image_id = image['id']
                src_file_path = image['file_path']
                
                # Get detections for this image
                cursor.execute("""
                    SELECT category, x, y, width, height, hard
                    FROM detections 
                    WHERE image_id = ? AND deleted = 0
                    AND (hard = 0 OR hard IS NULL)
                    AND confidence >= 0.20
                    ORDER BY confidence DESC
                """, (image_id,))
                detections_rows = cursor.fetchall()
                
                # Convert detections to list of dictionaries
                detections = []
                for det in detections_rows:
                    detections.append({
                        'category': det['category'],
                        'x': det['x'],
                        'y': det['y'],
                        'width': det['width'],
                        'height': det['height'],
                        'hard': bool(det['hard']) if det['hard'] is not None else False
                    })
                
                # Skip images with no valid detections
                if not detections:
                    print(f"Warning: No detections found for {src_file_path}, skipping")
                    continue
                
                # Construct full input path
                if os.path.isabs(src_file_path):
                    src_path = src_file_path
                else:
                    src_path = os.path.join(images_dir, src_file_path)
                
                # Check if source file exists
                if not os.path.exists(src_path):
                    print(f"Warning: Source file not found: {src_path}")
                    continue
                
                # Generate sequential filename: image_000001.jpg, image_000002.jpg, etc.
                dst_filename = f"image_{copied_count + 1:06d}.jpg"
                dst_path = os.path.join(output_dir, dst_filename)
                
                try:
                    # Copy the file
                    shutil.copy2(src_path, dst_path)
                    
                    # Add metadata entry
                    metadata.append({
                        'file_path': dst_filename,
                        'detections': detections
                    })
                    
                    copied_count += 1
                    
                    if copied_count % 100 == 0:
                        print(f"Exported {copied_count}/{len(all_selected_images)} images...")
                        
                except Exception as e:
                    print(f"Error copying {src_path} to {dst_path}: {e}")
                    continue
            
            # Write metadata to JSON file
            metadata_path = os.path.join(output_dir, 'metadata.json')
            try:
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                print(f"Metadata saved to: {metadata_path}")
            except Exception as e:
                print(f"Error writing metadata file: {e}")
            
            print(f"Successfully exported {copied_count} images")
            print(f"Total detections exported: {sum(len(img['detections']) for img in metadata)}")
        
        return 0
        
    except Exception as e:
        print(f"Database error: {e}")
        return 1
    
    finally:
        conn.close()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Export images from GGSort database')
    parser.add_argument('--db-file', required=True, help='SQLite database file')
    parser.add_argument('--images-dir', required=True, 
                        help='Base directory containing input image files')
    parser.add_argument('--output-dir', required=True,
                        help='Output directory for exported images')
    parser.add_argument('--max-images', type=int,
                        help='Maximum number of images to export')
    parser.add_argument('--target-gang-gang', type=int,
                        help='Target number of Gang Gang images to export, distributed evenly across locations')
    parser.add_argument('--target-possum', type=int,
                        help='Target number of Possum images to export, distributed evenly across locations')
    parser.add_argument('--include-locations', type=str,
                        help='Comma-separated list of locations to include (only these locations will be considered)')
    parser.add_argument('--exclude-locations', type=str,
                        help='Comma-separated list of locations to exclude from the dataset')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.db_file):
        print(f"Error: Database file not found: {args.db_file}")
        return 1
    
    if not os.path.exists(args.images_dir):
        print(f"Error: Images directory not found: {args.images_dir}")
        return 1
    
    if args.max_images and args.max_images <= 0:
        print(f"Error: --max-images must be a positive integer")
        return 1
    
    print(f"Exporting from database: {args.db_file}")
    print(f"Input images directory: {args.images_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.max_images:
        print(f"Maximum images: {args.max_images}")
    
    return export_images(args.db_file, args.images_dir, args.output_dir, args.max_images, args.target_gang_gang, args.target_possum, args.include_locations, args.exclude_locations)

if __name__ == "__main__":
    sys.exit(main())
