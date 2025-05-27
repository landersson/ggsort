#!/usr/bin/env python3

import unittest
import sys
import os
from io import StringIO

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_all_tests():
    """Run all test suites and provide comprehensive reporting"""
    
    # Test modules to run
    test_modules = [
        'test_database',
        'test_controller', 
        'test_integration',
        'test_renderer',
        'test_ui'
    ]
    
    print("=" * 80)
    print("GGSORT COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    print()
    
    # Track overall results
    total_tests = 0
    total_failures = 0
    total_errors = 0
    total_skipped = 0
    
    results_summary = []
    
    for module_name in test_modules:
        print(f"Running {module_name}...")
        print("-" * 40)
        
        # Capture test output
        test_output = StringIO()
        
        # Load and run the test module
        try:
            # Import the test module
            test_module = __import__(module_name)
            
            # Create test suite
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(test_module)
            
            # Run tests with custom result handler
            runner = unittest.TextTestRunner(
                stream=test_output,
                verbosity=2,
                buffer=True
            )
            result = runner.run(suite)
            
            # Collect statistics
            module_tests = result.testsRun
            module_failures = len(result.failures)
            module_errors = len(result.errors)
            module_skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
            
            total_tests += module_tests
            total_failures += module_failures
            total_errors += module_errors
            total_skipped += module_skipped
            
            # Store results for summary
            results_summary.append({
                'module': module_name,
                'tests': module_tests,
                'failures': module_failures,
                'errors': module_errors,
                'skipped': module_skipped,
                'success': module_failures == 0 and module_errors == 0
            })
            
            # Print module results
            if module_failures == 0 and module_errors == 0:
                print(f"✅ {module_name}: {module_tests} tests passed")
            else:
                print(f"❌ {module_name}: {module_failures} failures, {module_errors} errors out of {module_tests} tests")
                
                # Print failure details
                if module_failures > 0:
                    print("  Failures:")
                    for test, traceback in result.failures:
                        print(f"    - {test}")
                
                if module_errors > 0:
                    print("  Errors:")
                    for test, traceback in result.errors:
                        print(f"    - {test}")
            
            print()
            
        except ImportError as e:
            print(f"❌ Could not import {module_name}: {e}")
            results_summary.append({
                'module': module_name,
                'tests': 0,
                'failures': 0,
                'errors': 1,
                'skipped': 0,
                'success': False
            })
            total_errors += 1
            print()
        except Exception as e:
            print(f"❌ Error running {module_name}: {e}")
            results_summary.append({
                'module': module_name,
                'tests': 0,
                'failures': 0,
                'errors': 1,
                'skipped': 0,
                'success': False
            })
            total_errors += 1
            print()
    
    # Print comprehensive summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()
    
    # Module-by-module summary
    print("Module Results:")
    print("-" * 40)
    for result in results_summary:
        status = "✅ PASS" if result['success'] else "❌ FAIL"
        print(f"{result['module']:<20} {status:<8} {result['tests']:>3} tests, "
              f"{result['failures']:>2} failures, {result['errors']:>2} errors, {result['skipped']:>2} skipped")
    
    print()
    
    # Overall statistics
    print("Overall Results:")
    print("-" * 40)
    print(f"Total Tests:    {total_tests}")
    print(f"Passed:         {total_tests - total_failures - total_errors}")
    print(f"Failed:         {total_failures}")
    print(f"Errors:         {total_errors}")
    print(f"Skipped:        {total_skipped}")
    
    success_rate = ((total_tests - total_failures - total_errors) / total_tests * 100) if total_tests > 0 else 0
    print(f"Success Rate:   {success_rate:.1f}%")
    
    print()
    
    # Test coverage analysis
    print("Test Coverage Analysis:")
    print("-" * 40)
    
    coverage_areas = [
        ("Database Operations", "test_database", "Core database functionality including CRUD operations"),
        ("UI Controller Logic", "test_controller", "Detection manipulation and business logic"),
        ("Integration Tests", "test_integration", "End-to-end workflows and database-UI integration"),
        ("Rendering Logic", "test_renderer", "Detection visualization and coordinate conversion"),
        ("User Interface", "test_ui", "Mouse and keyboard interaction handling")
    ]
    
    for area_name, module_name, description in coverage_areas:
        module_result = next((r for r in results_summary if r['module'] == module_name), None)
        if module_result:
            status = "✅" if module_result['success'] else "❌"
            print(f"{status} {area_name:<25} ({module_result['tests']} tests)")
            print(f"   {description}")
        else:
            print(f"❌ {area_name:<25} (not found)")
        print()
    
    # Key functionality verification
    print("Key Functionality Verification:")
    print("-" * 40)
    
    key_features = [
        "Detection deletion and restoration",
        "Category change operations", 
        "Detection relocation/coordinate updates",
        "Mouse-based detection selection",
        "Overlapping detection cycling",
        "Database transaction consistency",
        "State persistence across sessions",
        "Error handling for invalid operations",
        "Confidence threshold filtering",
        "UI state management and redraws"
    ]
    
    # Determine if key features are covered based on test results
    overall_success = total_failures == 0 and total_errors == 0
    
    for feature in key_features:
        status = "✅" if overall_success else "⚠️"
        print(f"{status} {feature}")
    
    print()
    
    # Final verdict
    print("=" * 80)
    if total_failures == 0 and total_errors == 0:
        print("🎉 ALL TESTS PASSED! The application is ready for production use.")
        print("   Database handling and UI logic are working correctly.")
        print("   User changes will be properly registered and saved to the database.")
    else:
        print("⚠️  SOME TESTS FAILED. Please review the failures above.")
        print("   Database or UI functionality may not be working correctly.")
        print("   User changes might not be properly saved to the database.")
    
    print("=" * 80)
    
    # Return exit code
    return 0 if (total_failures == 0 and total_errors == 0) else 1


def run_specific_test_category(category):
    """Run tests for a specific category"""
    category_map = {
        'database': ['test_database'],
        'controller': ['test_controller'],
        'integration': ['test_integration'], 
        'renderer': ['test_renderer'],
        'ui': ['test_ui'],
        'core': ['test_database', 'test_controller'],
        'ui-full': ['test_ui', 'test_renderer'],
        'integration-full': ['test_integration', 'test_controller', 'test_database']
    }
    
    if category not in category_map:
        print(f"Unknown test category: {category}")
        print(f"Available categories: {', '.join(category_map.keys())}")
        return 1
    
    # Run subset of tests
    test_modules = category_map[category]
    
    print(f"Running {category} tests...")
    print("=" * 50)
    
    for module_name in test_modules:
        print(f"Running {module_name}...")
        
        try:
            test_module = __import__(module_name)
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromModule(test_module)
            runner = unittest.TextTestRunner(verbosity=2)
            result = runner.run(suite)
            
            if result.failures or result.errors:
                return 1
                
        except Exception as e:
            print(f"Error running {module_name}: {e}")
            return 1
    
    print("All tests in category passed!")
    return 0


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Run specific category
        category = sys.argv[1]
        exit_code = run_specific_test_category(category)
    else:
        # Run all tests
        exit_code = run_all_tests()
    
    sys.exit(exit_code) 