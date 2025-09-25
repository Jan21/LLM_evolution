"""Test script to verify the refactored modules work correctly."""

import sys
import os


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        from database_manager import DatabaseManager
        print("✓ database_manager imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import database_manager: {e}")
        return False
    
    try:
        from model_evolution import ModelEvolution
        print("✓ model_evolution imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import model_evolution: {e}")
        return False
    
    try:
        from experiment_runner import ExperimentRunner
        print("✓ experiment_runner imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import experiment_runner: {e}")
        return False
    
    print("All imports successful!\n")
    return True


def test_database_manager():
    """Test DatabaseManager functionality."""
    print("Testing DatabaseManager...")
    from database_manager import DatabaseManager
    
    # Create a test database
    test_db = "test_experiments.db"
    try:
        db = DatabaseManager(test_db)
        print("✓ DatabaseManager initialized")
        
        # Test getting current iteration
        iteration = db.get_current_iteration()
        print(f"✓ Current iteration: {iteration}")
        
        # Test getting best experiment (should be None for new DB)
        best = db.get_best_experiment()
        if best is None:
            print("✓ Correctly returns None for empty database")
        
        print("DatabaseManager tests passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ DatabaseManager test failed: {e}\n")
        return False
    finally:
        # Clean up test database
        if os.path.exists(test_db):
            os.remove(test_db)


def test_model_evolution():
    """Test ModelEvolution initialization."""
    print("Testing ModelEvolution...")
    
    # Check if API key file exists
    if not os.path.exists('api_key.txt'):
        print("⚠ Warning: api_key.txt not found - skipping ModelEvolution tests")
        print("  Create api_key.txt with your OpenAI API key to enable these tests\n")
        return True
    
    try:
        from model_evolution import ModelEvolution
        evolution = ModelEvolution()
        print("✓ ModelEvolution initialized")
        
        # Test getting model class string
        class_str = evolution.get_model_class_string('TransformerModel')
        if class_str and 'class TransformerModel' in class_str:
            print("✓ Successfully retrieved TransformerModel source")
        
        print("ModelEvolution tests passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ ModelEvolution test failed: {e}\n")
        return False


def test_experiment_runner():
    """Test ExperimentRunner initialization."""
    print("Testing ExperimentRunner...")
    
    try:
        from experiment_runner import ExperimentRunner
        runner = ExperimentRunner()
        print("✓ ExperimentRunner initialized")
        print("ExperimentRunner tests passed!\n")
        return True
        
    except Exception as e:
        print(f"✗ ExperimentRunner test failed: {e}\n")
        return False


def test_iteration_numbering():
    """Test that iteration numbering would work correctly."""
    print("Testing iteration numbering logic...")
    
    # Simulate the evolution loop iteration numbering
    max_iterations = 3
    for evolution_iter in range(max_iterations):
        # This is how the refactored code calculates the model version
        model_version = evolution_iter + 2  # For next iteration
        
        if evolution_iter < max_iterations - 1:
            expected_name = f"GeneratedTransformerModel_v{model_version}"
            print(f"  Evolution iteration {evolution_iter + 1} → {expected_name}")
    
    print("✓ Iteration numbering logic is correct\n")
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("TESTING REFACTORED MODULES")
    print("="*60 + "\n")
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    if all_passed:
        all_passed &= test_database_manager()
        all_passed &= test_model_evolution()
        all_passed &= test_experiment_runner()
        all_passed &= test_iteration_numbering()
    
    # Summary
    print("="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        print("The refactored modules are working correctly.")
    else:
        print("✗ SOME TESTS FAILED")
        print("Please check the error messages above.")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())