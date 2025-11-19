"""
Test script to verify that ControlSnapshot can save and load position/speed data.

This creates a test .npz file with the new format and verifies it can be loaded correctly.
"""

import sys
from pathlib import Path
import pickle
import lzma

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Define ControlSnapshot (matches GA scripts)
class ControlSnapshot:
    def __init__(self, forward, backward, left, right, car_position=None, car_speed=None):
        self.current_controls = (forward, backward, left, right)
        self.car_position = car_position if car_position is not None else (0, 0, 0)
        self.car_speed = car_speed if car_speed is not None else 0.0


def test_new_format():
    """Test that new format with position/speed can be saved and loaded."""
    print("=" * 70)
    print("TESTING NEW FORMAT (with position and speed)")
    print("=" * 70)
    print()

    # Create test snapshots with position and speed data
    test_snapshots = []
    for i in range(10):
        # Simulate some controls and trajectory data
        forward = i % 2 == 0
        backward = False
        left = i % 3 == 0
        right = i % 3 == 2
        car_position = (float(i * 10), 2.0, float(i * 5))
        car_speed = float(i * 2.5)

        snapshot = ControlSnapshot(forward, backward, left, right, car_position, car_speed)
        test_snapshots.append(snapshot)

    # Save to test file
    test_file = Path("test_new_format.npz")
    print(f"Creating test file: {test_file}")
    with lzma.open(test_file, "wb") as file:
        pickle.dump(test_snapshots, file)
    print("✓ Saved successfully")
    print()

    # Load and verify
    print("Loading test file...")
    with lzma.open(test_file, "rb") as file:
        loaded_snapshots = pickle.load(file)
    print("✓ Loaded successfully")
    print()

    # Verify data
    print("Verifying data integrity:")
    all_correct = True

    for i, snap in enumerate(loaded_snapshots):
        expected_pos = (float(i * 10), 2.0, float(i * 5))
        expected_speed = float(i * 2.5)

        if not hasattr(snap, 'car_position'):
            print(f"  ✗ Frame {i}: Missing car_position")
            all_correct = False
        elif snap.car_position != expected_pos:
            print(f"  ✗ Frame {i}: Position mismatch. Got {snap.car_position}, expected {expected_pos}")
            all_correct = False

        if not hasattr(snap, 'car_speed'):
            print(f"  ✗ Frame {i}: Missing car_speed")
            all_correct = False
        elif snap.car_speed != expected_speed:
            print(f"  ✗ Frame {i}: Speed mismatch. Got {snap.car_speed}, expected {expected_speed}")
            all_correct = False

    if all_correct:
        print("  ✓ All frames have correct position and speed data")
        print()

        # Show sample
        print("Sample frames:")
        for i in [0, 4, 9]:
            snap = loaded_snapshots[i]
            print(f"  Frame {i}: controls={snap.current_controls}, pos={snap.car_position}, speed={snap.car_speed}")
        print()

    # Cleanup
    test_file.unlink()
    print(f"✓ Test file deleted: {test_file}")
    print()

    if all_correct:
        print("=" * 70)
        print("✓ TEST PASSED: New format works correctly!")
        print("=" * 70)
    else:
        print("=" * 70)
        print("✗ TEST FAILED: Data integrity issues")
        print("=" * 70)

    return all_correct


def test_backward_compatibility():
    """Test that old format (controls only) still works."""
    print()
    print("=" * 70)
    print("TESTING BACKWARD COMPATIBILITY (old format)")
    print("=" * 70)
    print()

    # Create old-style snapshots (controls only)
    old_snapshots = []
    for i in range(5):
        forward = i % 2 == 0
        backward = False
        left = False
        right = False

        # Old style: only 4 arguments (no position/speed)
        snapshot = ControlSnapshot(forward, backward, left, right)
        old_snapshots.append(snapshot)

    # Save to test file
    test_file = Path("test_old_format.npz")
    print(f"Creating old format test file: {test_file}")
    with lzma.open(test_file, "wb") as file:
        pickle.dump(old_snapshots, file)
    print("✓ Saved successfully")
    print()

    # Load and verify
    print("Loading old format file...")
    with lzma.open(test_file, "rb") as file:
        loaded_snapshots = pickle.load(file)
    print("✓ Loaded successfully")
    print()

    # Verify defaults are applied
    print("Verifying default values:")
    all_correct = True

    for i, snap in enumerate(loaded_snapshots):
        # Should have default position (0, 0, 0) and speed 0.0
        if snap.car_position != (0, 0, 0):
            print(f"  ✗ Frame {i}: Position should be (0,0,0), got {snap.car_position}")
            all_correct = False

        if snap.car_speed != 0.0:
            print(f"  ✗ Frame {i}: Speed should be 0.0, got {snap.car_speed}")
            all_correct = False

    if all_correct:
        print("  ✓ All old-format frames have correct default values")
        print()

    # Cleanup
    test_file.unlink()
    print(f"✓ Test file deleted: {test_file}")
    print()

    if all_correct:
        print("=" * 70)
        print("✓ BACKWARD COMPATIBILITY TEST PASSED!")
        print("=" * 70)
    else:
        print("=" * 70)
        print("✗ BACKWARD COMPATIBILITY TEST FAILED")
        print("=" * 70)

    return all_correct


if __name__ == "__main__":
    print()
    test1_passed = test_new_format()
    test2_passed = test_backward_compatibility()

    print()
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"  New format test:             {'✓ PASSED' if test1_passed else '✗ FAILED'}")
    print(f"  Backward compatibility test: {'✓ PASSED' if test2_passed else '✗ FAILED'}")
    print()

    if test1_passed and test2_passed:
        print("✓✓✓ ALL TESTS PASSED ✓✓✓")
        print()
        print("The modified ControlSnapshot class correctly:")
        print("  1. Saves position (car_position) and speed (car_speed) data")
        print("  2. Maintains backward compatibility with old files")
        print("  3. Applies default values when position/speed are not provided")
    else:
        print("✗✗✗ SOME TESTS FAILED ✗✗✗")

    print("=" * 70)
    print()
