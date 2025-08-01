#!/usr/bin/env python3
"""
Test script for Green Context Simple Library
"""

def test_green_context():
    try:
        print("=== Testing Green Context Simple Library ===")
        
        # Import the library
        import green_context_simple as gc
        print(f"✅ Library imported successfully, version: {gc.__version__}")
        
        # Test 1: Using the class interface
        print("\n--- Test 1: Class Interface ---")
        manager = gc.GreenContextManager(device_id=0)
        print("✅ GreenContextManager created")
        
        # Create green context with 120 SMs for primary partition
        primary_stream, remaining_stream = manager.create_green_context_and_streams(120)
        print(f"✅ Green Context created:")
        print(f"   Primary stream handle: 0x{primary_stream:x}")
        print(f"   Remaining stream handle: 0x{remaining_stream:x}")
        
        # Get SM counts
        primary_sms, remaining_sms = manager.get_sm_counts()
        print(f"✅ SM distribution:")
        print(f"   Primary partition: {primary_sms} SMs")
        print(f"   Remaining partition: {remaining_sms} SMs")
        
        # Clean up
        manager.destroy_streams()
        print("✅ Streams destroyed")
        
        # Test 2: Using the standalone function
        print("\n--- Test 2: Standalone Function ---")
        primary_stream2, remaining_stream2 = gc.create_green_context_and_streams(
            intended_primary_partition_sm_count=100,
            primary_stream_priority=-1,
            remaining_stream_priority=0,
            device_id=0
        )
        print(f"✅ Standalone function works:")
        print(f"   Primary stream handle: 0x{primary_stream2:x}")
        print(f"   Remaining stream handle: 0x{remaining_stream2:x}")
        
        print("\n🎉 All tests passed!")
        
        # Usage example
        print("\n--- Python Usage Example ---")
        print("""
# Method 1: Using class (recommended for multiple operations)
import green_context_simple as gc

manager = gc.GreenContextManager()
primary_stream, remaining_stream = manager.create_green_context_and_streams(120)
# Use the streams...
primary_sms, remaining_sms = manager.get_sm_counts()
manager.destroy_streams()

# Method 2: Using standalone function (for one-time use)
primary, remaining = gc.create_green_context_and_streams(120)
        """)
        
    except ImportError as e:
        print(f"❌ Failed to import library: {e}")
        print("Make sure you have compiled the library with: bash build.sh")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_green_context()
    exit(0 if success else 1) 