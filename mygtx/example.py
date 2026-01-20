#!/usr/bin/env python3
"""
Green Context Simple Library Usage Example

This example demonstrates how to use the Green Context library 
that was compiled from C++ with Python bindings.
"""

import sys
import os

def test_green_context_simple():
    """Test the Green Context Simple library"""
    
    try:
        # Import the compiled library
        print("=== Green Context Simple Library Example ===")
        print("Importing green_context_simple library...")
        
        import green_context_simple as gc
        print(f"✅ Library imported successfully!")
        print(f"📦 Library version: {gc.__version__}")
        print()
        
        # Test 1: Using the class interface
        print("--- Test 1: Class Interface (GreenContextManager) ---")
        
        # Create a Green Context manager for device 0
        print("Creating GreenContextManager for device 0...")
        manager = gc.GreenContextManager(device_id=0)
        print("✅ GreenContextManager created successfully!")
        
        # Create Green Context and streams
        print("\nCreating Green Context with 120 SMs for primary partition...")
        intended_sm_count = 120
        primary_priority = -1  # Higher priority
        remaining_priority = 0  # Normal priority
        
        primary_stream, remaining_stream = manager.create_green_context_and_streams(
            intended_primary_partition_sm_count=intended_sm_count,
            primary_stream_priority=primary_priority,
            remaining_stream_priority=remaining_priority
        )
        
        print("✅ Green Context and streams created successfully!")
        print(f"   🔵 Primary partition stream handle:   0x{primary_stream:016x}")
        print(f"   🟢 Remaining partition stream handle: 0x{remaining_stream:016x}")
        
        # Get SM distribution information
        print("\nQuerying SM distribution...")
        primary_sms, remaining_sms = manager.get_sm_counts()
        total_sms = primary_sms + remaining_sms
        
        print(f"✅ SM Distribution:")
        print(f"   📊 Total SMs:     {total_sms}")
        print(f"   🔵 Primary SMs:   {primary_sms} ({primary_sms/total_sms*100:.1f}%)")
        print(f"   🟢 Remaining SMs: {remaining_sms} ({remaining_sms/total_sms*100:.1f}%)")
        
        # Clean up
        print("\nDestroying streams...")
        manager.destroy_streams()
        print("✅ Streams destroyed successfully!")
        print()
        
        # Test 2: Using the standalone function
        print("--- Test 2: Standalone Function Interface ---")
        
        print("Creating Green Context using standalone function...")
        intended_sm_count = 100
        primary_priority = -2  # Even higher priority
        remaining_priority = 1  # Lower priority
        
        primary_stream2, remaining_stream2 = gc.create_green_context_and_streams(
            intended_primary_partition_sm_count=intended_sm_count,
            primary_stream_priority=primary_priority,
            remaining_stream_priority=remaining_priority,
            device_id=0
        )
        
        print("✅ Standalone function works!")
        print(f"   🔵 Primary partition stream handle:   0x{primary_stream2:016x}")
        print(f"   🟢 Remaining partition stream handle: 0x{remaining_stream2:016x}")
        print()
        
        # Test 3: Stream handle usage example
        print("--- Test 3: Stream Handle Usage Example ---")
        print("Note: Stream handles can be used with other CUDA libraries")
        print("For example, you could:")
        print(f"  - Use primary stream (0x{primary_stream:x}) for high-priority compute kernels")
        print(f"  - Use remaining stream (0x{remaining_stream:x}) for background/communication tasks")
        print(f"  - Pass these handles to CuPy, Numba, or other CUDA Python libraries")
        print()
        
        # Success!
        print("🎉 All tests completed successfully!")
        print()
        print("💡 Tips for production use:")
        print("   1. Keep the GreenContextManager instance alive while using streams")
        print("   2. Always call destroy_streams() when done to free resources")
        print("   3. Stream handles are 64-bit integers that can be passed to other CUDA libraries")
        print("   4. Consider using context managers (with statements) for automatic cleanup")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import green_context_simple: {e}")
        print()
        print("Troubleshooting:")
        print("1. Make sure the .so file is in the current directory or Python path")
        print("2. Check that the .so file was compiled for the correct Python version")
        print("3. Verify CUDA drivers and libraries are available")
        return False
        
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False

def advanced_usage_example():
    """Advanced usage example with context manager pattern"""
    
    print("\n" + "="*60)
    print("Advanced Usage: Context Manager Pattern")
    print("="*60)
    
    try:
        import green_context_simple as gc
        
        class GreenContextManager:
            """Context manager wrapper for automatic resource cleanup"""
            
            def __init__(self, device_id=0):
                self.device_id = device_id
                self.manager = None
                self.streams_created = False
                
            def __enter__(self):
                self.manager = gc.GreenContextManager(self.device_id)
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.streams_created:
                    self.manager.destroy_streams()
                    
            def create_streams(self, sm_count, primary_priority=0, remaining_priority=0):
                """Create streams and return handles"""
                result = self.manager.create_green_context_and_streams(
                    sm_count, primary_priority, remaining_priority)
                self.streams_created = True
                return result
                
            def get_sm_counts(self):
                """Get SM distribution"""
                return self.manager.get_sm_counts()
        
        # Use with context manager for automatic cleanup
        print("Using context manager for automatic resource management...")
        with GreenContextManager(device_id=0) as gc_mgr:
            # Create streams
            primary_stream, remaining_stream = gc_mgr.create_streams(
                sm_count=150, 
                primary_priority=-1, 
                remaining_priority=0
            )
            
            print(f"✅ Streams created with automatic cleanup:")
            print(f"   Primary: 0x{primary_stream:x}")
            print(f"   Remaining: 0x{remaining_stream:x}")
            
            # Get SM info
            p_sms, r_sms = gc_mgr.get_sm_counts()
            print(f"   SM distribution: {p_sms} + {r_sms} = {p_sms + r_sms}")
            
            # Streams will be automatically destroyed when exiting the 'with' block
            
        print("✅ Context manager automatically cleaned up resources!")
        
    except ImportError:
        print("❌ green_context_simple not available for advanced example")

if __name__ == "__main__":
    print("Green Context Simple Library - Usage Example")
    print("=" * 50)
    
    # Check if we're in the right directory
    so_files = [f for f in os.listdir('.') if f.startswith('green_context_simple') and f.endswith('.so')]
    if so_files:
        print(f"📁 Found library file: {so_files[0]}")
    else:
        print("⚠️  No .so file found in current directory")
        print("   Make sure you're running this from the directory containing the compiled library")
    
    print()
    
    # Run basic tests
    success = test_green_context_simple()
    
    if success:
        # Run advanced example
        advanced_usage_example()
        
        print("\n" + "="*60)
        print("🎉 Example completed successfully!")
        print("You can now use green_context_simple in your own projects!")
        print("="*60)
    else:
        print("\n❌ Basic tests failed. Please check the library compilation and environment.")
        sys.exit(1) 