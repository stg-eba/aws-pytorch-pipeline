import sys
import os

print("--- PYTHON PATH DEBUGGING ---")
print(f"Current Working Directory: {os.getcwd()}")

try:
    import sagemaker
    print(f"\n✅ FOUND 'sagemaker' module.")
    print(f"📁 LOCATION: {sagemaker.__file__}")  # <--- THIS IS THE ANSWER
    
    # Check if it's a folder or file
    if hasattr(sagemaker, '__path__'):
        print(f"📂 PACKAGE PATH: {sagemaker.__path__}")
        
    print(f"❓ Has __version__? {hasattr(sagemaker, '__version__')}")
    
except ImportError:
    print("\n❌ Could not import sagemaker at all.")
except Exception as e:
    print(f"\n❌ Error during import: {e}")

print("\n--- SEARCH PATHS (sys.path) ---")
for p in sys.path:
    print(p)