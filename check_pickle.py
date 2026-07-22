
import os
import pickle

def inspect_pickle(path):
    print(f"Checking {path}")
    print(f"Size: {os.path.getsize(path)} bytes")
    with open(path, 'rb') as f:
        data = f.read()
        print(f"First 20 bytes: {data[:20]}")
        print(f"Has \\r (0x0d): {b'\x0d' in data}")
    try:
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        print("Successfully loaded!")
        print(f"Type: {type(obj)}")
        print(f"Keys: {list(obj.keys()) if hasattr(obj, 'keys') else obj}")
    except Exception as e:
        print(f"Error loading: {type(e)} - {e}")

inspect_pickle('weights/svm_rbf.pkl')
print("\n" + "="*50 + "\n")
inspect_pickle('weights/svm_linear.pkl')
