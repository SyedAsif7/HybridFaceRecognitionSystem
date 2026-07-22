
def fix_pickle(input_path, output_path):
    with open(input_path, 'rb') as f:
        data = f.read()
    fixed_data = data.replace(b'\r', b'')
    with open(output_path, 'wb') as f:
        f.write(fixed_data)
    print(f"Fixed {input_path} -> {output_path}")

fix_pickle('weights/svm_rbf.pkl', 'weights/svm_rbf_fixed.pkl')
fix_pickle('weights/svm_linear.pkl', 'weights/svm_linear_fixed.pkl')

import os
os.replace('weights/svm_rbf_fixed.pkl', 'weights/svm_rbf.pkl')
os.replace('weights/svm_linear_fixed.pkl', 'weights/svm_linear.pkl')
print("Replaced original pickle files with fixed versions")
