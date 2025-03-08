import zipfile
import os


dataset_file_path = 'brain-tumor-mri-dataset.zip' 
output_dir = 'Brain_Tumor_MRI_Dataset'  
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# extract
with zipfile.ZipFile(dataset_file_path, 'r') as zip_ref:
    zip_ref.extractall(output_dir)

print(f"Dataset has been extracted to: {output_dir}")

