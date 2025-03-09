import zipfile
import os
import gdown


google_drive_link = 'https://drive.google.com/uc?id=13OOP0r6coWO-dLGYQUMHMm6QEjgPuiuE'

dataset_file_path = 'brain-tumor-mri-dataset.zip' 
output_dir = 'Brain_Tumor_MRI_Dataset'  

gdown.download(google_drive_link, dataset_file_path, quiet=False)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

with zipfile.ZipFile(dataset_file_path, 'r') as zip_ref:
    zip_ref.extractall(output_dir)

print(f"Dataset has been extracted to: {output_dir}")
