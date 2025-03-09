import os
import gdown

file_urls = {
    "test_data.pt":   "https://drive.google.com/uc?id=1-Hd8VBM906w_CHLkW13TY0zEy_v6eSOR",
    "train_data.pt":  "https://drive.google.com/uc?id=1YQ5yAiGCKmxPuO19Aif3ai_AEEnAQ-4a",
    "val_data.pt":    "https://drive.google.com/uc?id=1-XycI_M-QuCHtp42ztP3Xal5bE4S4NNP" 
}

output_dir = 'outputs/data'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


for file_name, url in file_urls.items():
    output_path = os.path.join(output_dir, file_name)
    print(f"Downloading {file_name}...")
    gdown.download(url, output_path, quiet=False)

print("Dataset download completed!")
