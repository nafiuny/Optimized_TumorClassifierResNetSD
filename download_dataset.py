import os
import gdown


file_urls = [
    "test_data.pt":   "https://drive.google.com/file/d/1_1bBU9HdHkXWVlFf3_6XKM-b91soBkzY/view?usp=sharing", 
    "train_data.pt":  "hhttps://drive.google.com/file/d/1FX6hqOOOT9ecEf2bZtISZmEz9Xg096EG/view?usp=sharing", 
    "val_data.pt":    "https://drive.google.com/file/d/17vkZUrEnC17XBJRtuMPTGM32jUZlyggv/view?usp=sharing"    
]

output_dir = 'outputs/data'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# دانلود هر فایل
for file_name, url in file_urls.items():
    output_path = os.path.join(output_dir, file_name)
    print(f"Downloading {file_name}...")
    gdown.download(url, output_path, quiet=False)

print("Dataset download completed!")
