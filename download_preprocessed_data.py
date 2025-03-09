import os
import gdown


# file_urls = {
#     "test_data.pt":   "https://drive.google.com/uc?id=1_1bBU9HdHkXWVlFf3_6XKM-b91soBkzY", 
#     "train_data.pt":  "hhttps://drive.google.com/uc?id=1FX6hqOOOT9ecEf2bZtISZmEz9Xg096EG", 
#     "val_data.pt":    "https://drive.google.com/uc?id=17vkZUrEnC17XBJRtuMPTGM32jUZlyggv"    
# }
file_urls = {
    "test_data.pt":   "https://drive.google.com/uc?id=1YRl08ewX8sdEFhGQNpHiVVHF3wBpzVfS",
    "train_data.pt":  "hhttps://drive.google.com/uc?id=1YQ5yAiGCKmxPuO19Aif3ai_AEEnAQ-4a",
    "val_data.pt":    "https://drive.google.com/uc?id=1YQYhxs90jMBSQiDPnhsQ7t17RfV0JCSC" 
}

output_dir = 'outputs/data'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


for file_name, url in file_urls.items():
    output_path = os.path.join(output_dir, file_name)
    print(f"Downloading {file_name}...")
    gdown.download(url, output_path, quiet=False)

print("Dataset download completed!")
