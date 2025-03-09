import zipfile
import os
import gdown


google_drive_link = 'https://drive.google.com/uc?id=1eqPFtKy-a6og1J5GSgXOrlFq7xnZSTge'

best_model_file_path = 'best_model_resnet_sd.pth' 
output_dir = 'outputs/models'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


output_path = os.path.join(output_dir, best_model_file_path)
gdown.download(google_drive_link, output_path, quiet=False)


print("Best model download completed!")
