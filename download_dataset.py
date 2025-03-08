import os
import gdown


file_urls = [
    "https://drive.google.com/uc?id=FILE_ID_1",  # لینک فایل 1
    "https://drive.google.com/uc?id=FILE_ID_2",  # لینک فایل 2
    "https://drive.google.com/uc?id=FILE_ID_3"   # لینک فایل 3
]

# پوشه مقصد برای ذخیره‌سازی
output_dir = './data'

# اگر پوشه مقصد وجود ندارد، آن را ایجاد می‌کنیم
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# دانلود فایل‌ها
for url in file_urls:
    # نام فایل را از URL استخراج می‌کنیم
    file_name = url.split('=')[-1] + '.zip'  # یا فرمت صحیح فایل شما
    output_path = os.path.join(output_dir, file_name)
    
    # دانلود فایل
    print(f"Downloading {file_name}...")
    gdown.download(url, output_path, quiet=False)

print("Dataset download completed!")
