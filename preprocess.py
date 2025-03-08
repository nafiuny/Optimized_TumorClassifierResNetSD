import os
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
import argparse
import random

def process_and_save_images(images, labels, output_data_path, output_label_path, batch_size=500):
    """
    Progressive processing and storage of images as tensors.
    """
    tensors = []
    tensor_labels = []
    for idx, (image_path, label) in enumerate(zip(images, labels)):
        image = Image.open(image_path).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        image_tensor = transform(image)
        tensors.append(image_tensor)
        tensor_labels.append(label)

        if (idx + 1) % batch_size == 0 or (idx + 1) == len(images):
            torch.save(torch.stack(tensors), f"{output_data_path}_part_{idx // batch_size}.pt")
            torch.save(torch.tensor(tensor_labels), f"{output_label_path}_part_{idx // batch_size}.pt")
            tensors = []
            tensor_labels = []
            print(f"Processed and saved batch {idx // batch_size + 1}")

def merge_tensors(output_data_path, output_label_path, num_batches):
    """
    Merge tensor files into one final file.
    """
    data_tensors = []
    label_tensors = []
    for i in range(num_batches):
        data_tensors.append(torch.load(f"{output_data_path}_part_{i}.pt"))
        label_tensors.append(torch.load(f"{output_label_path}_part_{i}.pt"))

    merged_data = torch.cat(data_tensors)
    merged_labels = torch.cat(label_tensors)

    torch.save(merged_data, f"{output_data_path}.pt")
    torch.save(merged_labels, f"{output_label_path}.pt")

    # Delete partial files
    for i in range(num_batches):
        os.remove(f"{output_data_path}_part_{i}.pt")
        os.remove(f"{output_label_path}_part_{i}.pt")

    print(f"Merged tensors saved to {output_data_path}.pt and {output_label_path}.pt")

def preprocess_data(train_dir, test_dir, output_dir):
    classes = ["glioma", "meningioma", "pituitary", "notumor"]

    def collect_images_labels(data_dir):
        images, labels = [], []
        for label, class_name in enumerate(classes):
            class_path = os.path.join(data_dir, class_name)
            for image_name in os.listdir(class_path):
                images.append(os.path.join(class_path, image_name))
                labels.append(label)
        return images, labels

    train_images, train_labels = collect_images_labels(train_dir)
    test_images, test_labels = collect_images_labels(test_dir)

    combined = list(zip(train_images, train_labels))
    random.shuffle(combined)
    train_images, train_labels = zip(*combined)

    train_images, val_images, train_labels, val_labels = train_test_split(
        train_images, train_labels, test_size=0.2, stratify=train_labels, random_state=42
    )

    os.makedirs(output_dir, exist_ok=True)
    process_and_save_images(train_images, train_labels, os.path.join(output_dir, "train_data"), os.path.join(output_dir, "train_labels"))
    process_and_save_images(val_images, val_labels, os.path.join(output_dir, "val_data"), os.path.join(output_dir, "val_labels"))
    process_and_save_images(test_images, test_labels, os.path.join(output_dir, "test_data"), os.path.join(output_dir, "test_labels"))

    merge_tensors(os.path.join(output_dir, "train_data"), os.path.join(output_dir, "train_labels"), len(train_images) // 500 + 1)
    merge_tensors(os.path.join(output_dir, "val_data"), os.path.join(output_dir, "val_labels"), len(val_images) // 500 + 1)
    merge_tensors(os.path.join(output_dir, "test_data"), os.path.join(output_dir, "test_labels"), len(test_images) // 500 + 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training data directory')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to testing data directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save processed data')
    args = parser.parse_args()

    preprocess_data(args.train_dir, args.test_dir, args.output_dir)