import sys
import os
import time  # Import time for measuring inference time

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import argparse
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from IPython.display import Image
from sklearn.metrics import precision_score, recall_score, f1_score

from models import TumorClassifierResNet, TumorClassifierResNetSD

def get_model(model_name, num_classes):
    # if model_name == "resnet":
    #     return TumorClassifierResNet(num_classes=num_classes)
    if model_name == "resnet_sd":
        return TumorClassifierResNetSD(num_classes=num_classes)
    
    else:
        raise ValueError(f"Model {model_name} is not defined")

def evaluate_model(model_name, checkpoint_path, test_data_path, test_labels_path, batch_size=32):
    # Load model
    model = get_model(model_name, num_classes=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load test data and labels
    test_data = torch.load(test_data_path)
    test_labels = torch.load(test_labels_path)

    test_dataset = TensorDataset(test_data, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Evaluate the model
    all_preds = []
    all_labels = []
    correct = 0
    total = 0

    # Initialize variables for inference time measurement
    inference_time_total = 0.0
    num_batches = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Synchronize GPU before starting timing
            torch.cuda.synchronize()
            start_time = time.time()


            outputs = model(inputs)

            torch.cuda.synchronize()
            end_time = time.time()

            inference_time_total += (end_time - start_time)
            num_batches += 1

            preds = outputs.argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    # Generate classification report
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Glioma", "Meningioma", "Pituitary", "No Tumor"]))

    accuracy = correct / total    
    precision = precision_score(all_labels, all_preds, average="macro") 
    recall = recall_score(all_labels, all_preds, average="macro")
    f1 = f1_score(all_labels, all_preds, average="macro")
    
    print()
    print()
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")


    # Generate confusion matrix
    matplotlib.use('Agg')
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Glioma", "Meningioma", "Pituitary", "No Tumor"], yticklabels=["Glioma", "Meningioma", "Pituitary", "No Tumor"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plot_path = os.path.join("outputs/plots", "confusion_matrix_{model_name}.png")
    plt.savefig(plot_path)
    
    print("Confusion Matrix saved at: outputs/plots/confusion_matrix_{model_name}.png")

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model (e.g., cnn)')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the saved model checkpoint')
    parser.add_argument('--test_data_path', type=str, required=True, help='Path to test data tensor')
    parser.add_argument('--test_labels_path', type=str, required=True, help='Path to test labels tensor')
    args = parser.parse_args()

    evaluate_model(
        model_name=args.model_name,
        checkpoint_path=args.checkpoint_path,
        test_data_path=args.test_data_path,
        test_labels_path=args.test_labels_path
    )
