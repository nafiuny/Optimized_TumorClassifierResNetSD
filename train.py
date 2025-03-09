import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import argparse
import os

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from models import TumorClassifierResNetSD

def get_model(model_name, num_classes, stochastic_depth1, stochastic_depth2):
    if model_name == "resnet_sd":
        return TumorClassifierResNetSD(num_classes=num_classes)
    
    else:
        raise ValueError(f"Model {model_name} is not defined")
        
def train_model(model_name, checkpoint_name, 
                train_data_path, train_labels_path, 
                val_data_path, val_labels_path, 
                num_epochs, lr, batch_size, 
                stochastic_depth1, stochastic_depth2):

    model = get_model(model_name=model_name, num_classes=4, 
                      stochastic_depth1=stochastic_depth1, 
                      stochastic_depth2=stochastic_depth2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_data = torch.load(train_data_path)
    train_labels = torch.load(train_labels_path)
    val_data = torch.load(val_data_path)
    val_labels = torch.load(val_labels_path)

    train_dataset = TensorDataset(train_data, train_labels)
    val_dataset = TensorDataset(val_data, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    start_epoch = 0
    
    checkpoint_path = f"outputs/models/{checkpoint_name}.pth"

    if os.path.exists(checkpoint_path):
      print(f"Resuming training from checkpoint: {checkpoint_path}")
      checkpoint = torch.load(checkpoint_path, map_location=device)
      model.load_state_dict(checkpoint["model_state_dict"])
      optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
      best_val_acc = checkpoint["best_val_acc"]

      # Preventing rewriting history
      prev_history = checkpoint["history"]
      history["train_loss"] = prev_history["train_loss"].copy()
      history["val_loss"] = prev_history["val_loss"].copy()
      history["train_acc"] = prev_history["train_acc"].copy()
      history["val_acc"] = prev_history["val_acc"].copy()

      start_epoch = len(history["train_loss"])  # Number of previous epochs

    for epoch in range(start_epoch,num_epochs):
        model.train()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += (outputs.argmax(1) == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects / len(train_loader.dataset)
        
        history["train_loss"].append(epoch_loss)
        history["train_acc"].append(epoch_acc)

        val_loss, val_corrects = 0.0, 0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
                val_corrects += (outputs.argmax(1) == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_corrects / len(val_loader.dataset)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        print(f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "history": history
            },  checkpoint_path)            
            print(f"Model saved at epoch {epoch + 1}, Path: {checkpoint_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True, help='Name of the model (resnet_sd)')
    parser.add_argument('--checkpoint_name', type=str, required=True, help='Name of the checkpoint file to save')
    parser.add_argument('--train_data_path', type=str, required=True, help='Path to training data tensor')
    parser.add_argument('--train_labels_path', type=str, required=True, help='Path to training labels tensor')
    parser.add_argument('--val_data_path', type=str, required=True, help='Path to validation data tensor')
    parser.add_argument('--val_labels_path', type=str, required=True, help='Path to validation labels tensor')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for DataLoader')
    parser.add_argument('--stochastic_depth1', type=float, default=0.6, help='Survival probability for first Stochastic Depth layer')
    parser.add_argument('--stochastic_depth2', type=float, default=0.7, help='Survival probability for second Stochastic Depth layer')
    args = parser.parse_args()

    train_model(
        model_name=args.model_name,
        checkpoint_name=args.checkpoint_name,
        train_data_path=args.train_data_path,
        train_labels_path=args.train_labels_path,
        val_data_path=args.val_data_path,
        val_labels_path=args.val_labels_path,
        num_epochs=args.num_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        stochastic_depth1=args.stochastic_depth1,
        stochastic_depth2=args.stochastic_depth2
    )
