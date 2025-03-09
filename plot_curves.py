import os
import torch
import matplotlib
import matplotlib.pyplot as plt
import argparse

def plot_training_curves(checkpoint_path, output_dir="outputs/plots"):
    checkpoint = torch.load(checkpoint_path)
    if "history" not in checkpoint:
        raise ValueError("Checkpoint does not contain training history.")
    
    history = checkpoint["history"]
    if not all(key in history for key in ["train_loss", "val_loss", "train_acc", "val_acc"]):
        raise ValueError("Training history is missing required keys (train_loss, val_loss, train_acc, val_acc).")
    
    train_loss = history["train_loss"]
    val_loss = history["val_loss"]
    train_acc = history["train_acc"]
    val_acc = history["val_acc"]
    
    model_name = os.path.basename(checkpoint_path).split('.')[0]
    

    os.makedirs(output_dir, exist_ok=True)
    
    #  Loss
    plt.figure()
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Train Loss", marker="o")
    plt.plot(range(1, len(val_loss) + 1), val_loss, label="Validation Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid()
    loss_plot_path = os.path.join(output_dir, f"Train_loss_{model_name}.png")
    plt.savefig(loss_plot_path)
    print(f"Loss plot saved at {loss_plot_path}")
    plt.close()
    
    # Accuracy
    plt.figure()
    plt.plot(range(1, len(train_acc) + 1), train_acc, label="Train Accuracy", marker="o")
    plt.plot(range(1, len(val_acc) + 1), val_acc, label="Validation Accuracy", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid()
    acc_plot_path = os.path.join(output_dir, f"Train_acc_{model_name}.png")
    plt.savefig(acc_plot_path)
    print(f"Accuracy plot saved at {acc_plot_path}")
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Training Curves from Checkpoint")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the checkpoint file')
    parser.add_argument('--output_dir', type=str, default="outputs/plots", help='Directory to save the plots')
    args = parser.parse_args()
    
    plot_training_curves(args.checkpoint_path, args.output_dir)

