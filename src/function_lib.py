from torch.utils.data import DataLoader
from torch import nn
import torch
import os
import numpy as np


def train_loop(dataloader: DataLoader, model: nn.Module, loss_fn, optimizer, device):
    model.train()
    for X, y in dataloader:
        X = X.to(device)
        y = y.to(device)

        preds = model(X)
        loss = loss_fn(preds, y)

        # radimo backpropagation - racunamo gradijente
        loss.backward()

        # ovde radimo X_new = x - lr * grad
        optimizer.step()

        # ne zelimo da sabiramo sve gradijente zato ih ponistimo posle svake iteracije
        optimizer.zero_grad()


def test_loop(dataloader: DataLoader, model: nn.Module, loss_fn, device):
    # moramo da naglasimo da testiramo
    model.eval()
    with torch.no_grad():
        # racunamo total loss i broj pogodjenih predvidjanja
        total_loss = 0
        num_same = 0

        # idemo kroz loader i provlacimo instance kroz model i dobijemo predikciju
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            outputs = model(X)
            loss = loss_fn(outputs, y)
            total_loss += loss.item()

            _, indices = torch.max(outputs, 1)
            num_same += sum(indices == y).item()

        print(f"Total loss: {total_loss}")
        acc = num_same / len(dataloader.dataset)
        print(f"Accuracy: {acc}")

        return total_loss, acc


def save_model(model: nn.Module, model_name: str, path: str = "../models"):
    extension = ".pth"
    if not os.path.exists(path="../models"):
        os.mkdir(path="../models")
        print("Created a models directory")
    else:
        print("models directory already exists")

    print("Saving model...")
    torch.save(model, os.path.join(path, model_name + extension))
    print(f"Model '{model_name}' saved successfully.")


# BUGFIX: not returning actual model whe being called from kernel
# def load_model(path: str) -> nn.Module:
#     if os.path.exists(path):
#         print(f"Loading model: {path.split('/')[-1]}")
#         model = torch.load(path)
#         return model
#     else:
#         print(f"Cant find the file: {path}")
#         print("Loading aborted")


def print_trainable_params(model: nn.Module):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {trainable_params}")


def train(
    model: nn.Module,
    train_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    loss_fn,
    optimizer,
    losses,
    accs,
    num_epochs: int = 2,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    model.to(device)
    for epoch in range(num_epochs):
        print(f"Training epoch: {epoch+1}...")
        train_loop(
            dataloader=train_dataloader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        (loss, acc) = test_loop(
            dataloader=validation_dataloader,
            model=model,
            loss_fn=loss_fn,
            device=device,
        )
        losses.append(loss)
        accs.append(acc)

    return losses, accs
