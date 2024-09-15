import sys
import argparse

# if len(sys.argv) != 3:
#     print("Usage: python3 file --iters <n>")
#     sys.exit(1)

parser = argparse.ArgumentParser(
    description="Retrain models require additionall input from user."
)

parser.add_argument(
    "--iters",
    type=int,
    required=True,
    help="Number of iterations of re-training(required)",
)

args = parser.parse_args()

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch import nn
import torch
import function_lib as flib
import CNN_lib
from torch import optim
import time

if __name__ == "__main__":

    epoches = int(args.iters)

    loading_transform = transforms.Compose(
        [transforms.Grayscale(), transforms.ToTensor()]
    )

    # load the data
    train_dataset = datasets.ImageFolder(
        root="../data/Training", transform=loading_transform
    )
    validation_dataset = datasets.ImageFolder(
        root="../data/Validation", transform=loading_transform
    )
    test_dataset = datasets.ImageFolder(
        root="../data/Test", transform=loading_transform
    )

    # Make dataloaders
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    validation_dataloader = DataLoader(
        dataset=validation_dataset, batch_size=32, shuffle=True
    )
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)

    # models
    fcnn = CNN_lib.FullyConnectedNN()
    scn = CNN_lib.SimpleConvNN()
    emoNet = CNN_lib.EmotionNet()
    eda_cnn = CNN_lib.EDA_CNN()

    fcnn.load_state_dict(
        torch.load("../models/fullyConnectedNN.pth", weights_only=True)
    )
    scn.load_state_dict(torch.load("../models/simpleConvNN.pth", weights_only=True))
    emoNet.load_state_dict(torch.load("../models/emoNet.pth", weights_only=True))
    eda_cnn.load_state_dict(torch.load("../models/EDA_CNN.pth", weights_only=True))

    print("All models loaded succesfully!")

    loss_fn = nn.CrossEntropyLoss()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fcnn_optimizer = optim.Adam(params=fcnn.parameters(), lr=0.0001)
    scn_optimizer = optim.Adam(params=scn.parameters(), lr=0.0001)
    emoNet_optimizer = optim.Adam(params=emoNet.parameters(), lr=0.0001)
    eda_cnn_optimizer = optim.Adam(params=eda_cnn.parameters(), lr=0.0001)

    # RETRAIN ALL MODELS
    start_time = time.time()

    losses = []
    accs = []

    (losses, accs) = flib.train(
        fcnn,
        train_dataloader,
        validation_dataloader,
        loss_fn,
        fcnn_optimizer,
        losses=losses,
        accs=accs,
        num_epochs=epoches,
    )
    flib.save_model(model=fcnn, model_name="fullyConnectedNN", path="../models")
    print("=====================================================")

    losses = []
    accs = []

    (losses, accs) = flib.train(
        scn,
        train_dataloader,
        validation_dataloader,
        loss_fn,
        scn_optimizer,
        losses=losses,
        accs=accs,
        num_epochs=epoches,
    )
    flib.save_model(model=scn, model_name="simpleConvNN", path="../models")
    print("=====================================================")

    losses = []
    accs = []
    (losses, accs) = flib.train(
        emoNet,
        train_dataloader,
        validation_dataloader,
        loss_fn,
        emoNet_optimizer,
        losses=losses,
        accs=accs,
        num_epochs=epoches,
    )
    flib.save_model(model=emoNet, model_name="emoNet", path="../models")
    print("=====================================================")

    losses = []
    accs = []
    (losses, accs) = flib.train(
        eda_cnn,
        train_dataloader,
        validation_dataloader,
        loss_fn,
        eda_cnn_optimizer,
        losses=losses,
        accs=accs,
        num_epochs=epoches,
    )
    flib.save_model(model=eda_cnn, model_name="EDA_CNN", path="../models")
    print("=====================================================")

    print("All models re-trained successfully!")
    print(f"Elapsed time: {(time.time() - start_time) / 60} minutes.")
