#Scott Landry
# video_classifier.py

import os
import time
import random
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from deepaction_extract import Model_Data, Video
from dataset import CNN, ClipData
import config

# Set random seed for reproducibility (if defined)
if hasattr(config, 'RANDOM_SEED'):
    random.seed(config.RANDOM_SEED)

# transforms from the torchvision library
tfms = transforms.Compose([
    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
])

# net: the neural network model
# loader: DataLoader for training data
# loss_fn: loss function (CrossEntropyLoss)
# optim: optimizer (Adam)
# dev: device (CPU or GPU)
def train_epoch(net, loader, loss_fn, optim, dev):
    net.train()
    correct = 0
    total = 0

    for i, (clips, targets) in enumerate(loader):
        clips   = clips.to(dev)
        targets = targets.to(dev).long()  # ensure labels are long integers

        optim.zero_grad()
        logits = net(clips)
        loss   = loss_fn(logits, targets)
        loss.backward()
        optim.step()

        preds = logits.argmax(1)
        correct += (preds == targets).sum().item()
        total   += targets.size(0)

        if i % 10 == 0:
            loss_val = loss.item()
            last_acc = acc
            acc = correct / total
            delta = acc - last_acc 
            #nice c style print statement for flair
            print("\tbatch %03d → loss %.4f | acc %.3f | change %.4f" % (i, loss_val, acc, delta))


            """
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1024**3
                print(f"    GPU memory: {allocated:.2f} GB")
            """
        
        # Clear references to free memory
        del logits, loss, preds
        clips = None  # Help GC

    if total == 0:
        print("warning: no training data processed")
        return 0.0
    
    return correct / total


# model: the neural network (nn) model
# entropy_loss: loss function (CrossEntropyLoss)
# device: CPU or GPU
# val_loader: DataLoader for validation data
def validate(model, entropy_loss, device, val_loader):
    model.eval()
    total_predictions = 0
    correct_predictions = 0
    incorrect_predictions = 0
    
    for i, (videos, labels) in enumerate(val_loader):
        videos = videos.to(device)
        labels = labels.to(device).long()  # ensure labels are long integers
        
        with torch.no_grad():
            outputs = model(videos)
            loss = entropy_loss(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct = (predicted == labels)
        correct_predictions += correct.sum().item()
        incorrect = (predicted != labels)
        incorrect_predictions += incorrect.sum().item()

        if i % 10 == 0:
            print(f"\tvalidation batch: {i} loss: {loss.item()}")

    if total_predictions == 0:
        print("warning: no validation data processed")
        return 0.0
    
    accuracy = correct_predictions / total_predictions
    print(f"accuracy: {accuracy}")
    return accuracy


#Driver code
if __name__ == "__main__":
    print("Indexing folders...")
    all_classes = []
    total_count = 0
    class_index = 0  # separate counter that only increments for actual directories

    for folder_name in sorted(os.listdir(config.ROOT)):
        full_dir = os.path.join(config.ROOT, folder_name)
        if not os.path.isdir(full_dir):
            continue     # only directories are allowed

        print(f"  → {folder_name:<20}", end="")
        container = Model_Data(model_index=class_index, model_path=full_dir, sample_size=config.NUM_FRAMES)
        n = container.get_size()
        print(f"{n} clips (class {class_index})")
        class_index += 1  # only increment when we actually create a Model_Data object
        total_count += n
        all_classes.append(container)

    print(f"\nGot {total_count} videos from {len(all_classes)} classes total\n")

    train_list = []
    val_list   = []
    test_list  = []

    for c in all_classes:
        train_list.extend(c.get_training_videos())
        val_list.extend(c.get_validation_videos())
        test_list.extend(c.get_test_videos())

    print(f"Train: {len(train_list)}")
    print(f"Val:   {len(val_list)}")
    print(f"Test:  {len(test_list)}\n")

    train_ds = ClipData(train_list, transform=tfms)
    val_ds   = ClipData(val_list,   transform=tfms)
    test_ds  = ClipData(test_list,  transform=tfms)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using %s\n" % device)

    # Clear GPU cache before training just in case 
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print(f"GPU memory cleared. Free: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")


    num_classes = len(all_classes)  # number of video classes
    model = CNN(n_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 2 workers seems to work well
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds,   batch_size=config.BATCH_SIZE, shuffle=False,
                              num_workers=2, pin_memory=True)

    print("=" * 60)
    print("Training...")
    print("=" * 60)

    Last_val_acc = 0.0

    #train the model
    #epoch is the number of times the model will see the entire training set
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.NUM_EPOCHS}")
        t0 = time.time()
        last_val_acc = val_acc
        train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc   = validate(model, criterion, device, val_loader)

        took = time.time() - t0
        delta = val_acc - last_val_acc
        print("→ train %.4f | val %.4f | change %.4f | %.1fs" % (train_acc, val_acc, delta, took))

    #save the model weights
    torch.save(model.state_dict(), "deepaction_cnn_weights.pth")
    print("\nsaved to deepaction_cnn_weights.pth")

    #example output: → train 0.9010 | val 0.6962 | 369.5s