import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os
import copy

# ==============================
# Configuration
# ==============================
IMG_SIZE = 224  # ResNet expects 224x224
BATCH_SIZE = 32
EPOCHS = 25
LR = 0.001
NUM_CLASSES = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

train_dir = "dataset/train"
test_dir = "dataset/test"
os.makedirs("model", exist_ok=True)

# ==============================
# Data Transforms (stronger augmentation)
# ==============================
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ==============================
# Datasets & DataLoaders
# ==============================
full_train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)

class_names = full_train_dataset.classes
print(f"Classes: {class_names}")
print(f"Total training images: {len(full_train_dataset)}")
print(f"Total test images: {len(test_dataset)}")

# Split train into train + validation (90/10)
val_size = int(0.1 * len(full_train_dataset))
train_size = len(full_train_dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(
    full_train_dataset, [train_size, val_size]
)

# Apply test_transform to validation set (no augmentation)
val_dataset_copy = copy.copy(val_dataset)
val_dataset_copy.dataset = datasets.ImageFolder(train_dir, transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset_copy, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ==============================
# Model: Pretrained ResNet50 (Transfer Learning)
# ==============================
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# Freeze early layers, fine-tune later layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze layer3 and layer4 for fine-tuning
for param in model.layer3.parameters():
    param.requires_grad = True
for param in model.layer4.parameters():
    param.requires_grad = True

# Replace final FC layer
model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(512, NUM_CLASSES)
)

model = model.to(DEVICE)

# ==============================
# Loss, Optimizer, Scheduler
# ==============================
criterion = nn.CrossEntropyLoss()

# Different LR for pretrained vs new layers
optimizer = optim.Adam([
    {"params": model.layer3.parameters(), "lr": LR * 0.1},
    {"params": model.layer4.parameters(), "lr": LR * 0.1},
    {"params": model.fc.parameters(), "lr": LR}
], weight_decay=1e-4)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', patience=3, factor=0.5
)

# ==============================
# Training Loop with Best Model Saving
# ==============================
train_acc_list, val_acc_list = [], []
train_loss_list, val_loss_list = [], []
best_val_acc = 0.0

for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # --- Validation ---
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_loss = val_loss / val_total
    val_acc = val_correct / val_total

    scheduler.step(val_loss)

    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)

    print(f"Epoch [{epoch+1}/{EPOCHS}] "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "model/alzheimer_best.pth")
        print(f"  >> Best model saved (Val Acc: {val_acc*100:.2f}%)")

print(f"\nBest Validation Accuracy: {best_val_acc*100:.2f}%")

# ==============================
# Load Best Model for Evaluation
# ==============================
model.load_state_dict(torch.load("model/alzheimer_best.pth"))
model.eval()

# ==============================
# Accuracy & Loss Graphs
# ==============================
plt.figure(figsize=(10, 5))
plt.plot(train_acc_list, label='Train')
plt.plot(val_acc_list, label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig("assets/accuracy.png", dpi=150, bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(train_loss_list, label='Train')
plt.plot(val_loss_list, label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig("assets/loss.png", dpi=150, bbox_inches='tight')
plt.show()

# ==============================
# Test Predictions
# ==============================
all_preds, all_labels = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ==============================
# Classification Report
# ==============================
print("\nClassification Report:\n")
report = classification_report(all_labels, all_preds, target_names=class_names)
print(report)

# ==============================
# Confusion Matrix
# ==============================
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig("assets/confusion_matrix.png", dpi=150, bbox_inches='tight')
plt.show()

# ==============================
# Test Accuracy
# ==============================
test_acc = np.sum(all_preds == all_labels) / len(all_labels)
print(f"\nTest Accuracy: {test_acc*100:.2f}%")