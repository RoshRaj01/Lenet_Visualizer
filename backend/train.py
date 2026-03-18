import torch
import torchvision
import torchvision.transforms as transforms
from model import LeNet
import os

# ── Windows multiprocessing guard ────────────────────────────
# Required on Windows — DataLoader with num_workers > 0 spawns new
# processes, which re-imports the main script. Without this guard
# it crashes with "bootstrapping phase" RuntimeError.
if __name__ == '__main__':

    # ── Device ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using:", device)

    # ── Transforms ──────────────────────────────────────────
    # Augmentation mimics hand-drawn canvas variation:
    #   • RandomRotation  — digits drawn at slight angles
    #   • RandomAffine    — slight shifts in position
    #   • Normalize       — MNIST mean/std; MUST match at inference
    #   • RandomErasing   — robustness to partial strokes
    train_transform = transforms.Compose([
        transforms.RandomRotation(12),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=5),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    # ── Datasets ────────────────────────────────────────────
    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=train_transform)
    valset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=val_transform)

    # num_workers=0 → single process, avoids Windows spawn issues
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=0)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=256, shuffle=False, num_workers=0)

    # ── Model ────────────────────────────────────────────────
    model = LeNet().to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)

    # ── Training loop ────────────────────────────────────────
    epochs   = 15
    best_acc = 0.0
    os.makedirs("saved_model", exist_ok=True)

    for epoch in range(epochs):
        # Train
        model.train()
        running_loss = 0.0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, _ = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validate
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in valloader:
                images, labels = images.to(device), labels.to(device)
                outputs, _ = model(images)
                val_loss += criterion(outputs, labels).item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total   += labels.size(0)

        acc     = 100.0 * correct / total
        avg_val = val_loss / len(valloader)
        print(f"Epoch {epoch+1:2d}/{epochs}  "
              f"train_loss={running_loss/len(trainloader):.4f}  "
              f"val_loss={avg_val:.4f}  val_acc={acc:.2f}%")

        scheduler.step(avg_val)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "saved_model/lenet.pth")
            print(f"  ✓ Best so far — saved (val_acc={acc:.2f}%)")

    print(f"\nDone. Best val accuracy: {best_acc:.2f}%")
    print("Saved to saved_model/lenet.pth — run export.py next.")