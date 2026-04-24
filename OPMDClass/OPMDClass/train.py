import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from submission.model import DentalClassifier


def fit(model, train_loader, epochs = 20):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
    lr = 1e-4   
    )

    least_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}")

    
    torch.save(model.state_dict(), "submission/best_model.pth")
    print(f"  ✅ New best model saved with loss: {train_loss:.4f}")

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406], 
        [0.229, 0.224, 0.225]
    )
])



data = ImageFolder("data/sample_test", transform = train_transform)
train_loader = DataLoader(data, batch_size = 8, shuffle = True)

model = DentalClassifier()
model.load_state_dict(torch.load("submission/best_model.pth", map_location = torch.device("cpu")))
fit(model, train_loader, epochs = 20)   

model.eval()

images, labels = next(iter(train_loader))
outputs = model(images)
preds = torch.argmax(outputs, dim=1)

print("Predictions:", preds.tolist())
print("Labels:     ", labels.tolist())

correct = 0
total = 0

with torch.no_grad():
    for images, labels in train_loader:
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

print("Accuracy:", correct / total)