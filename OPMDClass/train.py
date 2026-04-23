import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from submission.model import DentalClassifier


def fit(model, train_loader, val_loader, epochs = 10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)

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

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        
        val_loss /= len(val_loader.dataset) 

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < least_val_loss:
            least_val_loss = val_loss
            torch.save(model.state_dict(), "submission/best_model.pth")
            print(f"  ✅ New best model saved with val loss: {least_val_loss:.4f}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406], 
        [0.229, 0.224, 0.225]
    )
])

data = ImageFolder("data/sample_test", transform = transform)
data.targets = [1 - t for t in data.targets]
train_size = int(0.8 * len(data))
val_size = len(data) - train_size       
train_data, val_data = random_split(data, [train_size, val_size])
train_loader = DataLoader(train_data, batch_size = 32, shuffle = True)
val_loader = DataLoader(val_data, batch_size = 32, shuffle = False)
model = DentalClassifier()
fit(model, train_loader, val_loader, epochs = 10)   