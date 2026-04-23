import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
from submission.model import DentalClassifier

def fit(model, loader, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)

    least_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0.0
        for images, labels in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
        
        average_loss = total_loss / len(loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {average_loss:.4f}")

        if total_loss < least_loss: 
            least_loss = total_loss
            torch.save(model.state_dict(), "submission/best_model.pth")
            print(f"New best model saved with loss {least_loss:.4f}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
])

data = ImageFolder("data/sample_test", transform = transform)
loader = torch.utils.data.DataLoader(data, batch_size = 16, shuffle = True)

model = DentalClassifier()

fit(model, loader, epochs = 15)

