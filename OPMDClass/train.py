import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import datasets
from torchvision.transforms import transforms
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
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader.dataset):.4f}")

        if total_loss < least_loss: 
            least_loss = total_loss
            torch.save(model.state_dict(), "submission/best_model.pth")
            print(f"New best model saved with loss {least_loss:.4f}")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

data = datasets.ImageFolder("data/sample_test", transform = transform)
loader = torch.utils.data.DataLoader(data, batch_size = 16, shuffle = True)

model = DentalClassifier()

fit(model, loader, epochs = 10)

