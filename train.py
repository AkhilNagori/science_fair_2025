import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from model.crnn import CRNN
from utils import LabelConverter
import os

class OCRDataset(Dataset):
    def __init__(self, label_file, transform, converter):
        with open(label_file, 'r') as f:
            lines = f.read().splitlines()
        self.samples = [line.split(maxsplit=1) for line in lines]
        self.transform = transform
        self.converter = converter

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('L')
        image = self.transform(image)
        target = torch.tensor(self.converter.encode(label), dtype=torch.long)
        return image, target, len(target)

def collate_fn(batch):
    images, labels, lengths = zip(*batch)
    images = torch.stack(images)
    labels = torch.cat(labels)
    lengths = torch.tensor(lengths, dtype=torch.long)
    return images, labels, lengths

charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
converter = LabelConverter(charset)

transform = transforms.Compose([
    transforms.Resize((32, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = OCRDataset('data/labels.txt', transform, converter)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CRNN(32, 1, len(charset) + 1, 256).to(device)
criterion = nn.CTCLoss(blank=0)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    epoch_loss = 0
    for images, labels, lengths in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        input_lengths = torch.full((images.size(0),), images.size(3) // 4, dtype=torch.long)

        outputs = model(images).log_softmax(2)
        loss = criterion(outputs, labels, input_lengths, lengths)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(dataloader):.4f}")

torch.save(model.state_dict(), 'crnn_iam.pth')
