import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from squeezenet import SqueezeNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 设置随机种子以确保可重复性
torch.manual_seed(0)

# 定义数据预处理转换
transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建训练集和验证集
train_dataset = ImageFolder('/home/storm/project/MobileNetV3/data/splitData/train', transform=transform)
valid_dataset = ImageFolder('/home/storm/project/MobileNetV3/data/splitData/valid', transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

# 定义SqueezeNet模型
model = SqueezeNet(num_classes=17)
num_classes = len(train_dataset.classes)
print("num_classes : ", num_classes)
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练模型
for epoch in range(10):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1} - Training Loss: {epoch_loss:.4f}")

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f"Epoch {epoch+1} - Validation Accuracy: {accuracy:.4f}")

# 保存模型
torch.save(model.state_dict(), 'squeezenet_model.pth')