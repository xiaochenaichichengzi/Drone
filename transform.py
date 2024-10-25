import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score

# 自定义数据集类
class PoseDataset(Dataset):
    def __init__(self, joint_path, label_path, transform=None):
        self.joint_path = joint_path  # 存储路径而不是数据
        self.labels = np.load(label_path)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # 按需加载单个样本
        sample = np.load(self.joint_path, mmap_mode='r')[idx]  # 使用内存映射按需加载
        label = self.labels[idx]

        # 转换为Tensor
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            sample = self.transform(sample)

        return sample, label

# 模型定义
class PoseRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(PoseRecognitionModel, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=64, kernel_size=(3, 3, 3), padding=1)
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2))  # 减少空间维度

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1)
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 1))  # 只减少时间和高度维度，保持宽度为1

        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=1)
        self.bn3 = nn.BatchNorm3d(256)
        self.pool3 = nn.AdaptiveAvgPool3d((1, 1, 1))  # 全局池化

        self.fc1 = nn.Linear(256, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # 输入 x 的形状: (batch_size, 3, 300, 17, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)  # 形状: (batch_size, 256, 1, 1, 1)

        x = x.view(x.size(0), -1)  # 展平
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x

# 路径定义
train_joint_path = 'data/train_joint.npy'
train_label_path = 'data/train_label.npy'

# 创建数据集
dataset = PoseDataset(train_joint_path, train_label_path)

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# 创建数据加载器
batch_size = 32

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# 检查数据加载器
for batch_idx, (inputs, labels) in enumerate(train_loader):
    print(f'输入数据形状: {inputs.shape}')
    print(f'标签形状: {labels.shape}')
    break  # 只测试第一个batch

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'使用设备: {device}')

# 检查 CUDA 状态
print(f"CUDA 可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    print(f"当前设备: {torch.cuda.current_device()}")
    print(f"设备名称: {torch.cuda.get_device_name(torch.cuda.current_device())}")

# 定义模型、损失函数和优化器
num_classes = len(np.unique(dataset.labels))
model = PoseRecognitionModel(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    print("开始训练一个epoch...")

    for batch_idx, (inputs, labels) in enumerate(loader):
        print(f'  训练 batch {batch_idx + 1}/{len(loader)}')
        inputs = inputs.to(device)  # shape: (batch_size, 3, 300, 17, 2)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)  # shape: (batch_size, num_classes)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc

# 验证函数
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    print("开始验证一个epoch...")

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
            print(f'  验证 batch {batch_idx + 1}/{len(loader)}')
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)

    return epoch_loss, epoch_acc

# 训练过程
num_epochs = 200
best_val_acc = 0.0

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch+1}/{num_epochs}:')
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print(f'  训练 Loss: {train_loss:.4f}, 准确率: {train_acc:.4f}')
    print(f'  验证 Loss: {val_loss:.4f}, 准确率: {val_acc:.4f}')

    # 保存最好的模型
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_pose_model.pth')
        print('  保存最优模型')

print('训练完成')
