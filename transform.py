# -*- coding: utf-8 -*-
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms as transforms

# 自定义数据集类
class MultiModalPoseDataset(Dataset):
    def __init__(self, joint_path, bone_path, motion_path, label_path, transform=None):
        self.joint_data = np.load(joint_path, mmap_mode='r')
        self.bone_data = np.load(bone_path, mmap_mode='r')
        self.motion_data = np.load(motion_path, mmap_mode='r')
        self.labels = np.load(label_path)
        self.transform = transform

        # 数据标准化
        self.scaler = StandardScaler()
        self.joint_data = self.scaler.fit_transform(self.joint_data.reshape(-1, self.joint_data.shape[-1])).reshape(self.joint_data.shape)
        self.bone_data = self.scaler.fit_transform(self.bone_data.reshape(-1, self.bone_data.shape[-1])).reshape(self.bone_data.shape)
        self.motion_data = self.scaler.fit_transform(self.motion_data.reshape(-1, self.motion_data.shape[-1])).reshape(self.motion_data.shape)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        joint_sample = self.joint_data[idx]
        bone_sample = self.bone_data[idx]
        motion_sample = self.motion_data[idx]
        label = self.labels[idx]

        # 合并模态数据
        sample = np.concatenate((joint_sample, bone_sample, motion_sample), axis=-1)

        # 转换为Tensor
        sample = torch.tensor(sample, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            sample = self.transform(sample)

        return sample, label


# Transformer 模型定义
class MultiModalPoseTransformerModel(nn.Module):
    def __init__(self, num_classes, embed_dim=512, num_heads=8, num_layers=6, dropout=0.3):  # 增加 dropout
        super(MultiModalPoseTransformerModel, self).__init__()

        # 输入数据的维度变换
        self.input_embedding = nn.Linear(3 * 17 * 2 * 3, embed_dim)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # 分类头
        self.fc = nn.Linear(embed_dim, num_classes)

        self.dropout = nn.Dropout(p=dropout)  # 添加 Dropout

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, 300, -1)

        x = self.input_embedding(x)
        x = self.transformer_encoder(x)
        x = self.global_pool(x.permute(0, 2, 1)).squeeze(-1)
        x = self.dropout(x)  # 在分类前应用 Dropout
        x = self.fc(x)

        return x


# 路径定义
train_joint_path = 'data/train_joint.npy'
train_bone_path = 'data/train_bone.npy'
train_motion_path = 'data/train_joint_motion.npy'
train_label_path = 'data/train_label.npy'
test_joint_path = 'data/test_joint_B.npy'
# 创建数据集
dataset = MultiModalPoseDataset(train_joint_path, train_bone_path, train_motion_path, train_label_path)

# 划分训练集和验证集
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'使用设备: {device}')

# 定义模型、损失函数和优化器
num_classes = len(np.unique(dataset.labels))
model = MultiModalPoseTransformerModel(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 尝试更小的学习率

# 学习率调度器
scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)

# 训练和验证函数
def train(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
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


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(loader):
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
num_epochs = 20
best_val_acc = 0.0
best_train_log = None  # 记录最佳的训练日志

for epoch in range(num_epochs):
    print(f'\nEpoch {epoch + 1}/{num_epochs}:')
    train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device)

    print(f'  训练 Loss: {train_loss:.4f}, 准确率: {train_acc:.4f}')
    print(f'  验证 Loss: {val_loss:.4f}, 准确率: {val_acc:.4f}')

    # 更新学习率
    scheduler.step(val_loss)

    # 如果当前验证准确率最好，则记录最佳日志
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_train_log = f"Best Training Log:\n 训练 Loss: {train_loss:.4f}, 准确率: {train_acc:.4f}\n 验证 Loss: {val_loss:.4f}, 准确率: {val_acc:.4f}"
        torch.save(model.state_dict(), 'best_pose_model.pth')
        print('  保存最优模型')

# 输出最佳的训练日志
print('\n' + best_train_log)
print('训练完成')
