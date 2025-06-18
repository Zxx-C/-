import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve, auc
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PatchEmbedding(nn.Module):
    """将图像分割成patches并进行embedding"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x):
        # x: (batch_size, channels, height, width)
        x = self.projection(x)  # (batch_size, embed_dim, n_patches_sqrt, n_patches_sqrt)
        x = x.flatten(2)  # (batch_size, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch_size, n_patches, embed_dim)
        return x

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, embed_dim=768, n_heads=12, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.attention_dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.projection_dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, embed_dim = x.shape
        
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, n_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 计算注意力分数
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # 应用注意力权重
        out = torch.matmul(attention_weights, v)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        
        out = self.projection(out)
        out = self.projection_dropout(out)
        
        return out

class TransformerBlock(nn.Module):
    """Transformer编码器块"""
    def __init__(self, embed_dim=768, n_heads=12, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # 注意力机制 + 残差连接
        x = x + self.attention(self.norm1(x))
        # MLP + 残差连接
        x = x + self.mlp(self.norm2(x))
        return x

class VisionTransformer(nn.Module):
    """Vision Transformer模型"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3, n_classes=2, 
                 embed_dim=768, depth=12, n_heads=12, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # 类别token和位置编码
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.patch_embedding.n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        
        # Transformer编码器层
        self.transformer = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Patch embedding
        x = self.patch_embedding(x)
        
        # 添加类别token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 添加位置编码
        x = x + self.pos_embedding
        x = self.dropout(x)
        
        # Transformer编码器
        for transformer in self.transformer:
            x = transformer(x)
        
        x = self.norm(x)
        
        # 分类头（使用CLS token）
        cls_token_final = x[:, 0]
        x = self.head(cls_token_final)
        
        return x

def get_data_loaders(data_dir, batch_size=32, img_size=224):
    """创建数据加载器"""
    # 数据增强和预处理
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 加载数据集
    train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_transform)
    val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform=val_transform)
    test_dataset = datasets.ImageFolder(os.path.join(data_dir, 'test'), transform=val_transform)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader, test_loader, train_dataset.classes

def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc='Training', leave=False)
    for inputs, labels in progress_bar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # 更新进度条
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100.*correct/total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc

def validate_epoch(model, val_loader, criterion, device):
    """验证一个epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='Validation', leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 保存预测结果用于指标计算
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc, all_preds, all_labels, all_probs

def plot_training_curves(train_losses, train_accs, val_losses, val_accs):
    """绘制训练曲线"""
    epochs = range(1, len(train_losses) + 1)
    
    # 训练和验证损失
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')  # 英文标题
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("已保存损失曲线图: loss_curve.png")  # 中文注释
    
    # 训练和验证准确率
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('accuracy_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("已保存准确率曲线图: accuracy_curve.png")

def plot_confusion_matrix(y_true, y_pred, class_names):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("已保存混淆矩阵图: confusion_matrix.png")

def plot_roc_curve(y_true, y_probs, class_names):
    """绘制ROC曲线"""
    plt.figure(figsize=(8, 6))
    
    # 对于二分类，绘制ROC曲线
    if len(class_names) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_probs[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("已保存ROC曲线图: roc_curve.png")

def plot_precision_recall_curve(y_true, y_probs, class_names):
    """绘制Precision-Recall曲线"""
    if len(class_names) == 2:
        precision, recall, _ = precision_recall_curve(y_true, y_probs[:, 1])
        pr_auc = auc(recall, precision)
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='blue', lw=2, 
                label=f'PR curve (AUC = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('precision_recall_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("已保存Precision-Recall曲线图: precision_recall_curve.png")

def plot_class_distribution(train_loader, class_names):
    """绘制类别分布图"""
    class_counts = defaultdict(int)
    for _, labels in train_loader:
        for label in labels:
            class_counts[label.item()] += 1
    
    classes = [class_names[i] for i in sorted(class_counts.keys())]
    counts = [class_counts[i] for i in sorted(class_counts.keys())]
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(classes, counts, color=['skyblue', 'lightcoral'])
    plt.title('Class Distribution in Training Set')
    plt.xlabel('Classes')
    plt.ylabel('Number of Samples')
    
    # 在柱状图上显示数值
    for bar, count in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                str(count), ha='center', va='bottom')
    
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("已保存类别分布图: class_distribution.png")

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 超参数设置
    batch_size = 16  # 减小batch size适应内存
    img_size = 224
    n_epochs = 20
    learning_rate = 3e-4
    
    # 加载数据
    print("正在加载数据...")
    data_dir = 'chest_xray'
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        data_dir, batch_size, img_size)
    
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")
    print(f"测试集样本数: {len(test_loader.dataset)}")
    print(f"类别: {class_names}")
    
    # 绘制类别分布
    plot_class_distribution(train_loader, class_names)
    
    # 创建模型
    print("正在创建Vision Transformer模型...")
    model = VisionTransformer(
        img_size=img_size,
        patch_size=16,
        in_channels=3,
        n_classes=len(class_names),
        embed_dim=384,  # 减小模型参数
        depth=6,        # 减少层数
        n_heads=6,
        mlp_ratio=4,
        dropout=0.1
    ).to(device)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数数: {total_params:,}")
    print(f"可训练参数数: {trainable_params:,}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # 训练历史记录
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0.0
    
    print("开始训练...")
    # 训练循环
    for epoch in range(n_epochs):
        print(f'\nEpoch {epoch+1}/{n_epochs}')
        print('-' * 50)
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # 验证
        val_loss, val_acc, val_preds, val_labels, val_probs = validate_epoch(
            model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        
        # 记录历史
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_vit_model.pth')
            print(f"保存最佳模型，验证准确率: {val_acc:.2f}%")
        
        print(f'训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.2f}%')
        print(f'验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.2f}%')
        print(f'当前学习率: {optimizer.param_groups[0]["lr"]:.6f}')
    
    # 绘制训练曲线
    plot_training_curves(train_losses, train_accs, val_losses, val_accs)
    
    # 在测试集上评估
    print("\n在测试集上评估...")
    model.load_state_dict(torch.load('best_vit_model.pth'))
    test_loss, test_acc, test_preds, test_labels, test_probs = validate_epoch(
        model, test_loader, criterion, device)
    
    print(f'测试准确率: {test_acc:.2f}%')
    
    # 转换为numpy数组
    test_probs = np.array(test_probs)
    
    # 绘制各种指标图
    plot_confusion_matrix(test_labels, test_preds, class_names)
    plot_roc_curve(test_labels, test_probs, class_names)
    plot_precision_recall_curve(test_labels, test_probs, class_names)
    
    # 打印详细分类报告
    print("\n详细分类报告:")
    print(classification_report(test_labels, test_preds, target_names=class_names))
    
    print("\n所有可视化图表已保存完成！")

if __name__ == '__main__':
    main() 