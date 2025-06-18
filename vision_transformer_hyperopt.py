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
import json
import time
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

def train_epoch(model, train_loader, criterion, optimizer, device, config_name):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    progress_bar = tqdm(train_loader, desc=f'Training {config_name}', leave=False)
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

def validate_epoch(model, val_loader, criterion, device, config_name):
    """验证一个epoch"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc=f'Validation {config_name}', leave=False)
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

def plot_hyperopt_comparison(results):
    """绘制超参数优化比较图"""
    config_names = list(results.keys())
    test_accs = [results[config]['test_acc'] for config in config_names]
    roc_aucs = [results[config]['roc_auc'] for config in config_names]
    pr_aucs = [results[config]['pr_auc'] for config in config_names]
    
    # 性能对比柱状图
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # 测试准确率对比
    bars1 = ax1.bar(config_names, test_accs, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    ax1.set_title('Test Accuracy Comparison')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(0, 100)
    for bar, acc in zip(bars1, test_accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.2f}%', ha='center', va='bottom')
    
    # ROC AUC对比
    bars2 = ax2.bar(config_names, roc_aucs, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    ax2.set_title('ROC AUC Comparison')
    ax2.set_ylabel('ROC AUC')
    ax2.set_ylim(0, 1)
    for bar, auc_val in zip(bars2, roc_aucs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{auc_val:.3f}', ha='center', va='bottom')
    
    # PR AUC对比
    bars3 = ax3.bar(config_names, pr_aucs, color=['skyblue', 'lightcoral', 'lightgreen', 'gold'])
    ax3.set_title('PR AUC Comparison')
    ax3.set_ylabel('PR AUC')
    ax3.set_ylim(0, 1)
    for bar, auc_val in zip(bars3, pr_aucs):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{auc_val:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('hyperopt_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("已保存超参数优化对比图: hyperopt_comparison.png")

def plot_training_curves_comparison(all_results):
    """绘制所有配置的训练曲线对比"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    colors = ['blue', 'red', 'green', 'orange']
    
    # 训练损失对比
    for i, (config_name, result) in enumerate(all_results.items()):
        epochs = range(1, len(result['train_losses']) + 1)
        ax1.plot(epochs, result['train_losses'], color=colors[i], label=config_name, linewidth=2)
    ax1.set_title('Training Loss Comparison')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 验证损失对比
    for i, (config_name, result) in enumerate(all_results.items()):
        epochs = range(1, len(result['val_losses']) + 1)
        ax2.plot(epochs, result['val_losses'], color=colors[i], label=config_name, linewidth=2)
    ax2.set_title('Validation Loss Comparison')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 训练准确率对比
    for i, (config_name, result) in enumerate(all_results.items()):
        epochs = range(1, len(result['train_accs']) + 1)
        ax3.plot(epochs, result['train_accs'], color=colors[i], label=config_name, linewidth=2)
    ax3.set_title('Training Accuracy Comparison')
    ax3.set_xlabel('Epochs')
    ax3.set_ylabel('Accuracy (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 验证准确率对比
    for i, (config_name, result) in enumerate(all_results.items()):
        epochs = range(1, len(result['val_accs']) + 1)
        ax4.plot(epochs, result['val_accs'], color=colors[i], label=config_name, linewidth=2)
    ax4.set_title('Validation Accuracy Comparison')
    ax4.set_xlabel('Epochs')
    ax4.set_ylabel('Accuracy (%)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("已保存训练曲线对比图: training_curves_comparison.png")

def save_results_table(results):
    """保存结果对比表格"""
    print("\n" + "="*80)
    print("超参数优化结果汇总")
    print("="*80)
    print(f"{'配置':<15} {'测试准确率':<12} {'ROC AUC':<10} {'PR AUC':<10} {'训练时间':<12} {'参数量':<12}")
    print("-"*80)
    
    for config_name, result in results.items():
        print(f"{config_name:<15} {result['test_acc']:<11.2f}% {result['roc_auc']:<9.3f} "
              f"{result['pr_auc']:<9.3f} {result['training_time']:<11.1f}s {result['total_params']:<12,}")
    
    print("-"*80)
    
    # 找出最佳配置
    best_config = max(results.keys(), key=lambda x: results[x]['test_acc'])
    print(f"最佳配置: {best_config} (测试准确率: {results[best_config]['test_acc']:.2f}%)")
    print("="*80)

def train_with_config(config, data_dir, device, class_names):
    """使用指定配置训练模型"""
    config_name = config['name']
    print(f"\n开始训练配置: {config_name}")
    print(f"参数设置: {config}")
    
    start_time = time.time()
    
    # 创建数据加载器
    train_loader, val_loader, test_loader, _ = get_data_loaders(
        data_dir, config['batch_size'], config['img_size'])
    
    # 创建模型
    model = VisionTransformer(
        img_size=config['img_size'],
        patch_size=16,
        in_channels=3,
        n_classes=len(class_names),
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        n_heads=config['n_heads'],
        mlp_ratio=4,
        dropout=config['dropout']
    ).to(device)
    
    # 计算模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数数量: {total_params:,}")
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['n_epochs'])
    
    # 训练历史记录
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    best_val_acc = 0.0
    
    # 训练循环
    epoch_progress = tqdm(range(config['n_epochs']), desc=f'Epochs for {config_name}')
    for epoch in epoch_progress:
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, config_name)
        
        # 验证
        val_loss, val_acc, val_preds, val_labels, val_probs = validate_epoch(
            model, val_loader, criterion, device, config_name)
        
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
            torch.save(model.state_dict(), f'best_model_{config_name}.pth')
        
        # 更新epoch进度条
        epoch_progress.set_postfix({
            'Train Acc': f'{train_acc:.2f}%',
            'Val Acc': f'{val_acc:.2f}%',
            'Best Val': f'{best_val_acc:.2f}%'
        })
    
    # 在测试集上评估
    model.load_state_dict(torch.load(f'best_model_{config_name}.pth'))
    test_loss, test_acc, test_preds, test_labels, test_probs = validate_epoch(
        model, test_loader, criterion, device, config_name)
    
    # 计算指标
    test_probs = np.array(test_probs)
    
    # ROC AUC
    fpr, tpr, _ = roc_curve(test_labels, test_probs[:, 1])
    roc_auc = auc(fpr, tpr)
    
    # PR AUC
    precision, recall, _ = precision_recall_curve(test_labels, test_probs[:, 1])
    pr_auc = auc(recall, precision)
    
    training_time = time.time() - start_time
    
    print(f"{config_name} 完成 - 测试准确率: {test_acc:.2f}%, ROC AUC: {roc_auc:.3f}, PR AUC: {pr_auc:.3f}")
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'test_acc': test_acc,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'training_time': training_time,
        'total_params': total_params,
        'test_preds': test_preds,
        'test_labels': test_labels,
        'test_probs': test_probs
    }

def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 定义四种超参数配置
    hyperparameter_configs = [
        {
            'name': 'Config_A_Small',
            'batch_size': 16,
            'img_size': 224,
            'n_epochs': 15,
            'learning_rate': 3e-4,
            'weight_decay': 0.01,
            'embed_dim': 256,
            'depth': 4,
            'n_heads': 4,
            'dropout': 0.1
        },
        {
            'name': 'Config_B_Medium',
            'batch_size': 12,
            'img_size': 224,
            'n_epochs': 15,
            'learning_rate': 2e-4,
            'weight_decay': 0.05,
            'embed_dim': 384,
            'depth': 6,
            'n_heads': 6,
            'dropout': 0.15
        },
        {
            'name': 'Config_C_Large',
            'batch_size': 8,
            'img_size': 224,
            'n_epochs': 15,
            'learning_rate': 1e-4,
            'weight_decay': 0.1,
            'embed_dim': 512,
            'depth': 8,
            'n_heads': 8,
            'dropout': 0.2
        },
        {
            'name': 'Config_D_Fast',
            'batch_size': 24,
            'img_size': 192,
            'n_epochs': 15,
            'learning_rate': 5e-4,
            'weight_decay': 0.01,
            'embed_dim': 320,
            'depth': 5,
            'n_heads': 5,
            'dropout': 0.1
        }
    ]
    
    # 加载数据以获取类别信息
    data_dir = 'chest_xray'
    _, _, _, class_names = get_data_loaders(data_dir, 16, 224)
    
    print(f"数据集类别: {class_names}")
    print(f"开始超参数优化，共测试 {len(hyperparameter_configs)} 种配置")
    
    # 存储所有结果
    all_results = {}
    
    # 训练每种配置
    for i, config in enumerate(hyperparameter_configs):
        print(f"\n{'='*60}")
        print(f"进度: {i+1}/{len(hyperparameter_configs)}")
        result = train_with_config(config, data_dir, device, class_names)
        all_results[config['name']] = result
    
    # 保存详细结果
    with open('hyperopt_results.json', 'w') as f:
        # 将numpy数组转换为列表以便JSON序列化
        json_results = {}
        for config_name, result in all_results.items():
            json_results[config_name] = {
                'train_losses': result['train_losses'],
                'train_accs': result['train_accs'],
                'val_losses': result['val_losses'],
                'val_accs': result['val_accs'],
                'test_acc': result['test_acc'],
                'roc_auc': result['roc_auc'],
                'pr_auc': result['pr_auc'],
                'training_time': result['training_time'],
                'total_params': result['total_params']
            }
        json.dump(json_results, f, indent=2)
    
    # 生成对比图表
    plot_hyperopt_comparison(all_results)
    plot_training_curves_comparison(all_results)
    
    # 打印结果汇总表格
    save_results_table(all_results)
    
    # 保存最佳配置的详细结果
    best_config_name = max(all_results.keys(), key=lambda x: all_results[x]['test_acc'])
    best_result = all_results[best_config_name]
    
    print(f"\n正在为最佳配置 {best_config_name} 生成详细可视化...")
    
    # 为最佳配置生成混淆矩阵
    cm = confusion_matrix(best_result['test_labels'], best_result['test_preds'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Best Config ({best_config_name})')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_best_{best_config_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"已保存最佳配置混淆矩阵: confusion_matrix_best_{best_config_name}.png")
    print(f"\n超参数优化完成！最佳配置为: {best_config_name}")
    print(f"所有结果已保存到 hyperopt_results.json")

if __name__ == '__main__':
    main() 