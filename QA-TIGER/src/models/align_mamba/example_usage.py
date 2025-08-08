"""
AlignMamba使用示例

展示如何使用AlignMamba进行不同类型的多模态任务，包括：
1. 多模态分类任务
2. 多模态回归任务  
3. 特征提取任务
4. 模型训练和评估流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt

from align_mamba_implementation import (
    AlignMamba, 
    create_align_mamba_classifier,
    create_align_mamba_regressor,
    create_align_mamba_feature_extractor
)


def generate_synthetic_multimodal_data(
    num_samples: int = 1000,
    audio_dim: int = 128,
    video_dim: int = 256,
    language_dim: int = 768,
    audio_len: int = 50,
    video_len: int = 30,
    language_len: int = 20,
    num_classes: int = 5,
    task_type: str = "classification"
) -> Dict[str, torch.Tensor]:
    """
    生成合成的多模态数据用于演示
    
    Args:
        num_samples: 样本数量
        audio_dim, video_dim, language_dim: 各模态特征维度
        audio_len, video_len, language_len: 各模态序列长度
        num_classes: 分类任务的类别数
        task_type: 任务类型 ("classification" 或 "regression")
    
    Returns:
        包含音频、视频、语言特征和标签的字典
    """
    # 生成随机特征
    audio_features = torch.randn(num_samples, audio_len, audio_dim)
    video_features = torch.randn(num_samples, video_len, video_dim)
    language_features = torch.randn(num_samples, language_len, language_dim)
    
    # 生成标签
    if task_type == "classification":
        labels = torch.randint(0, num_classes, (num_samples,))
    else:  # regression
        labels = torch.randn(num_samples)
    
    return {
        "audio": audio_features,
        "video": video_features,
        "language": language_features,
        "labels": labels
    }


def train_align_mamba(
    model: AlignMamba,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    learning_rate: float = 1e-3,
    device: str = "cuda"
) -> Dict[str, List[float]]:
    """
    训练AlignMamba模型
    
    Args:
        model: AlignMamba模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        num_epochs: 训练轮数
        learning_rate: 学习率
        device: 设备
    
    Returns:
        训练历史记录
    """
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    history = {
        "train_loss": [],
        "train_task_loss": [],
        "train_align_loss": [],
        "val_loss": [],
        "val_task_loss": [],
        "val_align_loss": []
    }
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_losses = {"total": [], "task": [], "align": []}
        
        for batch in train_loader:
            audio, video, language, labels = [x.to(device) for x in batch]
            
            optimizer.zero_grad()
            outputs = model(audio, video, language, labels=labels)
            
            loss = outputs["loss"]
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # 记录损失
            train_losses["total"].append(loss.item())
            if "task_loss" in outputs:
                train_losses["task"].append(outputs["task_loss"].item())
            train_losses["align"].append(outputs["alignment_loss"].item())
        
        # 验证阶段
        model.eval()
        val_losses = {"total": [], "task": [], "align": []}
        
        with torch.no_grad():
            for batch in val_loader:
                audio, video, language, labels = [x.to(device) for x in batch]
                outputs = model(audio, video, language, labels=labels)
                
                val_losses["total"].append(outputs["loss"].item())
                if "task_loss" in outputs:
                    val_losses["task"].append(outputs["task_loss"].item())
                val_losses["align"].append(outputs["alignment_loss"].item())
        
        # 更新学习率
        scheduler.step()
        
        # 记录平均损失
        history["train_loss"].append(np.mean(train_losses["total"]))
        history["train_align_loss"].append(np.mean(train_losses["align"]))
        history["val_loss"].append(np.mean(val_losses["total"]))
        history["val_align_loss"].append(np.mean(val_losses["align"]))
        
        if train_losses["task"]:
            history["train_task_loss"].append(np.mean(train_losses["task"]))
            history["val_task_loss"].append(np.mean(val_losses["task"]))
        
        # 打印进度
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
        print(f"  Val Loss: {history['val_loss'][-1]:.4f}")
        print(f"  Align Loss: {history['train_align_loss'][-1]:.4f}")
        if history["train_task_loss"]:
            print(f"  Task Loss: {history['train_task_loss'][-1]:.4f}")
        print()
    
    return history


def evaluate_classification_model(
    model: AlignMamba,
    test_loader: DataLoader,
    device: str = "cuda"
) -> Dict[str, float]:
    """评估分类模型性能"""
    model.eval()
    model = model.to(device)
    
    correct = 0
    total = 0
    all_losses = []
    
    with torch.no_grad():
        for batch in test_loader:
            audio, video, language, labels = [x.to(device) for x in batch]
            outputs = model(audio, video, language, labels=labels)
            
            predictions = outputs["logits"].argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            all_losses.append(outputs["loss"].item())
    
    accuracy = correct / total
    avg_loss = np.mean(all_losses)
    
    return {"accuracy": accuracy, "loss": avg_loss}


def evaluate_regression_model(
    model: AlignMamba,
    test_loader: DataLoader,
    device: str = "cuda"
) -> Dict[str, float]:
    """评估回归模型性能"""
    model.eval()
    model = model.to(device)
    
    all_predictions = []
    all_labels = []
    all_losses = []
    
    with torch.no_grad():
        for batch in test_loader:
            audio, video, language, labels = [x.to(device) for x in batch]
            outputs = model(audio, video, language, labels=labels)
            
            all_predictions.extend(outputs["logits"].cpu().numpy().flatten())
            all_labels.extend(labels.cpu().numpy().flatten())
            all_losses.append(outputs["loss"].item())
    
    predictions = np.array(all_predictions)
    labels = np.array(all_labels)
    
    mse = np.mean((predictions - labels) ** 2)
    mae = np.mean(np.abs(predictions - labels))
    correlation = np.corrcoef(predictions, labels)[0, 1]
    avg_loss = np.mean(all_losses)
    
    return {
        "mse": mse,
        "mae": mae,
        "correlation": correlation,
        "loss": avg_loss
    }


def visualize_alignment_matrices(
    model: AlignMamba,
    sample_batch: tuple,
    device: str = "cuda"
):
    """可视化对齐矩阵"""
    model.eval()
    model = model.to(device)
    
    audio, video, language, labels = [x.to(device) for x in sample_batch]
    
    with torch.no_grad():
        outputs = model(audio, video, language, return_alignments=True)
    
    # 获取第一个样本的传输矩阵
    Ma2l = outputs["transport_matrices"]["audio_to_lang"][0].cpu().numpy()  # (T_a, T_l)
    Mv2l = outputs["transport_matrices"]["video_to_lang"][0].cpu().numpy()  # (T_v, T_l)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 音频到语言的对齐
    im1 = ax1.imshow(Ma2l, cmap='Blues', aspect='auto')
    ax1.set_title('Audio-to-Language Alignment')
    ax1.set_xlabel('Language Tokens')
    ax1.set_ylabel('Audio Frames')
    plt.colorbar(im1, ax=ax1)
    
    # 视频到语言的对齐
    im2 = ax2.imshow(Mv2l, cmap='Reds', aspect='auto')
    ax2.set_title('Video-to-Language Alignment')
    ax2.set_xlabel('Language Tokens')
    ax2.set_ylabel('Video Frames')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.show()


def plot_training_history(history: Dict[str, List[float]]):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # 总损失
    axes[0].plot(history["train_loss"], label="Train Loss", color='blue')
    axes[0].plot(history["val_loss"], label="Val Loss", color='red')
    axes[0].set_title("Total Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)
    
    # 对齐损失
    axes[1].plot(history["train_align_loss"], label="Train Align Loss", color='green')
    axes[1].plot(history["val_align_loss"], label="Val Align Loss", color='orange')
    if history["train_task_loss"]:
        axes[1].plot(history["train_task_loss"], label="Train Task Loss", color='purple')
        axes[1].plot(history["val_task_loss"], label="Val Task Loss", color='brown')
    axes[1].set_title("Component Losses")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()


def main():
    """主函数：演示AlignMamba的完整使用流程"""
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # === 示例1: 多模态分类任务 ===
    print("=" * 50)
    print("示例1: 多模态分类任务")
    print("=" * 50)
    
    # 生成数据
    data = generate_synthetic_multimodal_data(
        num_samples=1000,
        task_type="classification",
        num_classes=5
    )
    
    # 创建数据加载器
    dataset = TensorDataset(data["audio"], data["video"], data["language"], data["labels"])
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 创建分类模型
    classifier = create_align_mamba_classifier(
        dim_audio=128,
        dim_video=256,
        dim_language=768,
        num_classes=5,
        d_model=256,
        n_layers=4,
        lambda_align=0.1
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in classifier.parameters()):,}")
    
    # 训练模型
    print("开始训练分类模型...")
    history = train_align_mamba(
        classifier, train_loader, val_loader,
        num_epochs=5, learning_rate=1e-3, device=device
    )
    
    # 评估模型
    results = evaluate_classification_model(classifier, test_loader, device)
    print(f"测试准确率: {results['accuracy']:.4f}")
    print(f"测试损失: {results['loss']:.4f}")
    
    # 可视化对齐矩阵
    sample_batch = next(iter(test_loader))
    print("生成对齐矩阵可视化...")
    visualize_alignment_matrices(classifier, sample_batch, device)
    
    # === 示例2: 多模态回归任务 ===
    print("=" * 50)
    print("示例2: 多模态回归任务")
    print("=" * 50)
    
    # 生成回归数据
    reg_data = generate_synthetic_multimodal_data(
        num_samples=800,
        task_type="regression"
    )
    
    reg_dataset = TensorDataset(
        reg_data["audio"], reg_data["video"], 
        reg_data["language"], reg_data["labels"]
    )
    
    reg_train_size = int(0.8 * len(reg_dataset))
    reg_test_size = len(reg_dataset) - reg_train_size
    
    reg_train_dataset, reg_test_dataset = torch.utils.data.random_split(
        reg_dataset, [reg_train_size, reg_test_size]
    )
    
    reg_train_loader = DataLoader(reg_train_dataset, batch_size=32, shuffle=True)
    reg_test_loader = DataLoader(reg_test_dataset, batch_size=32, shuffle=False)
    
    # 创建回归模型
    regressor = create_align_mamba_regressor(
        dim_audio=128,
        dim_video=256,
        dim_language=768,
        output_dim=1,
        d_model=256,
        lambda_align=0.05
    )
    
    # 训练回归模型
    print("开始训练回归模型...")
    reg_history = train_align_mamba(
        regressor, reg_train_loader, reg_test_loader,
        num_epochs=5, learning_rate=1e-3, device=device
    )
    
    # 评估回归模型
    reg_results = evaluate_regression_model(regressor, reg_test_loader, device)
    print(f"测试MSE: {reg_results['mse']:.4f}")
    print(f"测试MAE: {reg_results['mae']:.4f}")
    print(f"相关系数: {reg_results['correlation']:.4f}")
    
    # === 示例3: 特征提取 ===
    print("=" * 50)
    print("示例3: 特征提取")
    print("=" * 50)
    
    # 创建特征提取器
    feature_extractor = create_align_mamba_feature_extractor(
        dim_audio=128,
        dim_video=256,
        dim_language=768,
        d_model=256
    ).to(device)
    
    # 提取特征
    sample_audio, sample_video, sample_language, _ = sample_batch
    sample_audio = sample_audio[:2].to(device)  # 取前2个样本
    sample_video = sample_video[:2].to(device)
    sample_language = sample_language[:2].to(device)
    
    with torch.no_grad():
        features = feature_extractor(
            sample_audio, sample_video, sample_language,
            return_features=True
        )
    
    print(f"提取的特征形状: {features['features'].shape}")
    print(f"对齐损失: {features['alignment_loss'].item():.4f}")
    
    # 绘制训练历史
    print("绘制训练历史...")
    plot_training_history(history)
    
    print("演示完成！")


if __name__ == "__main__":
    main()