import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import matplotlib.pyplot as plt

# 导入你的模块
from dataset_build import FlightDataset, get_adjacency_matrix
from model import PhyGAT
from torch.utils.data import random_split
# ================= 配置 =================
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_PATH = 'best_phygat_model.pth'
DATA_PATH = 'dataset/flight_dataset.npy' # 确保这个路径对

# ================= 损失函数 (NLL Loss) =================
def heteroscedastic_loss(true, mean, log_var):
    """
    高斯负对数似然损失 (Negative Log Likelihood)
    L = 0.5 * exp(-s) * (y - y_hat)^2 + 0.5 * s
    其中 s = log(sigma^2)
    """
    precision = torch.exp(-log_var)
    mse = (true - mean) ** 2
    loss = 0.5 * precision * mse + 0.5 * log_var
    return loss.mean()

# ================= 训练流程 =================
def train():
    print(f"Using device: {DEVICE}")
    
    # 1. 准备数据
    # 确保调用 dataset.py 里的函数
    try:
        adj_matrix = get_adjacency_matrix().to(DEVICE)
        # train_ds = FlightDataset(DATA_PATH, mode='train')
        # test_ds = FlightDataset(DATA_PATH, mode='test')
        
        # train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        # test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        # print(f"Data Loaded: Train {len(train_ds)}, Test {len(test_ds)}")
        # === 修改开始 ===
        
        # 1. 加载全部数据 (不再让 dataset 内部切分)
        # 注意：这里我们传入 mode='all'，让它加载整个 .npy
        # (前提是你的 dataset.py 里处理了 else的情况，之前的代码是包含 else: self.data = self.raw_data 的)
        full_ds = FlightDataset(DATA_PATH, mode='all')
        
        # 2. 随机切分 (Random Split)
        # 这会打乱时间顺序，但在开发阶段用来验证模型收敛性是完全合法的
        train_size = int(0.8 * len(full_ds))
        test_size = len(full_ds) - train_size
        
        # 固定随机种子，保证每次跑结果一致
        generator = torch.Generator().manual_seed(42)
        train_ds, test_ds = random_split(full_ds, [train_size, test_size], generator=generator)
        
        # 3. 创建 Loader
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
        
        # === 修改结束 ===
    except Exception as e:
        print(f"[错误] 数据加载失败: {e}")
        return

    # 2. 初始化模型
    model = PhyGAT(num_nodes=6, in_dim=3).to(DEVICE)
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    # 修改 optimizer 定义
    # 将 weight_decay 从 1e-5 增加到 1e-3
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    
    # 3. 训练循环
    best_val_loss = float('inf')
    train_history = []
    val_history = []
    
    print("\nStarting Training...")
    
    for epoch in range(EPOCHS):
        # --- Training ---
        model.train()
        total_train_loss = 0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            
            # Forward
            mu, log_var, _ = model(x_batch, adj_matrix)
            
            # Loss Calculation (只针对物理存在的节点计算Loss，这里全算)
            loss = heteroscedastic_loss(y_batch, mu, log_var)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # 梯度裁剪
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                
                mu, log_var, _ = model(x_batch, adj_matrix)
                loss = heteroscedastic_loss(y_batch, mu, log_var)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(test_loader)
        
        train_history.append(avg_train_loss)
        val_history.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # --- Save Best Model ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), SAVE_PATH)
            # print("  -> Model Saved!")
            
    print(f"\nTraining Complete. Best Val Loss: {best_val_loss:.4f}")
    print(f"Model saved to: {SAVE_PATH}")
    
    # 4. 画训练曲线
    plt.figure(figsize=(10, 5))
    plt.plot(train_history, label='Train Loss')
    plt.plot(val_history, label='Val Loss')
    plt.title('PhyGAT Training NLL Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_curve.png')
    plt.show()

if __name__ == "__main__":
    train()