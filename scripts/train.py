import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# ================= 新增：对比损失函数 =================
def physics_contrast_loss(attentions, adj_mask, conflict_mask=None):
    """
    物理对比损失 - 完全重写版
    
    目标：
    1. 物理边的权重应该 > 0.3
    2. 非物理边的权重应该 < 0.1
    3. 冲突时，标记的边权重应该 < 0.1
    """
    B, N, _ = attentions.shape
    device = attentions.device
    
    if conflict_mask is None:
        # === 正常样本 ===
        # 构建物理边掩码（排除自连接）
        eye_mask = torch.eye(N).to(device)
        phy_edges = (adj_mask > 0) & (eye_mask == 0)
        non_phy_edges = (adj_mask == 0)
        
        # Loss 1: 物理边权重不足的惩罚
        # 期望每条物理边的权重 > 0.3
        phy_weights = attentions * phy_edges.unsqueeze(0)  # (B, N, N)
        phy_violations = torch.relu(0.3 - phy_weights)  # 低于0.3就惩罚
        loss_phy = phy_violations.sum() / (phy_edges.sum() * B + 1e-6)
        
        # Loss 2: 非物理边权重过高的惩罚
        # 期望每条非物理边的权重 < 0.1
        non_phy_weights = attentions * non_phy_edges.unsqueeze(0)
        non_phy_violations = torch.relu(non_phy_weights - 0.1)  # 高于0.1就惩罚
        loss_non_phy = non_phy_violations.sum() / (non_phy_edges.sum() * B + 1e-6)
        
        total_loss = loss_phy + loss_non_phy
        
    else:
        # === 冲突样本 ===
        # 标记为冲突的边权重应该 < 0.1
        conflict_weights = attentions * conflict_mask
        violations = torch.relu(conflict_weights - 0.1)
        total_loss = violations.sum() / (conflict_mask.sum() + 1e-6)
    
    return total_loss


def create_conflict_batch(x_batch, conflict_ratio=0.5):
    """
    人工制造物理冲突样本 - 增强版
    
    策略1：随机交换 Accel 数据（破坏 Accel-Baro 一致性）
    策略2：随机交换 Gyro 数据（破坏 Gyro-Mag 一致性）
    
    返回:
    - x_conflict: 冲突样本
    - conflict_mask: (B, N, N) 标记哪些边是冲突的
    """
    B, T, N, F = x_batch.shape
    x_conflict = x_batch.clone()
    conflict_mask = torch.zeros(B, N, N).to(x_batch.device)
    
    # 选择更多样本进行破坏（从30%提升到50%）
    n_conflict = int(B * conflict_ratio)
    conflict_indices = torch.randperm(B)[:n_conflict]
    
    for idx in conflict_indices:
        # 随机选择破坏策略
        strategy = torch.rand(1).item()
        
        if strategy < 0.5:
            # 策略1：交换 Accel (Node 1)
            swap_idx = torch.randint(0, B, (1,)).item()
            if swap_idx == idx:
                swap_idx = (idx + 1) % B
            
            x_conflict[idx, :, 1, :] = x_batch[swap_idx, :, 1, :]
            # 标记冲突边
            conflict_mask[idx, 3, 1] = 1.0  # Accel -> Baro
            conflict_mask[idx, 4, 1] = 1.0  # Accel -> GPS
        else:
            # 策略2：交换 Gyro (Node 2)
            swap_idx = torch.randint(0, B, (1,)).item()
            if swap_idx == idx:
                swap_idx = (idx + 1) % B
            
            x_conflict[idx, :, 2, :] = x_batch[swap_idx, :, 2, :]
            # 标记冲突边
            conflict_mask[idx, 5, 2] = 1.0  # Gyro -> Mag
            conflict_mask[idx, 1, 2] = 1.0  # Gyro -> Accel
    
    return x_conflict, conflict_mask


# ================= 修改后的训练函数 =================
def heteroscedastic_loss(true, mean, log_var):
    """
    高斯负对数似然损失 - 数值稳定版
    强制 sigma 不能过小
    """
    # 限制 log_var 的范围，防止 sigma 崩塌到0
    # log_var 范围: [-2, 2] 对应 sigma 范围: [0.37, 2.72]
    log_var = torch.clamp(log_var, min=-2, max=2)
    
    precision = torch.exp(-log_var)
    mse = (true - mean) ** 2
    loss = 0.5 * precision * mse + 0.5 * log_var
    
    # 确保损失非负
    loss = torch.clamp(loss, min=0)
    return loss.mean()


def train_with_contrast(model, train_loader, test_loader, adj_mask, 
                       epochs=100, lr=1e-3, lambda_contrast=0.5, device='cuda'):
    """
    带对比学习的训练流程
    
    lambda_contrast: 对比损失的权重（提升到0.5）
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)  # 增强到1e-2
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    best_val_loss = float('inf')
    train_history = []
    val_history = []
    patience_counter = 0
    early_stop_patience = 20  # 20个epoch验证集不降就停
    
    for epoch in range(epochs):
        model.train()
        total_nll = 0
        total_contrast = 0
        
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # === 正常样本前向传播 ===
            mu, log_var, attn = model(x_batch, adj_mask, return_all_steps=False)
            loss_nll = heteroscedastic_loss(y_batch, mu, log_var)
            
            # === 对比学习：正常样本 ===
            loss_contrast_normal = physics_contrast_loss(attn, adj_mask)
            
            # === 对比学习：冲突样本 ===
            x_conflict, conflict_mask = create_conflict_batch(x_batch, conflict_ratio=0.3)
            _, _, attn_conflict = model(x_conflict, adj_mask, return_all_steps=False)
            loss_contrast_conflict = physics_contrast_loss(attn_conflict, adj_mask, conflict_mask)
            
            # === 总损失 ===
            loss_total = loss_nll + lambda_contrast * (loss_contrast_normal + loss_contrast_conflict)
            
            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_nll += loss_nll.item()
            total_contrast += (loss_contrast_normal + loss_contrast_conflict).item()
        
        avg_nll = total_nll / len(train_loader)
        avg_contrast = total_contrast / len(train_loader)
        
        # === 验证 ===
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in test_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                mu, log_var, _ = model(x_batch, adj_mask)
                loss = heteroscedastic_loss(y_batch, mu, log_var)
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(test_loader)
        
        train_history.append(avg_nll)
        val_history.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"NLL: {avg_nll:.4f} | Contrast: {avg_contrast:.4f} | Val: {avg_val_loss:.4f}")
        
        # === 添加异常检测 ===
        if avg_nll < 0 or avg_contrast < 0:
            print(f"  ⚠️ WARNING: Negative loss detected! This indicates numerical instability.")
        
        # 监控对比学习的效果
        if epoch == 0:
            print(f"  📊 First epoch - Contrast Loss: {avg_contrast:.4f}")
        if epoch == 20:
            print(f"  📊 Epoch 20 - Contrast Loss should be decreasing")
            if avg_contrast > 1.0:
                print(f"  ⚠️ Contrast Loss still high! Model may not be learning edge patterns.")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model_contrast.pth')
            print(f"  → Model saved! (Val improved)")
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"\n⚠️ Early stopping at epoch {epoch+1}")
                break
        
        # 学习率调度
        scheduler.step(avg_val_loss)
    
    print(f"Training Complete. Best Val Loss: {best_val_loss:.4f}")
    
    # === 保存训练曲线 ===
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(train_history, label='Train NLL')
    plt.plot(val_history, label='Val Loss')
    plt.title('PhyGAT Training with Contrast Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('training_curve.png')
    print("Saved training curve to 'training_curve.png'")


# ================= 使用示例 =================
if __name__ == "__main__":
    from dataset_build import FlightDataset, get_adjacency_matrix
    from model import PhyGAT_Fixed
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载数据
    train_ds = FlightDataset('dataset/flight_dataset.npy', mode='train')
    test_ds = FlightDataset('dataset/flight_dataset.npy', mode='test')
    
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    # 加载邻接矩阵
    adj_matrix = get_adjacency_matrix().to(DEVICE)
    
    # 初始化模型
    model = PhyGAT_Fixed(num_nodes=6, in_dim=3).to(DEVICE)
    
    # 训练
    train_with_contrast(
        model, train_loader, test_loader, adj_matrix,
        epochs=100, lr=1e-3, lambda_contrast=0.1, device=DEVICE
    )