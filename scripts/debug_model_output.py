import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset_build import FlightDataset, get_adjacency_matrix
from model import PhyGAT_Fixed

# 配置
DEVICE = torch.device('cpu')
MODEL_PATH = 'best_model_contrast.pth'
DATA_PATH = 'dataset/flight_dataset.npy'

print("=" * 60)
print("Debugging PhyGAT Model Outputs")
print("=" * 60)

# 加载数据
test_ds = FlightDataset(DATA_PATH, mode='test')
test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

# 加载模型
adj_matrix = get_adjacency_matrix().to(DEVICE)
model = PhyGAT_Fixed(num_nodes=6, in_dim=3).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# 取一个batch
x_batch, y_batch = next(iter(test_loader))
x_batch = x_batch.to(DEVICE)

print(f"\n📊 Input Data:")
print(f"   Shape: {x_batch.shape}")
print(f"   Range: [{x_batch.min():.3f}, {x_batch.max():.3f}]")
print(f"   Mean: {x_batch.mean():.3f}, Std: {x_batch.std():.3f}")

# 推理
with torch.no_grad():
    mu, log_var, attn = model(x_batch, adj_matrix, return_all_steps=True)
    sigma = torch.exp(0.5 * log_var)

print(f"\n📊 Model Outputs:")
print(f"   Mu shape: {mu.shape}")
print(f"   Mu range: [{mu.min():.3f}, {mu.max():.3f}]")
print(f"   Mu mean: {mu.mean():.3f}, std: {mu.std():.3f}")

print(f"\n📊 Uncertainty (Sigma):")
print(f"   Log_var range: [{log_var.min():.3f}, {log_var.max():.3f}]")
print(f"   Sigma range: [{sigma.min():.6f}, {sigma.max():.6f}]")
print(f"   Sigma mean: {sigma.mean():.6f}")

# 关键诊断：Sigma是否过小
if sigma.mean() < 0.01:
    print("\n❌ CRITICAL: Sigma is too small!")
    print("   This means the model is overconfident.")
    print("   Residual-based detection will fail.")
else:
    print("\n✓ Sigma is in reasonable range")

print(f"\n📊 Attention Weights:")
print(f"   Attn shape: {attn.shape}")
print(f"   Attn range: [{attn.min():.3f}, {attn.max():.3f}]")

# 检查物理边的注意力权重
phy_edges = [(1, 3), (0, 1), (0, 2)]  # Accel->Baro, Act->Accel, Act->Gyro
attn_mean = attn.mean(dim=(0, 1))  # 平均到 (N, N)

print(f"\n📊 Key Physical Edge Weights:")
for src, tgt in phy_edges:
    weight = attn_mean[tgt, src].item()
    print(f"   Edge {src}->{tgt}: {weight:.4f}")

# 计算残差
residual = torch.abs(x_batch - mu)
print(f"\n📊 Residuals:")
print(f"   Range: [{residual.min():.3f}, {residual.max():.3f}]")
print(f"   Mean: {residual.mean():.3f}")

# 计算实际的异常得分（模拟检测器）
S_res = (residual / (sigma + 1e-6)).mean(dim=(2, 3))
print(f"\n📊 Anomaly Score (S_res):")
print(f"   Range: [{S_res.min():.3f}, {S_res.max():.3f}]")
print(f"   Mean: {S_res.mean():.3f}")

if S_res.mean() < 1.0:
    print("\n❌ CRITICAL: Anomaly scores are too low!")
    print("   With threshold=5.0, nothing will be detected.")
    print("\n💡 SOLUTION: Need to increase sigma or change loss function")

print("\n" + "=" * 60)
print("Diagnosis Complete")
print("=" * 60)