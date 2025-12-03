import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
from torch.utils.data import DataLoader
from dataset_build import FlightDataset, get_adjacency_matrix
from model import PhyGAT

# ================= 配置 =================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'best_phygat_model.pth'
DATA_PATH = 'dataset/flight_dataset.npy'
SCALER_PATH = 'dataset/scaler_params.pkl'

def evaluate():
    print(f"Loading model from {MODEL_PATH}...")
    
    # 1. 加载数据和参数
    # 加载归一化参数，用于反归一化 (还原成真实物理单位)
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    mean = scaler['mean'].values
    std = scaler['std'].values
    
    # 加载测试集
    test_ds = FlightDataset(DATA_PATH, mode='test')
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False) # Batch=1 方便画图
    
    # 加载邻接矩阵
    adj_matrix = get_adjacency_matrix().to(DEVICE)

    # 2. 加载模型
    model = PhyGAT(num_nodes=6, in_dim=3).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False))
    model.eval()
    
    # 3. 推理 (取前 200 个时间步画图)
    preds_mu = []
    preds_sigma = []
    truths = []
    attentions = None  # 初始化为 None，稍后会被赋值
    
    print("Running inference on test set...")
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            if i >= 200: break # 只看前200帧
            
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            # Forward
            mu, log_var, attn = model(x, adj_matrix)
            sigma = torch.exp(0.5 * log_var) # log_var -> std_dev
            
            preds_mu.append(mu.cpu().numpy().squeeze())
            preds_sigma.append(sigma.cpu().numpy().squeeze())
            truths.append(y.cpu().numpy().squeeze())
            
            # 保存最后一层的 Attention
            # attn 形状是 (B*T, N, N)，其中 B=1, T=50(窗口大小)
            # 我们需要将其转换为 (N, N) 用于热力图
            if i == 50: # 随便取第50帧的注意力来看看
                attn_np = attn.cpu().numpy()  # (50, 6, 6)
                # 对时间步维度求平均，得到平均注意力矩阵 (6, 6)
                attentions = attn_np.mean(axis=0)

    # 转为数组
    preds_mu = np.array(preds_mu)       # (T, 6, 3)
    preds_sigma = np.array(preds_sigma) # (T, 6, 3)
    truths = np.array(truths)           # (T, 6, 3)

    # 4. 反归一化 (只针对 Baro 和 Accel_Z 演示)
    # 节点索引: 1:Accel, 3:Baro
    # 特征索引: Accel的Z是[2], Baro只有[0]
    
    # 提取 Baro (Node 3, Feat 0)
    # 对应的 mean/std 索引需要根据 scaler 的列顺序找，这里简化处理：
    # 假设我们之前 scaler 的列顺序是按 dataset 节点顺序来的
    # Baro 对应的是 ordered_cols 中的第 10 列 (4 Act + 3 Acc + 3 Gyro = 10)
    # *注意：为了简单，这里我们直接画归一化后的数据看趋势即可，趋势对就说明训练对了*
    
    # ================= 画图 1: 预测性能 =================
    plt.figure(figsize=(12, 8))
    
    # 子图 1: Barometer (高度)
    plt.subplot(2, 1, 1)
    # Node 3 (Baro), Feat 0
    mu_baro = preds_mu[:, 3, 0]
    std_baro = preds_sigma[:, 3, 0]
    true_baro = truths[:, 3, 0]
    
    plt.plot(true_baro, 'k-', label='Ground Truth', linewidth=1.5)
    plt.plot(mu_baro, 'r--', label='Prediction', linewidth=1.5)
    # 画不确定性包络 (3 sigma)
    plt.fill_between(range(len(mu_baro)), 
                     mu_baro - 3*std_baro, 
                     mu_baro + 3*std_baro, 
                     color='r', alpha=0.2, label='Uncertainty (3$\sigma$)')
    plt.title('Performance Check: Barometer (Z-Axis)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图 2: Accel Z (加速度)
    plt.subplot(2, 1, 2)
    # Node 1 (Accel), Feat 2 (Z轴)
    mu_acc = preds_mu[:, 1, 2]
    true_acc = truths[:, 1, 2]
    
    plt.plot(true_acc, 'k-', label='Ground Truth')
    plt.plot(mu_acc, 'b--', label='Prediction')
    plt.title('Performance Check: Accel Z')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('eval_performance.png')
    print("Saved prediction plot to 'eval_performance.png'")

    # ================= 画图 2: 物理注意力热力图 =================
    if attentions is not None:
        plt.figure(figsize=(8, 6))
        import seaborn as sns
        
        # 节点标签
        labels = ['Act', 'Accel', 'Gyro', 'Baro', 'GPS', 'Mag']
        
        sns.heatmap(attentions, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=labels, yticklabels=labels)
        plt.title('PhyGAT Attention Weights (Snapshot)')
        plt.xlabel('Source Node (j)')
        plt.ylabel('Target Node (i)')
        plt.savefig('eval_attention.png')
        print("Saved attention heatmap to 'eval_attention.png'")
    else:
        print("Warning: Attention weights not captured (i=50 not reached)")
    
    # 在无GUI环境中（如WSL），plt.show()会发出警告，图片已保存到文件，无需显示
    # plt.show()  # 已注释：图片已保存，无需显示

if __name__ == "__main__":
    evaluate()