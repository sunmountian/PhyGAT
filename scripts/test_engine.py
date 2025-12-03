import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import argparse
from torch.utils.data import DataLoader
from dataset_build import FlightDataset, get_adjacency_matrix
from model import PhyGAT

# ================= 1. 核心配置区域 =================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'best_phygat_model.pth'      # 训练好的模型
SCALER_PATH = 'dataset/scaler_params.pkl'         # 训练集的归一化参数

# 默认攻击配置（可通过参数修改）
DEFAULT_ATTACK_CONFIG = {'enable': False}  # 默认关闭在线注入

# --- 场景 C: 使用正常数据 + 在线模拟攻击 (调试用) ---
# ATTACK_CONFIG = {
#     'enable': True,       # 开启在线注入
#     'type': 'ramp',       # 攻击类型
#     'target_node': 3,     # 3: Baro
#     'target_feat': 0,     # 0: Alt
#     'start_idx': 300,     # 第 300 帧开始攻击
#     'magnitude': 0.1      # 斜率 (归一化后的数值，约等于 0.1 sigma/step)
# }

# ================= 2. 辅助函数 =================

def inject_attack(x, attack_cfg):
    """
    (可选) 在线注入攻击逻辑
    """
    if not attack_cfg.get('enable', False):
        return x
    
    x_attacked = x.clone()
    # 简单的 Ramp 注入：从 start_idx 开始累积偏差
    start = attack_cfg['start_idx']
    node = attack_cfg['target_node']
    feat = attack_cfg['target_feat']
    mag = attack_cfg['magnitude']
    
    # 获取序列长度
    # x shape: (Batch, Window, Nodes, Features)
    # 这里我们只对当前 Batch 内超过 start 的时间步注入
    # 注意：这只是简易演示。严谨的注入应在 dataset 层面完成。
    
    # 这里的逻辑是：假设 input x 是滑动窗口，我们在窗口的最后一帧注入偏差
    # 如果要模拟连续攻击，需要在外部循环维护一个累积 drift
    return x_attacked

# ================= 3. 主测试逻辑 =================

def run_test(data_path, attack_config=None):
    """
    运行测试
    
    Args:
        data_path: 数据文件路径（必需）
        attack_config: 攻击配置字典（可选）
    """
    if attack_config is None:
        attack_config = DEFAULT_ATTACK_CONFIG
    
    print(f"Running Test on: {data_path}")
    print(f"Attack Config: {attack_config}")
    
    # 1. 加载资源
    if not os.path.exists(data_path):
        print(f"[Error] 找不到文件: {data_path}")
        return

    adj_matrix = get_adjacency_matrix().to(DEVICE)
    model = PhyGAT(num_nodes=6, in_dim=3).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"[Error] 模型加载失败: {e}")
        return
        
    model.eval()
    
    # 2. 加载数据
    # mode='all' 表示加载该文件的所有数据，不再进行 train/test 切分
    ds = FlightDataset(data_path, mode='all') 
    # shuffle=False 保证时间顺序，Batch=1 方便逐帧分析
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    
    # 3. 记录器
    scores = []
    attentions_acc_baro = [] # 记录 Accel -> Baro 的权重
    predictions = []
    truths = []
    uncertainties = []
    
    drift = 0.0 # 用于在线注入的累积偏差
    
    print("Starting Inference...")
    
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            # 为了画图清晰，只跑前 1000 帧 (或者根据你的数据量调整)
            if i >= 1000: break 
            
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            
            # --- 在线模拟攻击 (如果开启) ---
            if attack_config.get('enable', False):
                if i >= attack_config['start_idx']:
                    drift += attack_config['magnitude']
                    # 修改 Baro (Node 3, Feat 0) 的输入
                    # 修改窗口内最后一帧的数据
                    x[:, -1, 3, 0] += drift
            
            # --- 模型推理 ---
            # mu: (B, N, F), log_var: (B, N, F), attn: (B, N, N)
            mu, log_var, attn = model(x, adj_matrix)
            sigma = torch.exp(0.5 * log_var)
            
            # --- 提取关键数据 (针对 Baro 节点) ---
            # Node 3 (Baro), Feat 0 (Alt)
            pred_val = mu[:, 3, 0].item()
            true_val = y[:, 3, 0].item() # 注意：如果是在线注入，这里的 y 是没被污染的真值
            sigma_val = sigma[:, 3, 0].item()
            
            # 如果是在线注入，输入 x 变了，但 y (target) 还是原始的下一帧
            # 异常检测逻辑：Residual = |观测值(可能被攻击) - 预测值(基于历史推断)|
            # 但这里我们的输入 x 是过去，y 是未来。
            # 实际上，我们需要对比的是：当前时刻的观测值 vs 当前时刻的预测值
            # 简化起见，我们直接对比 y (真实观测) 和 pred (预测)
            
            # 如果是已有攻击数据文件 (test_attack.npy)，y 已经是被攻击过的数据了
            # 模型会根据过去的历史(x)预测正常的下一步，而 y 是异常的
            # 所以 Residual 会变大
            
            residual = abs(true_val - pred_val)
            score = residual / (sigma_val + 1e-6)
            
            # 提取 Attention: Target=Baro(3), Source=Accel(1)
            # 注意力矩阵通常形状是 (Batch, Target_Nodes, Source_Nodes)
            # 请根据 model.py 里的实现确认维度，通常 dim=1 是 Target, dim=2 是 Source
            attn_val = attn[0, 3, 1].item() 
            
            # 记录
            scores.append(score)
            attentions_acc_baro.append(attn_val)
            predictions.append(pred_val)
            truths.append(true_val)
            uncertainties.append(sigma_val)

    # ================= 4. 可视化绘图 =================
    print("Plotting results...")
    
    plt.figure(figsize=(12, 12))
    
    # 子图 1: 异常得分 (Anomaly Score)
    plt.subplot(3, 1, 1)
    plt.plot(scores, 'r-', label='Anomaly Score ($|y-\hat{y}| / \sigma$)', linewidth=1.5)
    plt.axhline(y=3.0, color='g', linestyle='--', label='Threshold (3.0)')
    if attack_config.get('enable'):
        plt.axvline(x=attack_config['start_idx'], color='k', linestyle='--', label='Attack Start')
    plt.title('1. Anomaly Detection Score')
    plt.ylabel('Score')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # 子图 2: 物理注意力权重 (Physical Attention)
    plt.subplot(3, 1, 2)
    plt.plot(attentions_acc_baro, 'b-', label='Attention: Accel $\\to$ Baro', linewidth=1.5)
    if attack_config.get('enable'):
        plt.axvline(x=attack_config['start_idx'], color='k', linestyle='--')
    plt.title('2. Structural Attention Evolution (Physical Consistency)')
    plt.ylabel('Attention Weight')
    plt.ylim(0, 1.0) # Attention 范围 0-1
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # 子图 3: 原始数据对比 (Prediction vs Observation)
    plt.subplot(3, 1, 3)
    plt.plot(truths, 'k-', label='Observation (Input)', alpha=0.7)
    plt.plot(predictions, 'r--', label='PhyGAT Prediction', alpha=0.9)
    # 画不确定性带
    arr_pred = np.array(predictions)
    arr_sigma = np.array(uncertainties)
    plt.fill_between(range(len(predictions)), 
                     arr_pred - 3*arr_sigma, 
                     arr_pred + 3*arr_sigma, 
                     color='r', alpha=0.1, label='Uncertainty (3$\sigma$)')
    
    plt.title('3. Prediction vs Observation (Barometer)')
    plt.xlabel('Time Step')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 根据输入文件名保存图片
    output_img = f"result_{os.path.basename(data_path).replace('.npy', '')}.png"
    plt.savefig(output_img)
    print(f"Result saved to: {output_img}")
    plt.show()

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PhyGAT 测试脚本')
    parser.add_argument('--data_path', type=str, required=True,
                        help='测试数据文件路径（必需）')
    parser.add_argument('--enable_attack', action='store_true',
                        help='启用在线攻击注入')
    parser.add_argument('--attack_type', type=str, default='ramp',
                        help='攻击类型（默认: ramp）')
    parser.add_argument('--target_node', type=int, default=3,
                        help='目标节点（默认: 3, Baro）')
    parser.add_argument('--target_feat', type=int, default=0,
                        help='目标特征（默认: 0, Alt）')
    parser.add_argument('--start_idx', type=int, default=300,
                        help='攻击开始索引（默认: 300）')
    parser.add_argument('--magnitude', type=float, default=0.1,
                        help='攻击幅度（默认: 0.1）')
    
    args = parser.parse_args()
    
    # 构建攻击配置
    attack_config = {
        'enable': args.enable_attack,
        'type': args.attack_type,
        'target_node': args.target_node,
        'target_feat': args.target_feat,
        'start_idx': args.start_idx,
        'magnitude': args.magnitude
    }
    
    return args.data_path, attack_config

if __name__ == "__main__":
    data_path, attack_config = parse_args()
    run_test(data_path, attack_config)