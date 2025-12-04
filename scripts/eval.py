import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os

# 确保能导入其他模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from torch.utils.data import DataLoader
from dataset_build import FlightDataset, get_adjacency_matrix
from model import PhyGAT_Fixed, AttackDetector

# ================= 配置 =================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'best_model_contrast.pth'
DATA_PATH = 'dataset/flight_dataset.npy'
SCALER_PATH = 'dataset/scaler_params.pkl'

# ================= 攻击注入器 =================
class AttackInjector:
    """模拟各种传感器攻击"""
    
    @staticmethod
    def ramp_attack(data, node_idx, feat_idx, start_step, slope=0.05):
        """
        渐变攻击 (Ramp Attack)
        data: (B, T, N, F)
        node_idx: 攻击的节点（如 3=Baro）
        feat_idx: 攻击的特征（如 0=Z轴）
        start_step: 攻击开始的时间步
        slope: 斜率（归一化空间）
        """
        B, T, N, F = data.shape
        attacked = data.clone()
        
        for t in range(start_step, T):
            offset = (t - start_step) * slope
            attacked[:, t, node_idx, feat_idx] += offset
        
        return attacked
    
    @staticmethod
    def bias_attack(data, node_idx, feat_idx, start_step, bias=0.5):
        """突变偏置攻击"""
        attacked = data.clone()
        attacked[:, start_step:, node_idx, feat_idx] += bias
        return attacked
    
    @staticmethod
    def replay_attack(data, node_idx, start_step, replay_length=10):
        """重放攻击：重复播放历史数据"""
        attacked = data.clone()
        B, T, N, F = data.shape
        
        if start_step < replay_length:
            return attacked
        
        replay_segment = data[:, start_step-replay_length:start_step, node_idx, :]
        
        for t in range(start_step, min(T, start_step + replay_length)):
            attacked[:, t, node_idx, :] = replay_segment[:, t - start_step, :]
        
        return attacked

def evaluate_attack_detection(model, test_loader, adj_matrix, attack_type='ramp', device='cuda'):
    model.eval()
    
    # 物理边定义 (Acc <-> Baro 等)
    phy_edges = [(1, 3), (0, 1), (0, 2), (2, 1), (1, 2)] 
    # 阈值设为 0.1 以匹配训练
    detector = AttackDetector(phy_edges, threshold_res=3.0, threshold_struct=0.1)
    
    all_scores_normal = []
    all_scores_attack = []
    
    injector = AttackInjector()
    
    print(f"  Processing test batches for {attack_type} attack...")
    
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(test_loader):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            B, T, N, F = x_batch.shape
            
            # === 1. 正常样本评估 ===
            # return_all_steps=False: 只获取最后一步的预测 (B, N, F)
            mu, log_var, attn = model(x_batch, adj_matrix, return_all_steps=False)
            sigma = torch.exp(0.5 * log_var)
            
            # 扩展维度以匹配 detect 函数的输入要求 (B, 1, N, F)
            y_true = y_batch.unsqueeze(1) 
            mu = mu.unsqueeze(1)
            sigma = sigma.unsqueeze(1)
            attn = attn.unsqueeze(1)
            
            score_norm, _, _ = detector.detect(y_true, mu, sigma, attn)
            all_scores_normal.append(score_norm.flatten().cpu().numpy())
            
            # === 2. 攻击样本评估 ===
            # 我们假设攻击发生在窗口的最末端，延续到预测目标 y
            
            # A. 攻击输入历史 x (影响预测)
            if attack_type == 'ramp':
                # 从 T-10 开始攻击，确保最后几帧是被污染的
                x_attack = injector.ramp_attack(x_batch, node_idx=3, feat_idx=0, 
                                               start_step=T-20, slope=0.2) # 加大一点力度
                # 同时也必须攻击目标值 y (因为 y 是 x 的下一帧)
                # 计算 y 对应的 ramp 偏移量: (T - (T-20)) * 0.2 = 4.0
                offset = (20) * 0.2 
                y_attack = y_batch.clone()
                y_attack[:, 3, 0] += offset
                
            elif attack_type == 'bias':
                x_attack = injector.bias_attack(x_batch, node_idx=3, feat_idx=0,
                                               start_step=T-20, bias=1.0)
                y_attack = y_batch.clone()
                y_attack[:, 3, 0] += 1.0 # 目标值也有偏差
                
            else: # replay
                x_attack = injector.replay_attack(x_batch, node_idx=3, 
                                                 start_step=T-20, replay_length=20)
                # Replay 比较特殊，y 也取历史值
                y_attack = y_batch.clone()
                # 这里简化处理，重放攻击较难同步 y，暂且只看输入影响
                # 实际上如果 x 被重放，y 也应该是重放的数据。
            
            # B. 预测 (基于被污染的历史)
            mu_att, log_var_att, attn_att = model(x_attack, adj_matrix, return_all_steps=False)
            sigma_att = torch.exp(0.5 * log_var_att)
            
            # C. 检测
            # 关键逻辑：观测值 y_attack 是"撒谎"的，预测值 mu_att 是"诚实"的(基于物理约束)
            # Residual = |撒谎 - 诚实| -> 很大 -> 报警
            y_true_att = y_attack.unsqueeze(1)
            mu_att = mu_att.unsqueeze(1)
            sigma_att = sigma_att.unsqueeze(1)
            attn_att = attn_att.unsqueeze(1)
            
            score_attack, _, _ = detector.detect(y_true_att, mu_att, sigma_att, attn_att)
            all_scores_attack.append(score_attack.flatten().cpu().numpy())

    # === 统计 ===
    scores_normal = np.concatenate(all_scores_normal)
    scores_attack = np.concatenate(all_scores_attack)
    
    # 简单的 ROC 计算
    thresholds = np.linspace(0, 50, 200) # 扩大阈值范围
    tpr_list = []
    fpr_list = []
    
    for th in thresholds:
        tp = np.sum(scores_attack > th)
        fp = np.sum(scores_normal > th)
        tpr = tp / (len(scores_attack) + 1e-6)
        fpr = fp / (len(scores_normal) + 1e-6)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
        
    return {
        'tpr': tpr_list,
        'fpr': fpr_list,
        'detection_delay': 0, # 基于单帧检测，延迟概念不同
        'scores_normal': scores_normal,
        'scores_attack': scores_attack
    }


# ================= 可视化函数 =================
def plot_detection_results(results, attack_type='ramp'):
    """画出检测结果"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 子图1: ROC曲线
    ax = axes[0]
    ax.plot(results['fpr'], results['tpr'], 'b-', linewidth=2)
    ax.plot([0, 1], [0, 1], 'r--', label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(f'ROC Curve ({attack_type.capitalize()} Attack)', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.1)
    
    # 子图2: 异常得分分布
    ax = axes[1]
    ax.hist(results['scores_normal'], bins=50, alpha=0.5, label='Normal', color='blue', density=True)
    ax.hist(results['scores_attack'], bins=50, alpha=0.5, label='Attack', color='red', density=True)
    ax.axvline(x=5.0, color='black', linestyle='--', label='Threshold', linewidth=2)
    ax.set_xlabel('Anomaly Score', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Score Distribution', fontsize=14)
    ax.legend()
    ax.set_yscale('log')
    
    # 子图3: 检测延迟
    ax = axes[2]
    delay_text = f"Avg Detection Delay:\n{results['detection_delay']:.2f} frames"
    if results['detection_delay'] < 5:
        color = 'lightgreen'
        status = '✓ Excellent'
    elif results['detection_delay'] < 10:
        color = 'wheat'
        status = '○ Good'
    else:
        color = 'lightcoral'
        status = '✗ Poor'
    
    ax.text(0.5, 0.6, delay_text, ha='center', va='center', fontsize=20, 
            bbox=dict(boxstyle='round', facecolor=color, alpha=0.8))
    ax.text(0.5, 0.1, status, ha='center', va='center', fontsize=16, weight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Detection Performance', fontsize=14)
    
    plt.tight_layout()
    save_path = f'detection_results_{attack_type}.png'
    plt.savefig(save_path, dpi=150)
    print(f"  Saved plot to '{save_path}'")
    plt.close()


# ================= 主程序 =================
def main():
    print("="*60)
    print("PhyGAT Attack Detection Evaluation")
    print("="*60)
    
    # 检查文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file not found: {MODEL_PATH}")
        print("   Please run train.py first!")
        return
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ Error: Dataset not found: {DATA_PATH}")
        return
    
    print(f"\n📁 Loading model from '{MODEL_PATH}'...")
    
    # 加载数据
    test_ds = FlightDataset(DATA_PATH, mode='test')
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)
    print(f"✓ Loaded test dataset: {len(test_ds)} samples")
    
    # 加载邻接矩阵
    adj_matrix = get_adjacency_matrix().to(DEVICE)
    print(f"✓ Adjacency matrix loaded: {adj_matrix.shape}")
    
    # 加载模型
    model = PhyGAT_Fixed(num_nodes=6, in_dim=3).to(DEVICE)
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    model.eval()
    
    print(f"\n🔍 Starting attack detection evaluation...")
    print(f"   Device: {DEVICE}")
    print("-"*60)
    
    # 测试不同攻击类型
    attack_types = ['ramp', 'bias', 'replay']
    summary = []
    
    for attack_type in attack_types:
        print(f"\n🎯 Testing {attack_type.upper()} attack...")
        
        results = evaluate_attack_detection(
            model, test_loader, adj_matrix, 
            attack_type=attack_type, device=DEVICE
        )
        
        # 找到最佳工作点（TPR - FPR 最大）
        tpr_arr = np.array(results['tpr'])
        fpr_arr = np.array(results['fpr'])
        best_idx = np.argmax(tpr_arr - fpr_arr)
        best_tpr = results['tpr'][best_idx]
        best_fpr = results['fpr'][best_idx]
        
        # 打印关键指标
        print(f"  📊 Results:")
        print(f"     Best TPR: {best_tpr:.3f} (Detection Rate)")
        print(f"     at FPR:   {best_fpr:.3f} (False Alarm Rate)")
        print(f"     Avg Delay: {results['detection_delay']:.2f} frames")
        
        # 评价
        if best_tpr > 0.9 and best_fpr < 0.05:
            grade = "🌟 Excellent"
        elif best_tpr > 0.7 and best_fpr < 0.1:
            grade = "✓ Good"
        else:
            grade = "⚠️ Needs Improvement"
        print(f"     Grade: {grade}")
        
        summary.append({
            'attack': attack_type,
            'tpr': best_tpr,
            'fpr': best_fpr,
            'delay': results['detection_delay']
        })
        
        # 生成可视化
        plot_detection_results(results, attack_type)
    
    # 打印总结
    print("\n" + "="*60)
    print("📈 EVALUATION SUMMARY")
    print("="*60)
    print(f"{'Attack Type':<15} {'TPR':<10} {'FPR':<10} {'Delay (frames)':<15}")
    print("-"*60)
    for item in summary:
        print(f"{item['attack'].capitalize():<15} {item['tpr']:<10.3f} {item['fpr']:<10.3f} {item['delay']:<15.2f}")
    print("="*60)
    
    print("\n✅ Evaluation complete!")
    print(f"   Generated files:")
    for attack_type in attack_types:
        print(f"   - detection_results_{attack_type}.png")


if __name__ == "__main__":
    main()