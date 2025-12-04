import torch
import torch.nn as nn
import torch.nn.functional as F

class PhysicsGATLayer(nn.Module):
    """物理约束图注意力层 - 支持逐帧计算"""
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super(PhysicsGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        self.W = nn.Linear(in_features, out_features, bias=False)
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj_mask):
        """
        h: (Batch_Size, Nodes, In_Features)
        adj_mask: (Nodes, Nodes)
        返回: (B, N, Out), (B, N, N) - 输出特征和注意力权重
        """
        B, N, _ = h.size()
        h_prime = self.W(h)

        # 构建节点对
        h_prime_repeat_1 = h_prime.unsqueeze(2).repeat(1, 1, N, 1)
        h_prime_repeat_2 = h_prime.unsqueeze(1).repeat(1, N, 1, 1)
        h_concat = torch.cat([h_prime_repeat_1, h_prime_repeat_2], dim=-1)
        
        e = self.leakyrelu(self.a(h_concat).squeeze(-1))

        # 应用物理掩码
        zero_vec = -1e9 * torch.ones_like(e)
        attention = torch.where(adj_mask > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        h_output = torch.matmul(attention, h_prime)
        return h_output, attention


class PhyGAT_Fixed(nn.Module):
    """修复版PhyGAT - 逐帧预测 + 可检测的架构"""
    def __init__(self, num_nodes=6, in_dim=3, hidden_dim=64, rnn_dim=128, dropout=0.2):
        super(PhyGAT_Fixed, self).__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.rnn_dim = rnn_dim
        
        # 1. 特征嵌入
        self.embedding = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. 物理GAT层
        self.gat = PhysicsGATLayer(hidden_dim, hidden_dim, dropout=dropout)
        
        # 3. 节点级GRU（每个节点独立建模）
        # 关键修改：不再展平全图，而是每个节点维护独立的RNN
        self.node_rnn = nn.GRU(
            input_size=hidden_dim,
            hidden_size=rnn_dim,
            num_layers=1,
            batch_first=True,
            dropout=0
        )
        
        # 4. 预测头（每个节点独立预测）
        self.predictor = nn.Linear(rnn_dim, in_dim * 2)  # mu + log_var

    def forward(self, x, adj_mask, return_all_steps=False):
        """
        x: (Batch, Window, Nodes, Features)
        adj_mask: (Nodes, Nodes)
        return_all_steps: 是否返回所有时间步的预测（检测时需要）
        
        返回:
        - mu: (B, T, N, F) 或 (B, N, F)
        - log_var: 同上
        - attentions: (B, T, N, N) 或 (B, N, N)
        """
        B, T, N, F = x.size()
        
        # 存储每个时间步的输出
        all_mu = []
        all_log_var = []
        all_attentions = []
        
        # 节点级隐状态初始化
        h_rnn = torch.zeros(1, B * N, self.rnn_dim).to(x.device)
        
        # 逐帧处理
        for t in range(T):
            x_t = x[:, t, :, :]  # (B, N, F)
            
            # Step 1: 空间特征提取
            h_emb = self.embedding(x_t)  # (B, N, Hidden)
            h_spatial, attn = self.gat(h_emb, adj_mask)  # (B, N, Hidden), (B, N, N)
            
            # Step 2: 时序特征提取（节点独立）
            # 重塑: (B, N, Hidden) -> (B*N, 1, Hidden)
            h_spatial_flat = h_spatial.view(B * N, 1, self.hidden_dim)
            
            # 跑GRU
            rnn_out, h_rnn = self.node_rnn(h_spatial_flat, h_rnn)  # (B*N, 1, RNN_Dim)
            rnn_out = rnn_out.squeeze(1)  # (B*N, RNN_Dim)
            
            # Step 3: 预测
            pred = self.predictor(rnn_out)  # (B*N, F*2)
            pred = pred.view(B, N, F, 2)
            
            mu_t = pred[..., 0]
            log_var_t = pred[..., 1]
            
            all_mu.append(mu_t)
            all_log_var.append(log_var_t)
            all_attentions.append(attn)
        
        # 堆叠结果
        all_mu = torch.stack(all_mu, dim=1)  # (B, T, N, F)
        all_log_var = torch.stack(all_log_var, dim=1)
        all_attentions = torch.stack(all_attentions, dim=1)  # (B, T, N, N)
        
        if return_all_steps:
            return all_mu, all_log_var, all_attentions
        else:
            # 训练时只返回最后一步
            return all_mu[:, -1], all_log_var[:, -1], all_attentions[:, -1]


# class AttackDetector:
#     """基于双重陷阱的异常检测器 - 修复版"""
#     def __init__(self, phy_edges, threshold_res=3.0, threshold_struct=0.1):
#         self.phy_edges = phy_edges
#         self.threshold_res = threshold_res
#         self.threshold_struct = threshold_struct
    
#     def detect(self, y_true, mu, sigma, attentions):
#         """
#         y_true: (B, T, N, F) 真实值
#         mu: (B, T, N, F) 预测均值
#         sigma: (B, T, N, F) 预测标准差
#         attentions: (B, T, N, N) 注意力权重
        
#         返回: (B, T) 异常得分
#         """
#         B, T, N, F = y_true.shape
        
#         # 1. 残差异常得分 S_res - 使用标准化残差
#         residual = torch.abs(y_true - mu)
#         residual_normalized = (residual - residual.mean()) / (residual.std() + 1e-6)
#         S_res = residual_normalized.mean(dim=(2, 3))  # (B, T)
        
#         # 2. 结构异常得分 S_struct
#         S_struct = torch.zeros(B, T).to(y_true.device)
        
#         for src, tgt in self.phy_edges:
#             alpha_phy = attentions[:, :, tgt, src]  # (B, T)
#             # 降低期望阈值，更容易触发
#             violation = torch.relu(0.1 - alpha_phy)
#             S_struct += violation
        
#         S_struct = S_struct / len(self.phy_edges)
        
#         # 3. 融合判决 - 增加权重
#         anomaly_score = 2.0 * S_res + 5.0 * S_struct
        
#         return anomaly_score, S_res, S_struct

class AttackDetector:
    """基于双重陷阱的异常检测器 - 修复版"""
    # 建议将 threshold_struct 默认值改为 0.1，与训练保持一致
    def __init__(self, phy_edges, threshold_res=3.0, threshold_struct=0.1):
        self.phy_edges = phy_edges
        self.threshold_res = threshold_res
        self.threshold_struct = threshold_struct
    
    def detect(self, y_true, mu, sigma, attentions):
        """
        y_true: (B, T, N, F) 真实值
        mu: (B, T, N, F) 预测均值
        sigma: (B, T, N, F) 预测标准差 (不确定性)
        attentions: (B, T, N, N) 注意力权重
        """
        B, T, N, F = y_true.shape
        
        # === 1. 残差异常得分 S_res ===
        # 错误做法 (原代码): (residual - mean) / std -> 导致攻击被抹平
        # 正确做法: 使用预测的 sigma 进行标准化 (Mahalanobis 距离的思想)
        # Score = |True - Pred| / Sigma
        
        residual = torch.abs(y_true - mu)
        
        # 加上 1e-6 防止除零
        # 维度: (B, T, N, F)
        z_score = residual / (sigma + 1e-6) 
        
        # 对所有节点和特征取平均，得到每个时间步的异常分
        # (B, T)
        S_res = z_score.mean(dim=(2, 3)) 
        
        # === 2. 结构异常得分 S_struct ===
        S_struct = torch.zeros(B, T).to(y_true.device)
        
        for src, tgt in self.phy_edges:
            alpha_phy = attentions[:, :, tgt, src]  # (B, T)
            # 阈值建议与训练时保持一致 (0.1)
            violation = torch.relu(self.threshold_struct - alpha_phy)
            S_struct += violation
        
        S_struct = S_struct / (len(self.phy_edges) + 1e-6)
        
        # === 3. 融合判决 ===
        # 动态调整权重，通常残差是最直接的证据
        anomaly_score = S_res + 10.0 * S_struct
        
        return anomaly_score, S_res, S_struct