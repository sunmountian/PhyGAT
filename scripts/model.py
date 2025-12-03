import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PhysicsGATLayer(nn.Module):
    """
    物理约束图注意力层 (Dense Implementation)
    强制应用 Adjacency Mask，切断非物理连接的梯度流。
    """
    def __init__(self, in_features, out_features, dropout=0.2, alpha=0.2):
        super(PhysicsGATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha

        # 线性变换 W
        self.W = nn.Linear(in_features, out_features, bias=False)
        # 注意力向量 a
        self.a = nn.Linear(2 * out_features, 1, bias=False)
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj_mask):
        """
        h: (Batch_Size * Time, Nodes, In_Features)
        adj_mask: (Nodes, Nodes) - 物理邻接矩阵 (0/1)
        """
        batch_size, N, _ = h.size()
        
        # 1. 线性变换: h_prime = hW
        h_prime = self.W(h) # (B*T, N, Out_Features)

        # 2. 构建节点对特征: [Wh_i || Wh_j]
        # 这里使用广播机制构建所有对
        # h_prime_repeat_1: (B*T, N, N, Out) -> 重复 N 次 (行)
        # h_prime_repeat_2: (B*T, N, N, Out) -> 重复 N 次 (列)
        h_prime_repeat_1 = h_prime.unsqueeze(2).repeat(1, 1, N, 1)
        h_prime_repeat_2 = h_prime.unsqueeze(1).repeat(1, N, 1, 1)
        
        # 拼接
        h_concat = torch.cat([h_prime_repeat_1, h_prime_repeat_2], dim=-1) # (B*T, N, N, 2*Out)
        
        # 3. 计算原始注意力分数 e_ij
        e = self.leakyrelu(self.a(h_concat).squeeze(-1)) # (B*T, N, N)

        # 4. === 核心创新: 应用物理 Mask ===
        # adj_mask 是 (N, N)，我们需要广播到 (B*T, N, N)
        # 物理上无连接的地方 (Mask=0)，分数设为 -1e9 (Softmax后为0)
        zero_vec = -1e9 * torch.ones_like(e)
        attention = torch.where(adj_mask > 0, e, zero_vec)
        
        # 5. Softmax 归一化
        attention = F.softmax(attention, dim=-1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # 6. 加权求和
        h_output = torch.matmul(attention, h_prime) # (B*T, N, Out)
        
        return h_output, attention

class PhyGAT(nn.Module):
    def __init__(self, num_nodes=6, in_dim=3, hidden_dim=64, rnn_dim=128, dropout=0.2):
        super(PhyGAT, self).__init__()
        
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        
        # 1. 特征嵌入 (Input -> Hidden)，将输入特征映射到高维隐藏空间
        self.embedding = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. 物理 GAT 层，进行空间特征提取，考虑物理连接约束
        self.gat = PhysicsGATLayer(hidden_dim, hidden_dim, dropout=dropout)
        
        # 3. 时序提取层 (GRU)
        # 输入维度: Nodes * Hidden_Dim (将图展平)
        # 这种设计让 GRU 能捕捉全图的全局时序演变
        # 将图结构在每个时间步展平，使用GRU捕捉全局时序演变
        self.rnn = nn.GRU(
            input_size=num_nodes * hidden_dim, 
            hidden_size=rnn_dim, 
            num_layers=2, 
            batch_first=True, 
            dropout=dropout
        )
        
        # 4. 预测头 (Output Head)
        # 输出维度: Nodes * In_Dim * 2 
        # *2 是因为同时输出均值(mu)和方差(log_var)
        # 输出每个节点的预测值（mu）和不确定性（log_var）
        self.head = nn.Linear(rnn_dim, num_nodes * in_dim * 2)

    def forward(self, x, adj_mask):
        """
        x: (Batch, Window, Nodes, Features)
        adj_mask: (Nodes, Nodes)
        """
        B, T, N, F = x.size()
        
        # --- 步骤 1: 空间特征提取 (GAT) ---
        # 为了高效计算，合并 Batch 和 Time 维度
        x_flat = x.view(B * T, N, F)
        
        # Embedding
        h = self.embedding(x_flat) # (B*T, N, Hidden)
        
        # GAT (输入物理掩码)
        h_spatial, attn_weights = self.gat(h, adj_mask) # (B*T, N, Hidden)
        
        # 恢复时间维度
        h_spatial = h_spatial.view(B, T, N, self.hidden_dim)
        
        # --- 步骤 2: 时序特征提取 (GRU) ---
        # 展平节点维度: (B, T, N * Hidden)
        # 让 RNN 看到每一时刻的整张图
        rnn_input = h_spatial.view(B, T, -1)
        
        # 跑 GRU
        rnn_out, _ = self.rnn(rnn_input) # (B, T, RNN_Dim)
        
        # 取最后一个时间步 (Many-to-One)
        last_step_feature = rnn_out[:, -1, :] # (B, RNN_Dim)
        
        # --- 步骤 3: 预测输出 ---
        output = self.head(last_step_feature) # (B, N*F*2)
        
        # 重塑为 (B, N, F, 2)
        # 最后一维: 0 -> mu (预测值), 1 -> log_var (不确定性)
        output = output.view(B, N, F, 2)
        
        mu = output[..., 0]
        log_var = output[..., 1]
        
        return mu, log_var, attn_weights