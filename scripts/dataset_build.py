import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ================= 配置区域 =================
# 时间窗口大小
# 50Hz 采样率下，50 代表看过去 1秒 的数据
WINDOW_SIZE = 50 

# 预测未来步数
# 1 代表预测紧接着的下一帧
PREDICTION_HORIZON = 1 

# 训练集占比 (前 80% 训练，后 20% 测试)
TRAIN_RATIO = 0.8 

# ================= 1. 定义物理邻接矩阵 (核心创新点) =================
def get_adjacency_matrix():
    """
    定义 6-DoF 无人机的物理连接关系图。
    节点索引: 
    0: Actuators (电机)
    1: Accel (加速度计)
    2: Gyro (陀螺仪)
    3: Baro (气压计)
    4: GPS (GPS速度)
    5: Mag (磁力计)
    """
    # 初始化 6x6 矩阵，对角线为1 (自连接，节点通过自注意力保留自身信息)
    # adj[i, j] = 1 表示从节点 i 指向 节点 j 的边
    adj = np.eye(6, dtype=np.float32)

    # 【新增】强制切断 Baro 的自连接 (Blindfold the Baro)
    # 3 是 Baro 的索引
    adj[3, 3] = 0
    
    # --- 定义物理因果边 (Source -> Target) ---
    
    # ✅ 正确做法：降低但不完全切断
    adj[3, 3] = 0.3  # 保留部分自连接（记忆历史）
    
    # 物理因果边
    adj[0, 1] = 1  # Act -> Accel
    adj[0, 2] = 1  # Act -> Gyro
    adj[2, 1] = 1  # Gyro <-> Accel 耦合
    adj[1, 2] = 1
    adj[1, 3] = 1  # Accel_Z -> Baro（关键边）
    adj[1, 4] = 1  # Accel_XY -> GPS
    adj[2, 5] = 1  # Gyro_Z -> Mag
    
    # 新增：Baro反向验证边（弱连接）
    adj[3, 1] = 0.5  # Baro -> Accel（反向验证）
    adj = adj.T
    return torch.FloatTensor(adj)

# ================= 2. PyTorch 数据集类 =================
class FlightDataset(Dataset):
    def __init__(self, data_path, window_size=WINDOW_SIZE, mode='train'):
        """
        data_path: .npy 文件路径
        window_size: 滑窗大小
        mode: 'train' 或 'test'
        """
        # 加载归一化后的数据 (N, 6, 3)
        self.raw_data = np.load(data_path)
        self.raw_data = self.raw_data.astype(np.float32) # 转为 float32
        
        self.window_size = window_size
        self.horizon = PREDICTION_HORIZON
        
        # 按时间顺序划分数据集
        total_len = len(self.raw_data)
        split_idx = int(total_len * TRAIN_RATIO)
        
        if mode == 'train':
            self.data = self.raw_data[:split_idx]
        elif mode == 'test':
            self.data = self.raw_data[split_idx:]
        else: # 'all'
            self.data = self.raw_data
            
        # 计算有效样本数
        # 样本 i 需要: data[i : i+window] 作为输入, data[i+window] 作为标签
        self.n_samples = len(self.data) - self.window_size - self.horizon + 1
        
        if self.n_samples <= 0:
            raise ValueError(f"数据量太少 ({len(self.data)})，不足以构建长度为 {window_size} 的窗口。")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # 输入 X: 过去的 window_size 帧
        # 形状: (Window, 6, 3)
        x_window = self.data[idx : idx + self.window_size]
        
        # 标签 Y: 紧接着的下一帧 (或者是未来 horizon 帧)
        # 形状: (6, 3)
        # 我们预测的是窗口结束后的第 horizon 帧
        y_target = self.data[idx + self.window_size + self.horizon - 1]
        
        return torch.from_numpy(x_window), torch.from_numpy(y_target)

# ================= 测试代码 (直接运行此脚本) =================
if __name__ == "__main__":
    import sys
    import os
    
    # 假设你的 .npy 文件路径
    NPY_PATH = 'dataset/flight_dataset.npy'  # 请根据实际情况修改
    
    if not os.path.exists(NPY_PATH):
        # 如果 dataset 文件夹下没有，试试当前目录
        NPY_PATH = 'flight_dataset.npy'
    
    if not os.path.exists(NPY_PATH):
        print(f"[错误] 找不到 {NPY_PATH}。请先运行数据处理脚本生成 .npy 文件。")
        sys.exit(1)

    print(f"正在加载数据集: {NPY_PATH} ...")
    
    # 1. 实例化 Dataset
    try:
        train_ds = FlightDataset(NPY_PATH, mode='train')
        test_ds = FlightDataset(NPY_PATH, mode='test')
        
        print(f"[OK] 训练集样本数: {len(train_ds)}")
        print(f"[OK] 测试集样本数: {len(test_ds)}")
        
        # 2. 实例化 DataLoader (模拟训练时的行为)
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        
        # 3. 取出一个 Batch 检查形状
        # x_batch: (Batch, Window, Nodes, Features)
        # y_batch: (Batch, Nodes, Features)
        x_batch, y_batch = next(iter(train_loader))
        
        print("\n--- 数据形状检查 ---")
        print(f"Input Batch (X): {x_batch.shape}")
        print(f"Target Batch (Y): {y_batch.shape}")
        
        expected_x_shape = (32, WINDOW_SIZE, 6, 3)
        expected_y_shape = (32, 6, 3)
        
        if x_batch.shape == expected_x_shape and y_batch.shape == expected_y_shape:
            print("\n✅ 数据管道 (Pipeline) 测试通过！")
            print("   可以开始编写 GAT 模型代码了。")
        else:
            print(f"\n❌ 形状不匹配！预期: {expected_x_shape}, 实际: {x_batch.shape}")

        # 4. 检查邻接矩阵
        adj = get_adjacency_matrix()
        print("\n--- 物理邻接矩阵 ---")
        print(adj)
        print(f"Adj Shape: {adj.shape}")

    except Exception as e:
        print(f"\n[错误] 数据集加载失败: {e}")
        import traceback
        traceback.print_exc()