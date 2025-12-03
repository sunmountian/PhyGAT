import pandas as pd
import numpy as np
import os
import glob
import pickle
import matplotlib.pyplot as plt

# ================= 配置区域 =================

LOG_ROOT_DIR = './logs'  # 你的数据根目录
TARGET_FREQ = '20ms'    # 50Hz
ARMED_THRESHOLD = 1000  # 解锁阈值

# 特征配置 (已根据你的 CSV 文件修正)
FEATURES_CONFIG = {
    'actuators': {
        'file_pattern': 'actuator_outputs.csv',
        'columns': ['output[0]', 'output[1]', 'output[2]', 'output[3]'], 
        'resample_method': 'mean',
        'fill_method': 'ffill'
    },
    # --- 修正点：IMU 处理逻辑 ---
    'accel': {
        'file_pattern': 'vehicle_imu.csv',
        # 注意：这里我们先读取 delta 值
        'columns': ['delta_velocity[0]', 'delta_velocity[1]', 'delta_velocity[2]', 'delta_velocity_dt'],
        'rename': ['dv_x', 'dv_y', 'dv_z', 'dt_v'],
        'resample_method': 'special_imu_accel' # 使用特殊处理函数
    },
    'gyro': {
        'file_pattern': 'vehicle_imu.csv',
        # 注意：这里我们先读取 delta 值
        'columns': ['delta_angle[0]', 'delta_angle[1]', 'delta_angle[2]', 'delta_angle_dt'],
        'rename': ['da_x', 'da_y', 'da_z', 'dt_a'],
        'resample_method': 'special_imu_gyro' # 使用特殊处理函数
    },
    # -------------------------
    'baro': {
        'file_pattern': 'vehicle_air_data.csv',
        'columns': ['baro_alt_meter'],
        'rename': ['baro_alt'],
        'resample_method': 'interpolate'
    },
    'gps': {
        'file_pattern': 'vehicle_gps_position.csv',
        'columns': ['vel_n_m_s', 'vel_e_m_s'],
        'rename': ['gps_vel_n', 'gps_vel_e'],
        'resample_method': 'interpolate'
    },
    'mag': {
        'file_pattern': 'vehicle_magnetometer.csv',
        'columns': ['magnetometer_ga[0]', 'magnetometer_ga[1]', 'magnetometer_ga[2]'],
        'rename': ['mag_x', 'mag_y', 'mag_z'],
        'resample_method': 'interpolate'
    }
}

# ================= 核心处理函数 =================

def process_single_flight(folder_path, flight_id):
    dfs = []
    
    for node_name, config in FEATURES_CONFIG.items():
        search_path = os.path.join(folder_path, config['file_pattern'])
        files = glob.glob(search_path)
        
        if not files:
            # print(f"  [Warn] Missing {node_name} in {folder_path}")
            return None
        
        file_path = files[0]
        try:
            df = pd.read_csv(file_path)
            
            # 1. 检查列名是否存在
            missing_cols = [c for c in config['columns'] if c not in df.columns]
            if missing_cols:
                # print(f"  [Error] {node_name} missing columns: {missing_cols}")
                return None
            
            # 2. 提取并重命名
            cols = ['timestamp'] + config['columns']
            df = df[cols].copy()
            if 'rename' in config:
                rename_map = dict(zip(config['columns'], config['rename']))
                df = df.rename(columns=rename_map)
            
            # 3. 设置时间索引
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')
            df = df.set_index('timestamp')
            
            # 4. 特殊处理 IMU 数据 (Delta -> Rate)
            if config['resample_method'] == 'special_imu_accel':
                # 加速度 = delta_velocity / (dt * 1e-6)
                # 防止除以0，加一个小量
                dt_sec = df['dt_v'] * 1e-6
                df_res = pd.DataFrame()
                df_res['acc_x'] = df['dv_x'] / dt_sec
                df_res['acc_y'] = df['dv_y'] / dt_sec
                df_res['acc_z'] = df['dv_z'] / dt_sec
                # 转换完后再降采样
                df_res = df_res.resample(TARGET_FREQ).mean()
                
            elif config['resample_method'] == 'special_imu_gyro':
                # 角速度 = delta_angle / (dt * 1e-6)
                dt_sec = df['dt_a'] * 1e-6
                df_res = pd.DataFrame()
                df_res['gyro_x'] = df['da_x'] / dt_sec
                df_res['gyro_y'] = df['da_y'] / dt_sec
                df_res['gyro_z'] = df['da_z'] / dt_sec
                df_res = df_res.resample(TARGET_FREQ).mean()
                
            # 5. 常规处理
            elif config['resample_method'] == 'mean':
                df_res = df.resample(TARGET_FREQ).mean()
            else: # interpolate
                df_res = df.resample(TARGET_FREQ).mean().interpolate(method='linear')
            
            if 'fill_method' in config:
                df_res = df_res.fillna(method=config['fill_method'])
            
            dfs.append(df_res)
            
        except Exception as e:
            # print(f"  [Error] Processing {node_name}: {e}")
            return None

    # 合并
    df_merged = pd.concat(dfs, axis=1).dropna()
    if len(df_merged) == 0: return None

    # 过滤未解锁
    actuator_cols = ['output[0]', 'output[1]', 'output[2]', 'output[3]']
    motor_mean = df_merged[actuator_cols].mean(axis=1)
    df_flight = df_merged[motor_mean > ARMED_THRESHOLD].copy()
    
    if len(df_flight) < 50: return None
    
    df_flight['flight_id'] = flight_id
    return df_flight

# ================= 主程序 =================
if __name__ == "__main__":
    all_flights_data = []
    flight_count = 0
    
    print(f"开始扫描根目录: {LOG_ROOT_DIR}")
    
    for root, dirs, files in os.walk(LOG_ROOT_DIR):
        # 只要包含 actuator_outputs 且包含 vehicle_imu 就尝试处理
        if any('actuator_outputs' in f for f in files) and any('vehicle_imu' in f for f in files):
            print(f"Processing: {root} ...", end=" ")
            df_single = process_single_flight(root, flight_count)
            
            if df_single is not None:
                all_flights_data.append(df_single)
                flight_count += 1
                print(f"[OK] 样本数: {len(df_single)}")
            else:
                print("[Skip] 数据不完整或无效")

    if all_flights_data:
        print(f"\n正在合并 {len(all_flights_data)} 次飞行数据...")
        df_master = pd.concat(all_flights_data)
        
        # 创建输出目录
        output_dir = 'dataset'
        os.makedirs(output_dir, exist_ok=True)
        
        # 排除 flight_id 进行归一化
        cols_to_norm = [c for c in df_master.columns if c != 'flight_id']
        
        # 确保列的顺序一致
        # Actuator(4) -> Accel(3) -> Gyro(3) -> Baro(1) -> GPS(2) -> Mag(3)
        # 这里的列名必须和 process_single_flight 生成的一致
        ordered_cols = [
            'output[0]', 'output[1]', 'output[2]', 'output[3]', # Act
            'acc_x', 'acc_y', 'acc_z',                          # Accel
            'gyro_x', 'gyro_y', 'gyro_z',                       # Gyro
            'baro_alt',                                         # Baro
            'gps_vel_n', 'gps_vel_e',                           # GPS
            'mag_x', 'mag_y', 'mag_z'                           # Mag
        ]
        
        # 检查是否所有列都存在
        if not all(col in df_master.columns for col in ordered_cols):
             print(f"[Error] 缺少列。当前列: {df_master.columns}")
        else:
            df_for_norm = df_master[ordered_cols]
            mean = df_for_norm.mean()
            std = df_for_norm.std()
            
            with open(os.path.join(output_dir, 'scaler_params.pkl'), 'wb') as f:
                pickle.dump({'mean': mean, 'std': std}, f)
                
            df_norm = (df_for_norm - mean) / (std + 1e-6)
            
            # 构建 Tensor (N, 6, 3)
            # 注意：Actuator有4个，GPS有2个，Baro有1个。
            # 策略：
            # Act: 取前3个 或者 压缩 (这里取前3个示例)
            # Baro/GPS: 补0
            
            data_npy = np.zeros((len(df_norm), 6, 3))
            
            # Node 0: Actuators (取前3个电机)
            data_npy[:, 0, :] = df_norm[['output[0]', 'output[1]', 'output[2]']].values
            # Node 1: Accel
            data_npy[:, 1, :] = df_norm[['acc_x', 'acc_y', 'acc_z']].values
            # Node 2: Gyro
            data_npy[:, 2, :] = df_norm[['gyro_x', 'gyro_y', 'gyro_z']].values
            # Node 3: Baro
            data_npy[:, 3, 0] = df_norm['baro_alt'].values
            # Node 4: GPS
            data_npy[:, 4, 0] = df_norm['gps_vel_n'].values
            data_npy[:, 4, 1] = df_norm['gps_vel_e'].values
            # Node 5: Mag
            data_npy[:, 5, :] = df_norm[['mag_x', 'mag_y', 'mag_z']].values
            
            output_path = os.path.join(output_dir, 'flight_dataset.npy')
            np.save(output_path, data_npy)
            print(f"\n[成功] 数据集已生成: {output_path} {data_npy.shape}")
    else:
        print("\n[错误] 未生成任何数据。")