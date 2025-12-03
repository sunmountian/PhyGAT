import pandas as pd
import numpy as np
import os
import glob
import pickle
import argparse

# ================= 配置 =================
TARGET_FREQ = '20ms'
ARMED_THRESHOLD = 1000

# 特征配置 (与训练保持一致)
FEATURES_CONFIG = {
    'actuators': {'file_pattern': 'actuator_outputs.csv', 'columns': ['output[0]', 'output[1]', 'output[2]', 'output[3]'], 'resample_method': 'mean', 'fill_method': 'ffill'},
    'accel': {'file_pattern': 'vehicle_imu.csv', 'columns': ['delta_velocity[0]', 'delta_velocity[1]', 'delta_velocity[2]', 'delta_velocity_dt'], 'rename': ['dv_x', 'dv_y', 'dv_z', 'dt_v'], 'resample_method': 'special_imu_accel'},
    'gyro': {'file_pattern': 'vehicle_imu.csv', 'columns': ['delta_angle[0]', 'delta_angle[1]', 'delta_angle[2]', 'delta_angle_dt'], 'rename': ['da_x', 'da_y', 'da_z', 'dt_a'], 'resample_method': 'special_imu_gyro'},
    'baro': {'file_pattern': 'vehicle_air_data.csv', 'columns': ['baro_alt_meter'], 'rename': ['baro_alt'], 'resample_method': 'interpolate'},
    'gps': {'file_pattern': 'vehicle_gps_position.csv', 'columns': ['vel_n_m_s', 'vel_e_m_s'], 'rename': ['gps_vel_n', 'gps_vel_e'], 'resample_method': 'interpolate'},
    'mag': {'file_pattern': 'vehicle_magnetometer.csv', 'columns': ['magnetometer_ga[0]', 'magnetometer_ga[1]', 'magnetometer_ga[2]'], 'rename': ['mag_x', 'mag_y', 'mag_z'], 'resample_method': 'interpolate'}
}

def process_single_flight(folder_path):
    dfs = []
    for node_name, config in FEATURES_CONFIG.items():
        search_path = os.path.join(folder_path, config['file_pattern'])
        files = glob.glob(search_path)
        if not files: return None
        file_path = files[0]
        try:
            df = pd.read_csv(file_path)
            cols = ['timestamp'] + config['columns']
            if not all(c in df.columns for c in cols): return None
            df = df[cols].copy()
            if 'rename' in config: df = df.rename(columns=dict(zip(config['columns'], config['rename'])))
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='us')
            df = df.set_index('timestamp')

            if config['resample_method'] == 'special_imu_accel':
                dt_sec = df['dt_v'] * 1e-6
                df_res = pd.DataFrame({'acc_x': df['dv_x']/dt_sec, 'acc_y': df['dv_y']/dt_sec, 'acc_z': df['dv_z']/dt_sec}, index=df.index).resample(TARGET_FREQ).mean()
            elif config['resample_method'] == 'special_imu_gyro':
                dt_sec = df['dt_a'] * 1e-6
                df_res = pd.DataFrame({'gyro_x': df['da_x']/dt_sec, 'gyro_y': df['da_y']/dt_sec, 'gyro_z': df['da_z']/dt_sec}, index=df.index).resample(TARGET_FREQ).mean()
            elif config['resample_method'] == 'mean':
                df_res = df.resample(TARGET_FREQ).mean()
            else:
                df_res = df.resample(TARGET_FREQ).mean().interpolate(method='linear')
            if 'fill_method' in config: 
                df_res = df_res.fillna(method=config['fill_method'])
            else:
                df_res = df_res.fillna(method='ffill').fillna(method='bfill')
            dfs.append(df_res)
        except: return None
    
    df_merged = pd.concat(dfs, axis=1).dropna()
    if len(df_merged) < 50: return None
    
    # 过滤未解锁
    actuator_cols = ['output[0]', 'output[1]', 'output[2]', 'output[3]']
    if all(col in df_merged.columns for col in actuator_cols):
        motor_mean = df_merged[actuator_cols].mean(axis=1)
        if motor_mean.mean() < ARMED_THRESHOLD:
            df_merged = df_merged[motor_mean > ARMED_THRESHOLD]
    
    return df_merged

def main(input_dir, output_name, scaler_path):
    # 如果没有提供输出名称，从输入目录名中提取
    if output_name is None:
        # 提取输入目录的最后一个目录名作为文件名
        input_basename = os.path.basename(os.path.normpath(input_dir))
        output_name = f"dataset_{input_basename}.npy"
    else:
        # 如果提供了输出名称，确保有 .npy 扩展名
        if not output_name.endswith('.npy'):
            output_name = f"{output_name}.npy"
    
    print(f"处理数据: {input_dir} -> {output_name}")
    with open(scaler_path, 'rb') as f: scaler = pickle.load(f)
    mean, std = scaler['mean'], scaler['std']
    
    all_flights = []
    for root, dirs, files in os.walk(input_dir):
        if any('actuator_outputs' in f for f in files) and any('vehicle_imu' in f for f in files):
            df = process_single_flight(root)
            if df is not None: all_flights.append(df)

    if not all_flights:
        print("[Error] 未找到有效数据")
        return

    df_master = pd.concat(all_flights)
    cols = ['output[0]', 'output[1]', 'output[2]', 'output[3]', 'acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'baro_alt', 'gps_vel_n', 'gps_vel_e', 'mag_x', 'mag_y', 'mag_z']
    
    # 检查列是否存在
    missing_cols = [c for c in cols if c not in df_master.columns]
    if missing_cols:
        print(f"[Error] 缺少列: {missing_cols}")
        print(f"[Info] 当前列: {df_master.columns.tolist()}")
        return
    
    # === 关键：使用训练集的 mean/std 进行归一化 ===
    df_norm = (df_master[cols] - mean) / (std + 1e-6)
    
    data_npy = np.zeros((len(df_norm), 6, 3))
    data_npy[:, 0, :] = df_norm[['output[0]', 'output[1]', 'output[2]']].values
    data_npy[:, 1, :] = df_norm[['acc_x', 'acc_y', 'acc_z']].values
    data_npy[:, 2, :] = df_norm[['gyro_x', 'gyro_y', 'gyro_z']].values
    data_npy[:, 3, 0] = df_norm['baro_alt'].values
    data_npy[:, 4, 0] = df_norm['gps_vel_n'].values; data_npy[:, 4, 1] = df_norm['gps_vel_e'].values
    data_npy[:, 5, :] = df_norm[['mag_x', 'mag_y', 'mag_z']].values
    
    # 创建输出目录
    output_dir = 'dataset'
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, output_name)
    np.save(output_path, data_npy)
    print(f"[成功] 生成: {output_path}, 形状: {data_npy.shape}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='输入数据目录路径')
    parser.add_argument('--output', required=False, default=None, help='输出文件名（可选，默认从输入目录名自动生成）')
    parser.add_argument('--scaler', default='dataset/scaler_params.pkl', help='归一化参数文件路径')
    args = parser.parse_args()
    main(args.input, args.output, args.scaler)