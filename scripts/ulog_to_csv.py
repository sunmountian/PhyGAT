#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
功能：
1. 自动将 PX4 的 .ulg 文件分解为多个 .csv；
"""

from pyulog import ULog
import pandas as pd
import os
import sys

def ulog_to_csv(ulog_path):
    # 加载 ULog 文件
    ulog = ULog(ulog_path)

    # 输出目录：与输入的.ulg文件在同一目录
    ulog_dir = os.path.dirname(os.path.abspath(ulog_path))
    base_name = os.path.splitext(os.path.basename(ulog_path))[0]
    output_dir = os.path.join(ulog_dir, base_name + "_csv")
    os.makedirs(output_dir, exist_ok=True)

    print(f"开始解析：{ulog_path}")
    print(f"输出目录：{output_dir}")

    # 遍历 ULog 数据列表
    for topic in ulog.data_list:
        name = topic.name            # topic 名称，如 "vehicle_gps_position"
        multi_id = topic.multi_id    # 实例号 0,1,...
        data_dict = topic.data       # dict

        # dict → DataFrame（关键修复点）
        df = pd.DataFrame(data_dict)

        # 构造文件名
        if multi_id == 0:
            filename = f"{name}.csv"
        else:
            filename = f"{name}_{multi_id}.csv"

        csv_path = os.path.join(output_dir, filename)

        # 保存 CSV
        df.to_csv(csv_path, index=False)

        print(f"导出：{csv_path}")

    print("全部导出完成！")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python ulog_to_csv.py <文件名.ulg>")
        sys.exit(1)

    ulog_to_csv(sys.argv[1])
