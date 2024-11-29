import pandas as pd
from datetime import datetime

# 假设数据存储在一个文本文件中
file_path = 'CollegeMsg.txt'

# 读取数据
data = pd.read_csv(file_path, sep=' ', header=None, names=['source', 'target', 'timestamp'])

# 将 Unix 时间戳转换为日期
data['date'] = pd.to_datetime(data['timestamp'], unit='s')

# 找到最早的日期
min_date = data['date'].min()

# 确保 min_date 是 datetime 类型
min_date = pd.to_datetime(min_date)

# 定义一个函数来计算时间差
def calculate_time_difference(data, min_date, unit='days'):
    if unit == 'minutes':
        return (data['date'] - min_date).dt.total_seconds() / 60
    elif unit == 'hours':
        return (data['date'] - min_date).dt.total_seconds() / 3600
    elif unit == 'days':
        return (data['date'] - min_date).dt.days
    elif unit == 'months':
        return (data['date'].dt.to_period('M') - min_date.to_period('M')).apply(lambda x: x.n)
    elif unit == 'years':
        return (data['date'].dt.to_period('Y') - min_date.to_period('Y')).apply(lambda x: x.n)
    else:
        raise ValueError("Unsupported time unit. Choose from 'minutes', 'hours', 'days', 'months', 'years'.")

# 选择时间单位
time_unit = 'days'  # 可以更改为 'minutes', 'hours', 'days', 'months', 'years'

# 计算时间差
data['time_difference'] = calculate_time_difference(data, min_date, unit=time_unit) + 1

# 将结果保存到新的文本文件
output_file_path = 'output.txt'
data[['source', 'target', 'time_difference']].to_csv(output_file_path, sep=' ', index=False, header=False)