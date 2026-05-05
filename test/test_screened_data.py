import pandas as pd

# 替换为你的文件路径和列名
file_path = 'F:/Desktop/Final/workspace/preprocessed_data/screened_preprocessed_train.csv'
column_name = 'meter_reading'

# 核心技巧：使用 usecols=[column_name] 只读取需要的列，极大节省内存
# 提示：如果文件包含中文或特殊字符，可能需要加上 encoding='utf-8' 或 'gbk'
print("正在读取数据...")
df = pd.read_csv(file_path, usecols=[column_name])

# 计算统计值
print("正在计算...")
max_val = df[column_name].max()
min_val=df[column_name].min()
mean_val = df[column_name].mean()
median_val = df[column_name].median()

# 输出结果
print("-" * 20)
print(f"列名: {column_name}")
print(f"最大值 (Max): {max_val}")
print(f"最小值 (Min):{min_val}")
print(f"平均值 (Mean): {mean_val}")
print(f"中位数 (Median): {median_val}")