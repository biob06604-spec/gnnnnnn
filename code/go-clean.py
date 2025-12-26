import pandas as pd

# 1. 设置文件名 (按照你的要求)
input_file = 'go_data.txt'  # 原始数据文件
output_file = 'cleaned_go_data.csv'  # 输出的新文件

# 2. 读取数据 (关键修改在这里)
# dtype={'goid': str} 强制告诉 pandas：把 'goid' 这一列当成纯文本读取，不要把它变成数字
try:
    df = pd.read_csv(input_file, sep='\t', dtype={'goid': str})
except:
    # 如果分隔符不是制表符，尝试自动匹配空格
    df = pd.read_csv(input_file, sep=r'\s+', dtype={'goid': str})

print(f"原始数据读取成功，共 {len(df)} 行。")

# 3. 数据清洗与筛选
# ---------------------------------------------------------
# 步骤 A: (可选) 过滤低质量数据
# 这里的逻辑是：只保留 rank <= 5 的数据。
# 如果你想保留所有数据，可以在下一行代码前加 # 注释掉
df_clean = df[df['ARGOT_rank'] <= 5].copy()

# 如果你不需要过滤，只想保留特定列，请取消下面这行的注释，并注释掉上面那行：
# df_clean = df.copy()

# 步骤 B: 只保留对 GNN 有用的列
selected_columns = ['qpid', 'goid', 'ontology', 'ARGOT_PPV']
df_final = df_clean[selected_columns]
# ---------------------------------------------------------

# 4. 检查数据类型 (验证前导零是否保留)
print("\n数据预览 (注意检查 goid 是否保留了 0):")
print(df_final.head())

# 5. 保存到新文件
# quoting=1 (csv.QUOTE_ALL) 或保持默认，只要是字符串，pandas保存时通常很安全
# 这里的 index=False 表示不保存行索引
df_final.to_csv(output_file, index=False, sep=',')

print(f"\n处理完成！")

print(f"文件已保存至: {output_file}")
