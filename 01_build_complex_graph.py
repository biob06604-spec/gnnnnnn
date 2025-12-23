import pandas as pd
import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

# ==========================================
# 1. 读取并清洗数据
# ==========================================
print("STEP 1: 读取数据...")

# --- A. 读取 PPI ---
try:
    ppi_df = pd.read_csv('cleaned_ppi.csv')
    ppi_df.columns = ['ID_A', 'ID_B', 'weight', 'type']
    print(f"PPI 数据加载成功: {len(ppi_df)} 条边")
except FileNotFoundError:
    print("❌ 找不到 cleaned_ppi.csv，请检查路径")
    exit()

# --- B. 读取 GO 数据 (根据调试结果修正) ---
try:
    # 【关键修正 1】sep=',' (逗号分隔)
    # 【关键修正 2】dtype={'goid': str} (防止 0006397 变成 6397)
    go_df = pd.read_csv('cleaned_go_data.csv', sep=',', dtype={'goid': str})

    print(f"GO 数据加载成功: {len(go_df)} 条注释")

    # 【关键修正 3】清洗 Gene ID: 去掉 "_P01" 等后缀
    # 你的列名是 'qpid', 'goid', 'ontology', 'ARGOT_PPV'
    go_df['gene_clean'] = go_df['qpid'].astype(str).str.split('_').str[0]

    # 规范化 GO ID: 确保是字符串
    # 如果原始数据里已经是 0006397，这里不动它
    # 如果你想加上 'GO:' 前缀，可以取消下面这行的注释
    # go_df['goid'] = 'GO:' + go_df['goid']

    print("GO 数据预览 (清洗后):")
    print(go_df[['gene_clean', 'goid', 'ARGOT_PPV']].head(3))

except Exception as e:
    print(f"❌ 读取 GO 数据失败: {e}")
    exit()

# ==========================================
# 2. 建立统一的索引 (Mapping)
# ==========================================
print("\nSTEP 2: 构建索引...")

# 1. 所有的基因 ID (PPI 的两列 + GO 清洗后的一列)
all_genes = set(ppi_df['ID_A']).union(set(ppi_df['ID_B'])).union(set(go_df['gene_clean']))
all_genes = sorted(list(all_genes))
gene_to_idx = {name: i for i, name in enumerate(all_genes)}

# 2. 所有的 GO ID
all_gos = sorted(list(set(go_df['goid'])))
go_to_idx = {name: i for i, name in enumerate(all_gos)}

print(f"Gene 节点数量: {len(all_genes)}")
print(f"GO 节点数量: {len(all_gos)}")

# ==========================================
# 3. 构建 HeteroData
# ==========================================
print("\nSTEP 3: 构建异构图...")
data = HeteroData()

# 设置节点数
data['gene'].num_nodes = len(all_genes)
data['go'].num_nodes = len(all_gos)

# --- A. 构建 Gene-PPI-Gene 边 ---
src_ppi = [gene_to_idx[g] for g in ppi_df['ID_A']]
dst_ppi = [gene_to_idx[g] for g in ppi_df['ID_B']]
# 边索引
data['gene', 'interacts', 'gene'].edge_index = torch.tensor([src_ppi, dst_ppi], dtype=torch.long)
# 边权重 (PPI score)
data['gene', 'interacts', 'gene'].edge_weight = torch.tensor(ppi_df['weight'].values, dtype=torch.float)

# --- B. 构建 Gene-GO 边 ---
# 筛选有效数据 (防止有些基因在 GO 表里有但在 gene_to_idx 里没有)
# 这一步非常重要，只保留在 PPI 网络中出现过的基因的注释
valid_go_df = go_df[go_df['gene_clean'].isin(gene_to_idx)]

src_go = [gene_to_idx[g] for g in valid_go_df['gene_clean']]
dst_go = [go_to_idx[g] for g in valid_go_df['goid']]

# 边索引
data['gene', 'annotated_with', 'go'].edge_index = torch.tensor([src_go, dst_go], dtype=torch.long)
# 边权重 (ARGOT_PPV)
data['gene', 'annotated_with', 'go'].edge_weight = torch.tensor(valid_go_df['ARGOT_PPV'].values, dtype=torch.float)

# 转换无向图 (让信息双向流动)
data = T.ToUndirected()(data)

print("\n✅ 复杂网络构建完成！")
print(data)

# 保存
torch.save(data, 'complex_hetero_data.pt')
torch.save({'gene_map': gene_to_idx, 'go_map': go_to_idx}, 'complex_mapping.pt')