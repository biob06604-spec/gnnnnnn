import torch
import torch.nn.functional as F

# ==========================================
# 1. 配置区域 (修改这里！)
# ==========================================
# 输入你想查询的那个基因的 ID
TARGET_GENE = "GRMZM2G161097"

# 显示前多少个相似基因？
TOP_K = 20

# ==========================================
# 2. 加载数据
# ==========================================
print(f"🔍 正在启动单基因搜索引擎: {TARGET_GENE} ...")

try:
    # 加载映射字典
    mapping = torch.load('complex_mapping.pt', weights_only=False)
    if 'gene_map' in mapping:
        gene_to_idx = mapping['gene_map']
    else:
        gene_to_idx = mapping.get('gene_to_idx')  # 兼容旧版

    # ID 转 名字 的字典
    idx_to_gene = {v: k for k, v in gene_to_idx.items()}

    # 加载向量
    all_z = torch.load('complex_gene_embeddings.pt', weights_only=False).cpu()
    print("✅ 数据加载完成")

except FileNotFoundError:
    print("❌ 错误：找不到 complex_mapping.pt 或 complex_gene_embeddings.pt")
    exit()

# ==========================================
# 3. 获取目标基因向量
# ==========================================
if TARGET_GENE not in gene_to_idx:
    print(f"\n❌ 抱歉，基因 {TARGET_GENE} 不在当前的图谱网络中。")
    print("可能原因：该基因在原始数据清洗时被过滤掉了。")
    exit()

target_idx = gene_to_idx[TARGET_GENE]
target_vector = all_z[target_idx]  # 获取这唯一的向量

print(f"✅ 找到基因，索引 ID: {target_idx}")

# ==========================================
# 4. 全局搜索 (核心计算)
# ==========================================
print("\n正在计算相似度...")

# unsqueeze(0) 把向量形状从 [32] 变成 [1, 32]，以便和全量数据对比
sim_scores = F.cosine_similarity(target_vector.unsqueeze(0), all_z)

# 获取前 K+1 个结果 (因为第 1 名肯定是它自己，相似度 1.0)
top_values, top_indices = torch.topk(sim_scores, k=TOP_K + 1)

# ==========================================
# 5. 打印结果
# ==========================================
print(f"\n📊 === 搜索结果: 与 {TARGET_GENE} 最像的基因 ===")
print("-" * 60)
print(f"{'Rank':<5} | {'Gene ID':<20} | {'Score':<8} | {'Note'}")
print("-" * 60)

count = 0
for i in range(len(top_indices)):
    idx = top_indices[i].item()
    score = top_values[i].item()
    gene_name = idx_to_gene[idx]

    # 跳过它自己 (相似度肯定是 1.0)
    if gene_name == TARGET_GENE:
        continue

    # 简单的标注
    note = ""
    if score > 0.99:
        note = "🔥 极度相似 (可能同源)"
    elif score > 0.95:
        note = "🌟 强相关"

    print(f"{count + 1:<5} | {gene_name:<20} | {score:.4f}   | {note}")

    count += 1
    if count >= TOP_K:
        break

print("-" * 60)