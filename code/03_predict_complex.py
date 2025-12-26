import torch
import torch.nn.functional as F

# ==========================================
# 1. 配置区域 (Configuration)
# ==========================================

KNOWN_TRAIT_GENES = [
    "GRMZM2G161097",  # ZmVPP1 (发在 Nature Genetics 的抗旱基因)
    "GRMZM2G127379",  # ZmNAC111 (幼苗抗旱)
    "GRMZM2G070054",  # ZmDREB2A (经典抗逆转录因子)
    "GRMZM2G051283",  # ZmARGOS1 (乙烯敏感性，抗旱)
    "GRMZM2G014902",  # ZmABA1 (ABA合成，气孔调节)
]

TRAIT_NAME = "抗旱性 (Drought Resistance)"

# 相似度过滤器：是否隐藏分数过高(>0.999)的基因？
# True = 隐藏（这通常是同源基因/家族基因，如果你想找新机制，建议设为 True）
# False = 显示所有（如果你想看所有相关基因，设为 False）
FILTER_CLONES = False

# ==========================================
# 2. 加载复杂模型数据
# ==========================================
print(f"正在启动 {TRAIT_NAME} 预测引擎...")

try:
    # A. 加载映射字典
    # 注意：文件名是 complex_mapping.pt
    mapping = torch.load('complex_mapping.pt', weights_only=False)

    # 提取基因映射 (Name -> ID)
    if 'gene_map' in mapping:
        gene_to_idx = mapping['gene_map']
    else:
        # 兼容旧版本
        gene_to_idx = mapping.get('gene_to_idx', None)

    if gene_to_idx is None:
        raise ValueError("字典中找不到 gene_map，请检查 complex_mapping.pt")

    # 反转字典 (ID -> Name) 用于最后打印名字
    idx_to_gene = {v: k for k, v in gene_to_idx.items()}

    # B. 加载训练好的向量
    # 注意：文件名是 complex_gene_embeddings.pt
    all_z = torch.load('complex_gene_embeddings.pt', weights_only=False).cpu()

    print("✅ 复杂异构模型数据加载完成！")
    print(f"   - 基因总数: {len(gene_to_idx)}")
    print(f"   - 向量维度: {all_z.shape}")

except FileNotFoundError:
    print("❌ 错误：找不到文件。请确保你运行完了 01_build_complex_graph.py 和 02_train_complex.py")
    exit()

# ==========================================
# 3. 计算“性状中心” (Trait Centroid)
# ==========================================
print("\n正在计算特征中心...")

valid_indices = []
print("种子基因状态:")
for gene in KNOWN_TRAIT_GENES:
    if gene in gene_to_idx:
        idx = gene_to_idx[gene]
        valid_indices.append(idx)
        print(f"  [√] 找到: {gene} (ID: {idx})")
    else:
        print(f"  [x] 未找到: {gene} (可能不在 PPI 网络中)")

if len(valid_indices) == 0:
    print("❌ 错误：所有种子基因都不在网络中，无法预测。请更换种子基因。")
    exit()

# 提取种子向量
seed_vectors = all_z[valid_indices]

# 计算平均向量 (中心点)
centroid = torch.mean(seed_vectors, dim=0)

# ==========================================
# 4. 全局搜索 (Global Search)
# ==========================================
print(f"\n正在全基因组 ({len(all_z)} 个基因) 中搜索潜在候选者...")

# 计算余弦相似度
sim_scores = F.cosine_similarity(centroid.unsqueeze(0), all_z)

# 取出前 50 名
top_k = 50
top_values, top_indices = torch.topk(sim_scores, k=top_k)

# ==========================================
# 5. 展示结果
# ==========================================
print(f"\n🏆 === {TRAIT_NAME} 预测结果 (Top Candidates) ===")
if FILTER_CLONES:
    print("   (注：已过滤掉相似度 > 0.999 的高度同源基因)")

print("-" * 65)
print(f"{'Rank':<5} | {'Gene ID':<20} | {'Score':<8} | {'Status'}")
print("-" * 65)

count = 0
for i in range(len(top_indices)):
    idx = top_indices[i].item()
    score = top_values[i].item()
    gene_name = idx_to_gene[idx]

    # 1. 跳过种子基因自己
    if gene_name in KNOWN_TRAIT_GENES:
        continue

    # 2. (可选) 过滤掉分数过高的克隆基因
    if FILTER_CLONES and score > 0.999:
        continue

    # 打印结果
    status = "🌟 新发现"
    # 如果分数特别高，标记为强相关
    if score > 0.98: status += " (强相关)"

    print(f"{count + 1:<5} | {gene_name:<20} | {score:.4f}   | {status}")

    count += 1
    if count >= 15:  # 只显示前 15 个
        break

print("-" * 65)
print("💡 建议：")
print("1. 复制上面的 'Gene ID' 去 MaizeGDB 或 NCBI 搜索。")
print("2. 重点关注 GO 注释与'胁迫响应(Stress Response)'相关的基因。")
