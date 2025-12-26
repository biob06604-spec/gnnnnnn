import torch
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
import gc

# ==========================================
# 1. 读取异构图数据
# ==========================================
print("STEP 1: 读取异构图数据...")
try:
    data = torch.load('complex_hetero_data.pt', weights_only=False)
    print("✅ 数据加载成功")
except FileNotFoundError:
    print("❌ 找不到 complex_hetero_data.pt")
    exit()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


# ==========================================
# 2. 定义模型 (修复版)
# ==========================================
class GNN_Backbone(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        # 这里的 x 和 edge_index 只是占位符
        # to_hetero 会自动把它变成字典操作
        x = self.conv1(x, edge_index)
        x = x.relu()  # 直接对 x 进行 relu，不需要循环！
        x = self.conv2(x, edge_index)
        return x


# --- B. 定义主模型 (包含 Embedding 和 骨架) ---
class HeteroModel(torch.nn.Module):
    def __init__(self, num_genes, num_gos, hidden_channels, out_channels, metadata):
        super().__init__()
        # 1. 定义 Embedding
        self.gene_emb = torch.nn.Embedding(num_genes, hidden_channels)
        self.go_emb = torch.nn.Embedding(num_gos, hidden_channels)

        # 2. 实例化骨架并转换
        # 先创建一个简单的 GNN
        backbone = GNN_Backbone(hidden_channels, out_channels)
        # 【关键】使用 to_hetero 转换骨架
        # 这会自动处理 relu 和字典传递，不会报错
        self.gnn = to_hetero(backbone, metadata, aggr='sum')

    def forward(self, edge_index_dict):
        # 构造输入特征字典
        x_dict = {
            'gene': self.gene_emb.weight,
            'go': self.go_emb.weight
        }
        # 传给转换后的 GNN
        return self.gnn(x_dict, edge_index_dict)


# ==========================================
# 3. 初始化与训练
# ==========================================
HIDDEN_DIM = 32

# 初始化我们的新模型
model = HeteroModel(
    num_genes=data['gene'].num_nodes,
    num_gos=data['go'].num_nodes,
    hidden_channels=HIDDEN_DIM,
    out_channels=HIDDEN_DIM,
    metadata=data.metadata()  # 传入元数据供转换使用
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 将数据移至设备
# 注意：如果显存不足，HeteroData 可能需要保留在 CPU，只在计算时切片
# 这里假设你的显存足够放下数据
data = data.to(device)


def train():
    model.train()
    optimizer.zero_grad()

    # 1. 前向传播 (直接调用模型，它内部会处理 Embedding)
    out_dict = model(data.edge_index_dict)

    # 2. 计算 Loss (链路预测)
    # 随机采样一些正样本边
    gene_edge_index = data['gene', 'interacts', 'gene'].edge_index
    # 随机取 10000 条边用于计算 Loss (全量计算太慢)
    perm = torch.randperm(gene_edge_index.size(1))[:10000]
    edge_batch = gene_edge_index[:, perm]

    src = edge_batch[0]
    dst = edge_batch[1]

    # 获取 Gene 的输出向量
    gene_out = out_dict['gene']

    # 正样本得分
    pos_score = (gene_out[src] * gene_out[dst]).sum(dim=-1)

    # 负样本采样
    neg_src = torch.randint(0, data['gene'].num_nodes, (10000,), device=device)
    neg_dst = torch.randint(0, data['gene'].num_nodes, (10000,), device=device)
    neg_score = (gene_out[neg_src] * gene_out[neg_dst]).sum(dim=-1)

    # Loss
    scores = torch.cat([pos_score, neg_score])
    labels = torch.cat([torch.ones_like(pos_score), torch.zeros_like(neg_score)])
    loss = F.binary_cross_entropy_with_logits(scores, labels)

    loss.backward()
    optimizer.step()
    return loss.item()


print("\nSTEP 2: 开始训练异构图模型...")
print(f"{'Epoch':<10} | {'Loss':<10}")
print("-" * 30)

for epoch in range(1, 101):
    loss = train()
    if epoch % 10 == 0:
        print(f"{epoch:<10} | {loss:.4f}")

print("-" * 30)
print("✅ 训练完成！")

# ==========================================
# 4. 保存结果
# ==========================================
model.eval()
with torch.no_grad():
    # 获取最终向量
    final_dict = model(data.edge_index_dict)
    # 只保存 Gene 的向量用于预测
    gene_embeddings = final_dict['gene'].cpu()

torch.save(gene_embeddings, 'complex_gene_embeddings.pt')
print("✅ 最终基因向量已保存为 'complex_gene_embeddings.pt'")
