import torch
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score


# ==========================================
# 1. 数据加载与划分
# ==========================================
def get_data_splits(path='complex_hetero_data.pt', val_ratio=0.1, test_ratio=0.2):
    print(f"Loading data from {path}...")
    try:
        data = torch.load(path, weights_only=False)
    except FileNotFoundError:
        print(f"❌ Error: {path} not found.")
        exit()

    transform = T.RandomLinkSplit(
        num_val=val_ratio,
        num_test=test_ratio,
        is_undirected=True,
        neg_sampling_ratio=1.0,
        edge_types=[('gene', 'interacts', 'gene')],
        rev_edge_types=[('gene', 'interacts', 'gene')]
    )
    return transform(data)


# ==========================================
# 2. 模型定义 (【关键修改】加入 Dropout)
# ==========================================
class GNN_Backbone(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        # 第一层卷积
        x = self.conv1(x, edge_index)
        x = x.relu()

        # 【修改 1】加入 Dropout (防止过拟合的核心)
        # p=0.5 表示随机丢弃 50% 的神经元
        # training=self.training 确保只在训练时丢弃，测试时全开
        x = F.dropout(x, p=0.5, training=self.training)

        # 第二层卷积
        x = self.conv2(x, edge_index)
        return x


class HeteroModel(torch.nn.Module):
    def __init__(self, num_genes, num_gos, hidden_channels, out_channels, metadata):
        super().__init__()
        self.gene_emb = torch.nn.Embedding(num_genes, hidden_channels)
        self.go_emb = torch.nn.Embedding(num_gos, hidden_channels)

        backbone = GNN_Backbone(hidden_channels, out_channels)
        self.gnn = to_hetero(backbone, metadata, aggr='sum')

    def forward(self, x_dict, edge_index_dict):
        return self.gnn(x_dict, edge_index_dict)

    def decode(self, z, edge_label_index):
        src = edge_label_index[0]
        dst = edge_label_index[1]
        return (z[src] * z[dst]).sum(dim=-1)


# ==========================================
# 3. 初始化
# ==========================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_data, val_data, test_data = get_data_splits()

HIDDEN_DIM = 32
model = HeteroModel(
    num_genes=train_data['gene'].num_nodes,
    num_gos=train_data['go'].num_nodes,
    hidden_channels=HIDDEN_DIM,
    out_channels=HIDDEN_DIM,
    metadata=train_data.metadata()
).to(device)

# 【修改 2】加入 weight_decay (L2 正则化)
# 5e-4 是一个经验值，专门惩罚过大的参数
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


# ==========================================
# 4. 训练与测试函数
# ==========================================
def train_step(data_batch):
    model.train()  # 这一步很重要，它会开启 Dropout
    optimizer.zero_grad()

    x_dict = {'gene': model.gene_emb.weight, 'go': model.go_emb.weight}
    out_dict = model(x_dict, data_batch.edge_index_dict)

    pred = model.decode(out_dict['gene'], data_batch['gene', 'interacts', 'gene'].edge_label_index)
    target = data_batch['gene', 'interacts', 'gene'].edge_label

    loss = F.binary_cross_entropy_with_logits(pred, target)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def test_step(data_batch):
    model.eval()  # 这一步很重要，它会关闭 Dropout，让预测更稳定

    x_dict = {'gene': model.gene_emb.weight, 'go': model.go_emb.weight}
    out_dict = model(x_dict, data_batch.edge_index_dict)

    pred_logits = model.decode(out_dict['gene'], data_batch['gene', 'interacts', 'gene'].edge_label_index)
    pred_probs = pred_logits.sigmoid()

    y_true = data_batch['gene', 'interacts', 'gene'].edge_label.cpu().numpy()
    y_pred = pred_probs.cpu().numpy()

    auc = roc_auc_score(y_true, y_pred)
    y_pred_binary = (y_pred > 0.5).astype(int)
    f1 = f1_score(y_true, y_pred_binary)
    acc = accuracy_score(y_true, y_pred_binary)

    return auc, f1, acc


# ==========================================
# 5. 主训练循环
# ==========================================
print(f"\n🚀 Start Robust Training on {device}...")
best_val_auc = 0
patience = 0

train_data = train_data.to(device)
val_data = val_data.to(device)
test_data = test_data.to(device)

for epoch in range(1, 201):
    loss = train_step(train_data)

    if epoch % 5 == 0:
        val_auc, val_f1, val_acc = test_step(val_data)
        print(f"Epoch {epoch:03d} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            # 保存最佳模型
            model.eval()
            with torch.no_grad():
                x_dict = {'gene': model.gene_emb.weight, 'go': model.go_emb.weight}
                full_out = model(x_dict, train_data.edge_index_dict)
                torch.save(full_out['gene'].cpu(), 'complex_gene_embeddings.pt')

            print(f"   💾 Best model saved! (AUC: {best_val_auc:.4f})")
            patience = 0
        else:
            patience += 1
            if patience > 30:  # 稍微增加一点耐心，因为加了 Dropout 收敛变慢是正常的
                print("🛑 Early stopping triggered.")
                break

print("\n🔍 Final Evaluation on Test Set...")
test_auc, test_f1, test_acc = test_step(test_data)
print(f"🏆 Robust Test Results:")
print(f"   - Accuracy: {test_acc:.4f}")
print(f"   - F1 Score: {test_f1:.4f}")
print(f"   - AUC:      {test_auc:.4f}")
