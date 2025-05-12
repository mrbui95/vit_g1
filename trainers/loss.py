import torch
import torch.nn.functional as F

def contrastive_koleo_loss(z, y, beta=0.5, top_k=5):
    """
    z: (N, D) - normalized features
    y: (N,)   - labels
    """
    z = F.normalize(z, dim=1)
    N = z.size(0)
    z = z[:, 0, :]   
    z = F.normalize(z, dim=1)  # chuẩn hóa nếu cần
    sim_matrix = torch.matmul(z, z.T)  # (N, N)

    loss_contr = 0.0
    for i in range(N):
        same_mask = (y == y[i])
        diff_mask = (y != y[i])

        # 1 - similarity with different class
        neg_sim = 1 - sim_matrix[i][diff_mask]
        # similarity with same class minus beta
        pos_sim = sim_matrix[i][same_mask]
        pos_sim = F.relu(pos_sim - beta)

        loss_contr += neg_sim.sum() + pos_sim.sum()

    loss_contr /= N

    # KoLeo regularizer (maximize entropy of distances)
    with torch.no_grad():
        dists = torch.cdist(z, z, p=2)  # pairwise Euclidean distances
        _, indices = torch.topk(dists, k=top_k+1, largest=False)  # +1 to skip self
        knn_dists = dists.gather(1, indices[:, 1:])  # skip self

    rho = knn_dists.mean(dim=1) + 1e-6  # avoid log(0)
    loss_koleo = -torch.log(rho).mean()

    return loss_contr + loss_koleo, loss_contr.item(), loss_koleo.item()

def contrastive_koleo_loss_cosin(z, y, beta=0.5, top_k=5):
    """
    z: (N, D) - raw features (will be normalized)
    y: (N,)   - labels
    """
    z = z[:, 0, :]  # Nếu input là (N, 1, D) thì lấy ra đúng shape (N, D)
    z = F.normalize(z, dim=1)  # chuẩn hóa để dùng cosine similarity
    N = z.size(0)

    sim_matrix = torch.matmul(z, z.T)  # cosine similarity matrix (N, N)

    loss_contr = 0.0
    for i in range(N):
        same_mask = (y == y[i])
        diff_mask = (y != y[i])

        neg_sim = 1 - sim_matrix[i][diff_mask]  # muốn khác lớp thì similarity thấp
        pos_sim = sim_matrix[i][same_mask]      # muốn cùng lớp thì similarity cao
        pos_sim = F.relu(beta - pos_sim)        # phạt nếu similarity chưa đủ cao

        loss_contr += neg_sim.sum() + pos_sim.sum()

    loss_contr /= N

    # --- KoLeo regularizer với cosine distance ---
    with torch.no_grad():
        cos_sim = torch.matmul(z, z.T)
        cos_dist = 1 - cos_sim  # cosine distance: càng nhỏ càng gần
        _, indices = torch.topk(cos_dist, k=top_k+1, largest=False)  # lấy hàng xóm gần nhất
        knn_dists = cos_dist.gather(1, indices[:, 1:])  # bỏ self (self-dist = 0)

    rho = knn_dists.mean(dim=1) + 1e-6  # tránh log(0)
    loss_koleo = -torch.log(rho).mean()

    return loss_contr + loss_koleo, loss_contr.item(), loss_koleo.item()
