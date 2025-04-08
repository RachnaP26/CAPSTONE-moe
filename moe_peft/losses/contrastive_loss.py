import torch
import torch.nn.functional as F

def multi_positive_contrastive_loss(embeddings, labels, temperature=0.1):
    device = embeddings.device
    batch_size = embeddings.size(0)

    embeddings = F.normalize(embeddings, dim=1)
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature

    self_mask = torch.eye(batch_size, dtype=torch.bool).to(device)
    pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)
    pos_mask = pos_mask & ~self_mask

    exp_sim = torch.exp(sim_matrix)
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
    pos_log_prob = (log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)

    return -pos_log_prob.mean()
