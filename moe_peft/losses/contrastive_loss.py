import torch
import torch.nn.functional as F

def multi_positive_contrastive_loss(embeddings, labels, temperature=0.1): # #embeddings: the model’s internal representations of inputs
#labels: expert IDs (used as pseudo-labels)
#temperature: a scaling factor for sharpness of similarity
    device = embeddings.device
    batch_size = embeddings.size(0)

    embeddings = F.normalize(embeddings, dim=1) #Normalize the embeddings — make their lengths 1, like turning vectors into unit arrows. This ensures fair cosine similarity comparison.
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature ##Compute a similarity matrix: how similar each embedding is to every other.

    self_mask = torch.eye(batch_size, dtype=torch.bool).to(device) #Prevent comparing an embedding with itself.
    pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1) ##Build a mask where 1s represent embeddings that came from the same expert — these are the positives we want to pull closer.
    pos_mask = pos_mask & ~self_mask

    exp_sim = torch.exp(sim_matrix) ##Compute the log probability of picking a positive sample vs. all others.
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True))
    pos_log_prob = (log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1) #Keep only the positives (same expert), take their average log-probability.

    return -pos_log_prob.mean() #Return the negative average log-likelihood — this is your contrastive loss.
