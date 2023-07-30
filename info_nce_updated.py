import torch
import torch.nn.functional as F
from torch import nn

class InfoNCE(nn.Module):
    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode

    def forward(self, query, positive_keys, negative_keys=None):
        return info_nce(query, positive_keys, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)
def info_nce(query, positive_keys, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_keys.dim() != 3:
        raise ValueError('<positive_keys> must have 3 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_keys):
        raise ValueError('<query> and <positive_keys> must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have the same number of components.
    if query.shape[-1] != positive_keys.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_keys> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_keys, negative_keys = normalize(query, positive_keys, negative_keys)

    if negative_keys is not None:
        positive_logits = (query.unsqueeze(1) * positive_keys).sum(-1)
        num_positives = positive_logits.shape[1]

        if negative_mode == 'unpaired':
            # Cosine similarity between all query-negative combinations
            negative_logits = torch.bmm(query.unsqueeze(1), negative_keys.transpose(1, 2)).squeeze(1)
        elif negative_mode == 'paired':
            # Cosine similarity between all query-negative pairs
            negative_logits = (query.unsqueeze(1) * negative_keys).sum(-1)

        num_negatives = negative_logits.shape[1] if negative_keys is not None else 0

        # Concatenation along the second dimension
        logits = torch.cat([positive_logits, negative_logits], dim=1)

        # Labels: 1 for positive pairs, 0 for negative pairs
        labels_single = torch.cat([torch.ones(num_positives), torch.zeros(num_negatives)], dim=0).to(query.device)
        labels = labels_single.unsqueeze(0).repeat(query.shape[0], 1)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = torch.bmm(query.unsqueeze(1), positive_keys.transpose(1, 2)).squeeze(1)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)

def transpose(x):
    return x.transpose(-2, -1)

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]