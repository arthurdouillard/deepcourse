def cross_entropy(probs, targets):
    return -torch.sum(targets * torch.log(probs + 1e-8), dim=1)

probs = torch.tensor([
    [0.9, 0.1],
    [0.7, 0.3],
    [0.2, 0.8],
    [0.6, 0.4]
])

targets = torch.eye(2)[torch.tensor([0, 0, 0, 1])]
cross_entropy(probs, targets)
