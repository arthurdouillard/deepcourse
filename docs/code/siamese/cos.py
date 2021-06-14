def tensor_cosine(x, y):
  x = F.normalize(x, p=2, dim=1)
  y = F.normalize(y, p=2, dim=1)

  return torch.sum(x * y, dim=1)
