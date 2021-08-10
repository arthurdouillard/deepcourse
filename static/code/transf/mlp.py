class MLP(nn.Module):
  def __init__(self, in_features, hid_features):
    super().__init__()

    self.mlp = nn.Sequential(
        nn.Linear(in_features, hid_features),
        nn.GELU(),
        nn.Linear(hid_features, in_features),
        nn.GELU(),
    )

  def forward(self, x):
    return self.mlp(x)
