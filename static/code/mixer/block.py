class Mixer(nn.Module):
  def __init__(self, embed_dim, nb_patches, mlp_ratio=(0.5, 4.0)):
    super().__init__()

    # These represent the hidden dimensions when transforming
    # the tokens and channels dimensions respectively
    tokens_dim = int(mlp_ratio[0] * embed_dim)
    channels_dim = int(mlp_ratio[1] * embed_dim)

    self.norm1 = nn.LayerNorm(embed_dim)
    self.norm2 = nn.LayerNorm(embed_dim)

    self.mlp_tokens = MLP(nb_patches, tokens_dim)
    self.mlp_channels = MLP(embed_dim, channels_dim)

  def forward(self, x):
    x = self.norm1(x)         # B, N, C
    res = x.transpose(1, 2)   # B, C, N
    res = self.mlp_tokens(res)  # B, C, N
    res = res.transpose(1, 2)   # B, N, C
    x = x + res

    x = x + self.mlp_channels(self.norm2(x))

    return x
