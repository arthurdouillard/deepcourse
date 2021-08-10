class Block(nn.Module):
  def __init__(self, embed_dim, mlp_ratio=4):
    super().__init__()

    self.attention = SelfAttention(embed_dim)
    self.norm1 = nn.LayerNorm(embed_dim)
    self.mlp = MLP(embed_dim, embed_dim * mlp_ratio)
    self.norm2 = nn.LayerNorm(embed_dim)

  def forward(self, x):
    x = x + self.attention(self.norm1(x))
    x = x + self.mlp(self.norm2(x))
    return x
