class SelfAttention(nn.Module):
  def __init__(self, embed_dim):
    super().__init__()

    self.scale = embed_dim ** -0.5
    self.q = nn.Linear(embed_dim, embed_dim, bias=False)
    self.k = nn.Linear(embed_dim, embed_dim, bias=False)
    self.v = nn.Linear(embed_dim, embed_dim, bias=False)
    self.projection = nn.Linear(embed_dim, embed_dim)

  def forward(self, x):
    B, N, C = x.shape

    q = self.q(x)
    k = self.k(x)
    v = self.v(x)

    attention = torch.matmul(q, k.transpose(1, 2)) * self.scale
    attention = torch.softmax(attention, dim=-1)

    x = torch.matmul(attention, v)
    x = self.projection(x)

    return x
