class MultiHeadsSelfAttention(nn.Module):
  def __init__(self, embed_dim, num_heads):
    super().__init__()

    head_dim = embed_dim // num_heads
    self.scale = head_dim ** -0.5
    self.num_heads = num_heads

    self.q = nn.Linear(embed_dim, embed_dim, bias=False)
    self.k = nn.Linear(embed_dim, embed_dim, bias=False)
    self.v = nn.Linear(embed_dim, embed_dim, bias=False)
    self.projection = nn.Linear(embed_dim, embed_dim)

  def forward(self, x):
    B, N, C = x.shape

    q = self.q(x)
    k = self.k(x)
    v = self.v(x)

    q = q.reshape(B, N, self.num_heads, C // self.num_heads)
    k = k.reshape(B, N, self.num_heads, C // self.num_heads)
    v = v.reshape(B, N, self.num_heads, C // self.num_heads)

    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)

    attention = torch.matmul(q, k.transpose(2, 3)) * self.scale
    attention = torch.softmax(attention, dim=-1)

    x = torch.matmul(attention, v)  # B, H, N, Hd
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(B, N, C)
    x = self.projection(x)

    return x
