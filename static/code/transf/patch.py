class PatchEmbed(nn.Module):
  def __init__(self, in_chan=1, patch_size=7, embed_dim=128):
    super().__init__()
    self.projection = nn.Conv2d(in_chan, embed_dim, kernel_size=patch_size, stride=patch_size)

  def forward(self, x):
    x = self.projection(x)

    B, C, H, W = x.shape
    x = x.view(B, C, H * W).transpose(1, 2)
    # Shape is now B, N, D
    # With B the batch size
    # With N the number of tokens (N = H * W)
    # With D the embedding dimension (previously C the number of channels)

    return x
