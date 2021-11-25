class VAE(nn.Module):
  def __init__(self):
    super().__init__()

    self.enc = Encoder()
    self.dec = Decoder()

  def forward(self, x):
    z, mean, log_var = self.enc(x)
    xhat = self.dec(z)
    return xhat, mean, log_var


VAE()(torch.randn(1, 1, 28, 28))[0].shape
