class Encoder(nn.Module):
  def __init__(self, latent_dim=2):
    super().__init__()

    self.latent_dim = latent_dim

    self.enc = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )

    self.fc = nn.Sequential(
        nn.Linear(7 * 7 * 64, 16),
        nn.ReLU(inplace=True)
    )

    self.fc_mean = nn.Linear(16, latent_dim)
    self.fc_log_var = nn.Linear(16, latent_dim)

  def forward(self, x):
    x = self.enc(x)
    x = x.view(len(x), -1)
    x = self.fc(x)

    mean = self.fc_mean(x)
    # Our encoder computes implicitly the log var
    # instead of the var, in order to avoid using
    # a numerically instable logarithm afterwards.
    log_var = self.fc_log_var(x)

    epsilon = torch.randn_like(mean)  # Gaussian with as much dimension as the mean
    # Remember that we have the *log* variance, thus we need to use an exponential
    # to get back the variance:
    z = mean + torch.exp(log_var) * epsilon

    return z, mean, log_var


