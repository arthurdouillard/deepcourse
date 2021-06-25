def reconstruction(xhat, x):
  return torch.pow(
      xhat - x, 2
  ).sum(dim=(1, 2, 3)).mean()


def kl_div_gauss(log_var, mean):
    kl_loss = -0.5 * (1 + log_var - torch.pow(mean, 2) - torch.exp(log_var))
    kl_loss = kl_loss.sum(dim=1).mean()
    return kl_loss
