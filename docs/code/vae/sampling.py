epsilon = torch.randn_like(mean)  # Gaussian with as much dimension as the mean
# Remember that we have the *log* variance, thus we need to use an exponential
# to get back the variance:
z = mean + torch.exp(log_var) * epsilon
