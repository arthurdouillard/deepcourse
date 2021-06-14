class DANN(nn.Module):
  def __init__(self):
    super().__init__()  # Important, otherwise will throw an error

    self.cnn = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),  # 32x28x28
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # 32x14x14
        nn.Conv2d(32, 48, kernel_size=5, stride=1, padding=2),  # 48x14x14
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # 48x7x7
    )

    self.classif = nn.Sequential(
        nn.Linear(7 * 7 * 48, 100),
        nn.ReLU(inplace=True),
        nn.Linear(100, 100),
        nn.ReLU(inplace=True),
        nn.Linear(100, 10)
    )

    self.domain = nn.Sequential(
        nn.Linear(7 * 7 * 48, 100),
        nn.ReLU(inplace=True),
        nn.Linear(100, 1)
    )

    self.grl = GradientReversal.apply

  def forward(self, x, factor=1):
    batch_size = len(x)

    x = self.cnn(x)
    x = x.view(batch_size, -1)

    class_pred = self.classif(x)
    domain_pred = self.domain(self.grl(x, -1 * factor))

    return class_pred, domain_pred
