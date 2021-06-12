class CNN(nn.Module):
  def __init__(self):
    super().__init__()  # Important, otherwise will throw an error

    self.cnn = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),  # 32x32x32
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # 32x16x16
        nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),  # 64x16x16
        nn.ReLU(inplace=True),  # faster with inplace, no need to copy
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # 64x8x8
        nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),  # 64x8x8
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),  # 64x4x4
    )

    self.fc1 = nn.Linear(4 * 4 * 64, 1000)
    self.act = nn.ReLU(inplace=True)
    self.fc2 = nn.Linear(1000, 10)

  def forward(self, x):
    x = self.cnn(x)
    b, c, h, w = x.shape
    x = x.view(b, c * h * w) # equivalent to x.view(b, -1)
    x = self.act(self.fc1(x))
    x = self.fc2(x)
    return x
