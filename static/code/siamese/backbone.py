class Backbone(nn.Module):
  def __init__(self):
    super().__init__()

    self.convs = nn.Sequential(
        ConvBlock(3, 16, 3),
        nn.MaxPool2d((2, 2)),
        ConvBlock(16, 32, 3),
        nn.MaxPool2d((2, 2)),
        ConvBlock(32, 64, 3),
        nn.MaxPool2d((2, 2)),
        ConvBlock(64, 128, 3),
        ConvBlock(128, 64, 3),
    )

    self.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(1024, 512),
        nn.Dropout(0.3),
        nn.Linear(512, 50)
    )

  def forward(self, x):
    x = self.convs(x)
    x = x.view(len(x), -1)
    return self.fc(x)
