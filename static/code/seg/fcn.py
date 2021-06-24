class FCN(nn.Module):
  def __init__(self, nb_classes):
    super().__init__()

    self.resnet = torchvision.models.resnet50(pretrained=True, replace_stride_with_dilation=[False, True, True])
    self.resnet.forward = forward_extract.__get__(
        self.resnet,
        torchvision.models.ResNet
    )  # monkey-patching

    self.fc = nn.Sequential(
        nn.Conv2d(2048, 512, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Conv2d(512, nb_classes, kernel_size=1)
    )

  def forward(self, x):
    x = self.resnet(x)
    x = self.fc(x)

    return F.interpolate(
        x, size=(224, 224), mode="bilinear", align_corners=False
    )
