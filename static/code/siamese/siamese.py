class Siamese(nn.Module):
  def __init__(self):
    super().__init__()
    self.backbone = Backbone()

  def forward(self, x1, x2):
    h1 = self.backbone(x1)
    h2 = self.backbone(x2)

    cos = tensor_cosine(h1, h2)

    return cos
