class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.img_branch = nn.Sequential(
            conv_disc_layer(1, d * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.label_branch = nn.Sequential(
            conv_disc_layer(10, d * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.classifier = nn.Sequential(
            conv_disc_layer(d * 4, d * 8, 4, 2, 1, bias=False),
            norm_layer(d * 8),
            nn.LeakyReLU(0.2, inplace=True),
            conv_disc_layer(d * 8, d * 16, 4, 2, 1, bias=False),
            norm_layer(d * 16),
            nn.LeakyReLU(0.2, inplace=True),
            conv_disc_layer(d * 16, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, y):
        # Expand label to have the size of the image
        y = y.expand(y.shape[0], y.shape[1], x.shape[2], x.shape[3])

        x = self.img_branch(x)
        y = self.label_branch(y)

        xy_concatenated = torch.cat([x, y], dim=1)
        o = self.classifier(xy_concatenated)

        return o.squeeze() # go from (bs, 1, 1, 1) to (bs,)


Discriminator()(torch.randn(10, 1, 32, 32), one_hot(y)).shape
