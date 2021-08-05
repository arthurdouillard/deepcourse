class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.noise_branch = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, d * 8, kernel_size=4, stride=1, padding=0, bias=False),
            norm_layer(d * 8),
            nn.ReLU(True)
        )
        self.label_branch = nn.Sequential(
            nn.ConvTranspose2d(10, d * 8, 4, 1, 0, bias=False),
            norm_layer(d * 8),
            nn.ReLU(True)
        )
        self.generator = nn.Sequential(
            nn.ConvTranspose2d( d * 16, d * 8, 4, 2, 1, bias=False),
            norm_layer(d * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(d * 8, d * 4, 4, 2, 1, bias=False),
            norm_layer(d * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(d * 4, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, y):
        z = self.noise_branch(z)
        y = self.label_branch(y)

        zy_concatenated = torch.cat([z, y], dim=1)
        o = self.generator(zy_concatenated)

        return o

Generator()(torch.randn(10, noise_dim, 1, 1), one_hot(y)).shape
