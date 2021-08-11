class MLPMixer(nn.Module):
  def __init__(self, embed_dim, nb_blocks, patch_size, nb_classes=10):
    super().__init__()

    nb_patches = (28 // patch_size) ** 2
    self.patch_embed = PatchEmbed(patch_size=patch_size, embed_dim=embed_dim)

    blocks = []
    for _ in range(nb_blocks):
      blocks.append(
          Mixer(embed_dim, nb_patches)
      )
    self.blocks = nn.Sequential(*blocks)

    self.norm = nn.LayerNorm(embed_dim)
    self.head = nn.Linear(embed_dim, nb_classes)

  def forward(self, x):
    x = self.patch_embed(x)
    x = self.blocks(x)
    x = self.norm(x)
    x = x.mean(dim=1)  # Pool alongside the tokens dimension
    return self.head(x)
