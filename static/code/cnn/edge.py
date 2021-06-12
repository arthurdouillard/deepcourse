w = torch.zeros(1, 3, 3, 3)

mat = torch.tensor([
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
])

w[0, 0, :, :] = mat
w[0, 1, :, :] = mat
w[0, 2, :, :] = mat

o = F.conv2d(img[None].mean(dim=0, keepdims=True), w)
o = 255 * (o - o.min()) / (o.max() - o.min())  # Rescale to [0, 255]
torch_to_jpg(o[0])
