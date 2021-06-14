w = torch.zeros(3, 3, 5, 5)
w[0, 0, 2, 2] = 1
w[1, 1, 2, 2] = 1
w[2, 2, 2, 2] = 1

o = F.conv2d(img[None], w)
torch_to_jpg(o[0])
