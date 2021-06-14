pooled = F.avg_pool2d(img[None], kernel_size=2)
plt.figure(figsize=(13, 8))
plt.subplot(1, 2, 1)
plt.imshow(torch_to_jpg(pooled[0]))
plt.title(f"w/ avg pooling, {pooled[0].shape}")

w = torch.zeros(3, 3, 2, 2)
w[0, 0, :, :] = 1 / 25
w[1, 1, :, :] = 1 / 25
w[2, 2, :, :] = 1 / 25
o = F.conv2d(img[None].mean(dim=0, keepdims=True), w, stride=2)
o = 255 * (o - o.min()) / (o.max() - o.min())  # Rescale to [0, 255]

plt.subplot(1, 2, 2)
plt.imshow(torch_to_jpg(o[0]))
plt.title(f"w/ convolution, {o[0].shape}");
