merged_attn = torch.ones(32, 47)

sizes = (112, 224, 512, 1024, 1500)
for size in sizes:
  image, attn = generate_attention("2dogs.jpg", (size, size))

  merged_attn *= F.interpolate(attn[None, None], (32, 47))[0, 0]

attn = torch.pow(merged_attn, 1 / len(sizes))

plt.figure(figsize=(10, 8))
ax = plt.subplot(1, 2, 1)
ax.axis("off")
plt.imshow(image)

ax = plt.subplot(1, 2, 2)
ax.axis("off")
plt.imshow(attn)
