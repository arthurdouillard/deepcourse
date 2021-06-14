gray_image = img.mean(dim=0)
print(gray_image.shape)
torch_to_jpg(gray_image)
