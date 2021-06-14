def generate_attention(path, size=(224, 224)):
  image = Image.open(path)
  image.thumbnail(size, Image.ANTIALIAS)

  tensor = transform(image)[None]
  with torch.no_grad():
    logits = net(tensor)[0]
  probs = torch.softmax(logits, dim=0)
  attention_map = probs[indexes].sum(dim=0)

  return image, attention_map
