def extract_features(loader, net):
  features, targets = [], []
  for images, labels in loader:
    with torch.no_grad():
      features.append(
          net(images.cuda()).cpu()
      )
    targets.append(labels)

  return torch.cat(features, dim=0), torch.cat(targets)
