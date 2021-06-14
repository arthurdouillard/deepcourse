conv1x1.weight.data = net.fc.weight.data[..., None, None]
conv1x1.bias.data = net.fc.bias.data
