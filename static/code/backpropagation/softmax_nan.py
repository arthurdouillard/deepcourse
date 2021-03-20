def softmax(x):
    maximum_value = x.max()
    e = torch.exp(x - maximum_value)
    return e / e.sum()

softmax(torch.tensor([234., 3., 4.]))
