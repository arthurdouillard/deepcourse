def softmax(x):
    maximum_value = x.max(dim=1, keepdims=True)[0]
    e = torch.exp(x - maximum_value)
    return e / e.sum(dim=1, keepdims=True)

x = torch.tensor([
    [1., 2., 3.],
    [4., 9., -12.]
])
probabilities = softmax(x)

print(probabilities.sum(dim=1))
