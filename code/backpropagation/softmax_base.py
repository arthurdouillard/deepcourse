def softmax(x):
    e = torch.exp(x)
    return e / e.sum()

print(softmax(torch.tensor([1., 2., 3.])))
print(softmax(torch.tensor([3., -0.12, -4.2, 9])))
