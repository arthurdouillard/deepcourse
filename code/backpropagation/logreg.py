class LogisticRegression:
    def __init__(self, input_size, nb_classes, learning_rate=0.01):
        self.w = torch.randn(input_size, nb_classes).float()
        self.b = torch.zeros(nb_classes).float()

        self.learning_rate = learning_rate

    def forward(self, x):
        return torch.mm(x, self.w) + self.b

    def fit(self, inputs, targets, train=True):
        probs = softmax(self.forward(inputs))
        if train:
            self.backward(inputs, probs, targets)

        loss = cross_entropy(probs, torch.eye(10)[targets]).mean()
        return loss

    def backward(self, inputs, probs, targets):
        batch_size = len(inputs)

        grad_logits = probs - torch.eye(10)[targets]
        grad_w = torch.mm(inputs.T, grad_logits) / batch_size
        grad_b = torch.sum(grad_logits, dim=0) / batch_size

        self.w = self.w - self.learning_rate * grad_w
        self.b = self.b - self.learning_rate * grad_b

    def accuracy(self, inputs, targets):
        y_pred = self.forward(inputs).argmax(dim=1)
        return torch.mean((y_pred == targets).float())
