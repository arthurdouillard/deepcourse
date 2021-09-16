class MLP:
    def __init__(self, input_size, hidden_size, nb_classes, learning_rate=0.01):
        self.w_hidden = Parameter(torch.randn(input_size, hidden_size).float())
        self.b_hidden = Parameter(torch.zeros(hidden_size).float())

        self.w_output = Parameter(torch.randn(hidden_size, nb_classes).float())
        self.b_output = Parameter(torch.zeros(nb_classes).float())

        self.learning_rate = learning_rate

    def forward(self, x):
        h_tilde = torch.mm(x, self.w_hidden) + self.b_hidden
        h = torch.tanh(h_tilde)
        logits = torch.mm(h, self.w_output) + self.b_output

        return logits, h_tilde, h

    def fit(self, inputs, targets, train=True):
        logits, *outputs = self.forward(inputs)
        probs = softmax(logits)
        loss = cross_entropy(probs, torch.eye(10)[targets]).sum()
        if train:
            self.backward(inputs, probs, targets, loss, *outputs)
        return loss

    def backward(self, inputs, probs, targets, loss, h_tilde, h):
        batch_size = len(probs)

        loss.backward()

        self.w_output.data = self.w_output.data - self.learning_rate * self.w_output.grad / batch_size
        self.b_output.data = self.b_output.data - self.learning_rate * self.b_output.grad / batch_size
        self.w_hidden.data = self.w_hidden.data - self.learning_rate * self.w_hidden.grad / batch_size
        self.b_hidden.data = self.b_hidden.data - self.learning_rate * self.b_hidden.grad / batch_size

    def accuracy(self, inputs, targets):
        y_pred = self.forward(inputs)[0].argmax(dim=1)
        y_true = targets

        return torch.mean((y_pred == y_true).float())
