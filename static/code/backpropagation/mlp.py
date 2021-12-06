class MLP:
    def __init__(self, input_size, hidden_size, nb_classes, learning_rate=0.5):
        self.w_hidden = torch.randn(input_size, hidden_size).float()
        self.b_hidden = torch.zeros(hidden_size).float()

        self.w_output = torch.randn(hidden_size, nb_classes).float()
        self.b_output = torch.zeros(nb_classes).float()

        self.learning_rate = learning_rate

    def forward(self, x):
        h_tilde = torch.mm(x, self.w_hidden) + self.b_hidden
        h = torch.tanh(h_tilde)
        logits = torch.mm(h, self.w_output) + self.b_output
        probs = softmax(logits)

        return h_tilde, h, logits, probs

    def fit(self, inputs, targets, train=True):
        outputs = self.forward(inputs)
        probs = outputs[-1]
        if train:
            self.backward(inputs, targets, *outputs)

        loss = cross_entropy(probs, torch.eye(10)[targets]).mean()
        return loss

    def backward(self, inputs, targets, h_tilde, h, logits, probs):
        batch_size = len(inputs)

        grad_logits = probs - torch.eye(10)[targets]
        grad_wo = torch.mm(h.T, grad_logits) / batch_size
        grad_bo = torch.sum(grad_logits, dim=0) / batch_size

        grad_h = torch.mm(grad_logits, self.w_output.T)
        grad_htilde = grad_h * grad_tanh(h)
        grad_wh = torch.mm(inputs.T, grad_htilde) / batch_size
        grad_bh = torch.sum(grad_htilde, dim=0) / batch_size

        self.w_output = self.w_output - self.learning_rate * grad_wo
        self.b_output = self.b_output - self.learning_rate * grad_bo
        self.w_hidden = self.w_hidden - self.learning_rate * grad_wh
        self.b_hidden = self.b_hidden - self.learning_rate * grad_bh

    def accuracy(self, inputs, targets):
        y_pred = self.forward(inputs)[-1].argmax(dim=1)
        y_true = targets

        return torch.mean((y_pred == y_true).float())
