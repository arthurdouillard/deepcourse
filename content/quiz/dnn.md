---
title: "Deep Neural Network"
layout: mainquiz
---


{{< quiz cards=`[
    {Front: "What are the two passes done in a neural network?", Back: "Forward and backward passes"},
    {Front: "What is the formula of a single neuron?", Back: "\\(f(x) = w x + b\\)"},
    {Front: "What is the shape of \\(w\\) in a single neuron?", Back: "Vector of shape (Nb_input_dims)"},
    {Front: "What is the shape of \\(b\\) in a single neuron?", Back: "Scalar value"},
    {Front: "What is the shape of \\(W\\) in a hidden layer", Back: "Matrix of shape (Nb_hidden_dims x Nb_input_dims)"},
    {Front: "What is the shape of \\(b\\) in a hidden layer", Back: "Vector of shape (Nb_hidden_dims)"},
    {Front: "Why do we use non-linear activation in Multi-Layer Perceptron?", Back: "To separate non-linear problems. And stacking multiple layers without activation is mathematically equivalent to a single layer."},
    {Front: "Name some non-linear activations?", Back: "Sigmoid, tanh, ReLU"},
    {Front: "What is the ReLU formula?", Back: "\\(max(0, x)\\)"},
    {Front: "What is the Sigmoid formula?", Back: "\\(\\frac{1}{1 + e^{-x}}\\)"},
    {Front: "What is the problem of the sigma in backpropagation?", Back: "The signal can vanish because sigmoid is satured at both extremum"},
    {Front: "What is the Sigmoid formula?", Back: "\\(\\frac{1}{1 + e^{-x}}\\)"},
    {Front: "What is the usual classification loss?", Back: "Cross entropy: \\(-\\Sigma_i y_i \\log \\hat{y}_i\\)"},
    {Front: "What happens if the learning rate is too high?", Back: "The network risks to diverge"},
    {Front: "What happens if the learning rate is too low?", Back: "The network risks to fall in bad local minima or converging too slowly"},
    {Front: "What is the difference between Batch Gradient Descent and Stochastic Gradient Descent?", Back: "1: The whole dataset goes through the model at once. 2: One sample at the time goes through the model"},
]` quizid=1 >}}

