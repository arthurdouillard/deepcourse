---
author: "Arthur Douillard"
title: "Deep Neural Network"
date: "1956-06-18"
description: "Neural Network and Backpropagation"
tags: []
hideMeta: false

ShowToc: true
hasQuiz: true
---

This lesson has more math than usual, but it's important that we don't shy away
the theory behind deep learning. It will help us build a strong understanding of
the algorithms we will code in later sections. If you feel overwhelmed by the math here,
please see this [review](/) first.

We are talking of neural networks, then what is a neuron? Certainly not the kind that is
in your brain. While biologically inspired, artificial neurons are very different
from the real counterpart, thus for now put aside any comparison with brains.

The neurons, or parameters, of our network are floating values. A neuron could
be 0.7964, -102.329, or 3912.20238. We group those neurons in matrices, often
denoted \\(W\\). These matrices are often accompanied by a bias \\(b\\).

# Single Output Network

Let's have an example of very simple network:

[Image of a Linear regression]

Let's decompose this network in three parts:

**Input**:
- The input is an image of a digit: \\(X \in \mathcal{R}^{W \times H \times 1}\\). The first two dimensions \\(W\\) & \\(H\\)
are respectively the width and height of the image. The last dimension correspond
to the number of channels. Because here the image is in grayscale, there is only one channel,
but with color images we have three channels: Red, Blue, and Green ([RGB](https://en.wikipedia.org/wiki/RGB_color_model)).
- The network we are learning right now only accepts vectors, not tensors. Thus we are flattening the input into a vector with
a single dimension of length \\(W \times H\\).

**Neurons / Parameters**:
- We have \\(W \in \mathcal{R}^{1 \times W \times H}\\) and \\(b \in \mathcal{R}\\), respectively the **weights**
and **bias**.
- The operation \\(W^T X\\) results in a single output value \\(z \in \mathcal{R}\\). Thus \\(z\\) can have a value from
\\(-\infty\\) to \\(+\infty\\).

**Activation**:
- We use a non-linear activation \\(\sigma(z) = \frac{1}{1 + e^{-z}} \in [0, 1]\\) called **sigmoid**
- It allows us to map \\(z\\) to a probability \\(o\\).

This is a single-layer neural network that can classify images into two classes.

# Multiple Outputs Network


# Multi-layers Networks


# Activation Functions


# Backpropagation


# Initialization


# Optimizers
