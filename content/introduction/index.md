---
author: "Arthur Douillard"
title: "Introduction"
date: "1956-06-18"
description: "Introduction to Deep Learning"
tags: ["deeplearning", "vision"]
hideMeta: false

hasQuiz: true
---

Welcome to this course on Deep Learning for Computer Vision!

This course will teach you the fundamentals of Deep Learning applied to images. Each topic
will be covered under three different type of materials:

- **Lessons**, such as this one, accompanied with slides and small quizzes between chapters;
- **Practices**, with Google Colab notebooks where we will code actual algorithms;
- And, **quizzes**, where you will test your recall on the important notions.

Lessons and practices are essential to learn new materials. However knowledge can
be brittle, it's then best to test it with **active recall** through quizzes. If you're
really motivated, you can also download the Anki decks I provide for each lessons.
Anki is a **space-repetition** tools to ingrain deeply knowledge in your memory, and
to basically never forget it. I'm not kidding, it's a marvel.

No more chitchat, let's start.

# A Bit Of History

In 1956, the summer workshop of [Darmouth](https://en.wikipedia.org/wiki/Dartmouth_workshop) was held with
prestigious participants (Minsky, Shannon, McCarthy, etc.). This event marked the
starting point of Artificial Intelligence as a field on its own. The hopes were high: beating a chess champion, prove math theorems, replace human's works.
Following this meeting, Rosenblatt created the **perceptron** in 1958, a very simplified model of a biological neuron that aimed to classify images.
Yes, "artificial neural networks" and "computer vision" aren't new.

Despite these initial progresses, the field suffered from severe cutbacks in funding leading to the
first AI winter in the 70s. The expectation were too great compared to what was achieved. This phenomenon
occurred again in the 80s. At this point, practical applications came mainly from other approaches than
neural networks, the Support-Vector Machines (SVMs) invented in 1992 proved to be useful for the decades to come.

During the same time, Le Cun developped the first **Convolutional Neural Network** (CNN) which was specialized to
work on images. Fast-forward a few years, **Deep CNNs** became the kind of image classification, first in 2011 with Ciresan,
and later in 2012 with **AlexNet** of Krizhevsky.

Here is first, and easy, quiz to test our knowledge on this chapter:

{{< quiz cards=`[
    {Front: "Are artificial neural networks recent?", Back: "No! It dates back from 1958 and Rosenblatt's Perceptron."},
    {Front: "What is the main reason why neural networks didn't work in the XXth century?", Back: "Because of a lack of <b>computational power</b>"},
    {Front: "What are the two parts of a classical machine learning pipeline?", Back: "A <b>features</b> extractor (SIFT, HOG, etc.) and a <b>model</b> (SVM, Random Forest, etc.)"},
    {Front: "What is the necessary math property of the individual part of a neural network?", Back: "They all must be differentiable"},
    {Front: "What are the two phases during a neural network training", Back: "Forward and Backward passes"},
    {Front: "During which pass is the neural network updated?", Back: "Backward passes"},
]` quizid=1 >}}


# Deep Learning 101

In this course, we will first code a perceptron, the early CNN of Le Cun, and models similar to AlexNet.
Then, we will explore the more recent approaches made to Deep Learning. At this point, you may be confused by the terminology:
artificial intelligence, machine learning, neural networks, deep learning... What's their differences?

{{< optimg src="aivsmlvsdl.png" alt="AI vs ML vs DL" >}}


Machine Learning is a subfield of Artificial Intelligence based on statistical methods. Deep Learning is a subfield of Machine Learning based on deep neural networks.

Deep Neural Networks, that I'll now abbreviate to DNNs, are made of different modules, exactly like Legos and can be seen as a single function \\(f\\):

{{< optimg src="graph.png" alt="Computation graph" >}}


On the left, \\(X\\) is the input. It can be an image, a text, a sound, etc. On the
right, \\(f(X)\\) is the result of the function \\(f\\) when applied on \\(X\\).
The DNN showcased here is made of four blocks: two **linear** operations
(\\(W_0\\), \\(W_1\\)) and two **non-linear** operations (\\(\sigma\\), \\(\sigma\\)).
The former holds **parameters** which are also known as the network's neurons. The latter
are without any parameters.

We want to learn the parameters stored in \\(W_0\\) and \\(W_1\\) in order to have
a function \\(f\\) suitable to our goal which could be classifying cats and dogs from an image.
Following this example, our result \\(f(X)\\) would be a float value between 0 and 1. 0 would indicate
that our network is certain that \\(X\\) is an image of cat, and respectively for 1 an image of dog.
Most of the time our network is not fully confident and \\(f(X)\\) could be 0.2, 0.4, or even 0.7.

So, we want to make sure that \\(f\\) is as certain as possible ---and right-- given an image \\(X\\).
To do so, we need a **loss** function (also called cost function). This new function
has to penalize our network if it makes a mistake (i.e. saying cat for a dog image) or if it is
too uncertain (i.e. 0.47 for a cat image):

{{< optimg src="lossfunction.png" alt="Example of the loss function" >}}

A small loss means that our model is not that bad, and we shouldn't change too much
its parameters. On the other hand, a high loss means that we should change a lot of
parameters in order to improve.

In this course, we will cover our to design our function \\(f\\), what loss function
we can choose, and how to actually update our parameters.


{{< quiz cards=`[
    {Front: "Are artificial neural networks recent?", Back: "No! It dates back from 1958 and Rosenblatt's Perceptron."},
    {Front: "What is the main reason why neural networks didn't work in the XXth century?", Back: "Because of a lack of <b>computational power</b>"},
    {Front: "What are the two parts of a classical machine learning pipeline?", Back: "A <b>features</b> extractor (SIFT, HOG, etc.) and a <b>model</b> (SVM, Random Forest, etc.)"},
    {Front: "What is the necessary math property of the individual part of a neural network?", Back: "They all must be differentiable"},
    {Front: "What are the two phases during a neural network training", Back: "Forward and Backward passes"},
    {Front: "During which pass is the neural network updated?", Back: "Backward passes"},
]` quizid=1 >}}

# Applications


{{< quiz cards=`[
    {Front: "What are non-structured data?", Back: "Data that doesn't fit in tables, such as text, images, or sound."},
    {Front: "Deep Learning excels on structured or non-structured data?", Back: "Non-structured data"},
]` quizid=2 >}}

# Ecosystem

{{< quiz cards=`[
    {Front: "Python is slow, then why Deep Learning mainly exists on Python?", Back: "Because Python is binded to efficient C++ and CUDA"},
    {Front: "What is the fastest hardware to train a neural network? CPU or GPU?", Back: "GPU"},
]` quizid=3 >}}
