# Licpi

Licpi is a light customizable pytorch-inspired framework

If you're familiar with PyTorch, you'll appreciate how we can derive all the partial derivatives by just calling a function on the loss function.

Example: 

Easily build neural networks with scalar values and automatic differentiation. This example demonstrates both data (left number in each node) and gradients (right number in each node):

```python
from micrograd import nn
n = nn.Neuron(2)
X = [
    [3.0, 1.0],
    [3.0, 1.0],
]
y = [-1.0, 1.0]
n = nn.MLP(2,[2,1])

y_pred = [n(x) for x in X]
loss = sum((ygt - yout)**2 for ygt, yout in zip(y, y_pred))
loss.backward()

dot = draw_dot(y)
```


![dag](https://github.com/user-attachments/assets/8d612c7d-0a1d-4fad-af11-384d7ca023fa)


This repository is built by referring to [micrograd](https://github.com/karpathy/micrograd/), from [Andrej Karpathy](https://karpathy.ai/), 

I highly recommend his remarkable [tutorial](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1) to those who aspire to learn deep learning.
