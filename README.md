# Licpi

Licpi is a <em>**Light-weight Customizable Pytorch-Inspired Framework.**</em>

If you're familiar with PyTorch, you'll appreciate how we can derive all the partial derivatives by just calling a function on the loss function.

---

<h3>Example</h3>

Easily build neural networks with scalar values and automatic differentiation. This example demonstrates both data (left number in each node) and gradients (right number in each node):

```python
from licpi import nn
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

![DAG](https://github.com/user-attachments/assets/a32586d8-2846-49b5-9897-52f2e423db4e)
> *The image is a segment of a Directed Acyclic Graph (DAG). It displays data and partial derivatives of the gradient in rectangular nodes, computations in oval nodes, and each edge indicates which pair of data and partial derivative was fed into the computation.

This repository is built by referring to [micrograd](https://github.com/karpathy/micrograd/), from [Andrej Karpathy](https://karpathy.ai/), 

I highly recommend his remarkable [tutorial](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ&index=1) to those who aspire to learn deep learning.
