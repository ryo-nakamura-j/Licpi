# licpi

licpi is a light customizable pytorch-inspired framework

If you are familiar with the implementation of pytorch library, 
You might be grateful for we can derive all the partial derivatives by just calling a function on the loss function.

Likewise, you can implement neural network with scalar values with autograd functionarity as follows.

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
