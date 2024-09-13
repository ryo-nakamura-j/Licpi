from licpi.engine import Value
from .engine import Tensor, NDArray, TensorOp,array_api
from typing import Tuple

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b
    
    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor | Tuple[Tensor]:
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
     
        return a ** self.scalar
 

    def gradient(self, out_grad, node):
        return node.inputs[0]**(self.scalar - 1) * out_grad * self.scalar


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], NDArray) or not isinstance(
            node.inputs[1], NDArray
        ):
            raise ValueError("Both inputs must be tensors (NDArray).")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * array_api.log(a.data)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad / rhs, - lhs * out_grad / rhs ** 2


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return array_api.divide(a, self.scalar)

    def gradient(self, out_grad, node):
        return out_grad / self.scalar


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
     
        if self.axes:
            return array_api.swapaxes(a, self.axes[0], self.axes[1])
        else:
            return array_api.swapaxes(a, a.ndim - 2, a.ndim - 1)
 

    def gradient(self, out_grad, node):
        shape = node.inputs[0].shape
        return array_api.transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)
 

    def gradient(self, out_grad, node):
     
        return out_grad.reshape(node.inputs[0].shape)
 


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
     
        return array_api.broadcast_to(a, self.shape)
 

    def gradient(self, out_grad, node):
        ori_shape = node.inputs[0].shape
        shrink_dims = [i for i in range(len(self.shape))]

        for i, (ori,cur) in enumerate(zip(reversed(ori_shape), reversed(self.shape))):
            if ori==cur:
                shrink_dims[- (i + 1)] = -1

        shrink_dims = tuple(filter(lambda x:x>=0, shrink_dims))
        return out_grad.sum(shrink_dims).reshape(ori_shape)

def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
     
        return array_api.sum(a, self.axes)
 

    def gradient(self, out_grad, node):
        new_shape = list(node.inputs[0].shape)
        axes = range(len(new_shape)) if self.axes is None else self.axes
        for axis in axes:
            new_shape[axis] = 1
        return out_grad.reshape(new_shape).broadcast_to(node.inputs[0].shape)

def summation(a, axes=None):
    return Summation(axes)(a)
class MatMul(TensorOp):
    def compute(self, a, b):
     
        return array_api.matmul(a, b)
 

    def gradient(self, out_grad, node):
     
        lhs, rhs = node.inputs
        return matmul(out_grad, rhs.transpose()),matmul(lhs.transpose(), out_grad)


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
     
        return array_api.negative(a)
 

    def gradient(self, out_grad, node):
     
        return array_api.negative(out_grad)


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
     
        return array_api.log(a)

 

    def gradient(self, out_grad, node):
     
        return out_grad / node.inputs[0]
 


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
     
        return array_api.exp(a)
 

    def gradient(self, out_grad, node):
     
        return out_grad * exp(node.inputs[0])
 


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
     
        out = array_api.copy(a)
        out[out < 0] = 0
        return out

    def gradient(self, out_grad, node):
    #######################################################################
    # The original solution is not numerically stable.
    #
    # grad = divide(relu(node.inputs[0]), node.inputs[0])
    # return (multiply(out_grad, grad),)
    #######################################################################
    # There seems to be no numerically stable solution
    # that solely calls needle operations. assistance of
    # `array_api` is a must.
        node_input = node.inputs[0]
        return multiply(out_grad,
                        Tensor(node_input.realize_cached_data() > 0,
                                device=node.device,
                                dtype=node.dtype,
                                required_grad=node.requires_grad))


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        max_z_original = array_api.max(Z, self.axes, keepdims=True) 
        max_z_reduce = array_api.max(Z, self.axes)
        return array_api.log(array_api.sum(array_api.exp(Z - max_z_original), self.axes)) + max_z_reduce 
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        z = node.inputs[0]
        max_z = z.realize_cached_data().max(self.axes, keepdims=True)
        exp_z = exp(z - max_z)
        sum_exp_z = summation(exp_z, self.axes)
        grad_sum_exp_z = out_grad / sum_exp_z
        expand_shape = list(z.shape)
        axes = range(len(expand_shape)) if self.axes is None else self.axes
        for axis in axes:
            expand_shape[axis] = 1
        grad_exp_z = grad_sum_exp_z.reshape(expand_shape).broadcast_to(z.shape)
        return grad_exp_z * exp_z
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)