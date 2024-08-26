import licpi
from typing import Optional, List, Any, Tuple, Union
from .backend_numpy import Device, cpu, all_devices
import numpy

from licpi import init

# needle version
LAZY_MODE = True
TENSOR_COUNTER = 0

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks
import numpy as array_api

NDArray = numpy.ndarray


class Op:
    def __call__(self, *args):
        raise ModuleNotFoundError()
        
    def compute(self, *args: Tuple[NDArray]):
        '''Calculate forward pass of operator
        
        Parameters
        ----------
        input: np.ndarray
            A list of input arrays to the function
            
        Returns
        -------
        output: nd.array
            Array output of the operation
            
        '''
        raise NotImplementedError()
        
    def gradient(self, out_grad: "Value", node: "Value") -> Union["Value", Tuple["Value"]]:
        """compute partial adjoing for each input value for a given output adjoint.

        Parameters
        ----------
        out_grad: Value
            The adjoint wrt to the output value.

        node: Value
            The value node of forward evaluation.

        Returns
        ------
        input_grads: Value or Tuple[Value]
            A list containing partial gradient adjoints to be propagated to each of the input node.
        """
        raise NotImplementedError()

    def gradient_as_tuple(self, out_grad: "Value", node: "Value") ->Tuple["Value"]:
        """ Convenience method to always return a tuple from gradient call"""
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            return (output,)

class TensorOp(Op):
    """ Op class specialized to output tensors, will be alternate subclasses for other structures"""
    def __call__(self, *args):
        return Tensor.make_from_op(self,args)


class Value:
    """A value in the computational graph."""

    # trace of computational graph
    op: Optional[Op]
    inputs: list["Value"]
    # The following fields are cached fields for 
    # dynamic computation
    cached_data: NDArray
    requires_grad: bool

    def realize_cached_data(self):
        """Run compute to realize the cached data."""
        # avoid recomputation
        if self.cached_data is not None:
            return self.cached_data
        # note: data implicitly calls realized cached data
        self.cached_data = self.op.compute(
            *[x.realize_cached_data() for x in self.inputs]
        )
        return self.cached_data

    def _init(self, op: Optional[Op], inputs: List["Value"], *, num_outputs: int = 1, cached_data: List[object] = None, requires_grad: Optional[bool]= None):
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs =inputs 
        self.num_outputs = num_outputs 
        self.cached_data = cached_data 
        self.requires_grad = requires_grad 

    def __repr__(self):
        return "licpi.Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), _op="*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out =  Value(self.data ** other, (self,), f"**{other}")

        def _backward():
            self.grad += (other * self.data ** (other-1)) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __truediv__(self, other):
        return self * other**-1

    def exp(self):
        out = Value(math.exp(self.data), (self, ), _op="exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        x = self.data 
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self, ), "tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)

        self.grad = 1
        
        for node in reversed(topo):
            node._backward()

    def __radd__(self, other): # other + self
        return self + other

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __rtruediv__(self, other): # other / self
        return other * self**-1


class Tensor(Value):
    def __init__(self, array, *, device: Optional[Device] = None, dtype = None, requires_grad = True, **kwags):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.drype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
        else:
            device = device if device else cpu()
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)
        
        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad
        )

    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        if array_api is numpy:
            return numpy.array(numpy_array, dtype=dtype)
        return array_api.array(numpy_array, device=device, dtype=dtype)

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        tensor.realize_cached_data()
        return tensor
    
    def backward(self, out_grad=None):
        out_grad = (
            out_grad
            if out_grad
            else init.ones(*self.shape, dtype=self.dtype, device=self.device)
        )
    
    def __add__(self, other):
        if isinstance(other, Tensor):
            return licpi.ops.EWiseAdd()(self, other)
        else:
            return licpi.ops.AddScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return licpi.ops.EWiseAdd()(self, licpi.ops.Negate()(other))
        else:
            return licpi.ops.AddScalar(-other)(self)

def compute_gradient_of_variables(output_tensor, out_grad):
    """Take gradient of output node with respect to each node in node_list.

    Store the computed result in the grad fielf of each variable. 
    """

    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list: Dist[Tensor, List[Tensor]] = {}
    
    node_to_output_grads_list[output_tensor] = [out_grad]

    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    for node in reverse_topo_order:
        ajoint = node_to_output_grads_list[node]
        node.grad = ajoint
        if node.op is None:
            continue
        partial_ajoints = node.op.gradient_as_tuple(ajoint, node)
        for in_node, partial_ajoint in zip(node.inputs, partial_ajoints):
            if in_node not in node_to_output_grads_list:
                node_to_output_grads_list[in_node] = []
            node_to_output_grads_list[in_node].append(partial_ajoint)


def find_topo_sort(node_list: List[Value]) -> List[Value]:
    visited = []
    topo_order = []

    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order

def topo_sort_dfs(node: Value, visited: List[Value], topo_order : List[Value]):
    if node not in visited:
        visited.append(node)
        for input in node.inputs:
            topo_sort_dfs(input, visited, topo_order)
        topo_order.append(node)
        