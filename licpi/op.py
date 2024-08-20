from typing import Any, Tuple, Union
import numpy
from engine import Value, Tensor

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




        