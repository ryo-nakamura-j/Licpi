from .engine import Tensor, NDArray, TensorOp
from typing import Tuple

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b
    
    def gradient(self, out_grad: Tensor, node: Tensor) -> Tensor | Tuple[Tensor]:
        return out_grad, out_grad