import torch
from pytriton.decorators import batch
import numpy as np
from pytriton.triton import Triton
from pytriton.model_config import ModelConfig, Tensor


class _InferFuncWrapper:
    def __init__(self, model: torch.nn.Module, device: str):
        self._model = model
        self._device = device

    @batch
    def __call__(self, **inputs):
        (input1_batch,) = inputs.values()
        input1_batch_tensor = torch.from_numpy(input1_batch).to(self._device)
        output1_batch_tensor = self._model(input1_batch_tensor)
        output1_batch = output1_batch_tensor.cpu().detach().numpy()
        return [output1_batch]


def _infer_function_factory(devices):
    infer_fns = []
    for device in devices:
        model = torch.nn.Linear(20, 30).to(device).eval()
        infer_fns.append(_InferFuncWrapper(model=model, device=device))

    return infer_fns


with Triton() as triton:
  triton.bind(
      model_name="Linear",
      infer_func=_infer_function_factory(devices=["cpu", "cpu"]),
      inputs=[
          Tensor(dtype=np.float32, shape=(-1,)),
      ],
      outputs=[
          Tensor(dtype=np.float32, shape=(-1,)),
      ],
      config=ModelConfig(max_batch_size=16),
  )

