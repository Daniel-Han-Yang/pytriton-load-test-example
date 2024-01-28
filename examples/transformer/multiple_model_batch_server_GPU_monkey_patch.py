import logging
import numpy as np
import time

from pytriton.model_config.triton_model_config import TritonModelConfig, TensorSpec, ResponseCache
from sentence_transformers import SentenceTransformer

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor, DeviceKind
from pytriton.triton import Triton, TritonConfig

logger = logging.getLogger("performance-testing-smallBert")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

def _get_triton_model_config(self) -> TritonModelConfig:
    """Generate ModelConfig from descriptor and custom arguments for Python model.

    Returns:
        ModelConfig object with configuration for Python model deployment
    """
    if not self._triton_model_config:
        triton_model_config = TritonModelConfig(
            model_name=self.model_name,
            model_version=self.model_version,
            batching=self.config.batching,
            batcher=self.config.batcher,
            max_batch_size=self.config.max_batch_size,
            decoupled=self.config.decoupled,
            backend_parameters={"workspace-path": self._workspace.path.as_posix()},
            instance_group={DeviceKind.KIND_GPU: len(self.infer_functions)},
        )
        inputs = []
        for idx, input_spec in enumerate(self.inputs, start=1):
            input_name = input_spec.name if input_spec.name else f"INPUT_{idx}"
            tensor = TensorSpec(
                name=input_name, dtype=input_spec.dtype, shape=input_spec.shape, optional=input_spec.optional
            )
            inputs.append(tensor)

        outputs = []
        for idx, output_spec in enumerate(self.outputs, start=1):
            output_name = output_spec.name if output_spec.name else f"OUTPUT_{idx}"
            tensor = TensorSpec(name=output_name, dtype=output_spec.dtype, shape=output_spec.shape)
            outputs.append(tensor)

        triton_model_config.inputs = inputs
        triton_model_config.outputs = outputs

        if self.config.response_cache:
            triton_model_config.response_cache = ResponseCache(enable=True)

        self._triton_model_config = triton_model_config

    return self._triton_model_config


def _infer_function_factory(devices):
    infer_fns = []
    for device in devices:
        model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1').to(device).eval()
        infer_fns.append(_InferFuncWrapper(model=model, device=device))

    return infer_fns


class _InferFuncWrapper:
    def __init__(self, model, device: str):
        self._model = model
        self._device = device

    @batch
    def __call__(self, **inputs):
        (sequence_batch,) = inputs.values()
        sequence_batch: np.ndarray = np.char.decode(sequence_batch.astype("bytes"), "utf-8")
        input_texts = [text for batch in sequence_batch for text in batch]

        ts = time.time()
        vector = self._model.encode(sentences=input_texts)
        print(f"Model runtime  : {int((time.time() - ts) * 1000)}ms")
        return [vector]


with Triton(TritonConfig) as triton:
    logger.info("Loading smallBert model.")
    triton.bind(
        model_name="smallBert",
        infer_func=_infer_function_factory(["cuda", "cuda", "cuda"]),
        inputs=[
            Tensor(name="text", dtype=np.bytes_, shape=(1,)),
        ],
        outputs=[
            Tensor(
                name="response",
                dtype=np.float32,
                shape=(-1,),
            ),
        ],
        config=ModelConfig(max_batch_size=50),
        strict=True,
    )
    logger.info("Serving inference")
    triton.serve()
