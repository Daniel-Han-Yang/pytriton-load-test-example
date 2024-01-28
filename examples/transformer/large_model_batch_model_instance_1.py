import logging
import numpy as np
import time

from sentence_transformers import SentenceTransformer

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

logger = logging.getLogger("performance-testing-smallBert")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")


def _infer_function_factory(devices):
    infer_fns = []
    for device in devices:
        model = SentenceTransformer('intfloat/e5-large-v2').to(device).eval()
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


with Triton() as triton:
    logger.info("Loading smallBert model.")
    triton.bind(
        model_name="smallBert",
        infer_func=_infer_function_factory(["cuda"] * 1),
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
        config=ModelConfig(max_batch_size=16),
        strict=True,
    )
    logger.info("Serving inference")
    triton.serve()
