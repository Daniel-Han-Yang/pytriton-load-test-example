
import logging
import numpy as np
import itertools
import time

from sentence_transformers import SentenceTransformer

from pytriton.decorators import batch
from pytriton.model_config import ModelConfig, Tensor
from pytriton.triton import Triton

logger = logging.getLogger("performance-testing-smallBert")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s: %(message)s")

sbertmodel = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

@batch
def _infer_fn(**inputs: np.ndarray):
    ts = time.time()
    (input_request,) = inputs.values()
    input_request: np.ndarray = np.char.decode(input_request.astype("bytes"), "utf-8")
    input_texts = [text for text in input_request]
    ts = time.time()
    print(input_texts)
    vector = sbertmodel.encode(sentences=input_texts)
    print(vector)
    print(f"Model runtime  : {int((time.time() - ts) * 1000)}ms")

    return [vector]


with Triton() as triton:
    logger.info("Loading smallBert model.")
    triton.bind(
        model_name="smallBert",
        infer_func=_infer_fn,
        inputs=[
            Tensor(name="text", dtype=np.bytes_, shape=(1,)),
        ],
        outputs=[
            Tensor(
                name="response",
                dtype=np.float32,
                shape=(1, 384),
            ),
        ],
        config=ModelConfig(batching=False),
        strict=True,
    )
    logger.info("Serving inference")
    triton.serve()
