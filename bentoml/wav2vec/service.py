import bentoml
import io
from typing import Any

wav2vec_runner = bentoml.models.get("wav2vec_model:latest").to_runner()

svc = bentoml.Service(
    name="wav2vec-service", runners=[wav2vec_runner]
)

@svc.api(input=bentoml.io.File(), output=bentoml.io.Text())
async def wav2vec(audioFile: io.BytesIO) -> str:
    audioBytes = audioFile.read()
    generated = await wav2vec_runner.async_run(audioBytes, max_length=3000)
    return generated
