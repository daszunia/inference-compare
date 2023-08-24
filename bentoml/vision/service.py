import bentoml

vis_runner = bentoml.models.get("vis_model:latest").to_runner()

svc = bentoml.Service(
    name="vis-service", runners=[vis_runner]
)

@svc.api(input=bentoml.io.Text(), output=bentoml.io.Text())
async def vis(text: str) -> str:
    generated = await vis_runner.async_run(text, max_length=3000)
    return generated
