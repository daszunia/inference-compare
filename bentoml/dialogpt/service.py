import bentoml

dialogpt_runner = bentoml.models.get("dialogpt_model:latest").to_runner()

svc = bentoml.Service(
    name="dialogpt-service", runners=[dialogpt_runner]
)

@svc.api(input=bentoml.io.Text(), output=bentoml.io.Text())
async def dialogpt(text: str) -> str:
    generated = await dialogpt_runner.async_run(text, max_length=3000)
    return generated
