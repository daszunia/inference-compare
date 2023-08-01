# inference-compare
Comparing major ML serving frameworks

# Torchserve
* Download model from GIT LFS. Check out the repo and download only one needed file.
```bash
git lfs install

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/microsoft/DialoGPT-medium

git lfs pull --include "pytorch_model.bin"
```

* Create `model_store` directory.
* Run toch archiver to transform model into *.mar format.
```bash
torch-model-archiver --model-name "dialogpt" --version 1.0 --serialized-file ./DialoGPT-medium/pytorch_model.bin --extra-files "./DialoGPT-medium/config.json,./DialoGPT-medium/vocab.json" --handler ./inference-compare/dialogpt-handler.py
```
* Run serve.
```bash
torchserve --start --model-store model_store/ --models dialogpt=dialogpt.mar
```

* Stop serve.
```bash
torchserve --stop
```

## For running in K8s
* Build docker image.
```bash
docker build --tag dialogpt-test:latest .
```

* If needed push the image to local image registry, for example in kind (latest tag is assumed).
```bash
kind load docker-image localhost:32000/dialogpt-test
```

* Apply provided deployment which will create service with http port 8080 and metrics port 8081.
```bash
kubectl apply -f deployment.yaml
```