# inference-compare
Comparing major ML serving frameworks

### Testing on models:
* Chatting model - https://huggingface.co/microsoft/DialoGPT-medium
* Speech to text - https://huggingface.co/facebook/wav2vec2-base-960h
* Image recognition - https://huggingface.co/google/vit-base-patch16-224

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

* Ask model for results (locally).
```bash
curl -X POST http://localhost:8080/predictions/{model_name} -T {test_file}
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

# KServe
* Deploy istio (for example 1.17.2)
* Deploy KServe related CRDs
* Build dockerfile, upload, apply deployment
* Port forward the pod
```bash
kubectl port-forward -n kserve kserve-custom-vision-predictor-00001-deployment-55bc9cc4c6658wn 8080:8080
```

* Ask for results (locally)
```bash
curl localhost:8080/v1/models/kserve-custom-dialogpt:predict -d "{ \"data\": \"How to get rich fast\" }"
```
