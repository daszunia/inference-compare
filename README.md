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
torch-model-archiver --model-name "dialogpt" --version 1.0 --serialized-file ./DialoGPT-medium/pytorch_model.bin --extra-files "./DialoGPT-medium/config.json,./DialoGPT-medium/vocab.json" --handler ./inference-compare/torchserve/dialogpt/dialogpt-handler.py
```
* Move the *.mar file into `model_store` directory.
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

* For sending wav files use different data flag
```bash
curl localhost:8080/v1/models/kserve-custom-wav2vec:predict --data-binary @./voice.wav
```

# Tensorflow

* Download pre-trained model
* Process the model to be saved in tensorflow accepted format.
* Be careful to check what should be the new class compatible with TF. \
  For example `AutoModelForCausalLM` becomes `TFGPT2LMHeadModel`.
* Prepare pre-processing and post-processing scripts.
* Inject them into saved model and export.
* Download tensorflow serving docker image.

Unfortunately, it's impossible to inject arbitrary Python objects and code into TFServing routines that serve our model. All operations need to be compatible with TensorFlow computational graphs. Because we rely on Huggingface tokenizers, it's impossible to use them out-of-the-box with TFServing without any considerable modifications. These modifications may involve converting Huggingface tokenizers into TensorFlow tokenizers[https://github.com/Hugging-Face-Supporter/tftokenizers], relying on a rather untested (5 GitHub stars) and unmaintained (last modifications > 1 year ago) library.

https://discuss.huggingface.co/t/is-that-possible-to-embed-the-tokenizer-into-the-model-to-have-it-running-on-gcp-using-tensorflow-serving/10532


# BentoML

To run locally:
* Install `bentoml` via pip.
* Save model to local bentoml repository. (run `dialogpt-save-model.py` script)
* Run local server with `bentoml serve service:svc` in project directory.
* Ask for result by running (url is a name of the async function in `service.py`)
```bash
curl -X 'POST' http://0.0.0.0:3000/dialogpt -H 'accept: text/plain' -H 'Content-Type: text/plain' -d 'Hello, my name is XX how are you'
```

To run on kubernetes:
* Run `bentoml build`
* Wrap into a docker image with `bentoml containerize vis-service:latest`
* Re-tag the image if needed
* Load the image into local registry, for example `kind load docker-image localhost:5001/vis-bento`
* Apply deployment
