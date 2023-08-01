# Download model from huggingface
FROM debian:bullseye-slim as model
WORKDIR /src
RUN apt-get update && \
  apt-get install -y sudo curl git && \
  curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && \
  sudo apt-get install git-lfs
RUN git lfs install
RUN GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/microsoft/DialoGPT-medium /src/model
RUN cd /src/model && git lfs pull --include "pytorch_model.bin"


FROM pytorch/torchserve:0.8.0-cpu
WORKDIR /src
COPY --link . .

RUN mkdir model_store
COPY --from=model /src/model/pytorch_model.bin .
COPY --from=model /src/model/config.json .
COPY --from=model /src/model/vocab.json .

RUN python -m pip install transformers

RUN torch-model-archiver \
    --model-name "dialogpt" \
    --version 1.0 \
    --serialized-file ./pytorch_model.bin \
    --extra-files "./config.json,./vocab.json" \
    --handler ./dialogpt-handler.py

RUN mv dialogpt.mar ./model_store/

CMD ["torchserve", "--start", "--model-store", "./model_store/", "--models", "dialogpt=dialogpt.mar"]