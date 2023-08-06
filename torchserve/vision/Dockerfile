# Download model from huggingface
FROM debian:bullseye-slim as model
WORKDIR /src
RUN apt-get update && \
  apt-get install -y sudo curl git && \
  curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash && \
  sudo apt-get install git-lfs
RUN git lfs install
RUN GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/google/vit-base-patch16-224 /src/model
RUN cd /src/model && git lfs pull --include "pytorch_model.bin"


FROM pytorch/torchserve:0.8.0-cpu
WORKDIR /src
COPY --link . .

RUN mkdir model_store
COPY --from=model /src/model/pytorch_model.bin .
COPY --from=model /src/model/config.json .
COPY --from=model /src/model/preprocessor_config.json .

RUN python -m pip install transformers

RUN torch-model-archiver \
    --model-name "vis" \
    --version 1.0 \
    --serialized-file ./pytorch_model.bin \
    --extra-files "./config.json,./preprocessor_config.json" \
    --handler ./vision-handler.py

RUN mv vis.mar ./model_store/

CMD ["torchserve", "--start", "--ncs", "--model-store", "./model_store/", "--models", "vis=vis.mar"]