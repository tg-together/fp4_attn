# syntax=docker/dockerfile:1.4
FROM nvcr.io/nvidia/pytorch:25.01-py3
RUN apt-get update && apt-get install -y --no-install-recommends

WORKDIR /workspace
COPY . .

RUN bash setup.sh
RUN pip install --no-cache-dir lm_eval pytest pytest-timeout einops

# sudo docker build -t fp4_attn .
# sudo docker run --rm -it --gpus all --shm-size=32g --ulimit memlock=-1 --ulimit stack=67108864 -v /home:/home -v /scratch:/scratch fp4_attn /bin/bash