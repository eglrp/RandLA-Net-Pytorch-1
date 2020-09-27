#!/usr/bin/env bash

docker run --gpus all \
    -d \
    --privileged \
    --name terminal \
    --hostname pytorch_dev \
    --pid=host \
    --gpus all \
    -v /home/terminal/Git:/home/terminal/Git \
    -p 12345:22 \
    -p 23456:8888 \
    --rm -ti --ipc=host \
    qinyi20060410/pytorch:local \
    /bin/bash