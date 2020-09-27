#!/usr/bin/env bash

xhost +local:root 1>/dev/null 2>&1
docker exec \
    -it terminal \
    /bin/bash
xhost -local:root 1>/dev/null 2>&1