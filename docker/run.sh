#!/bin/bash

#
# Copyright (c) 2025 Innodisk crop.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
#

docker run --rm -it \
    --net host \
    --privileged \
    -v /dev/:/dev \
    iq-vlm-demo
