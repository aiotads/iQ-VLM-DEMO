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
    --shm-size=2g \
    -v /dev/:/dev \
    -v /usr/lib:/host_lib \
    -v /usr/libexec:/host_libexec \
    iq-vlm-demo
