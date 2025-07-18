#!/bin/bash

export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:/host_lib
export GST_PLUGIN_PATH=/host_lib/gstreamer-1.0/
export GST_PLUGIN_SCANNER=/host_libexec/gstreamer-1.0/gst-plugin-scanner

export XDG_RUNTIME_DIR=/dev/socket/weston
export WAYLAND_DISPLAY=wayland-1
export PYTHONPATH=/host_lib/python3.10/site-packages/

"$@"
