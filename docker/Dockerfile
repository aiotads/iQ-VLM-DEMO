FROM docker.io/ubuntu:24.04


ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app
EXPOSE 4000/tcp

# Enable apt cache
RUN rm -f /etc/apt/apt.conf.d/docker-clean; \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

# Install run-time dependencies
# RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
#     --mount=type=cache,target=/var/lib/apt,sharing=locked \
RUN apt update && apt --no-install-recommends install -y \
        python3-venv \
        xkb-data \
        libgdk-pixbuf2.0-0 \
        fontconfig \
        ;

COPY requirements.txt /app/
RUN python3 -m venv venv && venv/bin/pip3 install -r requirements.txt

COPY resources/vlm_story_prompt.txt /app/iqs_vlm_prompt.txt
COPY videos /app/videos
COPY prompts /app/prompts
COPY webcam.py webcam_desktop.py async_safety_video.py docker/env.sh /app/


LABEL BSP_VERSION="0.0.1"
LABEL APP_VERSION="1.0.0"
WORKDIR  /workspace
ENTRYPOINT ["/app/env.sh", "/app/venv/bin/python3", "/app/async_safety_video.py"]
CMD [ "--runtime-dir", "/app/" ]
