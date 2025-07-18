FROM docker.io/ubuntu:22.04


ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /app
EXPOSE 4000/tcp

# Enable apt cache
RUN rm -f /etc/apt/apt.conf.d/docker-clean; \
    echo 'Binary::apt::APT::Keep-Downloaded-Packages "true";' > /etc/apt/apt.conf.d/keep-cache

# Install run-time dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt update && apt --no-install-recommends install -y \
        python3-venv \
        xkb-data \
        libgdk-pixbuf2.0-0 \
        fontconfig \
        ;

COPY requirements.txt /app/
RUN python3 -m venv venv && venv/bin/pip3 install -r requirements.txt

COPY webcam.py webcam_desktop.py docker/env.sh /app/
COPY resources/vlm_short_prompt.txt /app/vlm_prompt.txt

CMD [ "./env.sh", "venv/bin/python3", "webcam_desktop.py" ]

