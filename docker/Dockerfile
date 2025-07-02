FROM docker.io/alpine:3.20.6

WORKDIR /app
EXPOSE 4000/tcp

RUN apk add python3 py3-opencv gst-plugins-good

COPY requirements.txt /app/
RUN python3 -m venv --system-site-packages venv && venv/bin/pip3 install -r requirements.txt

COPY webcam.py  /app/
COPY resources/vlm_short_prompt.txt /app/vlm_prompt.txt

CMD [ "venv/bin/python3", "webcam.py" ]

