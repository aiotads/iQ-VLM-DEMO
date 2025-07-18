#!/usr/bin/env python3
#
# Copyright (c) 2025 Innodisk crop.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
#

import base64
import collections
import dataclasses
import glob
import sys
import textwrap
import threading
import time
from queue import Queue

import cv2
import httpx
import jinja2
import numpy as np
import numpy.typing as npt
import urllib3
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from loguru import logger

LLM_BASE_URL = "http://127.0.0.1:22434"

RUNTIME_PATH = "/opt/innodisk/ppes/llm"

WEBCAM_DEV = None  # Search for the first one
# WEBCAM_DEV = "/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e-video-index0"


urllib3.disable_warnings()


@dataclasses.dataclass
class Message:
    image: npt.NDArray = dataclasses.field(
        default_factory=lambda: np.array([])
    )  # raw BGR image
    response: str = ""
    vlm_image: npt.NDArray = dataclasses.field(
        default_factory=lambda: np.array([])
    )  # raw BGR image


bus: Queue[Message] = Queue()
stopped = threading.Event()

images = collections.deque(maxlen=1)  # JPEG image
new_image = threading.Event()


def put_wrapped_text(
    image,
    text,
    org,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=1.0,
    color=(255, 255, 255),
    thickness=2,
    bg_color=(0, 0, 0),
    max_width=400,
    line_spacing=1.2,
):
    """
    Draws wrapped text with a background rectangle on the image.

    Parameters:
        image (np.ndarray): The target image.
        text (str): The text to draw.
        org (tuple): Top-left corner (x, y) where text should start.
        font (int): OpenCV font.
        font_scale (float): Font scale factor.
        color (tuple): Text color (B, G, R).
        thickness (int): Thickness of the text.
        bg_color (tuple): Background rectangle color.
        max_width (int): Max width in pixels before wrapping.
        line_spacing (float): Spacing multiplier between lines.
    """
    x, y = org
    wrapped_lines = []

    # Estimate average character width in pixels
    avg_char_width = cv2.getTextSize("A", font, font_scale, thickness)[0][0]
    max_chars_per_line = max_width // avg_char_width

    wrapped_lines.extend(textwrap.wrap(text, width=max_chars_per_line))

    # Measure text block size
    line_height = int(
        cv2.getTextSize("Ag", font, font_scale, thickness)[0][1] * line_spacing
    )
    block_height = line_height * len(wrapped_lines)
    block_width = max(
        (
            cv2.getTextSize(line, font, font_scale, thickness)[0][0]
            for line in wrapped_lines
        ),
        default=1,
    )

    # Draw background rectangle
    cv2.rectangle(
        image, (x, y), (x + block_width, y + block_height), bg_color, thickness=-1
    )

    # Draw each line of text
    for i, line in enumerate(wrapped_lines):
        text_y = y + int((i + 1) * line_height - (line_height * (1 - 1 / line_spacing)))
        cv2.putText(
            image,
            line,
            (x, text_y),
            font,
            font_scale,
            color,
            thickness,
            lineType=cv2.LINE_AA,
        )

    return image


def process():
    response = "┃ "

    while not stopped.is_set():
        new_image.wait()

        image = images.pop()

        responses = vlm2(base64.b64encode(image).decode(), "")
        try:
            first_response = next(responses)
            response = f"{first_response}"

            vlm_image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            bus.put(Message(vlm_image=vlm_image, response=response))

            for text in responses:
                response += text
                bus.put(Message(response=response))

        except StopIteration:
            time.sleep(1)
            continue


def vlm2(image: str, description: str):
    llm = ChatOllama(
        model="llava2_7B_FT:latest",
        num_ctx=10000,
        temperature=0.8,
        num_predict=256,
        base_url=LLM_BASE_URL,
    )

    env = jinja2.Environment(loader=jinja2.FileSystemLoader([RUNTIME_PATH, "./"]))
    template = env.get_template("vlm_prompt.txt")
    prompt = template.render()

    messages = [
        SystemMessage(
            content=(
                "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions."
            )
        ),
        HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                },
            ]
        ),
    ]
    try:
        for chunk in llm.stream(messages):
            yield chunk.text()
    except httpx.HTTPError as e:
        message = "Unable to talk to the Ollama server"
        logger.error("{}: {} {}", message, e.request, e)


def search_webcam() -> None | str:
    cams = glob.glob("/dev/v4l/by-id/*-index0")
    if not cams:
        return None
    cams.sort()
    return cams[0]


def video_source():
    if not WEBCAM_DEV:
        webcam_dev = search_webcam()
        if not webcam_dev:
            logger.error("No webcam is detected")
            # nicegui.app.shutdown()
            sys.exit(500)
    else:
        webcam_dev = WEBCAM_DEV

    # pipeline = "pipewiresrc ! queue ! image/jpeg,width=1920,height=1080,framerate=30/1 ! appsink sync=false max-buffers=1 drop=true"
    pipeline = f"""
    v4l2src device={webcam_dev} !
    queue !
    image/jpeg,width=1920,height=1080,framerate=30/1 !
    appsink sync=false max-buffers=1 drop=true
    """

    video_capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not video_capture.isOpened():
        logger.error("Failed to open {}", webcam_dev)
        sys.exit(500)

    logger.info("{} is opened", webcam_dev)

    while not stopped.is_set():
        ret, frame = video_capture.read()
        if not ret:
            continue

        raw = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        bus.put(Message(image=raw))

        images.append(frame)
        new_image.set()


def paint():
    cv2.namedWindow("iq-vlm", cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow("iq-vlm", 1280, 720)
    scence = np.array([])
    image = np.array([])
    vlm_image = np.array([])
    response = " "
    while True:
        m = bus.get(timeout=2)
        if m.image.size > 0:
            image = m.image
        if image.size > 0 and m.vlm_image.size > 0:
            tmp = cv2.resize(m.vlm_image, None, fx=0.35, fy=0.35)
            vlm_image = cv2.copyMakeBorder(
                tmp,
                0,
                0,
                0,
                image.shape[1] - tmp.shape[1],
                cv2.BORDER_CONSTANT,
                0,
            )
        if m.response:
            response = m.response.strip()

        if vlm_image.size > 0:
            scence = np.concatenate((image, vlm_image), axis=0)
        else:
            scence = image

        if scence.size > 0:
            put_wrapped_text(
                scence,
                response,
                (int(scence.shape[1] * 0.03), int(scence.shape[0] * 0.03)),
                cv2.FONT_HERSHEY_DUPLEX,
                2.0,
                (255, 255, 255),
                2,
                max_width=int(scence.shape[1]),
                line_spacing=1.5,
            )
            cv2.imshow("iq-vlm", scence)
        if cv2.waitKey(1) == ord("q"):
            cv2.destroyAllWindows()
            return


t1 = threading.Thread(None, target=video_source, name="video_source")
t2 = threading.Thread(None, target=process, name="vlm")
t1.start()
t2.start()

paint()
stopped.set()
t1.join()
t2.join()
