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
import queue
import textwrap
import threading
import time

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


bus: queue.Queue[Message] = queue.Queue()
stopped = threading.Event()

images = collections.deque(maxlen=1)  # JPEG image
new_image = threading.Event()


def blend_with_background(overlay, background):
    # separate the alpha channel from the color channels
    alpha_channel = overlay[:, :, 3] / 255  # convert from 0-255 to 0.0-1.0
    overlay_colors = overlay[:, :, :3]

    # To take advantage of the speed of numpy and apply transformations to the entire
    # image with a single operation
    # the arrays need to be the same shape. However, the shapes currently
    # looks like this:
    #    - overlay_colors shape:(width, height, 3)  3 color values for each pixel,
    #                                               (red, green, blue)
    #    - alpha_channel  shape:(width, height, 1)  1 single alpha value for each pixel
    # We will construct an alpha_mask that has the same shape as the
    # overlay_colors by duplicate the alpha channel
    # for each color so there is a 1:1 alpha channel for each color channel
    alpha_mask = alpha_channel[:, :, np.newaxis]

    # combine the background with the overlay image weighted by alpha
    composite = background * (1 - alpha_mask) + overlay_colors * alpha_mask

    return composite


def put_wrapped_text(
    image,
    text,
    org,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=1.0,
    color=(255, 255, 255),
    thickness=2,
    bg_color=(19, 13, 157),
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
    line_height = (
        int(cv2.getTextSize("Ag", font, font_scale, thickness)[0][1] * line_spacing) + 3
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
        ret = new_image.wait(timeout=2)
        if not ret:
            return

        image = images.pop()

        responses = vlm2(base64.b64encode(image).decode(), "")
        try:
            first_response = next(responses)
            response = f"{first_response}"
            bus.put(Message(response=response))

            for text in responses:
                response += text
                bus.put(Message(response=response))

        except StopIteration:
            time.sleep(1)
            continue


def vlm2(image: str, _description: str):
    llm = ChatOllama(
        model="llava2_7B_FT:latest",
        num_ctx=10000,
        temperature=0.8,
        num_predict=256,
        base_url=LLM_BASE_URL,
    )

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(["./", RUNTIME_PATH]))
    template = env.get_template("iqs_vlm_prompt.txt")
    prompt = template.render()

    messages = [
        SystemMessage(
            content=(
                "A chat between a curious human and an artificial intelligence "
                "assistant. The assistant gives helpful, detailed, and polite answers "
                "to the human's questions."
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
            return
    else:
        webcam_dev = WEBCAM_DEV

    pipeline = f"""
    v4l2src device={webcam_dev} !
    queue !
    image/jpeg,width=1920,height=1080,framerate=30/1 !
    appsink sync=false max-buffers=1 drop=true
    """

    video_capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not video_capture.isOpened():
        logger.error("Failed to open {}", webcam_dev)
        return

    logger.info("{} is opened", webcam_dev)

    while not stopped.is_set():
        ret, frame = video_capture.read()
        if not ret:
            continue

        raw = cv2.imdecode(frame, cv2.IMREAD_COLOR)
        if raw is None:
            continue

        bus.put(Message(image=raw))

        images.append(frame)
        new_image.set()


def paint():
    WINDOW_NAME = "iqs-vlm-demo"
    OGOL = "iVBORw0KGgoAAAANSUhEUgAAAO4AAABiCAMAAABgQh+zAAAATlBMVEUAAADsGiPsGyPqGyPqGyPrGyP////94+P1jZH3qa3tOD/zcXX6xsjtKTH+8fHwVFrzcXb0f4P5uLv81NbyYmjvRk33m5/vRkz4qazycXbNN6odAAAABXRSTlMA35/fYAEB6DsAAANXSURBVHja7dnbcuIwDIDhbbuSHPmUE4S+/4tuk1oTE3sL24ENZfRfhWaM+AiEdPLrWXuBSq/KVa5ylatc5SpXucpVrnKVq1zlKle5P4z78qS9vVb6/QuetPrnVrnKVa5ylatc5SpXucpVrnKVq1zlKle5ylXuA3GjmRvh/o3rJG+W4IsOZq67NdfgXAP3r1knMS7BF1mcI+UqV7l1LrRurvvPXLd0Z+4+lVxJuT+Ge7DWxu9zvZX1Va531t+C29HcEWSzWV55T3N9hNRIHxkAYDt8Lshly18DfkTU242ip8+sW7lMS+tyIsKPAhGdLFe4ETGI+BZnZtkkz6eAqXDK32sCNoSS8yB1AbPokL+ZmFU/M7MLeNZQ4fY4R3xrLjrMcxk3HDFvSF4uSB2kHOJFrsFtVHKjLL4tt6wVbtk7zHnCbWGSQ3KZy3QNVybenUs+51YOb4spIgpnizq8gitvSaC5UOVyL097F24ga608as+4obfWnhKLs4NLdn4gi7p8j7MfUahyWVYfPmHWEm24fgxpdgd34Mq5pi25wY0wd8SlCQAOuNTDkhxRWtcMDEvR/Z0bQFp4OZfTOplxc+4JUlRw5bvD65FqcCmeP1+Y5DwV/JfXzFyHyMTgAkq9vwuXC275o9+nVYJyII1B+CR7LnPRcMktzxR7ceWl58dZonNuexUXyUmxyqXOw+Nwpy23u4Zb/7EauOS+C/YxuLzlumu5BktvyaUTPwcXjlg0ZlwpHOEJvrv1q8iVm2UegOuHDddvuM1X3FRnUk3iepkYjHEoHXfnyhZ5SHUo3CPOhXiZK8kabPOJcVi9e3PlBXbnBzewrMHmX7i24M4ZuYrk3bmMuXfq1/8bOaTtcQIfo6Eq99CYOEHKu8TaTDyQXFjtzfUJiK6LrUtCjGfnIHKEqYIb0zWGiTE2hEtUTOS0h/fmQgxY5GCOBywruWWmnNjKnr25cCi88pnzw7e4vS8myjnB+b25pXfwAOL9Btf52sQOl3h/LvhjBqb8+tZ3PWZR0/CGy03AvKHxUJkoS/hW93dl0/jzX/8pv9ta3quVzGcjFHHaNYG0ub8rk9bh5URZ4vXuvXKVq1zlKle5ylWucpWrXOUqV7nKVa5ylavcR+K+PGdvde4fzR5ttmCaatkAAAAASUVORK5CYII= "  # noqa: E501

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    ogol = cv2.imdecode(
        np.frombuffer(base64.b64decode(OGOL), np.uint8), cv2.IMREAD_UNCHANGED
    )
    scence = np.array([])
    image = np.array([])
    response = " "
    while True:
        try:
            m = bus.get(timeout=2)
        except queue.Empty:
            return

        if m.image.size > 0:
            image = m.image
            image[-ogol.shape[0] :, -ogol.shape[1] :] = blend_with_background(
                ogol, image[-ogol.shape[0] :, -ogol.shape[1] :]
            )
        if m.response:
            response = m.response.strip()

        scence = image

        if scence.size > 0:
            put_wrapped_text(
                scence,
                response,
                (int(scence.shape[1] * 0.03), int(scence.shape[0] * 0.03)),
                cv2.FONT_HERSHEY_SIMPLEX,
                2.0,
                (255, 255, 255),
                thickness=2,
                max_width=int(scence.shape[1]),
                line_spacing=1.5,
            )
            cv2.imshow(WINDOW_NAME, scence)
        if cv2.waitKey(3) == ord("q"):
            return


t1 = threading.Thread(None, target=video_source, name="video_source")
t2 = threading.Thread(None, target=process, name="vlm")
t1.start()
t2.start()

paint()
stopped.set()
t1.join()
t2.join()
cv2.destroyAllWindows()
