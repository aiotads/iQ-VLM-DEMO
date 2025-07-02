#!/usr/bin/env python3

#
# Copyright (c) 2025 Innodisk crop.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
#

import asyncio
import base64
import glob
from collections import deque

import cv2
import httpx
import jinja2
import nicegui
import urllib3
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from loguru import logger
from nicegui import html, ui

LLM_BASE_URL = "http://127.0.0.1:22434"
# LLM_BASE_URL = "http://192.168.3.130:11434"

RUNTIME_PATH = "/opt/innodisk/ppes/llm"

WEBCAM_DEV = None  # Search for the first one
# WEBCAM_DEV = "/dev/v4l/by-id/usb-046d_Logitech_Webcam_C930e-video-index0"


urllib3.disable_warnings()


@nicegui.binding.bindable_dataclass
class Model:
    response: str = ""
    image: str = ""  # base64-encoded JPEG
    vlm_image: str = ""  # base64-encoded JPEG
    raw: str = ""


event_received = asyncio.Event()
events: deque = deque(maxlen=1)

model = Model()


def notify(message: str, **kwds) -> None:
    for client in nicegui.Client.instances.values():
        if not client.has_socket_connection:
            continue
        with client:
            ui.notify(message, **kwds)


def search_webcam() -> None | str:
    cams = glob.glob("/dev/v4l/by-id/*-index0")
    if not cams:
        return None
    cams.sort()
    return cams[0]


async def webcam() -> None:
    if not WEBCAM_DEV:
        webcam_dev = search_webcam()
        if not webcam_dev:
            logger.error("No webcam is detected")
            nicegui.app.shutdown()
            return
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
        nicegui.app.shutdown()
        return

    logger.info("{} is opened", webcam_dev)

    while True:
        ret, frame = await asyncio.to_thread(video_capture.read)
        if not ret:
            continue
        model.image = base64.b64encode(frame).decode()

        events.append(model.image)
        event_received.set()


async def process():
    model.response = "┃ "

    while True:
        await event_received.wait()
        event_received.clear()

        image = events.pop()

        responses = vlm2(image, "")
        try:
            first_response = await anext(responses)
            model.vlm_image = image
            model.response = f"{first_response}"

            async for text in responses:
                model.response += text
        except StopAsyncIteration:
            await asyncio.sleep(1)
            continue
        except Exception as e:
            logger.error(
                "An error occurred while receiving the response from VLM. {}",
                str(e),
            )
            await asyncio.sleep(1)
            continue


async def vlm2(image: str, description: str):
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
        async for chunk in llm.astream(
            # f"{json.dumps(metadata['data']['metadata']['events'])}\n in 30 words, quickly"
            messages
        ):
            yield chunk.text()
    except httpx.HTTPError as e:
        message = "Unable to talk to the Ollama server"
        notify(message, type="warning")
        logger.error("{}: {} {}", message, e.request, e)


nicegui.app.on_startup(process())
nicegui.app.on_startup(webcam())


@ui.page("/")
def index():
    # Custom CSS to overlay text on video
    ui.add_css("""
    .video-container {
        position: relative;
    }

    .overlay-text {
      position: absolute;
      top: 10px;
      left: 10px;
      right: 10px;
      color: white;
      font-size: 2.2em;
      background-color: rgba(2, 1, 2, 0.8);
      padding: 5px 15px;
      border-radius: 8px;
    }
    """)

    with html.div().classes("video-container w-[75%]"):
        ui.interactive_image().classes(
            "w-full h-full video-container"
        ).bind_source_from(model, "image", lambda s: "data:image/jpg;base64," + s)

        ui.label("hello").bind_text_from(model, "response").classes("overlay-text")

    ui.image("image").bind_source_from(
        model, "vlm_image", lambda s: "data:image/jpg;base64," + s
    ).classes("w-[20%]")


# Run the NiceGUI app
ui.run(host="0.0.0.0", port=4000, show=False)
