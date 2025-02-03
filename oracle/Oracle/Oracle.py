import os
import logging, logging.config
import requests
import json
import firebase_admin
from firebase_admin import credentials, db
from openai import OpenAI
from threading import Thread, Event
import signal
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import base64
from bs4 import BeautifulSoup
from jinja2 import Template
import argparse
from types import SimpleNamespace
import asyncio
import logging
import os
import random
from datetime import datetime
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
import asyncio
from contextlib import asynccontextmanager
import mimetypes

config = {
    "web_host": "0.0.0.0",
    "web_port": 8000,
    "openai_config_path": "xanadu-secret-openai.json",
    "folder_path": "tmp_images",
    "html_path": ".",
    "temperature": 0.7,
    "model": "gpt-4o",
    "fb_key": "/xanadu/oracle/files",
    "thread_pool_size": 5,
    "firebase_config_path": "xanadu-secret-firebase-forwarder.json",
    "firebase_credential_path": "xanadu-secret-f5762-firebase-adminsdk-9oc2p-1fb50744fa.json",
    "basic_auth_config_path": "xanadu-secret-oracle-basic-auth.json",
    "html_oracle_waiting":  '<html><body><h1 style="color:#fff">Waiting for the Oracle of Chrysopoeia.<h1></body></html>',
    "html_oracle_busy": '<html><body><h1 style="color:#fff">The Oracle of Chrysopoeia is working for someone.<br/> Try your request in a bit. <h1></body></html>',
    "suppress_first_fb_run": True  # Skips a run based on the first keys encountered on startup
}

auth_config_path = Path(config["basic_auth_config_path"])
if auth_config_path.exists():
    with auth_config_path.open("r") as f:
        config.update(json.load(f))
else:
    logger.error(f"can't find {config['basic_auth_config_path']}")

oracle_status = {
    "runs": -1,
    "running": False
}

dynamic_html_cache: str = config["html_oracle_waiting"]
dynamic_html_clients: set[WebSocket] = set()
connected_terminal_websockets: set[WebSocket] = set()
main_loop = None  # async io


## Is there another place we can put this
#  main()
#
@asynccontextmanager
async def lifespan(app: FastAPI):
    title = "Choreography Oracle (Kinescope)"
    parser = argparse.ArgumentParser(description=title)
    args = parser.parse_args()
    logger.info(title)
    global main_loop
    main_loop = asyncio.get_running_loop()
    firebase_task = asyncio.create_task(firebase_listener_task())
    heartbeat_task = asyncio.create_task(heartbeat())
    try:
        yield
    finally:
        firebase_task.cancel()
        heartbeat_task.cancel()


######
import base64
import ipaddress
from fastapi import FastAPI, Request, Response, WebSocket, status
from starlette.middleware.base import BaseHTTPMiddleware

USERNAME = "admin"
PASSWORD = "secret"
ALLOWED_IPS = ["131.179.141.0/24"]
ALLOWED_NETWORKS = [ipaddress.ip_network(ip, strict=False) for ip in ALLOWED_IPS]


def check_auth(client_ip: str | None, auth: str | None) -> bool:
    if client_ip:
        try:
            ip_obj = ipaddress.ip_address(client_ip)
            for net in (ipaddress.ip_network(ip, strict=False) for ip in config["auth_allowed_ips"]):
                if ip_obj in net:
                    return True
        except Exception:
            pass
    if not auth or not auth.startswith("Basic "):
        return False
    try:
        decoded = base64.b64decode(auth.split(" ", 1)[1]).decode("utf-8")
        user, pwd = decoded.split(":", 1)
    except Exception:
        return False
    return user == config["auth_username"] and pwd == config["auth_password"]


class BasicAuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if not check_auth(request.client.host if request.client else None, request.headers.get("Authorization")):
            return Response("Unauthorized", status_code=status.HTTP_401_UNAUTHORIZED,
                            headers={"WWW-Authenticate": "Basic"})
        return await call_next(request)

    ######


app = FastAPI(lifespan=lifespan)
app.add_middleware(BasicAuthMiddleware)


## -- Logging

class WebSocketLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        message = self.format(record)
        global main_loop
        if main_loop is None:
            return
        # Schedule sending on the main event loop in a thread-safe way.
        for ws in list(connected_terminal_websockets):
            main_loop.call_soon_threadsafe(
                lambda ws=ws, message=message: asyncio.create_task(self.send_message(ws, message))
            )

    async def send_message(self, ws: WebSocket, message: str) -> None:
        try:
            await ws.send_text(message)
        except Exception:
            connected_terminal_websockets.discard(ws)


## Set up logger borrowed from Hermes
##
class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.INFO: "\033[97m",
        logging.WARNING: "\033[93m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[1;31m",
        logging.DEBUG: "\033[37m",
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        message = super().format(record)
        return f"{color}{message}{self.RESET}"


path = Path("logconfig.json")
with path.open("r", encoding="utf-8") as f:
    logconfig = json.load(f)
logging.config.dictConfig(logconfig)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Init OpenAI
with open(config['openai_config_path']) as f:
    openai_config = json.load(f)
config["openai_api_key"] = openai_config["apiKey"]
client = OpenAI(api_key=config["openai_api_key"])
if client is None:
    logging.error("Could not create OpenAI client.")
    sys.exit(0)


## Initialize Firebase
##
def initialize_firebase(config):
    with open(config['firebase_config_path']) as f:
        firebase_config = json.load(f)
    cred = credentials.Certificate(config['firebase_credential_path'])
    firebase_admin.initialize_app(cred, {
        'databaseURL': firebase_config['databaseURL']
    })


## FastAPI


@app.websocket("/log")
async def log_endpoint(websocket: WebSocket) -> None:
    if not check_auth(websocket.client.host if websocket.client else None, websocket.headers.get("Authorization")):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    connected_terminal_websockets.add(websocket)
    client_host, client_port = websocket.client
    logger.info(f"New connection from {client_host}:{client_port}")
    try:
        while True:
            await websocket.receive_text()  # Keep the connection open.
    except WebSocketDisconnect:
        connected_terminal_websockets.discard(websocket)


@app.websocket("/dynamic")
async def dynamic_html_endpoint(websocket: WebSocket):
    if not check_auth(websocket.client.host if websocket.client else None, websocket.headers.get("Authorization")):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    dynamic_html_clients.add(websocket)
    try:
        await websocket.send_text(dynamic_html_cache)
        while True:
            await asyncio.sleep(5)  # keep connection active.
    except WebSocketDisconnect:
        dynamic_html_clients.discard(websocket)


@app.get("/")
async def get_index(request: Request):
    # authenticated via the middleware
    return FileResponse(os.path.join("html", "index.html"))


app.mount("/static", StaticFiles(directory="html", html=True), name="static")


async def heartbeat() -> None:
    while True:
        clients = connected_terminal_websockets | dynamic_html_clients
        for ws in list(clients):
            try:
                await ws.send_text("<<ping>>")
            except Exception:
                connected_terminal_websockets.discard(ws)
                dynamic_html_clients.discard(ws)
        await asyncio.sleep(30)


@app.post("/urlinput")
async def submit_input(request: Request):
    client_host, client_port = request.client
    data = await request.json()
    user_input = data.get("input", "")
    logger.info(f"Received input from client: {user_input}")
    loop = asyncio.get_running_loop()
    # Offload proc_urls to an executor to avoid blocking the event loop.
    await loop.run_in_executor(
        None,
        proc_urls,
        SimpleNamespace(data=user_input),
        f"{client_host}:{client_port}"
    )
    return JSONResponse({"status": "ok"})


async def broadcast_html(new_html) -> None:
    global dynamic_html_cache
    dynamic_html_cache = new_html
    #logger.debug("Broadcasting")
    disconnected = set()
    for client in dynamic_html_clients:
        try:
            await client.send_text(dynamic_html_cache)
        except Exception:
            disconnected.add(client)
    # Remove any clients that failed.
    for client in disconnected:
        dynamic_html_clients.discard(client)


## Firebase
##
async def firebase_listener_task():
    initialize_firebase(config)
    ref = db.reference(config['fb_key'])
    if ref is None:
        logging.error("Could not create firebase client.")
        sys.exit(0)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def fb_listener(event):
        if config["suppress_first_fb_run"] and oracle_status["runs"] == -1:
            oracle_status["runs"] = 0
            return
        return proc_urls(event, config["fb_key"])

    listener_future = loop.run_in_executor(None, lambda: ref.listen(fb_listener))
    while True:
        await asyncio.sleep(5)  # TODO


# --------- ORACLE MAIN LOGIC

## Downloader images from provided URLs
#
def download_images(image_urls, folder_path, max_threads):
    if os.path.exists(folder_path):
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    else:
        os.makedirs(folder_path)

    def download_image(url, index):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                content_type = response.headers.get("Content-Type", "")
                if content_type.startswith("image/"):
                    extension = mimetypes.guess_extension(content_type)
                    image_path = os.path.join(folder_path, f"image_{index}{extension}")
                    with open(image_path, "wb") as file:
                        file.write(response.content)
                        logger.info(f"Downloaded {url} to {image_path}")
                else:
                    logger.warning(f"URL does not point to an image: {url}")


        except Exception as e:
            logger.error(f"Error downloading image: {e}")

    with ThreadPoolExecutor(max_threads) as executor:
        for i, url in enumerate(image_urls):
            executor.submit(download_image, url, i)

    files = []
    try:
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            if os.path.isfile(file_path):
                files.append(file_path)
    except Exception as e:
        logger.error(f"download_images Exception in local file management: {e}")
    return files


# Base64 encoding for uploaded images
def base64_encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


# Jinja2 templating for text prompts
def render_nested_vars(data, **variables):
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str):  # could check key here
                data[key] = Template(value).render(**variables)
            else:
                render_nested_vars(value, **variables)
    elif isinstance(data, list):
        for item in data:
            render_nested_vars(item, **variables)
    return data


# Output writer for last set of completion choices
# Cup of Beautiful Soup placeholder for cleanup and munging

def cleanHTML(intext):
    soup = BeautifulSoup(intext, "html.parser")
    return soup.find("html").prettify()


def writeHTML(choices, config):
    def writeCleanHTML(intext, outfile):
        with open(outfile, "w", encoding="utf-8") as html_file:
            html_content = cleanHTML(intext)
            html_file.write(html_content)
        return html_content

        # Write a file per choice

    #
    k = 0
    for choice in choices:
        filename = f"choreo_table{k}.html"
        path = (Path(config["html_path"]) / Path(filename)).resolve()
        logger.info(f"Writing choice {k} as {filename}.  {path.as_uri()} ")
        writeCleanHTML(choice.message.content, path)
        k += 1

## ChatGPT logic / processing chain
##
def process_pipeline(image_collection, client, config):
    logger.info(f"process_pipeline called with collection {image_collection}")

    ## Input: IMAGES
    ## Output: HTML WITH CHOREO
    steps = [
        {
            "description": "Generate an ancient Greek poem.",
            "prompt": [
                {"type": "text",
                 "text": """
                    Write a three-verse poem in ancient greek inspired by the content and form of the images.  
                    Be as specific to the images contents as possible, without discussing that they are images. 
                    Provide only the poem without explanation, elaboration, or translation.
                    """},

            ],
            "role": "You are a creative translation tool that is a modern-day Oracle. You take images in and produce ancient greek poetry output.",
            "f_explain": lambda completion_choices: f"Image Analysis:\n{completion_choices[0].message.content}\n",
            "f_side_effects": [],
        },
        {
            "description": "Create an English translation for reference.",
            "prompt": [
                {"type": "text",
                 "text": """
                    Translate this poem to English without explanation, additional formatting, or elaboration. 
                    Maintain the three verse section
                    """},
                {"type": "text", "text": "{{llm_out[n-1][0].message.content}}"}
            ],
            "role": "You are a translator, you take beautiful greek prose in and produce english outputs, without commentary",
            "f_explain": lambda completion_choices: f"Translation:\n{completion_choices[0].message.content}\n",
            "f_side_effects": [],
        },
        {
            "description": "Conceive of a choreography based on both the Greek and English versions",
            "prompt": [
                {"type": "text",
                 "text": """
                    Write a five-minute, three to four section jazzercise choreography for a broad audience in english inspired by this poem, 
                    no more than two moves per section. Do not elaborate or explain, provide the choreography in plain 
                    text only. Cite the lines of the poem that inspire the moves starting with 'Inspiration: ...' 
                    """},
                {"type": "text", "text": "{{llm_out[n-2][0].message.content}}"},
                {"type": "text", "text": "{{llm_out[n-1][0].message.content}}"}
            ],
            "role": "You are a machine choreographer.",
            "f_explain": lambda completion_choices: f"Jazzercise:\n{completion_choices[0].message.content}\n",
            "f_side_effects": [],
        },
        {
            "description": "Format as HTML.",
            "prompt": [
                {"type": "text",
                 "text": """
                    Convert the following five-minute, three to four section jazzercise choreography description into a 
                    complete, well-formatted HTML5 document, black background, white sans-serif text, containing a table 
                    (with gridlines) with section, moves described as bullets, and inspiration, for a human leader to on 
                    the flyrefer to in teaching a new class. Each section should have no more than two moves that are 
                    clearly and concisely describe.  Do not include a title.   Format the table to use the entire document 
                    width. The only CSS should be to set san-serif and yellow font color to identify specific body movements,
                    as well ensuring line spacing that is easily readable at a distance.
                    Do not use list formatting in the inspiration column, and put very short catchy section titles in the section column.
                    The inspirational poem is included for reference and should be provided verbatim in both English and Greek after 
                    the table, with no title, greek lines (color #999) interleaved with the english translation (color #eee), preserving original line breaks using the <br/> tag.   
                    """},
                {"type": "text", "text": "Choreography: {{llm_out[n-1][0].message.content}}"},
                {"type": "text", "text": "Inspirational poem: {{llm_out[n-2][0].message.content}}"},
                {"type": "text", "text": "Original greek: {{llm_out[n-3][0].message.content}}"}

            ],
            "role": "You are an HTML5 formatter.  You only respond in syntatically correct HTML5.",
            "f_explain": lambda completion_choices: f"Final html: \n{completion_choices[0].message.content[:150]}...\n",
            "f_side_effects": [lambda completion_choices: writeHTML(completion_choices, config)],
        }
    ]

    # Add images to step [0]
    base64_image = []
    for image_path in image_collection:
        base64_image = base64_encode_image(image_path)
        steps[0]["prompt"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

    ## ------------------
    ## Run our LLM CoT prompting
    ## Shouldn't need to edit here
    ## 
    llm_out = []
    for n in range(len(steps)):
        prompt_content = render_nested_vars(steps[n]["prompt"], base64_image=base64_image, n=n, llm_out=llm_out)
        logger.info(f"\n\tStep {n}: {steps[n]['description']}\n\tPrompt: {str(prompt_content)[:350]} ...")
        try:
            completion = client.chat.completions.create(model="gpt-4o",
                                                        messages=[
                                                            {"role": "system",
                                                             "content": steps[n]["role"]},
                                                            {"role": "user", "content": prompt_content},
                                                        ])
            llm_out.append(completion.choices)

            logger.debug(f"\n{steps[n]['f_explain'](llm_out[n])}")
            for F in steps[n]["f_side_effects"]: F(llm_out[n])
        except Exception as e:
            logger.error(f"Exception in process_pipeline LLM calls - {e}")

    try:
        return llm_out[-1][0].message.content  # just choice 0
    except e:
        logger.error("Exception in returning result", e)
        return None


# Firebase event listener
# Expected incoming format is either a string with a single image or a json-formatted array
#
def proc_urls(event, source="", broadcast=True, testURL=None):
    # Reset firebase listener first run suppression if we've been triggered at least once
    if oracle_status["running"]:
        logger.warning("Oracle is running. Discarding request.")
        return

    # Handle string and list inputs

    if not testURL is None:
        logger.warning(f"Invoked with test URL: {testURL}.")
        event = SimpleNamespace(data=testURL)
        oracle_status["runs"] = 0

    if not isinstance(event.data, str):
        logger.info(f"proc_urls {source}: Data not a string, returning.  Received: {event.data}")
        return
    try:
        if len(event.data.strip()) == 0:
            logger.info(f"proc_urls {source}: Data is empty.")
            return
        files = json.loads(event.data)
        if not isinstance(files, list):
            logger.info(
                f"proc_urls {source}: Data is JSON but {type(files)} and not a list, returning.  Received: {files}")
            return
    except:
        files = [event.data]

    oracle_status["running"] = True
    if oracle_status["runs"] < 0: oracle_status["runs"] = 0
    oracle_status["runs"] += 1
    result = None
    try:
        busy_html = config["html_oracle_busy"]
        if broadcast:
            main_loop.call_soon_threadsafe(
                lambda: asyncio.create_task(broadcast_html(busy_html))
            )

        logger.info(f"proc_urls {source}: Run {oracle_status['runs']} {event.data}")

        image_collection = download_images(files, config['folder_path'], config['thread_pool_size'])
        if len(image_collection) < 1:
            logger.error(f"No images obtained, aborting.")
            raise

        result = process_pipeline(image_collection, client, config)

        logger.info(f"proc_urls {source}: Completed process_pipeline")

        result = cleanHTML(result)  # Strip the ```html
        if broadcast:
            main_loop.call_soon_threadsafe(
                lambda: asyncio.create_task(broadcast_html(result))
            )
    except Exception as e:
        logger.error(f"Exception in Oracle processing (proc_urls) - {e}")
        if broadcast:
            main_loop.call_soon_threadsafe(
                lambda: asyncio.create_task(broadcast_html(config["html_oracle_waiting"]))
            )
    finally:
        oracle_status["running"] = False

    return result

if __name__ == "__main__":
    uvicorn.run(app, host=config["web_host"], port=config["web_port"])
