import os, sys, signal, argparse, random, base64, json, ipaddress
import logging, logging.config
from datetime import datetime
import time
import requests, mimetypes
import asyncio
from contextlib import asynccontextmanager
import firebase_admin
from firebase_admin import credentials, db
from openai import OpenAI
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from jinja2 import Template
from types import SimpleNamespace
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, Response, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
import uvicorn

## ChatGPT logic / processing chain
from oracle_completions import make_oracle_pipeline

main_loop = None  # async io

## Set up logger borrowed from Hermes

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

logger = logging.getLogger(__name__)
path = Path("logconfig.json")
with path.open("r", encoding="utf-8") as f:
    logconfig = json.load(f)
    logger.info(f"{path} loaded")
logging.config.dictConfig(logconfig)
logger.setLevel(logging.DEBUG)

# Load configuration files

config = {}
config_path = Path("oracle-config.json")
if config_path.exists():
    with config_path.open("r") as f:
        config.update(json.load(f))
    logger.info(f"{config_path} loaded")
else:
    logger.error(f"can't find {config_path}")
    sys.exit(0)

auth_config_path = Path(config["basic_auth_config_path"])
if auth_config_path.exists():
    with auth_config_path.open("r") as f:
        config.update(json.load(f))
    logger.info(f"{auth_config_path} loaded")
else:
    logger.error(f"can't find {config['basic_auth_config_path']}")



# FastAPI
#

# Is there another place we can put this
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

# Auth via known IPs or basic_auth configured in oracle-config.json
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

# Instantiate the app
dynamic_html_cache: str = config["html_oracle_waiting"]
dynamic_html_clients: set[WebSocket] = set()
connected_terminal_websockets: set[WebSocket] = set()
oracle_status = {
    "runs": -1,
    "running": False
}
app = FastAPI(lifespan=lifespan)
app.add_middleware(BasicAuthMiddleware)

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

# FastAPI Path handlers

# Terminal websocket
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

# Dynamic HTML (output) websocket
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

# Handle URL submissions
@app.post("/urlinput")
async def submit_input(request: Request):
    client_host, client_port = request.client
    data = await request.json()
    user_input = data.get("input", "")
    logger.info(f"Received input from client: {user_input}")
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(
        None,
        proc_urls,
        SimpleNamespace(data=user_input),
        f"{client_host}:{client_port}"
    )
    return JSONResponse({"status": "ok"})

# Serve the static index page
@app.get("/")
async def get_index(request: Request):
    # authenticated via the middleware
    return FileResponse(os.path.join("html", "index.html"))

# Mount other static files
app.mount("/static", StaticFiles(directory="html", html=True), name="static")

# Broadcast HTML updates to connected clients
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

# Task to send a heatbeat to the websockets
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

# Firebase listener
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

# Base64 encoding for uploaded images to send to OpenAI API
def base64_encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Jinja2 templating for text prompts that are in oracle_completions.py
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
    # write a file per choice
    # TODO: broadcast more than one?
    k = 0
    for choice in choices:
        filename = f"choreo_table{k}.html"
        path = (Path(config["html_path"]) / Path(filename)).resolve()
        logger.info(f"Writing choice {k} as {filename}.  {path.as_uri()} ")
        writeCleanHTML(choice.message.content, path)
        k += 1

## Run our LLM CoT prompting
## Shouldn't need to edit here - edit oracle_completions.py
## Take images in, produce HTML
def process_pipeline(image_collection, client, config):
    logger.info(f"process_pipeline called with collection {image_collection}")
    t0 = time.time()
    depends = SimpleNamespace(writeHTML = writeHTML, config = config)  # Pass in things we need
    oracle_pipeline = make_oracle_pipeline(depends)

    # Add images to step [0]
    base64_image = []
    for image_path in image_collection:
        base64_image = base64_encode_image(image_path)
        oracle_pipeline[0]["prompt"].append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}})

    llm_out = []
    for n in range(len(oracle_pipeline)):
        prompt_content = render_nested_vars(oracle_pipeline[n]["prompt"], base64_image=base64_image, n=n, llm_out=llm_out)
        logger.info(f"\n\tStep {n}: {oracle_pipeline[n]['description']}\n\tPrompt: {str(prompt_content)[:350]} ...")
        try:
            completion = client.chat.completions.create(model="gpt-4o",
                                                        messages=[
                                                            {"role": "system",
                                                             "content": oracle_pipeline[n]["role"]},
                                                            {"role": "user", "content": prompt_content},
                                                        ])
            llm_out.append(completion.choices)

            logger.debug(f"\n{oracle_pipeline[n]['f_explain'](llm_out[n])}")
            for F in oracle_pipeline[n]["f_side_effects"]: F(llm_out[n])
        except Exception as e:
            logger.error(f"Exception in process_pipeline LLM calls", exc_info=True)
    t1 = time.time()
    t = t1-t0
    logger.info (f"process_pipeline execution time {t:0.3f}s.")
    try:
        return llm_out[-1][0].message.content  # just choice 0
    except e:
        logger.error("Exception in returning result", exc_info=True)
        return None


# Triggered by firebase or clients
# Take image URLs in
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

        # Get the images
        image_collection = download_images(files, config['folder_path'], config['thread_pool_size'])
        if len(image_collection) < 1:
            logger.error(f"No images obtained, aborting.")
            raise

        # Run the LLM
        result = process_pipeline(image_collection, client, config)
        logger.info(f"proc_urls {source}: Completed process_pipeline")

        # Send out the HTML (was already saved in process_pipeline)
        result = cleanHTML(result)  # Strip the ```html
        if broadcast:
            main_loop.call_soon_threadsafe(
                lambda: asyncio.create_task(broadcast_html(result))
            )
    except Exception as e:
        logger.error(f"Exception in Oracle processing (proc_urls)", exc_info=True)
        if broadcast:
            main_loop.call_soon_threadsafe(
                lambda: asyncio.create_task(broadcast_html(config["html_oracle_waiting"]))
            )
    finally:
        oracle_status["running"] = False

    return result

if __name__ == "__main__":
    uvicorn.run(app, host=config["web_host"], port=config["web_port"], log_config=logconfig)
