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


config = {
    "openai_config_path": "xanadu-secret-openai.json",
    "folder_path": "tmp_images",
    "html_path": ".",
    "temperature": 0.7,
    "model": "gpt-4o",
    "fb_key": "/xanadu/oracle/files",
    "thread_pool_size": 5,
    "firebase_config_path": "xanadu-secret-firebase-forwarder.json",
    "firebase_credential_path": "xanadu-secret-f5762-firebase-adminsdk-9oc2p-1fb50744fa.json",
    "suppress_first_fb_run": True  # Skips a run based on the first keys encountered on startup
}
status = {
    "runs": -1
}

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
app = FastAPI(lifespan=lifespan)


# A global set to keep track of connected terminal WebSocket clients.
connected_terminal_websockets: set[WebSocket] = set()

# Global variable to hold the main event loop.
main_loop = None

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

# Add websocket logger
#logger = logging.getLogger("live_logger")
#logger.setLevel(logging.DEBUG)
# ws_handler = WebSocketLogHandler()
# ws_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
# logger.addHandler(ws_handler)

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

# --------- FAST API SERVER


@app.websocket("/log")
async def log_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    connected_terminal_websockets.add(websocket)
    client_host, client_port = websocket.client
    logger.info(f"New connection from {client_host}:{client_port}")
    try:
        while True:
            await websocket.receive_text()  # Keep the connection open.
    except WebSocketDisconnect:
        connected_terminal_websockets.discard(websocket)


# Global cache for the latest dynamic HTML fragment.
dynamic_html_cache: str = "<html><body></body></html>"

# A global set to keep track of connected dynamic HTML WebSocket clients.
dynamic_html_clients: set[WebSocket] = set()

@app.websocket("/dynamic")
async def dynamic_html_endpoint(websocket: WebSocket):
    await websocket.accept()
    dynamic_html_clients.add(websocket)
    try:
        await websocket.send_text(dynamic_html_cache)
        while True:
            await asyncio.sleep(5)  # keep connection active.
    except WebSocketDisconnect:
        dynamic_html_clients.discard(websocket)


# Serve the main HTML file (client code) from the root URL.
@app.get("/")
async def get_index():
    return FileResponse(os.path.join("html", "index.html"))

# Mount static assets under "/static" (if any).
app.mount("/static", StaticFiles(directory="html", html=True), name="static")

async def generate_test_logs() -> None:
    count = 0
    while True:
        if connected_terminal_websockets:  # Only log if at least one terminal client is connected.
            logger.info(f"Log message #{count}")
            count += 1
        await asyncio.sleep(0.25)


async def heartbeat() -> None:
    while True:
        # Send a "ping" to terminal clients.
        for ws in list(connected_terminal_websockets):
            try:
                await ws.send_text("<<ping>>")
            except Exception:
                connected_terminal_websockets.discard(ws)
        # Also ping dynamic clients if desired.
        for ws in list(dynamic_html_clients):
            try:
                await ws.send_text("<<ping>>")
            except Exception:
                dynamic_html_clients.discard(ws)
        await asyncio.sleep(30)

def gen_html() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    random_number = random.randint(1, 100)
    return f"""
    <!DOCTYPE html>
    <html>
      <head>
        <meta charset="UTF-8">
        <title>Dynamic HTML Update</title>
      </head>
      <body style="background-color:#000; color:#fff; margin:0; padding:1em;">
        <h1>Dynamic HTML Content</h1>
        <p>Updated at {now}</p>
        <p>Random number: {random_number}</p>
      </body>
    </html>
    """


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
                    image_path = os.path.join(folder_path, f"image_{index}.jpg")
                    with open(image_path, "wb") as file:
                        file.write(response.content)
                else:
                    logger.warning(f"URL does not point to an image: {url}")
            logger.info(f"Downloaded {url} to {image_path}")

        except Exception as e:
            logger.error(f"Error downloading image: {e}")

    with ThreadPoolExecutor(max_threads) as executor:
        for i, url in enumerate(image_urls):
            executor.submit(download_image, url, i)

    files = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            files.append(file_path)
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
        steps[0]["prompt"].append( {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"} } )

    ## ------------------
    ## Run our LLM CoT prompting
    ## Shouldn't need to edit here
    ## 
    llm_out = []
    for n in range(len(steps)):
        prompt_content = render_nested_vars(steps[n]["prompt"], base64_image=base64_image, n=n, llm_out=llm_out)
        logger.info(f"\n\tStep {n}: {steps[n]['description']}\n\tPrompt: {str(prompt_content)[:350]} ...")
        completion = client.chat.completions.create(model="gpt-4o",
                                                    messages=[
                                                        {"role": "system",
                                                         "content": steps[n]["role"]},
                                                        {"role": "user", "content": prompt_content},
                                                    ])
        llm_out.append(completion.choices)
        logger.debug(f"\n{steps[n]['f_explain'](llm_out[n])}")
        for F in steps[n]["f_side_effects"]: F(llm_out[n])

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
    if status["runs"] < 0: status["runs"] = 0

    ## Handle string and list inputs
    #
    if not testURL is None:
        logger.warning(f"Invoked with test URL: {testURL}.")
        event = SimpleNamespace(data=testURL)
        status["runs"] = 0

    if not isinstance(event.data, str):
        logger.info(f"proc_urls {source}: Data not a string, returning.  Received: {event.data}")
        return
    try:
        if len(event.data.strip()) == 0:
            logger.info(f"proc_urls {source}: Data is empty.")
            return
        files = json.loads(event.data)
        if not isinstance(files, list):
            logger.info(f"proc_urls {source}: Data is JSON but {type(files)} and not a list, returning.  Received: {files}")
            return
    except:
        files = [event.data]

    ## Run
    #
    status["runs"] +=1
    logger.info(f"proc_urls {source}: Run {status['runs']} {event.data}")

    image_collection = download_images(files, config['folder_path'], config['thread_pool_size'])

    result = process_pipeline(image_collection, client, config)
    logger.info(f"proc_urls {source}: Completed process_pipeline")


    result = cleanHTML(result)    # Strip the ```html
    if broadcast:
        main_loop.call_soon_threadsafe(
            lambda: asyncio.create_task(broadcast_html(result))
        )

    return result


async def firebase_listener_task():
    initialize_firebase(config)
    ref = db.reference(config['fb_key'])
    if ref is None:
        logging.error("Could not create firebase client.")
        sys.exit(0)

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def fb_listener(event):
        if config["suppress_first_fb_run"] and status["runs"]==-1:
            #logger.info("Suppressing first run")
            status["runs"]=0
            return
        return proc_urls(event, config["fb_key"])

    listener_future = loop.run_in_executor(None, lambda : ref.listen( fb_listener ))
    while True:
        await asyncio.sleep(5) #TODO

    def on_signal():
        logging.info("SIGINT received, stopping firebase listener task.")
        loop.call_soon_threadsafe(stop_event.set)
    loop.add_signal_handler(signal.SIGINT, on_signal)

    await stop_event.wait()
    logging.info("Firebase listener task stopped.")

async def broadcast_html(new_html) -> None:
    global dynamic_html_cache
    dynamic_html_cache = new_html
    logger.debug("Broadcasting")
    disconnected = set()
    for client in dynamic_html_clients:
        try:
            await client.send_text(dynamic_html_cache)
        except Exception:
            disconnected.add(client)
    # Remove any clients that failed.
    for client in disconnected:
        dynamic_html_clients.discard(client)


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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

