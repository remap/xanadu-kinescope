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


## Initialize Firebase
##
def initialize_firebase(config):
    with open(config['firebase_config_path']) as f:
        firebase_config = json.load(f)
    cred = credentials.Certificate(config['firebase_credential_path'])
    firebase_admin.initialize_app(cred, {
        'databaseURL': firebase_config['databaseURL']
    })


# ---------

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
            logger.info(f"Downladed {url} to {image_path}")

        except Exception as e:
            logger.error(f"Error downloading image: {e}")

    # TODO: How do we know it is complete?
    #
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

def writeHTML(choices):
    def writeCleanHTML(intext, outfile):
        with open(outfile, "w", encoding="utf-8") as html_file:
            soup = BeautifulSoup(intext, "html.parser")
            html_content = soup.find("html").prettify()  # encode_contents().decode("utf-8")
            html_file.write(html_content)
        return html_content

    # Write a file per choice
    #
    k = 0
    for choice in choices:
        filename = f"choreo_table{k}.html"
        logger.info(f"Writing choice {k} as {filename} ")
        writeCleanHTML(choice.message.content, filename)
        k += 1

def process_pipeline(image_collection, client, config):

    logger.info(f"process_pipeline called with collection {image_collection}")

    ## Input: IMAGES
    ## Ouput: HTML WITH CHOREO
    steps = [
        {
            "description": "Generate an ancient Greek poem. Images are added to this step later.",
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
            "f_side_effects": [lambda completion_choices: writeHTML(completion_choices)],
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
def listener(event, status, config, client):
    if config["suppress_first_run"] and status["runs"]==0:
        logger.info("Suppressing first run")
        return
    else:
        logger.info("Running on current data value, set 'suppress_first_run' to True in config to skip.")
        status["runs"] +=1
        logger.info(f"Listener, run {status['runs']} {event.data}")

    if not isinstance(event.data, str):
        logger.info(f"Listener data not a string, returning.  Received: {event.data}")
        return
    try:
        files = json.loads(event.data)
        if not isinstance(files, list):
            logger.info(f"Listener data is JSON but {type(files)} and not a list, returning.  Received: {files}")
            return
    except:
        files = [event.data]

    # Get the images
    image_collection = download_images(files, config['folder_path'], config['thread_pool_size'])

    # Run the chain
    result = process_pipeline(image_collection, client, config)
    logger.info("Completed process_pipeline, waiting for new urls...")


def signal_handler(sig, frame, stop_event, listener_thread):
    logger.info("Terminating listener...")
    stop_event.set()
    listener_thread.join()
    sys.exit(0)

def main():
    config = {
        "openai_config_path": "xanadu-secret-openai.json",
        "folder_path": "tmp_images",
        "temperature": 0.7,
        "model": "gpt-4o",
        "fb_key": "/xanadu/oracle/files",
        "thread_pool_size": 5,
        "firebase_config_path": "xanadu-secret-firebase-forwarder.json",
        "firebase_credential_path": "xanadu-secret-f5762-firebase-adminsdk-9oc2p-1fb50744fa.json",
        "suppress_first_run": False  # Skips a run based on the first keys encountered on startup
    }
    status = {
        "runs": 0
    }

    # Init firebase
    initialize_firebase(config)
    ref = db.reference(config['fb_key'])

    # Init OpenAI
    with open(config['openai_config_path']) as f:
        openai_config = json.load(f)
    config["openai_api_key"] = openai_config["apiKey"]
    client = OpenAI(api_key=config["openai_api_key"])

    # Listen for new URLs
    #
    stop_event = Event()
    listener_thread = Thread(target=ref.listen, args=(lambda event: listener(event, status, config, client),))
    listener_thread.start()
    signal.signal(signal.SIGINT, lambda sig, frame: signal_handler(sig, frame, stop_event, listener_thread))
    while not stop_event.is_set():
        stop_event.wait(1)

if __name__ == "__main__":
    main()


# def apitest(key):
#     client = OpenAI(api_key=key)
#     image_path = "test_image.jpg"
#     base64_image = base64_encode_image(image_path)
#     prompt_content = [
#         {"type": "text", "text": "Write a haiku inspired by this image."},
#         {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}, }
#     ]
#     completion = client.chat.completions.create(model="gpt-4o",
#                                                 messages=[
#                                                     {"role": "system", "content": "You are a creative assistant."},
#                                                     {"role": "user", "content": prompt_content},
#                                                 ])
#     print(completion.choices[0].message.content)