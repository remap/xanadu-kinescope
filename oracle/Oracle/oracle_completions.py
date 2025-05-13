

## Input: IMAGES
## Output: HTML WITH CHOREO

make_oracle_pipeline = lambda depends: [
    {
        "description": "Generate an ancient Greek poem.",
        "prompt": [
            {"type": "text",
             "text": """
                The attached image provides 14 thumbnails related to the muses of ancient greece who have
                appeared in the present day (2025). The top row of 7 thumbnails shows the seven muses as 
                drawn by their modern acolytes. 
                The bottom row of 7 thumbnails shows seven object offerings to the muses by those same
                modern-day supplicants.     
                Write a three-verse poem in ancient greek inspired by the specific content and form of the images.  
                Be as specific to the image contents as possible, without discussing that they are images.
                The goal here is not to talk about the muses, but about the offering of their audience / acolytes. 
                Nouns should refer to what is seen / implied by the images and adjectives should relate clearly to the aesthetics. 
                Focus the poem about the drawings and offerings that have been made, with the muses themselves applied. 
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
                Maintain the three verse section, and make the lines in the first and third line of each verse rhyme.
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
                I have a bank of 12 dance moves: 
                1) peace eyes with 60's arms: do peace signs over the eyes and then lift each arm up above head with a fist.
                2) Point to Self: Use thumbs to point as self, heels shift left and right. 
                3) Step Touch in place: Snap fingers, feet take turns step touching with a bit of a shoulder groove.
                4) Shoulder Groove: Step side to side as shoulders go up and down. 
                5) Sprinkler Point w/ Stomp: Point arm out for four counts with a heel stomp then switch sides.
                6) Sunshine Arms: Open arms up above head with a little hip shake. 
                7) Speaker to Mouth: Put hands near mouth and turn body to diagonals. 
                8) Hand to Ear w/ Neighbor: Put hand to ear and lean body towards left and right.
                9) Wipe with Step Touch: Swipe arms back and forth in front of torso and step side to side.
                10) Firework Arms: Burst fingers up left, up right, down left, down right with a little groove. 
                11) Punch Ups: Pump both arms up in the air and switch hips. 
                12) Rodeo Arms: Swirl arms above hips standing, Swirl arms with a plie. 
                Using moves from the above dance move bank, write a five-minute, three section jazzercise choreography for a 
                broad audience in english inspired by this poem, no more than one move per section. Do not elaborate or explain, 
                provide the choreography in plain text only. Cite the lines of the poem that inspire the moves starting with 'Inspiration: ...' 
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
                Convert the following five-minute, three section jazzercise choreography description into a 
                complete, well-formatted HTML5 document, black background, white sans-serif text, containing a table 
                (with gridlines) with section, moves described as bullets, and inspiration, for a human leader to on 
                the fly to refer to in teaching a new class. Each section should have no more than one move that is 
                clearly and concisely describe.  Do not include a title.   Format the table to use the entire document 
                width. The only CSS should be to set san-serif and yellow font color to identify specific body movements,
                as well ensuring line spacing that is easily readable at a distance.  Use 24px font and screen-width tables
                Do not use list formatting in the inspiration column, and put very short catchy section titles in the section column based off the line of inspiration from the poem.
                The inspirational poem is included for reference and should be provided verbatim in both English and Greek after 
                the table, with no title, greek lines (color #999) interleaved with the english translation (color #eee), preserving original line breaks using the <br/> tag.   
                """},
            {"type": "text", "text": "Choreography: {{llm_out[n-1][0].message.content}}"},
            {"type": "text", "text": "Inspirational poem: {{llm_out[n-2][0].message.content}}"},
            {"type": "text", "text": "Original greek: {{llm_out[n-3][0].message.content}}"}

        ],
        "role": "You are an HTML5 formatter.  You only respond in syntatically correct HTML5.",
        "f_explain": lambda completion_choices: f"Final html: \n{completion_choices[0].message.content[:150]}...\n",
        "f_side_effects": [lambda completion_choices: depends.writeHTML(completion_choices, depends.config)],
    }
]