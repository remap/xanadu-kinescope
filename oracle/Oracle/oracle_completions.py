

## Input: IMAGES
## Output: HTML WITH CHOREO

make_oracle_pipeline = lambda depends: [
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
                I have a bank of 12 dance moves: 
                1) peace eyes with 60's arms
                2) Point to Self
                3) Step Touch in place
                4) Shoulder Groove
                5) Sprinkler Point w/ Stomp
                6) Sunshine Arms
                7) Speaker to Mouth
                8) Hand to Ear w/ Neighbor
                9) Wipe with Step Touch
                10) Firework Arms
                11) Punch Ups
                12) Rodeo Arms.
                Using moves from the above dance move bank, write a five-minute, three to four section jazzercise choreography for a 
                broad audience in english inspired by this poem, no more than two moves per section. Do not elaborate or explain, 
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
        "f_side_effects": [lambda completion_choices: depends.writeHTML(completion_choices, depends.config)],
    }
]