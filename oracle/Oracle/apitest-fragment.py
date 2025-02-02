
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