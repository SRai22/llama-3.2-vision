import requests
import torch
from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
import gradio as gr
from threading import Thread
from transformers import TextIteratorStreamer
import time  # Import time module to measure inference time
from pathlib import Path

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Model ID or path for LLaMA
model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

# Check if CUDA is available
if torch.cuda.is_available():
    print(f"CUDA is available! Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Please install the correct version of CUDA.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = MllamaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_id)

# Define the function to be used in Gradio for streaming generation
    

# Gradio interface
def make_demo(model, processor):
    model_name = model.config._name_or_path

    example_image_urls = [
        ("https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/dd5105d6-6a64-4935-8a34-3058a82c8d5d", "small.png"),
        ("https://github.com/openvinotoolkit/openvino_notebooks/assets/29454499/1221e2a8-a6da-413a-9af6-f04d56af3754", "chart.png"),
    ]

    for url, file_name in example_image_urls:
        if not Path(file_name).exists():
            Image.open(requests.get(url, stream=True).raw).save(file_name)

    def bot_streaming(message, history):
        print(f"message is - {message}")
        print(f"history is - {history}")

        print("here 1")
        files = message["files"] if isinstance(message, dict) else message.files
        print("here 2")
        message_text = message["text"] if isinstance(message, dict) else message.text
        print("here 3")
        if files:
            # message["files"][-1] is a Dict or just a string
            if isinstance(files[-1], dict):
                print("here 4")
                image = files[-1]["path"]
            else:
                print("here 5")
                image = files[-1] if isinstance(files[-1], (list, tuple)) else files[-1].path
        else:
            # if there's no image uploaded for this turn, look for images in the past turns
            # kept inside tuples, take the last one
            print("here 6")
            for hist in history:
                if type(hist[0]) == tuple:
                    image = hist[0][0]

        print("here 7")
        try:
            if image is None:
                # Handle the case where image is None
                raise gr.Error("You need to upload an image for Llama-3.2-Vision to work. Close the error and try again with an Image.")
        except NameError:
            # Handle the case where 'image' is not defined at all
            raise gr.Error("You need to upload an image for Llama-3.2-Vision to work. Close the error and try again with an Image.")

        print("here 8")
        conversation = []
        flag = False
        for user, assistant in history:
            if assistant is None:
                flag = True
                conversation.extend([{"role": "user", "content": []}])
                continue
            if flag == True:
                conversation[0]["content"] = [{"type": "text", "text": f"<|image|>\n{user}"}]
                conversation.extend([{"role": "assistant", "content": [{"type": "text", "text": assistant}]}])
                flag = False
                continue
            conversation.extend(
                [{"role": "user", "content": [{"type": "text", "text": user}]}, {"role": "assistant", "content": [{"type": "text", "text": assistant}]}]
            )

        if len(history) == 0:
            conversation.append({"role": "user", "content": [{"type": "text", "text": f"<|image|>\n{message_text}"}]})
        else:
            conversation.append({"role": "user", "content": [{"type": "text", "text": message_text}]})
        print(f"prompt is -\n{conversation}")

        prompt = processor.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        print(f"prompt is -\n{prompt}")

        # Load and process the image
        image = Image.open(image)
        inputs = processor(image, prompt, return_tensors="pt").to(device)

        # Measure inference time
        start_time = time.time()  # Start measuring time

        streamer = TextIteratorStreamer(
            processor,
            **{
                "skip_special_tokens": True,
                "skip_prompt": True,
                "clean_up_tokenization_spaces": False,
            },
        )

        generation_kwargs = dict(
            inputs,
            streamer=streamer,
            max_new_tokens=1024,
            do_sample=False,
            # temperature=0.0,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        buffer = ""
        for new_text in streamer:
            buffer += new_text
            yield buffer

        # Measure the time after the generation
        end_time = time.time()  # End time
        inference_time = end_time - start_time  # Calculate inference time
        print(f"Inference took {inference_time:.2f} seconds")


    demo = gr.ChatInterface(
        fn=bot_streaming,
        title=f"{model_name}",
        examples=[
            {"text": "Describe the image", "files": ["./small.png"]},
            {"text": "What does this image show?", "files": ["./chart.png"]},
        ],
        description=f"Upload an image and start chatting with Llama-3.2-Vision.",
        stop_btn="Stop Generation",
        multimodal=True,
    )

    return demo

# Run the demo
processor.chat_template = processor.tokenizer.chat_template
demo = make_demo(model, processor)

try:
    demo.launch(debug=True)
except Exception:
    demo.launch(debug=True, share=True)
