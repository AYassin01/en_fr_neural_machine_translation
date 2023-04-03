
# import json
import logging
import os

import gradio as gr
# import numpy as np
# import tensorflow as tf
from flask import Flask, request, jsonify
from model.model import EnFrTranslator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



app = Flask(__name__)

# define model path and max input length
MODEL_PATH = "./model/en_fr_nmt_model.h5"
MAX_INPUT_LEN = 15

model = EnFrTranslator(
    model_path=MODEL_PATH,
    en_vocab_path="./model/vocab/en_vocab.json",
    fr_vocab_path="./model/vocab/fr_vocab.json",
    max_input_len=MAX_INPUT_LEN
)

logging.basicConfig(level=logging.INFO)



@app.route("/", methods=["GET"])
def translate_page():

    # Define the input and output fields for the interface
    input_text = gr.components.Textbox(label="Input Text")
    output_text = gr.components.Textbox(label="Output Text")

    # Create the interface with the translation function and the input/output fields
    return gr.Interface(
        model.translate,
        inputs=input_text,
        outputs=output_text,
        title="English to French Translator",
        description="Translate English text to French using a \
        neural machine translation model."
    ).launch(server_port=8000)





def main():
    """Run the Flask app."""
    app.run(host="0.0.0.0", port=8000, debug=False)


if __name__ == "__main__":
    main()


