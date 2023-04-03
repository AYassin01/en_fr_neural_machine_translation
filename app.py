import logging
import os

from flask import Flask, render_template, request

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
    max_input_len=MAX_INPUT_LEN,
)

logging.basicConfig(level=logging.INFO)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/", methods=["POST", "GET"])
def translate():
    input_text = request.form["input_text"]
    output_text = model.translate(input_text)
    return render_template("index.html", output=output_text)


def main():
    """Run the Flask app."""
    app.run(host="0.0.0.0", port=8000, debug=True)


if __name__ == "__main__":
    main()
