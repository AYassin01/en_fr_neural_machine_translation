import gradio as gr
from model.model import EnFrTranslator

# Load the model
translator = EnFrTranslator(
    model_path="model/en_fr_nmt_model.h5",
    en_vocab_path="model/vocab/en_vocab.json",
    fr_vocab_path="model/vocab/fr_vocab.json",
    max_input_len=15
)


def interface():
    # Define the input and output fields for the interface
    input_text = gr.components.Textbox(label="Input Text")
    output_text = gr.components.Textbox(label="Output Text")

    # Create the interface with the translation function and the input/output fields
    gr.Interface(
        translator.translate,
        inputs=input_text,
        outputs=output_text,
        title="English to French Translator",
        description="Translate English text to French using a \
        neural machine translation model."
    ).launch(server_port=8000)

if __name__ == "__main__":
    interface()

