from tensorflow.keras.layers import TextVectorization as TV
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import pad_sequences as pds
import tensorflow as tf
import logging
import json
import numpy as np

class EnFrTranslator:
    def __init__(self, model_path, en_vocab_path, fr_vocab_path, max_input_len):
        with open(en_vocab_path, "r") as f:
            en_vocab = json.load(f)
        with open(fr_vocab_path, "r") as f:
            fr_vocab = json.load(f)
        
        logging.info("Successfully loaded vocab files...")
        
        self.text_to_id = TV(
            vocabulary=list(en_vocab),
            output_mode="int",
            output_sequence_length=max_input_len
        )
        self.proba_to_ids = TV(vocabulary=list(fr_vocab))
        self.max_input_len = max_input_len
        self.model = load_model(model_path)
        
        logging.info("Successfully loaded model and vectorizers...")

    def translate(self, eng_sent):
        """Accepts an English sentence and returns the French translation."""
        
        logging.info("Input request received: {}".format(eng_sent))
        input_pred_sequences = self.text_to_id(
            tf.reshape(
                tf.ragged.constant(eng_sent),
                (-1, 1)
            )
        )
        input_pred_sequences = pds(
            sequences=list(input_pred_sequences),
            maxlen=self.max_input_len,
            padding="post",
            truncating="post"
        )                                                                                           
        
        prediction_ids = self.model.predict(input_pred_sequences)
        
        logging.info("prediction successful")
        
        output_vocab = np.array(self.proba_to_ids.get_vocabulary())
        
        return ' '.join(
            [output_vocab[prediction] for prediction \
            in np.argmax(prediction_ids[0], 1)]
        )

def main():
    translator = EnFrTranslator(
        model_path="./model/en_fr_nmt_model.h5",
        en_vocab_path="./model/vocab/en_vocab.json",
        fr_vocab_path="./model/vocab/fr_vocab.json",
        max_input_len=15
    )
    logging.info(translator.translate("hello world"))

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
