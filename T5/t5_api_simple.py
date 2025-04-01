from pathlib import Path

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils import inf_encode
from flask import Flask, request, jsonify
from flask_cors import CORS

from transformers import T5Tokenizer, T5ForConditionalGeneration


model_name = "t5-small"
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)
tokenizer = T5Tokenizer.from_pretrained(model_name)

source_len = 512
target_len = 512

app = Flask(__name__)
CORS(app)

torch.set_float32_matmul_precision("high")

@app.route('/api/ask', methods=['POST'])
def ask():
    data = request.json #['source_text': str]

    preds = inf_encode(model, tokenizer, data['source_text'], source_len, device)

    return jsonify({'answer': preds})


if __name__ == "__main__":
    app.run(port = 6000)
