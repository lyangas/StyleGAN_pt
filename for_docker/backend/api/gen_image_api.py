from gen_image_helper import generate_image
import torch

import flask
from flask import Flask, request, jsonify, send_file
import requests
import json
from io import StringIO, BytesIO

from PIL import Image

app = Flask(__name__)

import base64
def serve_pil_image(pil_img):
    buffered = BytesIO()
    pil_img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue())
    return img_str


@app.route('/api/test', methods=['GET'])
def api_test():
    return {'response': 'test'}


@app.route('/api/transfer_style', methods=['POST'])
def transfer_style():
    try:
        print(request.form)
        print(request.form['seed'])
        seed = int(request.form['seed'])
        
        styled_output = generate_image(seed=seed)
        
        res = serve_pil_image(styled_output)
        return res
        
    except Exception as e:
        print(str(e))
        return {'error': str(e)}
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5080)