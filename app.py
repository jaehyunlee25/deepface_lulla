from flask import Flask, render_template, send_file
from flask import request
from flask_cors import CORS, cross_origin
import json

from deepface import DeepFace
import matplotlib.pyplot as plt
import pandas as pd

app = Flask(__name__)
CORS(app)

@app.route('/find', methods=['POST'])
def find():
    param = json.loads(request.get_data(), encoding='utf-8')
    target = param['target']
    df = DeepFace.find(img_path = target, db_path = './images', enforce_detection=False)
    js = df.to_json(orient = 'records')
    return js

@app.route('/compare', methods=['POST'])
def compare():
    param = json.loads(request.get_data(), encoding='utf-8')
    img1 = param['img1']
    img2 = param['img2']
    result = DeepFace.verify(img1_path = img1, img2_path = img2, model_name = "Facenet")
    return result
@app.route('/detect/<img_name>')
def detect(img_name):
    img1 = DeepFace.detectFace(img_name)
    plt.imshow(img1)
    plt.savefig("det_" + img_name)
    return send_file("det_" + img_name)
@app.route('/analyze', methods=['POST'])
def main_page():
    param = json.loads(request.get_data(), encoding='utf-8')
    img_name = param['img_name']
    obj = DeepFace.analyze(img_name)
    return obj
@app.route('/image', methods=['POST'])
def post_image():
    param = json.loads(request.get_data(), encoding='utf-8')
    print(param)
    img_name = param['img_name']
    return send_file(img_name)
@app.route('/image/<img_name>', methods=['GET'])
def get_image(img_name):
    print(request)
    return send_file(img_name)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
