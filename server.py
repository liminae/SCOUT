import base64

import cv2
import numpy as np
from flask import Flask, jsonify, request

from geometry import landmarks
from vision import analyze

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 5*1024*1024


@app.post('/analyze')
def image():
    upload = request.files.get('image')
    if upload is None:
        return jsonify(error='Provide an image file'), 400
    data = np.frombuffer(upload.read(), dtype=np.uint8)
    if not data.size:
        return jsonify(error='Empty image'), 400
    try:
        frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if frame is None or frame.shape[0]*frame.shape[1]>12000000:
            raise ValueError('Invalid or oversized image')
        metrics, overlay = analyze(frame)
        ok, png = cv2.imencode('.png', overlay)
        if not ok:
            raise ValueError('Could not encode overlay')
    except (ValueError, cv2.error) as e:
        return jsonify(error=str(e)), 400
    return jsonify(experimental=True, metrics=metrics, overlay=base64.b64encode(png).decode())


@app.post('/landmarks')
def points():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify(error='Expected a JSON object'), 400
    try:
        metrics = landmarks(data.get('points'), float(data.get('width', 1)), float(data.get('height', 1)))
    except (TypeError, ValueError) as e:
        return jsonify(error=str(e)), 400
    return jsonify(experimental=True, metrics=metrics)


if __name__=='__main__':
    app.run(host='127.0.0.1', port=5000)
