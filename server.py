import json
from http.server import HTTPServer, SimpleHTTPRequestHandler

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = "wighets/best.pt"
PORT = 8000

model = YOLO(MODEL_PATH)


class Handler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.path = "/index.html"
        return super().do_GET()

    def do_POST(self):
        if self.path != "/predict":
            self.send_error(404)
            return

        data = self.rfile.read(int(self.headers["Content-Length"]))
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)

        move, conf = None, 0.0
        if frame is not None:
            boxes = model(frame, verbose=False)[0].boxes
            if len(boxes) > 0:
                i = int(boxes.conf.argmax())
                move = str(model.names[int(boxes.cls[i])]).capitalize()
                conf = float(boxes.conf[i])

        body = json.dumps({"move": move, "conf": conf}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


if __name__ == "__main__":
    print(f"Open http://localhost:{PORT}")
    HTTPServer(("localhost", PORT), Handler).serve_forever()
