from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, Namespace

from pendulum.double_pendulum import DoublePendulum
from mandelbrot.mandelbrot_set import MandelbrotSet

from threading import Thread
import time
from typing import List

from PIL import Image, ImageDraw
from io import BytesIO
import base64

app = Flask(__name__)

app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

PENDULA = {}
THREADS = {}
RK4_H = 0.005
FRAME_RATE = 60
N_FRAMES = (1.0 / FRAME_RATE) / RK4_H

@app.route("/")
def index():
    return render_template("index.html") 

@app.route("/pendulum")
def pendulum():
    return render_template("pendulum.html")

@app.route("/schrodinger")
def schrodinger():
    return render_template("schrodinger.html")

@app.route("/n-body")
def n_body():
    return render_template("coming_soon.html")

@app.route("/mandelbrot")
def mandelbrot():
    return render_template("mandelbrot.html")

import numpy as np

@socketio.on("update") 
def update(data):
    emit("update", {"theta_1": data["theta_1"], "theta_2": data["theta_2"]})

class PlayThread(Thread):
    def __init__(self, pendula: List[DoublePendulum]):
        super(PlayThread, self).__init__()
        self.pendula = pendula
        self.n_pendula = len(pendula)
        self.running = True
        self.index = 0
        self.start_time = time.time()
        self.n_iterations = 0
    
    def run(self):
        while self.running:
            if time.time() - self.start_time < 1.0 / FRAME_RATE:
                if self.n_iterations < N_FRAMES:
                    for i in range(self.n_pendula):
                        self.pendula[i].runge_kutta_4(RK4_H)
                    self.n_iterations += 1
                continue

            for i in range(max(0, N_FRAMES - self.n_iterations)):
                for i in range(self.n_pendula):
                    self.pendula[i].runge_kutta_4(RK4_H)
                
            socketio.emit("update", {"theta_1": [p.theta_1 for p in self.pendula], "theta_2": [p.theta_2 for p in self.pendula], "index": self.index})
            self.index += 1
            self.start_time = time.time()
            self.n_iterations = 0

@socketio.on("play")
def play(data):
    theta_1 = data["theta_1"]
    theta_2 = data["theta_2"]

    PENDULA[request.sid] = []

    for t1, t2 in zip(theta_1, theta_2):
        PENDULA[request.sid].append(DoublePendulum(1.0, 1.0, 1.0, 1.0, theta_1_0=t1, theta_2_0=t2, theta_1_p_0=0.0, theta_2_p_0=0.0))

    thread = PlayThread(PENDULA[request.sid])
    THREADS[request.sid] = thread
    thread.start()

@socketio.on("pause")
def pause():
    if request.sid in THREADS:
        THREADS[request.sid].running = False
        THREADS[request.sid].join()
        del THREADS[request.sid]

@socketio.on("disconnect")
def disconnect():
    print("Client disconnected")
    if request.sid in PENDULA:
        del PENDULA[request.sid]

    if request.sid in THREADS:
        THREADS[request.sid].running = False
        THREADS[request.sid].join()
        del THREADS[request.sid]

def add_axes(image, real_lower, real_upper, imag_lower, imag_upper, x_res, y_res):
    draw = ImageDraw.Draw(image)

    x_ints = np.linspace(0, x_res, 10)
    y_ints = np.linspace(y_res, 0, 10)

    x_floats = np.linspace(real_lower, real_upper, 10)
    y_floats = np.linspace(imag_lower, imag_upper, 10)

    for i, real in zip(x_ints, x_floats):
        draw.line([(i, 0), (i, y_res)], fill="white", width=1)
        draw.text((i, 0), "{:.2e}".format(real), fill="white")

    for i, imag in zip(y_ints, y_floats):
        draw.line([(0, i), (x_res, i)], fill="white", width=1)
        draw.text((0, i), "{:.2e}".format(imag), fill="white")

@socketio.on("render_mandelbrot")
def render_mandelbrot(data):
    real_lower = float(data["real_lower"])
    real_upper = float(data["real_upper"])
    imag_lower = float(data["imag_lower"])
    imag_upper = float(data["imag_upper"])
    x_res = int(data["x_res"])
    y_res = int(data["y_res"])
    n_iter = int(data["n_iter"])

    if request.sid in PENDULA:
        del PENDULA[request.sid]

    if request.sid in THREADS:
        THREADS[request.sid].running = False
        THREADS[request.sid].join()
        del THREADS[request.sid]

    success = False

    while not success:
        try:
            mandelbrot_set = MandelbrotSet(real_lower, real_upper, imag_lower, imag_upper, x_res, y_res)
            mandelbrot_set.iterate(n_iter)
            success = True
        except:
            pass
    

    image_array = mandelbrot_set.get_image((x_res, y_res))[::-1]
    try:
        image_array = image_array.get()
    except:
        pass

    image_array = image_array.astype(np.uint8)

    image = Image.fromarray(image_array)
    add_axes(image, real_lower, real_upper, imag_lower, imag_upper, x_res, y_res)
    

    buffer = BytesIO()
    image.save(buffer, format="PNG")
    image_data = buffer.getvalue()
    image_data_base64 = base64.b64encode(image_data).decode("utf-8")

    socketio.emit("rendered_mandelbrot", {"image": f"data:image/png;base64,{image_data_base64}"})

if __name__=="__main__":
    app.run(debug=True)
    