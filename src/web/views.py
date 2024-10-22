from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit, Namespace

from pendulum.double_pendulum import DoublePendulum

from threading import Thread
import time
from typing import List

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

if __name__=="__main__":
    app.run(debug=True)
    