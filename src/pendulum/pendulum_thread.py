import threading
from PyQt5.QtCore import QThread, pyqtSlot
from double_pendulum import DoublePendulum
from typing import List
from queue import Queue

class PendulumThread(QThread):
    def __init__(self, pendula : List[DoublePendulum], calculate_h, display_h):
        super(PendulumThread, self).__init__()
        self.running = True
        self.paused = False

        self.pendula = pendula
        self.theta_1s = Queue(maxsize=-1)
        self.theta_2s = Queue(maxsize=-1)

        self.rk4_h = calculate_h
        self.display_h = display_h
        self.t = 0
        self.N_recorded = 0

        self.buffer_size = 1000

        self.queue_lock = threading.Lock()

    def loop_function(self):
        if self.theta_1s.qsize() >= self.buffer_size:
            self.paused = True
        else:
            self.paused = False

        if not self.paused:
            self.t += self.rk4_h

            record = (self.t // self.display_h) > self.N_recorded

            if record:
                t1, t2 = [], []
            for p in self.pendula:
                p.runge_kutta_4(self.rk4_h)

                if record:
                    t1.append(p.theta_1)
                    t2.append(p.theta_2)

            if record:
                with self.queue_lock:
                    self.theta_1s.put(t1)
                    self.theta_2s.put(t2)
                    self.N_recorded += 1

    @pyqtSlot()
    def run(self):
        while self.running:
            self.loop_function()

            

    def stop(self):
        self.running = False

    