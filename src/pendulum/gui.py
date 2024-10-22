import sys

from PySide2.QtWidgets import (QWidget, QPushButton, QApplication, 
                             QVBoxLayout, QSlider, QLabel,
                             QCheckBox, QTabWidget, QLineEdit,
                             QHBoxLayout, QComboBox, QDoubleSpinBox)
from PySide2.QtCore import QTimer, QThreadPool
from PySide2.QtGui import QDoubleValidator
from PySide2 import QtCore, QtGui
import pyqtgraph as pg

import numpy as np

from double_pendulum import DoublePendulum
import time

THETA = "θ"
DELTA = "Δ"

class PendulumGUI(QWidget):
    def __init__(self, theta_1, theta_2, parent=None):
        super().__init__(parent)

        self.setGeometry(100, 100, 500, 778)
        self.setWindowTitle("Double Pendulum")

        self.grid = QVBoxLayout()

        self.m1, self.m2 = 1, 1
        self.L1, self.L2 = 1, 1


        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setYRange(-(1.1 * (self.L1 + self.L2)), (1.1 * (self.L1 + self.L2)))
        self.plot_widget.setXRange(-(1.1 * (self.L1 + self.L2)), (1.1 * (self.L1 + self.L2)))

        self.pendulum_bobs_1 = pg.ScatterPlotItem(
            size=5*np.sqrt(self.m1), brush=pg.mkBrush(255, 255, 255, 255)
        )
        self.pendulum_bobs_2 = pg.ScatterPlotItem(
            size=5*np.sqrt(self.m2), brush=pg.mkBrush(255, 255, 255, 255)
        )

        self.init_single_pendulum(theta_1, theta_2)

        self.plot_widget.addItem(self.pendulum_bobs_1)
        self.plot_widget.addItem(self.pendulum_bobs_2)

        self.grid.addWidget(self.plot_widget)

        self.tabs = QTabWidget()

        self.graph_tab = QWidget()
        self.graph_tab_grid = QVBoxLayout()

        self.show_traces = QCheckBox("Show Traces")
        self.graph_tab_grid.addWidget(self.show_traces)

        self.set_m1 = QWidget()
        self.set_m1.setLayout(QHBoxLayout())
        self.set_m1.layout().addWidget(QLabel("Mass 1"))
        self.set_m1_entry = QLineEdit()
        self.set_m1_entry.setValidator(QDoubleValidator())
        self.set_m1_entry.setText("1.0")
        self.set_m1_entry.returnPressed.connect(self.set_pendulum_properties)
        self.set_m1.layout().addWidget(self.set_m1_entry)

        self.set_L1 = QWidget()
        self.set_L1.setLayout(QHBoxLayout())
        self.set_L1.layout().addWidget(QLabel("Length 1"))
        self.set_L1_entry = QLineEdit()
        self.set_L1_entry.setValidator(QDoubleValidator())
        self.set_L1_entry.setText("1.0")
        self.set_L1_entry.returnPressed.connect(self.set_pendulum_properties)
        self.set_L1.layout().addWidget(self.set_L1_entry)

        self.set_m2 = QWidget()
        self.set_m2.setLayout(QHBoxLayout())
        self.set_m2.layout().addWidget(QLabel("Mass 2"))
        self.set_m2_entry = QLineEdit()
        self.set_m2_entry.setValidator(QDoubleValidator())
        self.set_m2_entry.setText("1.0")
        self.set_m2_entry.returnPressed.connect(self.set_pendulum_properties)
        self.set_m2.layout().addWidget(self.set_m2_entry)

        self.set_L2 = QWidget()
        self.set_L2.setLayout(QHBoxLayout())
        self.set_L2.layout().addWidget(QLabel("Length 2"))
        self.set_L2_entry = QLineEdit()
        self.set_L2_entry.setValidator(QDoubleValidator())
        self.set_L2_entry.setText("1.0")
        self.set_L2_entry.returnPressed.connect(self.set_pendulum_properties)
        self.set_L2.layout().addWidget(self.set_L2_entry)

        self.graph_tab_grid.addWidget(self.set_m1)
        self.graph_tab_grid.addWidget(self.set_L1)
        self.graph_tab_grid.addWidget(self.set_m2)
        self.graph_tab_grid.addWidget(self.set_L2)

        self.set_properties = QPushButton("Set Properties")
        self.set_properties.clicked.connect(self.set_pendulum_properties)

        self.freeze_button = QPushButton("Freeze")
        self.freeze_button.clicked.connect(self.freeze_all)
        
        self.graph_tab_grid.addWidget(self.set_properties)
        self.graph_tab_grid.addWidget(self.freeze_button)

        self.pendulum_tab = QWidget()
        self.pendulum_tab_grid = QVBoxLayout()

        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.pause)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.play)

        self.pendulum_tab_grid.addWidget(self.pause_button)
        self.pendulum_tab_grid.addWidget(self.play_button)

        self.slider_max = 1000

        self.theta_1_slider = QSlider(QtCore.Qt.Horizontal)
        self.theta_1_slider.setRange(0, self.slider_max)
        self.theta_1_slider.valueChanged.connect(self.change_theta_1)
        self.theta_2_slider = QSlider(QtCore.Qt.Horizontal)
        self.theta_2_slider.setRange(0, self.slider_max)
        self.theta_2_slider.valueChanged.connect(self.change_theta_2)

        self.theta_1_label = QLabel(f"Theta 1: {round(self.theta_1[0], 3)}")
        self.pendulum_tab_grid.addWidget(self.theta_1_label)
        self.pendulum_tab_grid.addWidget(self.theta_1_slider)
        self.theta_2_label = QLabel(f"Theta 2: {round(self.theta_2[0], 3)}")
        self.pendulum_tab_grid.addWidget(self.theta_2_label)
        self.pendulum_tab_grid.addWidget(self.theta_2_slider)

        self.chaos_tab = QWidget()
        self.chaos_tab_grid = QVBoxLayout()

        self.number_of_pendulum_entry = QWidget()
        self.number_of_pendulum_entry.setLayout(QHBoxLayout())
        self.number_of_pendulum_entry.layout().addWidget(QLabel("Number of Pendula"))
        self.number_of_pendulum_entry_list = QComboBox()
        self.number_of_pendulum_entry_list.addItems([str(n) for n in range(2, 11)])
        self.number_of_pendulum_entry.layout().addWidget(self.number_of_pendulum_entry_list)

        self.chaos_theta_1 = QWidget()
        self.chaos_theta_1.setLayout(QHBoxLayout())
        self.chaos_theta_1.layout().addWidget(QLabel("Theta 1"))
        self.chaos_theta_1_spin = QDoubleSpinBox()
        self.chaos_theta_1_spin.setRange(0, 2 * np.pi)
        self.chaos_theta_1_spin.setSingleStep(np.pi / 100)
        self.chaos_theta_1.layout().addWidget(self.chaos_theta_1_spin)

        self.chaos_theta_2 = QWidget()
        self.chaos_theta_2.setLayout(QHBoxLayout())
        self.chaos_theta_2.layout().addWidget(QLabel("Theta 2"))
        self.chaos_theta_2_spin = QDoubleSpinBox()
        self.chaos_theta_2_spin.setRange(0, 2 * np.pi)
        self.chaos_theta_2_spin.setSingleStep(np.pi / 100)
        self.chaos_theta_2.layout().addWidget(self.chaos_theta_2_spin)

        self.chaos_d_theta_1 = QWidget()
        self.chaos_d_theta_1.setLayout(QHBoxLayout())
        self.chaos_d_theta_1.layout().addWidget(QLabel(f"{DELTA}{THETA}1"))
        self.chaos_d_theta_1_spin = QDoubleSpinBox()
        self.chaos_d_theta_1_spin.setDecimals(5)
        self.chaos_d_theta_1_spin.setRange(0, np.pi / 50)
        self.chaos_d_theta_1_spin.setSingleStep(np.pi / 10000)
        self.chaos_d_theta_1.layout().addWidget(self.chaos_d_theta_1_spin)

        self.chaos_d_theta_2 = QWidget()
        self.chaos_d_theta_2.setLayout(QHBoxLayout())
        self.chaos_d_theta_2.layout().addWidget(QLabel(f"{DELTA}{THETA}2"))
        self.chaos_d_theta_2_spin = QDoubleSpinBox()
        self.chaos_d_theta_2_spin.setDecimals(5)
        self.chaos_d_theta_2_spin.setRange(0, np.pi / 50)
        self.chaos_d_theta_2_spin.setSingleStep(np.pi / 10000)
        self.chaos_d_theta_2.layout().addWidget(self.chaos_d_theta_2_spin)

        for item in [self.chaos_theta_1_spin, self.chaos_theta_2_spin, self.chaos_d_theta_1_spin, self.chaos_d_theta_2_spin]:
            item.valueChanged.connect(self.chaos_set)
        self.number_of_pendulum_entry_list.currentIndexChanged.connect(self.chaos_set)

        self.chaos_go = QPushButton("Go")
        self.chaos_go.clicked.connect(self.play)

        self.chaos_tab_grid.addWidget(self.number_of_pendulum_entry)
        self.chaos_tab_grid.addWidget(self.chaos_theta_1)
        self.chaos_tab_grid.addWidget(self.chaos_theta_2)
        self.chaos_tab_grid.addWidget(self.chaos_d_theta_1)
        self.chaos_tab_grid.addWidget(self.chaos_d_theta_2)
        self.chaos_tab_grid.addWidget(self.chaos_go)


        self.graph_tab.setLayout(self.graph_tab_grid)
        self.pendulum_tab.setLayout(self.pendulum_tab_grid)
        self.chaos_tab.setLayout(self.chaos_tab_grid)

        self.tab_titles = ["General Settings", "Pendulum Demo", "Chaos Demo"]

        self.tabs.addTab(self.graph_tab, "General Settings")
        self.tabs.addTab(self.pendulum_tab, "Pendulum Demo")
        self.tabs.addTab(self.chaos_tab, "Chaos Demo")

        self.tabs.currentChanged.connect(self.changed_tab)

        self.grid.addWidget(self.tabs)

        self.update_timer_timeout = 0.03
        self.dp_h = 0.005
        self.index = 0
        self.N_per_timeout = int(self.update_timer_timeout / self.dp_h)
        self.xpp_all = np.load("best.npy")

        self.show()

        self.pendulum_timer = QTimer()
        self.pendulum_timer.timeout.connect(self.update_pendulum)
        self.pendulum_timer.start(int(self.update_timer_timeout * 1000))

        # self.pendulum_thread = PendulumThread(self.double_pendula, self.dp_h, self.update_timer_timeout)
        # self.pendulum_thread.start()

        self.setLayout(self.grid)

        self.running = True
        self.t = 0

        self.trace_1_x = [[], ]
        self.trace_1_y = [[], ]
        self.trace_2_x = [[], ]
        self.trace_2_y = [[], ]
        self.max_trace_len = 1000

        self.trace_1_plots = [self.plot_widget.plot([], [], pen=pg.mkPen(color=(255, 0, 0))) for i in range(self.N_pendula)]
        self.trace_2_plots = [self.plot_widget.plot([], [], pen=pg.mkPen(color=(0, 0, 255))) for i in range(self.N_pendula)]

    def update_pendulum(self):
        self.pendulum_timer.stop()
        if self.running:
            for i in range(self.N_per_timeout):
                # self.double_pendula.x0_pp[:] = self.xpp_all[self.index]
                self.index += 1
                self.double_pendula.runge_kutta_4(self.dp_h)

                self.t += self.update_timer_timeout / self.N_per_timeout


            theta_1, theta_2 = self.double_pendula.get_angles()
            self.theta_1 = theta_1
            self.theta_2 = theta_2
            
            self.set_plot_data()
            if self.N_pendula == 1:
                self.set_labels()


        self.pendulum_timer.start(int(self.update_timer_timeout * 1000))


    def freeze_all(self):
        for i in range(self.N_pendula):
            self.double_pendula[i].freeze()

    def set_pendulum_properties(self):
        self.m1 = float(self.set_m1_entry.text())
        self.L1 = float(self.set_L1_entry.text())
        self.m2 = float(self.set_m2_entry.text())
        self.L2 = float(self.set_L2_entry.text())

        self.plot_widget.setYRange(-(1.1 * (self.L1 + self.L2)), (1.1 * (self.L1 + self.L2)))
        self.plot_widget.setXRange(-(1.1 * (self.L1 + self.L2)), (1.1 * (self.L1 + self.L2)))

        for i in range(self.N_pendula):
            self.double_pendula[i].m1 = self.m1
            self.double_pendula[i].L1 = self.L1
            self.double_pendula[i].m2 = self.m2
            self.double_pendula[i].L2 = self.L2

        self.pendulum_bobs_1.setData([], [])
        self.pendulum_bobs_2.setData([], [])

        self.plot_widget.removeItem(self.pendulum_bobs_1)
        self.plot_widget.removeItem(self.pendulum_bobs_2)

        self.pendulum_bobs_1 = pg.ScatterPlotItem(
            size=5*np.sqrt(self.m1), brush=pg.mkBrush(255, 255, 255, 255)
        )
        self.pendulum_bobs_2 = pg.ScatterPlotItem(
            size=5*np.sqrt(self.m2), brush=pg.mkBrush(255, 255, 255, 255)
        )

        self.plot_widget.addItem(self.pendulum_bobs_1)
        self.plot_widget.addItem(self.pendulum_bobs_2)

        self.set_plot_data()

    def init_single_pendulum(self, theta_1, theta_2):

        try:
            self.purge_plots()
        except:
            pass

        self.theta_1, self.theta_2 = np.array([theta_1, ]), np.array([theta_2, ])

        self.double_pendula = DoublePendulum(self.m1, self.L1, self.m2, self.L2, theta_1_0=self.theta_1, theta_2_0=self.theta_2, theta_1_p_0=np.zeros_like(self.theta_1), theta_2_p_0=np.zeros_like(self.theta_2))

        self.double_pendula.freeze()
        self.N_pendula = 1

        x1s, y1s, x2s, y2s = self.get_points()

        self.plots = [self.plot_widget.plot([0, x1s[i], x2s[i]], [0, y1s[i], y2s[i]]) for i in range(self.N_pendula)]

        self.pendulum_bobs_1.setData(x1s, y1s)
        self.pendulum_bobs_2.setData(x2s, y2s)

    def changed_tab(self, i):
        if self.tab_titles[i] == "Chaos Demo":
            self.chaos_set()
        if self.tab_titles[i] == "Pendulum Demo" and self.N_pendula > 1:
            value_1 = self.theta_1_slider.value()
            theta_1 = value_1 / self.slider_max * 2 * np.pi

            value_2 = self.theta_2_slider.value()
            theta_2 = value_2 / self.slider_max * 2 * np.pi

            self.init_single_pendulum(theta_1, theta_2)
        
    def chaos_set(self, *args):
        self.pause()

        N = int(self.number_of_pendulum_entry_list.currentText())
        self.N_pendula = N

        if len(self.double_pendula.theta_1) != N:
            self.theta_1 = np.zeros(N)
            self.theta_2 = np.zeros(N)
            self.double_pendula = DoublePendulum(self.m1, self.L2, self.m2, self.L2, theta_1_0=self.theta_1, theta_2_0=self.theta_2,
                                                                                    theta_1_p_0=np.zeros_like(self.theta_1), theta_2_p_0=np.zeros_like(self.theta_2))

        theta_1 = self.chaos_theta_1_spin.value()
        theta_2 = self.chaos_theta_2_spin.value()
        d_theta_1 = self.chaos_d_theta_1_spin.value()
        d_theta_2 = self.chaos_d_theta_2_spin.value()
        self.double_pendula.freeze()
        
        for i in range(N):
            t1 = theta_1 + i * d_theta_1
            t2 = theta_2 + i * d_theta_2
            self.theta_1[i] = t1
            self.theta_2[i] = t2

        self.double_pendula.theta_1 = self.theta_1
        self.double_pendula.theta_2 = self.theta_2

        self.set_plot_data()

    def get_points(self, x0=0):
        x0 = self.double_pendula.x0[0]
        x1, y1, x2, y2 = [], [], [], []
        for i in range(self.N_pendula):
            x1.append(x0 + self.L1 * np.sin(self.theta_1[i]))
            y1.append(-self.L1 * np.cos(self.theta_1[i]))

            x2.append(x1[-1] + self.L2 * np.sin(self.theta_2[i]))
            y2.append(y1[-1] - self.L2 * np.cos(self.theta_2[i]))

        return x1, y1, x2, y2
    
    def pause(self):
        self.running = False

    def play(self):
        self.running = True

    def rainbow_rgb(self, index):
        prop = index / self.N_pendula
        prop_mod_sixth = prop % (1 / 6) * 6

        if prop < (1/6):
            r = 255 * prop_mod_sixth
            g = 0
            b = 255
        elif prop < (2/6):
            r = 255
            g = 0
            b = 255 * (1 - prop_mod_sixth)
        elif prop < (3/6):
            r = 255
            g = 255 * prop_mod_sixth
            b = 0
        elif prop < (4/6):
            r = 255 * (1 - prop_mod_sixth)
            g = 255
            b = 0
        elif prop < (5/6):
            r = 0
            g = 255
            b = 255 * prop_mod_sixth
        else:
            r = 0
            g = 255 * (1 - prop_mod_sixth)
            b = 255

        return (r, g, b)

    def purge_plots(self):
        for plot in self.plots:
            plot.setData([], [])

        self.pendulum_bobs_1.setData([], [])
        self.pendulum_bobs_2.setData([], [])

        for tp in self.trace_1_plots:
            tp.setData([], [])
        for tp in self.trace_2_plots:
            tp.setData([], [])

    def set_plot_data(self):
        x1s, y1s, x2s, y2s = self.get_points()

        if len(self.plots) != self.N_pendula:
            self.purge_plots()
            self.plots = [self.plot_widget.plot([], []) for i in range(self.N_pendula)]

            self.trace_1_x = [[] for i in range(self.N_pendula)]
            self.trace_1_y = [[] for i in range(self.N_pendula)]
            self.trace_2_x = [[] for i in range(self.N_pendula)]
            self.trace_2_y = [[] for i in range(self.N_pendula)]

            self.trace_1_plots = [self.plot_widget.plot([], [], pen=pg.mkPen(
                color=(255, 0, 0))) for i in range(self.N_pendula)]
            self.trace_2_plots = [self.plot_widget.plot([], [], pen=pg.mkPen(color=self.rainbow_rgb(i))) for i in range(self.N_pendula)]

        for i in range(self.N_pendula):
            x1 = x1s[i]
            y1 = y1s[i]
            x2 = x2s[i]
            y2 = y2s[i]
            if self.show_traces.isChecked():
                self.trace_1_x[i].append(x1)
                self.trace_1_y[i].append(y1)
                self.trace_2_x[i].append(x2)
                self.trace_2_y[i].append(y2)

                for l in [self.trace_1_x[i], self.trace_1_y[i], self.trace_2_x[i], self.trace_2_y[i]]:
                    if len(l) > self.max_trace_len:
                        l.pop(0)
            else:
                self.trace_1_x[i] = []
                self.trace_1_y[i] = []
                self.trace_2_x[i] = []
                self.trace_2_y[i] = []

            self.plots[i].setData([self.double_pendula.x0[0], x1, x2], [0, y1, y2])

            self.trace_1_plots[i].setData(self.trace_1_x[i], self.trace_1_y[i])
            self.trace_2_plots[i].setData(self.trace_2_x[i], self.trace_2_y[i])

        self.pendulum_bobs_1.setData(x1s, y1s)
        self.pendulum_bobs_2.setData(x2s, y2s)

    def set_labels(self):
        self.theta_1_label.setText(f"Theta 1: {round(self.theta_1[0], 3)}")
        self.theta_2_label.setText(f"Theta 2: {round(self.theta_2[0], 3)}")

    def change_theta_1(self):
        if self.N_pendula == 1:
            value = self.theta_1_slider.value()
            theta_1 = value / self.slider_max * 2 * np.pi
            
            self.double_pendula.freeze()
            self.pause()
            self.double_pendula.theta_1[0] = theta_1
            self.theta_1 = [theta_1, ]
            self.set_plot_data()
            self.set_labels()

    def change_theta_2(self):
        if self.N_pendula == 1:
            value = self.theta_2_slider.value()
            theta_2 = value / self.slider_max * 2 * np.pi
            
            self.double_pendula.freeze()
            self.pause()
            self.double_pendula.theta_2[0] = theta_2
            self.theta_2 = [theta_2, ]
            self.set_plot_data()
            self.set_labels()

    def closeEvent(self, a0) -> None:
        return super().closeEvent(a0)


        


if __name__=="__main__":
    app = QApplication([])
    
    pendulum_gui = PendulumGUI(np.random.random() * 2 * np.pi, np.random.random() * 2 * np.pi)
    # pendulum_gui = PendulumGUI(0.0, 0.0)

    sys.exit(app.exec_())
