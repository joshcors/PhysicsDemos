import numpy as np

class DoublePendulum:
    def __init__(self, m1, L1, m2, L2, theta_1_0=0, theta_2_0=0, theta_1_p_0=0, theta_2_p_0=0):
        self.m1, self.L1 = m1, L1
        self.m2, self.L2 = m2, L2

        self.theta_1 = theta_1_0
        self.theta_1_p = theta_1_p_0

        self.theta_2 = theta_2_0
        self.theta_2_p = theta_2_p_0

        self.g = 9.81

        self.x0 = np.zeros_like(self.theta_1)
        self.x0_p = np.zeros_like(self.theta_1)
        self.x0_pp = np.zeros_like(self.theta_1)

    def theta_1_pp(self):
        numerator = -self.g * (2 * self.m1 + self.m2) * np.sin(self.theta_1)
        numerator -= self.m2 * self.g * np.sin(self.theta_1 - 2 * self.theta_2)
        numerator -= 2 * np.sin(self.theta_1 - self.theta_2) * self.m2 * (self.theta_2_p**2 * self.L2 + self.theta_1_p**2 * self.L1 * np.cos(self.theta_1 - self.theta_2))
        numerator -= (2 * self.m1 + self.m2) * self.x0_pp * np.cos(self.theta_1)
        numerator += self.m2 * self.x0_pp * np.cos(self.theta_1 - 2 * self.theta_2)

        denominator = self.L1 * (2 * self.m1 + self.m2 - self.m2 * np.cos(2 * self.theta_1 - 2 * self.theta_2))

        return numerator / denominator
    
    def theta_2_pp(self):
        numerator = self.theta_1_p**2 * self.L1 * (self.m1 + self.m2)
        numerator += self.g * (self.m1 + self.m2) * np.cos(self.theta_1)
        numerator += self.theta_2_p**2 * self.L2 * self.m2 * np.cos(self.theta_1 - self.theta_2)

        numerator *= 2 * np.sin(self.theta_1 - self.theta_2)
        numerator -= (self.m1 + self.m2) * self.x0_pp * np.cos(self.theta_2)
        numerator += (self.m1 + self.m2) * self.x0_pp * np.cos(2 * self.theta_1 - self.theta_2)

        denominator = self.L2 * (2 * self.m1 + self.m2 - self.m2 * np.cos(2 * self.theta_1 - 2 * self.theta_2))

        return numerator / denominator
    
    def runge_kutta_f(self, x):

        ot1, ot2, otp1, otp2 = self.theta_1, self.theta_2, self.theta_1_p, self.theta_2_p

        self.theta_1, self.theta_2, self.theta_1_p, self.theta_2_p = x

        f0 = self.theta_1_p
        f1 = self.theta_2_p
        f2 = self.theta_1_pp()
        f3 = self.theta_2_pp()

        self.theta_1, self.theta_2, self.theta_1_p, self.theta_2_p = ot1, ot2, otp1, otp2

        return np.array([f0, f1, f2, f3])
    
    def runge_kutta_4(self, h):

        x = np.array([self.theta_1, self.theta_2, self.theta_1_p, self.theta_2_p])

        self.x0_p += self.x0_pp * h
        self.x0 += self.x0_p * h

        k1 = self.runge_kutta_f(x)
        k2 = self.runge_kutta_f(x + 0.5 * k1 * h)
        k3 = self.runge_kutta_f(x + 0.5 * k2 * h)
        k4 = self.runge_kutta_f(x + k3 * h)

        x = x + h * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

        self.theta_1, self.theta_2, self.theta_1_p, self.theta_2_p = x

    def euler(self, h):
        t1pp = self.theta_1_pp()
        t2pp = self.theta_2_pp()

        self.theta_1_p += h * t1pp
        self.theta_2_p += h * t2pp

        self.theta_1 += h * self.theta_1_p
        self.theta_2 += h * self.theta_2_p

    def get_angles(self):
        return self.theta_1 % (2 * np.pi), self.theta_2 % (2 * np.pi)
    
    def freeze(self):
        if isinstance(self.theta_1_p, np.ndarray):
            self.theta_1_p = np.zeros_like(self.theta_1_p)
            self.theta_2_p = np.zeros_like(self.theta_2_p)
        else:
            self.theta_1_p = 0
            self.theta_2_p = 0

    def bottom(self):
        if isinstance(self.theta_1, np.ndarray):
            self.theta_1 = np.zeros_like(self.theta_1)
            self.theta_2 = np.zeros_like(self.theta_2)
        else:
            self.theta_1 = 0
            self.theta_2 = 0

    def reset(self):
        self.freeze()
        self.bottom()


        