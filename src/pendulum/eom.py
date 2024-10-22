from sympy import *


t1,   t2     = symbols("t1 t2")
t1_p, t2_p   = symbols("t1_p t2_p")
t1_pp, t2_pp = symbols("t1_pp t2_pp")

x0_pp = symbols("x0_pp")

L1, L2 = symbols("L1 L2")

x1_pp = x0_pp - t1_p**2 * L1 * sin(t1) + t1_pp * L1 * cos(t1)
y1_pp = t1_p**2 * L1 * cos(t1) + t1_pp * L1 * sin(t1)

x2_pp = x1_pp - t2_p**2 * L2 * sin(t2) + t2_pp * L2 * cos(t2)
y2_pp = y1_pp + t2_p**2 * L2 * cos(t2) + t2_pp * L2 * sin(t2)

m1, m2, g = symbols("m1 m2 g")

LS1 = sin(t1) * (m1 * y1_pp + m2 * y2_pp + m2 * g + m1 * g)
RS1 = -cos(t1) * (m1 * x1_pp + m2 * x2_pp)

LS2 = sin(t2) * (m2 * y2_pp + m2 * g)
RS2 = -cos(t2) * (m2 * x2_pp)

EQ1 = LS1 - RS1
EQ2 = LS2 - RS2

res = solve([EQ1, EQ2], [t1_pp, t2_pp])

my_t1_pp = simplify(res[t1_pp])
my_t2_pp = simplify(res[t2_pp])

print(my_t1_pp)
print()
print(my_t2_pp)