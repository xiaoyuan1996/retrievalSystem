import numpy as np
import time

def calc_sim():
    a = np.random.random((512))
    b = np.random.random((512))
    c = np.mean(np.multiply(a,b))
    return c



times = 100000

cs = []

t1 = time.time()
for i in range(times):
    c = calc_sim()
    cs.append(c.tolist())
t2 = time.time()
print(t2-t1)

cs = np.sort(cs)[::-1]
print(cs)
print(len(cs))
t1 = time.time()
print(t1-t2)
