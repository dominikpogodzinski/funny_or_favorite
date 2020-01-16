import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


class CelBody:
    def __init__(self, init_pos, init_vel, mass, name):
        self.pos = np.array(init_pos).astype(float)
        self.vel = np.array(init_vel).astype(float)
        self.mass = mass
        self.name = name
        self.trajectory = []

    def step(self):
        self.pos += SolarSystem.dt * self.vel
        self.trajectory.append([self.pos[0], self.pos[1]])
        if len(self.trajectory) >= 100:
            self.trajectory.pop(0)
        if self.pos[0] > 5 or self.pos[0] < -5:
            self.vel[0] = -self.vel[0]
        if self.pos[1] > 5 or self.pos[1] < -5:
            self.vel[1] = -self.vel[1]


class SolarSystem:
    dt = 0.0005

    def __init__(self, bodies):
        self.bodies = bodies

    def step(self):
        for i in self.bodies:
            for j in self.bodies:
                if i is not j:
                    force = self.gravity(i, j)
                    acc = force / i.mass
                    i.vel += acc * SolarSystem.dt
            i.step()

    @staticmethod
    def gravity(i, j):
        G = 0.00008
        # G = 0.0008
        vec = j.pos - i.pos
        norm_vec = vec / np.linalg.norm(vec)
        scalar = G * i.mass * j.mass / np.linalg.norm(vec) ** 2
        return scalar * norm_vec


fig, ax = plt.subplots()

init_all = [{'r': [0, 0], 'v': [0., 0], 'm': 2000000.0, 'n': 'sun'},
            {'r': [0.058, 0], 'v': [0, 47.4], 'm': 0.3, 'n': 'mercury'},
            {'r': [0.108, 0], 'v': [0, 35.0], 'm': 5.0, 'n': 'venus'},
            {'r': [0.15, 0], 'v': [0, 29.8], 'm': 6.0, 'n': 'earth'},
            {'r': [0.23, 0], 'v': [0, 24.1], 'm': 0.6, 'n': 'mars'},
            {'r': [0.8, 0], 'v': [0, 13.1], 'm': 1900.0, 'n': 'jupiter'},
            {'r': [1.4, 0], 'v': [0, 9.7], 'm': 568.0, 'n': 'saturn'},
            {'r': [2.8, 0], 'v': [0, 6.8], 'm': 87.0, 'n': 'uranus'},
            {'r': [4.5, 0], 'v': [0, 5.4], 'm': 102.0, 'n': 'neptune'}]

planet_colors = [[255, 223, 34, 255],
                 [236, 214, 126, 255],
                 [165, 124, 27, 255],
                 [79, 76, 176, 255],
                 [193, 68, 14, 255],
                 [216, 202, 157, 255],
                 [234, 214, 184, 255],
                 [209, 231, 231, 255],
                 [63, 84, 186, 255]]

s = SolarSystem([CelBody(p['r'], p['v'], p['m'], p['n']) for p in init_all])
# p1 = CelBody([2, 0], [0, 5], 60, 'planet 1')
# p2 = CelBody([1, 0], [0, 5], 60, 'planet 2')
#
# s = SolarSystem([p1, p2])

bodies = ax.scatter([], [], zorder=80)
trajs = []
for n, b in enumerate(s.bodies):
    t = ax.plot([p[0] for p in b.trajectory], [p[1] for p in b.trajectory],
                color=np.array(planet_colors[n]) / 255)[0]
    trajs.append(t)


def initialize():
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    return [bodies] + trajs


def update(frame):
    s.step()
    bodies.set_offsets([b.pos for b in s.bodies])
    bodies.set_edgecolors('k')
    bodies.set_facecolors([np.array(x) / 255 for x in planet_colors])
    # print(s.bodies[6].trajectory)
    for n, b in enumerate(s.bodies):
        trajs[n].set_xdata([p[0] for p in b.trajectory])
        trajs[n].set_ydata([p[1] for p in b.trajectory])
    return [bodies] + trajs


ani = FuncAnimation(fig, func=update,
                    init_func=initialize, blit=True,
                    frames=2000, interval=33)

plt.show()
# ani.save('animacja.gif', writer=PillowWriter(fps=20))
