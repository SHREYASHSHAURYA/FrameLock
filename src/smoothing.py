import numpy as np


class Trajectory:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.a = 0
        self.trajectory = []

    def update(self, dx, dy, da):
        self.x += dx
        self.y += dy
        self.a += da

        self.trajectory.append([self.x, self.y, self.a])

        return self.x, self.y, self.a

    def smooth(self, radius=30):
        smoothed = []

        for i in range(len(self.trajectory)):
            sum_x = 0
            sum_y = 0
            sum_a = 0
            count = 0

            for j in range(
                max(0, i - radius), min(len(self.trajectory), i + radius + 1)
            ):
                sum_x += self.trajectory[j][0]
                sum_y += self.trajectory[j][1]
                sum_a += self.trajectory[j][2]
                count += 1

            smoothed.append([sum_x / count, sum_y / count, sum_a / count])

        return np.array(smoothed)
