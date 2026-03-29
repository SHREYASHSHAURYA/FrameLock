import numpy as np


class Trajectory:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.a = 0
        self.trajectory = []

        self.kf_x = np.array([0.0, 0.0])
        self.kf_y = np.array([0.0, 0.0])
        self.kf_a = np.array([0.0, 0.0])

        self.P_x = np.eye(2)
        self.P_y = np.eye(2)
        self.P_a = np.eye(2)

        self.F = np.array([[1, 1], [0, 1]])
        self.H = np.array([[1, 0]])
        self.Q = np.eye(2) * 0.0001
        self.R = np.array([[0.1]])

    def _kalman_update(self, kf_state, P, measurement):

        x_pred = self.F @ kf_state
        P_pred = self.F @ P @ self.F.T + self.Q

        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        x_new = x_pred.reshape(-1, 1) + K @ (
            np.array([[measurement]]) - self.H @ x_pred.reshape(-1, 1)
        )
        x_new = x_new.flatten()
        P_new = (np.eye(2) - K @ self.H) @ P_pred

        return x_new, P_new

    def update(self, dx, dy, da):
        self.x += dx
        self.y += dy
        self.a += da

        self.trajectory.append([self.x, self.y, self.a])

        self.kf_x, self.P_x = self._kalman_update(self.kf_x, self.P_x, self.x)
        self.kf_y, self.P_y = self._kalman_update(self.kf_y, self.P_y, self.y)
        self.kf_a, self.P_a = self._kalman_update(self.kf_a, self.P_a, self.a)

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

    def smooth_kalman(self):
        smoothed = []
        kf_x = np.array([0.0, 0.0])
        kf_y = np.array([0.0, 0.0])
        kf_a = np.array([0.0, 0.0])
        P_x = np.eye(2)
        P_y = np.eye(2)
        P_a = np.eye(2)

        for point in self.trajectory:
            kf_x, P_x = self._kalman_update(kf_x, P_x, point[0])
            kf_y, P_y = self._kalman_update(kf_y, P_y, point[1])
            kf_a, P_a = self._kalman_update(kf_a, P_a, point[2])
            smoothed.append([kf_x[0], kf_y[0], kf_a[0]])

        return np.array(smoothed)
