import numpy as np

class KalmanFilter:
    def __init__(self, A, B, d, H, Q, R, P, x):
        self.A = A # state transition matrix
        self.B = B # control input matrix
        self.d = d # drift term
        self.H = H # observation matrix
        self.Q = Q # process noise covariance
        self.R = R # measurement noise covariance
        self.P = P # estimate error covariance
        self.x = x # state estimate

    def predict(self, u):
        self.x = self.A @ self.x + self.B @ u + self.d
        self.P = self.A @ self.P @ self.A.T + self.Q
        return self.x, self.P

    def update(self, z, H=None, R=None, c = None):
        #-----------------------------------------
        # Measurement Model
        # z = H x + c + v
        #------------------------------------------
        H = self.H if H is None else H
        R = self.R if R is None else R
        c = np.zeros((H.shape[0],)) if c is None else c

        y = z - (H @ self.x + c) #innovation
        S = H @ self.P @ H.T + R #innovation covariance

        K = self.P @ self.H.T @ np.linalg.inv(S) # Kalman gain    
        self.x = self.x + K @ y # update the state estimate
        
        # update the estimate covariance
        I = np.eye(self.P.shape[0])
        KH = K @ H
        self.P = (I - KH) @ self.P @ (I - KH).T + K @ R @ K.T # Joseph form to ensure positive semi-definiteness
        
        return self.x, self.P
