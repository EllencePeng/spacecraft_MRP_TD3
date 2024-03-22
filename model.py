import numpy as np
from numpy.linalg import inv

class spacecraft_simplified:
    def __init__(self, J):
        self.J = J

    def dynamics(self, t, X, Tr):
        omega = X[:3]
        p = X[3:6]
        p_cross = np.array([[0, -p[2], p[1]],
                            [p[2], 0, -p[0]],
                            [-p[1], p[0], 0]])
        
        Gp = (1- p.T@p)*np.eye(3) + 2*p_cross + 2*p.reshape(-1, 1) @ p.reshape(1, -1)
        dXdt = np.zeros(6)
        dXdt[:3] = inv(self.J) @ (- np.cross(omega, self.J@omega) + Tr) 
        dXdt[3:6] = 1/4 * Gp @ omega

        return dXdt

