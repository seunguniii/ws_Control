import numpy as np
from qpsolvers import solve_qp

class QP:
  def __init__(self, dt, N, x, x_ref, x_max, x_min):
    O3 = np.zeros((3,3))

    I3 = np.array([
      [1.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0]
    ])
    '''
    p0 = x[:3]
    p_ref = x_ref[:3]
    dp = (p_ref - p0)/N
    x_ref_stacked = \
    np.array([
      np.hstack((p0 + k*dp, x_ref[3:]))
      for k in range(1, N+1)
    ])
    x_ref_stacked = x_ref_stacked.reshape(-1, 1)
    '''
    x_ref_stacked = np.tile(x_ref, (N, 1))
    x_max_stacked = np.tile(x_max, (N, 1))
    x_min_stacked = np.tile(x_min, (N, 1))

    A = np.block([
      [I3, dt*I3],
      [O3, I3   ]
    ])
    nA = A.shape[0]
    A_stacked = np.zeros((nA*N, nA))
    A_power = np.eye(nA)
    for i in range(N):
      A_power = A_power@A
      A_stacked[i*nA:(i+1)*nA, :] = A_power

    B = np.block([
      [0.5*dt**2*I3],
      [dt*I3       ]
    ])
    nB, mB = B.shape
    B_stacked = np.zeros((nB*N, mB*N))
    for i in range(N):
      for j in range(i+1):
        B_stacked[i*nB:(i+1)*nB, j*mB:(j+1)*mB] = np.linalg.matrix_power(A, i-j) @ B

    Qp = np.diag([30, 30, 100])
    Qv = np.diag([2, 2, 5])
    Q = np.block([
      [Qp, O3],
      [O3, Qv]
    ])
    Q_stacked = np.kron(np.eye(N), Q)

    R = I3.copy()*0.05
    R_stacked = np.kron(np.eye(N), R)

    self.H = B_stacked.T@Q_stacked@B_stacked + R_stacked
    self.f = B_stacked.T@Q_stacked@(A_stacked@x - x_ref_stacked)

    self.G = np.block([
      [B_stacked], [-B_stacked]
    ])

    self.h = np.block([
      [x_max_stacked - A_stacked@x ],
      [-x_min_stacked + A_stacked@x]
    ])

  def solve(self):
    ans = solve_qp(self.H, self.f, G=None, h=None, solver="quadprog")
    return ans
