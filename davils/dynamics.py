import numpy as np
import scipy as sp
# from matplotlib import pyplot as plt


def shearframe(n, k, c, m, kg=0, relative_dampers=True):    # from Knut utilis
    #
    #   ================  --> u_1
    #   |              |
    #   |              |  k1
    #   ================  --> u_2
    #   |              |  k2
    #   :              :
    #   :              :
    #   :              :
    #   |              |
    #   ================  --> u_n
    #   |              | k_n
    #   |              |
    # /////          /////
    #

    if type(m) == int or type(m) == float:
        m = np.ones(n) * m

    if type(c) == int or type(c) == float:
        c = np.ones(n) * c

    if type(k) == int or type(k) == float:
        k = np.ones(n) * k

    if type(kg) == int or type(kg) == float:
        kg = np.ones(n) * kg

    if relative_dampers == False:
        C = np.diag(c)
    else:
        c_shift = np.insert(c[0:-1], 0, 0.0, axis=0)
        C = np.diag(c_shift + c)

    M = np.diag(m)

    k_shift = np.insert(k[0:-1], 0, 0.0, axis=0)
    kg_shift = np.insert(kg[0:-1], 0, 0.0, axis=0)

    K = np.diag(k_shift + k)
    Kg = np.diag(kg_shift + kg)

    for level in range(0, n - 1):
        K[level, level + 1] = -k[level]
        K[level + 1, level] = -k[level]
        Kg[level, level + 1] = -kg[level]
        Kg[level + 1, level] = -kg[level]

        if relative_dampers == True:
            C[level, level + 1] = -c[level]
            C[level + 1, level] = -c[level]

    return K, C, M, Kg


def get_state_space(m, k, c, dt=None, s_u=None, s_a=None, s_v=None, s_d=None):
    """
    Defines the state space representation given the system matrices K, M, and C and the selection matrices S_a, S,v,
    and S_d, according to the following system of ODEs:
        x'(t) = Ac*x(t) + Bc*u(t)
        y(t) = Cc*x(t) + Dc*u(t)
    where:
        x(t) = [x1(t), x2(t)] = [x1(t), x1'(t)]
        y(t) = [d(t), v(t), a(t)] = [S_d.T * x1(t), S_v.T * x2(t), S_a.T * x2'(t)]
        u(t) = S_u * f(t)
    Outputs are physical displacements, physical velocities, and physical accelerations (in this order). Cc and Dc
    matrices are defined accordingly.

    :param m: Mass matrix of the system (n_dof * n_dof)
    :param k: Stiffness matrix of the system (n_dof * n_dof)
    :param c: Damping matrix of the system (n_dof * n_dof)
    :param dt: Time-step for discrete formulation (s)
    :param s_u: Selection matrix of the input forces (n_dof * n_input)
    :param s_a: Selection matrix of the output accelerations (n_dof * n_output_disp)
    :param s_v: selection matrix of the output velocities (n_dof * n_output_vel)
    :param s_d: selection matrix of the output displacements (n_dof * n_output_acc)
    :return: Scipy continuous-time linear time invariant system base class
    """

    if s_u is None:
        s_u = np.eye(len(m))
    if s_a is None:
        s_a = np.eye(len(m))
    if s_v is None:
        s_v = np.eye(len(m))
    if s_d is None:
        s_d = np.eye(len(m))

    mi = np.linalg.inv(m)                                                # M^-1 (n_dof x n_dof)
    mi_k = mi @ k                                                   # M^-1 @ K (n_dof x n_dof)
    mi_c = mi @ c                                                   # M^-1 @ C (n_dof x n_dof)
    O = np.zeros_like(m)                                            # Identity matrix (n_dof x n_dof)
    I = np.eye(len(m))                                              # Zeros matrix (n_dof x n_dof)

    Ac = np.vstack((np.hstack((O, I)), np.hstack((-mi_k, -mi_c))))  # System (State) Matrix Ac (2*n_dof x 2*n_dof)

    Bc = np.vstack((O @ s_u, mi @ s_u))                             # Input Influence Matrix Bc (2*n_dof x n_input)

    c1 = np.hstack((s_d.T, s_d.T @ O))                              # displacements rows
    c2 = np.hstack((s_v.T @ O, s_v.T))                              # velocities rows
    c3 = np.hstack((s_a.T @ -mi_k, s_a.T @ -mi_c))                  # accelerations rows
    Cc = np.vstack((c1, c2, c3))                                    # Output Influence Matrix Cc (n_output*n_dof x 2*n_dof)

    Dc = np.vstack((s_d.T @ O @ s_u, s_v.T @ O @ s_u, s_a.T @ mi @ s_u))  # Direct Transmission Matrix Dc
                                                                    # (n_output*n_dof x n_input=n_dof)

    system_clti = sp.signal.lti(Ac, Bc, Cc, Dc)                     # Continuous-time linear time invariant system
    system_dlti = None

    if dt is not None:
        Ad = sp.linalg.expm(Ac * dt)
        Bd = (Ad - np.eye(len(Ad))) @ np.linalg.inv(Ac) @ Bc

        system_dlti = sp.signal.dlti(Ad, Bd, Cc, Dc, dt=dt)              # Discrete-time linear time invariant system

    return system_clti, system_dlti


def get_state_space_modal(omega, phi, gamma, mass=None, dt=None, s_u=None, s_a=None, s_v=None, s_d=None):
    """
    Defines the state space representation given the system matrices in modal coordinates Omega, Phi, and Gamma and the
    selection matrices S_a, S,v, and S_d, according to the following system of ODEs:
        x'(t) = Ac*x(t) + Bc*u(t)
        y(t) = Cc*x(t) + Dc*u(t)
    where:
        x(t) = [x1(t), x2(t)] = [x1(t), x1'(t)]
        y(t) = [d(t), v(t), a(t)] = [S_d.T * x1(t), S_v.T * x2(t), S_a.T * x2'(t)]
        u(t) = S_u * f(t)
    Outputs are *physical* displacements, *physical* velocities, and *physical* accelerations (in this order). Cc and Dc
    matrices are defined accordingly.

    :param omega: Mass matrix of the system (n_dof * n_dof)
    :param phi: Stiffness matrix of the system (n_dof * n_dof)
    :param gamma: Damping matrix of the system (n_dof * n_dof)
    :param dt: Time-step for discrete formulation (s)
    :param s_u: Selection matrix of the input forces (n_dof * n_input)
    :param s_a: Selection matrix of the output accelerations (n_dof * n_output_disp)
    :param s_v: selection matrix of the output velocities (n_dof * n_output_vel)
    :param s_d: selection matrix of the output displacements (n_dof * n_output_acc)
    :return: Scipy continuous-time linear time invariant system base class
    """

    if s_u is None:
        s_u = np.eye(len(phi))
    if s_a is None:
        s_a = np.eye(len(phi))
    if s_v is None:
        s_v = np.eye(len(phi))
    if s_d is None:
        s_d = np.eye(len(phi))
    if mass is None:
        m_i = np.eye(len(phi))
        print('Careful, mass matrix not provided. Mode shapes are assumed to be mass-normalized.')
    else:
        m_i = np.linalg.inv(mass)

    O = np.zeros_like(omega)                                             # Identity matrix (n_dof x n_dof)
    I = np.eye(len(omega))                                               # Zeros matrix (n_dof x n_dof)

    Ac = np.vstack((np.hstack((O, I)), np.hstack((-omega**2, -gamma))))  # System (State) Matrix Ac (2*n_dof x 2*n_dof)

    Bc = np.vstack((O @ phi.T @ s_u, phi.T @ m_i @ s_u))                 # Input Influence Matrix Bc (2*n_dof x n_input)

    c1 = np.hstack((s_d.T @ phi, s_d.T @ phi @ O))                       # Displacements rows
    c2 = np.hstack((s_v.T @ phi @ O, s_v.T @ phi))                       # Velocities rows
    c3 = np.hstack((s_a.T @ phi @ -omega**2, s_a.T @ phi @ -gamma))      # Accelerations rows
    Cc = np.vstack((c1, c2, c3))                                         # Output Influence Matrix Cc (n_output*n_dof x 2*n_dof)

    Dc = np.vstack((np.zeros_like(s_d.T @ s_u), np.zeros_like(s_v.T @ s_u), s_a.T @ phi @ phi.T @ s_u))  # Direct Transmission Matrix
                                                                         # Dc (n_output*n_dof x n_input=n_dof)

    system_clti = sp.signal.lti(Ac, Bc, Cc, Dc)                          # Continuous-time linear time invariant system
    system_dlti = None

    if dt is not None:
        Ad = sp.linalg.expm(Ac * dt)
        Bd = (Ad - np.eye(len(Ad))) @ np.linalg.inv(Ac) @ Bc

        system_dlti = sp.signal.dlti(Ad, Bd, Cc, Dc, dt=dt)              # Discrete-time linear time invariant system

    return system_clti, system_dlti


def mac(modes1, modes2):
    """
    Computes MAC matrix given two mode shapes matrices (n_dof x n_modes).

    :param modes1: First mode shape
    :param modes2: Second mode shape
    :return: MAC matrix
    """
    mac_matrix = np.zeros((np.shape(modes1)[1], np.shape(modes2)[1]))
    for i in range(np.shape(modes1)[1]):
        for j in range(np.shape(modes2)[1]):
            mac_matrix[i,j] = np.dot(modes1[:,i], modes2[:,j])**2 / \
                              np.dot(np.dot(modes1[:,i], modes1[:,i]), np.dot(modes2[:,j], modes2[:,j]))

    return mac_matrix



def simulate_lti(system, u, t, in_noise=None, out_noise=None):
    if system.dt is None:
        dt = t[1] - t[0]
        Ad = sp.linalg.expm(system.A * dt)
        Bd = (Ad - np.eye(len(Ad))) @ np.linalg.inv(system.A) @ system.B
        system = sp.signal.dlti(Ad, Bd, system.C, system.D, dt=dt)

    if in_noise is None:
        in_noise = np.zeros((len(system.A), len(t)))
    if out_noise is None:
        out_noise = np.zeros((len(system.C), len(t)))

    x = np.zeros((len(system.A), len(t)))
    y = np.zeros((len(system.C), len(t)))
    for k in range(len(t) - 1):
        x[:, k + 1] = system.A @ x[:, k] + system.B @ u[:, k] + in_noise[:, k]
        y[:, k] = system.C @ x[:, k] + system.D @ u[:, k] + out_noise[:, k]

    return t, y, x


def kalman_filter(system, u, y, t, Q, R, S=None):
    """

    :param system:
    :param u:
    :param y:
    :param t:
    :param Q:
    :param R:
    :param S:
    :return:
    """

    if system.dt is None:
        raise TypeError('Linea time-invariant system representation must be in discrete time.')

    D = system.D
    C = system.C
    A = system.A
    B = system.B
    x_e = np.zeros((len(A), len(t)))
    P = np.zeros((len(A), len(A), len(t)))

    if S is None:
        S = np.zeros_like(A)

    for k in range(len(t)-1):
        global iter_
        iter_ = k

        Si = C @ P[:, :, k] @ C.T + R
        Si = (Si + Si.T) / 2
        Si_i = np.linalg.inv(Si)
        x_e[:, k] += P[:, :, k] @ C.T @ Si_i @ (y[:, k] - C @ x_e[:, k] - D @ u[:, k])
        P[:, :, k] += -P[:, :, k] @ C.T @ Si_i @ C @ P[:, :, k]

        K = (A @ P[:, :, k] @ C.T + S) @ Si_i
        x_e[:, k + 1] = A @ x_e[:, k] + B @ u[:, k] + K @ (y[:, k] - C @ x_e[:, k] - D @ u[:, k])
        P[:, :, k + 1] = A @ P[:, :, k] @ A.T + K @ (A @ P[:, :, k] @ C.T + S).T + Q
        P[:, :, k + 1] = (P[:, :, k + 1] + P[:, :, k + 1].T)/2

    return x_e, P


def augmented_kalman_filter(t, y, p0, A, B, G, J, Q, R, E):
    """
    Estimates the state of a LINEAR system wwith GAUSSIAN distributed variables.
    Knowing the only some observations of the state.

    Parameters
    ----------
    t : array float
        time vector n.
    y : matrix float
        mxn observations.
    p0 : array float
        pxn initial inputs.
    A : matrix float
        NxN state matrix.
    B : matrix float
        Nxp state-input matrix, where p is the number of inputs.
    G : matrix float
        mxN state-observation matrix.
    J : matrix float
        mxp input-observation matrix.
    Q : metrix float
        NxN covariance noise in state.
    R : matrix float
        mxm covariance measurement noise.
    E : matrix float
        pxp covariance input noise.

    Returns
    -------
    x_hat : array float
        Nxn estimated states.
    P : matrix float
        NxNxn estimation uncertainty.

    """
    npp = B.shape[1]
    ns = A.shape[0]
    ny = G.shape[0]
    x_hat = np.zeros((ns + npp, len(t)))
    x_hat[ns:, 0] = p0
    P = np.zeros((ns + npp, ns + npp, len(t)))
    Aa = np.vstack((np.hstack((A, B)),
                    np.hstack((np.zeros((npp, ns)), np.eye(npp)))))
    Ga = np.hstack((G, J))
    Qa = np.vstack((np.hstack((Q, np.zeros((ns, npp)))),
                    np.hstack((np.zeros((npp, ns)), E))))
    for k in range(len(t) - 1):
        # measurement update
        Omega = Ga @ P[:, :, k] @ Ga.T + R
        Omega = (Omega + Omega.T) / 2  # force to be symmetric
        x_hat[:, k] = x_hat[:, k] + P[:, :, k] @ Ga.T @ npla.inv(Omega) @ (y_st[:, k] - Ga @ x_hat[:, k])
        P[:, :, k] = P[:, :, k] - P[:, :, k] @ Ga.T @ npla.inv(Omega) @ Ga @ P[:, :, k]
        # time update
        # K = (Aa @ P[:,:,k] @ Ga.T + S)@ npla.inv(Omega)
        x_hat[:, k + 1] = Aa @ x_hat[:, k]
        P[:, :, k + 1] = Aa @ P[:, :, k] @ Aa.T + Qa
        P[:, :, k + 1] = (P[:, :, k + 1] + P[:, :, k + 1].T) / 2  # force to be symmetric
    x = x_hat[0:ns, :]
    p = x_hat[ns:, :]
    return x, p, P


