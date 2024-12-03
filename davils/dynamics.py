import numpy as np
import scipy as sp
from matplotlib import pyplot as plt


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


def linear_newmark_krenk(M, C, K, f, u0, udot0, h, gamma, beta):
    # Initialize variables
    u = np.zeros((M.shape[0], f.shape[1]))
    udot = np.zeros((M.shape[0], f.shape[1]))
    u2dot = np.zeros((M.shape[0], f.shape[1]))

    # Insert initial conditions
    u[:, 0] = u0
    udot[:, 0] = udot0

    # Calculate "modified mass"
    if len(M) == 1:
        Mstar = M + gamma * h * C + beta * h ** 2 * K
    else:
        Mstar = M + gamma * h * C + beta * h ** 2 * K
        Mstar_inv = np.linalg.inv(Mstar)

    # Calculate initial accelerations
    if len(M) == 1:
        u2dot[:, 0] = (f[:, 0] - C @ udot[:, 0] - K @ u[:, 0]) / M
    else:
        u2dot[:, 0] = np.linalg.solve(M, f[:, 0] - C @ udot[:, 0] - K @ u[:, 0])

    for n in range(0, f.shape[1] - 1):

        # Predicion step
        udotstar = udot[:, n] + (1 - gamma) * h * u2dot[:, n]
        ustar = u[:, n] + h * udot[:, n] + (1 / 2 - beta) * h ** 2 * u2dot[:, n]
        # u2dot[:,n+1] = np.linalg.solve(Mstar, f[:,n+1] - C@udotstar - K@ustar)
        if len(M) == 1:
            u2dot[:, n + 1] = (f[:, n + 1] - C @ udotstar - K @ ustar) / Mstar
        else:
            u2dot[:, n + 1] = Mstar_inv @ (f[:, n + 1] - C @ udotstar - K @ ustar)

        # Correction step
        udot[:, n + 1] = udotstar + gamma * h * u2dot[:, n + 1]
        u[:, n + 1] = ustar + beta * h ** 2 * u2dot[:, n + 1]

    return u, udot, u2dot


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

    if dt is not None:
        Ad = sp.linalg.expm(Ac * dt)
        Bd = (Ad - np.eye(len(Ad))) @ np.linalg.inv(Ac) @ Bc
        return sp.signal.dlti(Ad, Bd, Cc, Dc, dt=dt)              # Discrete-time linear time invariant system

    else:
        return sp.signal.lti(Ac, Bc, Cc, Dc)                     # Continuous-time linear time invariant system


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

    :param omega: Modal matrix of the system (n_dof * n_dof)
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
    # make sure Ac is not singular
    # Ac = Ac + 1e-12 * np.eye(len(Ac))
    
    Bc = np.vstack((O @ phi.T @ s_u, phi.T @ m_i @ s_u))                 # Input Influence Matrix Bc (2*n_dof x n_input)

    c1 = np.hstack((s_d.T @ phi, s_d.T @ phi @ O))                       # Displacements rows
    c2 = np.hstack((s_v.T @ phi @ O, s_v.T @ phi))                       # Velocities rows
    c3 = np.hstack((s_a.T @ phi @ -omega**2, s_a.T @ phi @ -gamma))      # Accelerations rows
    Cc = np.vstack((c1, c2, c3))                                         # Output Influence Matrix Cc (n_output*n_dof x 2*n_dof)

    Dc = np.vstack((np.zeros_like(s_d.T @ s_u), np.zeros_like(s_v.T @ s_u), s_a.T @ phi @ phi.T @ s_u))  # Direct Transmission Matrix
                                                                         # Dc (n_output*n_dof x n_input=n_dof)

    if dt is not None:
        Ad = sp.linalg.expm(Ac * dt)
        Bd = (Ad - np.eye(len(Ad))) @ np.linalg.inv(Ac) @ Bc

        system_dlti = sp.signal.dlti(Ad, Bd, Cc, Dc, dt=dt)              # Discrete-time linear time invariant system
        return system_dlti
    else:
        system_clti = sp.signal.lti(Ac, Bc, Cc, Dc)  # Continuous-time linear time invariant system
        return system_clti


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
            mac_matrix[i,j] = (np.abs(modes1[:,i].T @ np.conj(modes2[:,j]))**2) / \
                              ((modes1[:,i].T @ np.conj(modes1[:,i])) * (modes2[:,j].T @ np.conj(modes2[:,j])))
            # mac_matrix[i,j] = np.dot(modes1[:,i], modes2[:,j])**2 / \
            #                   np.dot(np.dot(modes1[:,i], modes1[:,i]), np.dot(modes2[:,j], modes2[:,j]))

    return mac_matrix


def simulate_lti(system, u, t, in_noise=None, out_noise=None):
    """
    Simulates a Linear Time Invariant (LTI) system in state-space form.

    Parameters:
    - system (scipy.signal.lti or scipy.signal.dlti): The LTI system to be simulated. If the system is continuous-time (lti),
      it will be converted to a discrete-time system (dlti) using zero-order hold with the given time vector `t`.
    - u (ndarray): The input signal array of shape (m, len(t)), where m is the number of inputs and len(t) is the number of time steps.
    - t (ndarray): The time vector.
    - in_noise (ndarray, optional): The input noise array of shape (n, len(t)), where n is the number of states. If None, it defaults to zero noise.
    - out_noise (ndarray, optional): The output noise array of shape (p, len(t)), where p is the number of outputs. If None, it defaults to zero noise.

    Returns:
    - t (ndarray): The time vector.
    - y (ndarray): The output signal array of shape (p, len(t)), where p is the number of outputs.
    - x (ndarray): The state trajectory array of shape (n, len(t)), where n is the number of states.
    """

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

    A = system.A
    B = system.B
    C = system.C
    D = system.D
    for k in range(len(t) - 1):
        x[:, k + 1] = A @ x[:, k] + B @ u[:, k] + in_noise[:, k]
        y[:, k] = C @ x[:, k] + D @ u[:, k] + out_noise[:, k]

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
        
        # Si can be inverted? buh...
        
        
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


def augmented_kalman_filter(t, y, p0, A, B, G, J, Q, R, E): # TO BE CORRECTED/MODIFIED
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


def joint_inputstate(system, y, t, Q, R, S=None):
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
    u_e = np.zeros((len(D.T), len(t)))
    Px = np.zeros((len(A), len(A), len(t)))
    Pu = np.zeros((len(D.T), len(D.T), len(t)))
    Pxu = np.zeros((len(A), len(D.T), len(t)))
    Pux = np.zeros((len(D.T), len(A), len(t)))

    if S is None:
        S = np.zeros((len(A), len(C)))

    for k in range(len(t)-1):       
        # Input estimation
        V = C @ Px[:, :, k] @ C.T + R
        V = (V + V.T) / 2                                           # Force symmetry
        V_i = np.linalg.inv(V)
        M = np.linalg.inv(D.T @ V_i @ D) @ D.T @ V_i
        u_e[:,k] = M @ (y[:,k] - C @ x_e[:,k])
        Pu[:,:,k] = np.linalg.inv(D.T @ V_i @ D)
        Pu[:, :, k] = (Pu[:, :, k] + Pu[:, :, k].T)/2               # Force symmetry
        
        # Measurement update
        K = Px[:,:,k] @ C.T @ V_i
        x_e[:, k] += K @ (y[:, k] - C @ x_e[:, k] - D @ u_e[:, k])
        Px[:,:,k] += -K @ (V - D @ Pu[:,:,k] @ D.T) @ K.T
        Pxu[:,:,k] = -K @ D @ Pu[:,:,k]
        Pux[:,:,k] = Pxu[:,:,k].T
        
        # Time update
        x_e[:, k + 1] = A @ x_e[:, k] + B @ u_e[:, k]
        N = A @ K @ (np.eye(len(C)) - D @ M) + B @ M
        Px[:,:,k+1] = (np.hstack((A,B)) @ np.hstack((np.vstack((Px[:, :, k], Pux[:, :, k])), np.vstack((Pxu[:, :, k], Pu[:, :, k])))) 
                       @ np.vstack((A.T, B.T)) + Q - N @ S.T - S @ N.T)
        Px[:, :, k + 1] = (Px[:, :, k + 1] + Px[:, :, k + 1].T)/2   # Force symmetry

    return x_e, u_e, Px, Pu, Pxu


def dual_kalman_filter(system, y, t, Q, R, S=None):  # beta, not working
    """
    ### understaind why it does not work well

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

    D = system.D  ## J
    C = system.C  ## G
    A = system.A
    B = system.B
    x_e = np.zeros((len(A), len(t)))
    u_e = np.zeros((len(D.T), len(t)))
    Px = np.zeros((len(A), len(A), len(t)))
    Pu = np.zeros((len(D.T), len(D.T), len(t)))
    Qu = Q[:len(D.T), :len(D.T)]

    if S is None:
        S = np.zeros_like(A)

    for k in range(1, len(t)-1):
        # Input prediction
        u = u_e[:,k-1]         ###
        Pu[:,:,k] = Pu[:,:,k-1] + Qu

        # Input update
        Kp = Pu[:,:,k] @ D.T @ np.linalg.inv(D @ Pu[:,:,k] @ D.T + R)
        u_e[:,k] = u + Kp @ (y[:, k] - C @ x_e[:, k-1] - D @ u)
        Pu[:,:,k] += -Kp @ D @ Pu[:,:,k]

        # State prediction
        x_e[:, k] = A @ x_e[:, k-1] + B @ u_e[:, k]
        Px[:,:,k] = A @ Px[:,:,k-1] @ A.T + Q

        # State update
        Kx = Px[:,:,k] @ C.T @ np.linalg.inv(C @ Px[:,:,k] @ C.T + R)
        x_e[:,k] += Kx @ (y[:, k] - C @ x_e[:, k] - D @ u_e[:, k])
        Px[:,:,k] += -Kx @ C @ Px[:,:,k]

    return x_e, u_e, Px, Pu


def xmacmat_alt(phi1, phi2=None, conjugates=True, return_alternative=False):
    """
    Alternative implementation. Modal assurance criterion numbers, cross-matrix between two modal transformation matrices (modes stacked as columns).

    Arguments
    ---------------------------
    phi1 : double
        reference modes
    phi2 : double, optional
        modes to compare with, if not given (i.e., equal default value None), phi1 vs phi1 is assumed
    conjugates : True, optional
        check the complex conjugates of all modes as well (should normally be True)

    Returns
    ---------------------------
    macs : boolean
        matrix of MAC numbers
    """

    if phi2 is None:
        phi2 = phi1

    if phi1.ndim == 1:
        phi1 = phi1[:, np.newaxis]

    if phi2.ndim == 1:
        phi2 = phi2[:, np.newaxis]

    A = np.sum(phi1.T * np.conj(phi1.T), axis=1)[:, np.newaxis]
    B = np.sum(phi2.T * np.conj(phi2.T), axis=1)[:, np.newaxis]
    norms = np.abs(A @ B.T)

    if conjugates:
        macs = np.maximum(abs(phi1.T @ np.conj(phi2)) ** 2 / norms, abs(phi1.T @ phi2) ** 2 / norms)  # maximum = element-wise max
    else:
        macs = abs(phi1.T @ np.conj(phi2)) ** 2 / norms

    # Alternative formulation
    if return_alternative:
        macs2 = np.zeros((len(phi1.T),len(phi2.T)))
        for i, mode1 in enumerate(phi1.T):
            for k, mode2 in enumerate(phi2.T):
                if conjugates:
                    macs2[i,k] = np.maximum(np.abs(mode1 @ np.conj(mode2))**2, np.abs(mode1 @ mode2)**2)/((mode1.T @ np.conj(mode1)) * (mode2.T @ np.conj(mode2)))
                else:
                    macs2[i,k] = np.abs(mode1 @ np.conj(mode2))**2/((mode1.T @ np.conj(mode1)) * (mode2.T @ np.conj(mode2)))
        macs2 = np.real(macs2)
        macs = np.real(macs)
        return macs, macs2

    macs = np.real(macs)
    return macs


def mpcval(phi):
    # Based on the current paper:
    # Pappa, R. S., Elliott, K. B., & Schenk, A. (1993). 
    # Consistent-mode indicator for the eigensystem realization algorithm. 
    # Journal of Guidance, Control, and Dynamics, 16(5), 852–858.

    # Ensure on matrix format
    if phi.ndim == 1:
        phi = phi[:, np.newaxis]

    n_modes = np.shape(phi)[1]
    mpc_val = [None] * n_modes

    for mode in range(0, n_modes):
        phin = phi[:, mode]
        Sxx = np.dot(np.real(phin), np.real(phin))
        Syy = np.dot(np.imag(phin), np.imag(phin))
        Sxy = np.dot(np.real(phin), np.imag(phin))

        if 2 * Sxy != 0:
            eta = (Syy - Sxx) / (2 * Sxy)
            lambda1 = (Sxx + Syy) / 2 + Sxy * np.sqrt(eta ** 2 + 1)
            lambda2 = (Sxx + Syy) / 2 - Sxy * np.sqrt(eta ** 2 + 1)
            mpc_val[mode] = ((lambda1 - lambda2) / (lambda1 + lambda2)) ** 2
        else:
            mpc_val[mode] = np.nan  # Set MPC value to NaN if 2*Sxy is zero

    mpc_val = np.array(mpc_val)
    return mpc_val


def xmpcmat(phi1, phi2):
    # Ensure both inputs are in matrix format
    if phi1.ndim == 1:
        phi1 = phi1[:, np.newaxis]
    if phi2.ndim == 1:
        phi2 = phi2[:, np.newaxis]

    # Check that both phi1 and phi2 have the same number of modes
    n_modes1 = np.shape(phi1)[1]
    n_modes2 = np.shape(phi2)[1]
    cross_mpc_matrix = np.zeros((n_modes1, n_modes2))

    for mode1 in range(n_modes1):
        phin1 = phi1[:, mode1]

        for mode2 in range(n_modes2):
            phin2 = phi2[:, mode2]

            # Compute Sxx, Syy, and Sxy using phi1 and phi2
            Sxx = np.dot(np.real(phin1), np.real(phin2))
            Syy = np.dot(np.imag(phin1), np.imag(phin2))
            Sxy = np.dot(np.real(phin1), np.imag(phin2))

            # Calculate cross-MPC
            if 2 * Sxy != 0:
                eta = (Syy - Sxx) / (2 * Sxy)
                lambda1 = (Sxx + Syy) / 2 + Sxy * np.sqrt(eta ** 2 + 1)
                lambda2 = (Sxx + Syy) / 2 - Sxy * np.sqrt(eta ** 2 + 1)
                cross_mpc_matrix[mode1, mode2] = ((lambda1 - lambda2) / (lambda1 + lambda2)) ** 2
            else:
                cross_mpc_matrix[mode1, mode2] = np.nan  # Set MPC value to NaN if 2*Sxy is zero

    return cross_mpc_matrix


def xmpcmat_alt(phi1, phi2):
    # Based on the current paper:
    # Pappa, R. S., Elliott, K. B., & Schenk, A. (1993).
    # Consistent-mode indicator for the eigensystem realization algorithm.
    # Journal of Guidance, Control, and Dynamics, 16(5), 852–858.

    # Ensure on matrix format
    if phi1.ndim == 1:
        phi1 = phi1[:, np.newaxis]
    if phi2.ndim == 1:
        phi2 = phi2[:, np.newaxis]

    n_modes1 = np.shape(phi1)[1]
    n_modes2 = np.shape(phi2)[1]
    mpc_mat = np.zeros((n_modes1, n_modes2))

    for mode1 in range(0, n_modes1):
        phin1 = phi1[:, mode1]
        for mode2 in range(0, n_modes2):
            Sxx = np.dot(np.real(phin1), np.real(phin2))
            Syy = np.dot(np.imag(phin1), np.imag(phin2))
            Sxy = np.dot(np.real(phin1), np.imag(phin2))

            if Sxy != 0:
                eta = (Syy - Sxx) / (2 * Sxy)
                lambda1 = (Sxx + Syy) / 2 + Sxy * np.sqrt(eta ** 2 + 1)
                lambda2 = (Sxx + Syy) / 2 - Sxy * np.sqrt(eta ** 2 + 1)
                mpc_mat[mode1, mode2] = ((lambda1 - lambda2) / (lambda1 + lambda2)) ** 2
            else:
                mpc_mat[mode1, mode2] = np.nan  # Set MPC value to NaN if 2*Sxy is zero

    return mpc_mat


def normalized_cross_mpcval(phi1, phi2):
    # Ensure both inputs are in matrix format
    if phi1.ndim == 1:
        phi1 = phi1[:, np.newaxis]
    if phi2.ndim == 1:
        phi2 = phi2[:, np.newaxis]

    # Check that both phi1 and phi2 have the same number of modes
    n_modes1 = np.shape(phi1)[1]
    n_modes2 = np.shape(phi2)[1]
    cross_mpc_matrix = np.zeros((n_modes1, n_modes2))

    for mode1 in range(n_modes1):
        phin1 = phi1[:, mode1]
        norm1 = np.linalg.norm(phin1)  # Norm of phin1 for normalization

        for mode2 in range(n_modes2):
            phin2 = phi2[:, mode2]
            norm2 = np.linalg.norm(phin2)  # Norm of phin2 for normalization

            # Compute Sxx, Syy, and Sxy using phi1 and phi2 and normalize them
            Sxx = np.dot(np.real(phin1), np.real(phin2)) / (norm1 * norm2)
            Syy = np.dot(np.imag(phin1), np.imag(phin2)) / (norm1 * norm2)
            Sxy = np.dot(np.real(phin1), np.imag(phin2)) / (norm1 * norm2)

            # Calculate normalized cross-MPC
            if 2 * Sxy != 0:
                eta = (Syy - Sxx) / (2 * Sxy)
                lambda1 = (Sxx + Syy) / 2 + Sxy * np.sqrt(eta ** 2 + 1)
                lambda2 = (Sxx + Syy) / 2 - Sxy * np.sqrt(eta ** 2 + 1)
                cross_mpc_matrix[mode1, mode2] = ((lambda1 - lambda2) / (lambda1 + lambda2)) ** 2
            else:
                cross_mpc_matrix[mode1, mode2] = 0.0  # Set to 0 if 2*Sxy is zero

    return cross_mpc_matrix


def maxreal(phi, normalize=False):
    """
    Rotate complex vectors (stacked column-wise) such that the absolute values of the real parts are maximized.

    Arguments
    ---------------------------
    phi : double
        complex-valued modal transformation matrix (column-wise stacked mode shapes)

    Returns
    ---------------------------
    phi_max_real : boolean
        complex-valued modal transformation matrix, with vectors rotated to have maximum real parts
    """   
    # Check if phi is a 1D array
    if phi.ndim == 1:
        phi = phi[:, np.newaxis]
        flatten = True
    else:
        flatten = False
        
    angles = np.expand_dims(np.arange(0,np.pi, 0.01), axis=0)
    phi_max_real = np.zeros(np.shape(phi)).astype('complex')
    for mode in range(0,np.shape(phi)[1]):
        rot_mode = np.dot(np.expand_dims(phi[:, mode], axis=1), np.exp(angles*1j))
        max_angle_ix = np.argmax(np.sum(np.real(rot_mode)**2,axis=0), axis=0)

        phi_max_real[:, mode] = phi[:, mode] * np.exp(angles[0, max_angle_ix]*1j)*np.sign(sum(np.real(phi[:, mode])))
    
    if normalize:
        for mode in range(0,np.shape(phi)[1]):
            phi_max_real[:, mode] = phi_max_real[:, mode]/np.max(np.abs(phi_max_real[:, mode]))
    
    if flatten:
        phi_max_real = phi_max_real.flatten()
        
    return phi_max_real


def force_collinearity(phi):
    """
    Rotate complex vectors (stacked column-wise) such that the absolute values of the real parts are maximized and rotate each component to cancel the imaginary part.

    Arguments
    ---------------------------
    phi : double
        complex-valued modal transformation matrix (column-wise stacked mode shapes)

    Returns
    ---------------------------
    phi_max_real : boolean
        complex-valued modal transformation matrix, with vectors rotated to have maximum real parts, and each component rotated to cancel the imaginary part
    """
    # Check if phi is a 1D array
    if phi.ndim == 1:
        phi = phi[:, np.newaxis]
        flatten = True
    else:
        flatten = False

    angles = np.expand_dims(np.arange(0,np.pi, 0.01), axis=0)
    phi_max_real = np.zeros(np.shape(phi)).astype('complex')
    for mode in range(0,np.shape(phi)[1]):
        rot_mode = np.dot(np.expand_dims(phi[:, mode], axis=1), np.exp(angles*1j))
        max_angle_ix = np.argmax(np.sum(np.real(rot_mode)**2,axis=0), axis=0)

        phi_max_real[:, mode] = phi[:, mode] * np.exp(angles[0, max_angle_ix]*1j)*np.sign(sum(np.real(phi[:, mode])))
        for el in range(0, np.shape(phi)[0]):
            if np.angle(phi_max_real[el, mode]) < np.pi/2 or np.angle(phi_max_real[el, mode]) >= 3/2*np.pi:
                phi_max_real[el, mode] = phi_max_real[el, mode] * np.exp(1j* -np.angle(phi_max_real[el, mode]))
            else:
                phi_max_real[el, mode] = phi_max_real[el, mode] * np.exp(1j* (np.pi - np.angle(phi_max_real[el, mode])))
                print('hey')
    if flatten:
        phi_max_real = phi_max_real.flatten()

    return phi_max_real
