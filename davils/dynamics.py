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





