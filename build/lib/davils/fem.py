import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib import animation as animplt


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def truss_element(E, A, rho, x1, x2):

    L = ((x2 - x1) @ (x2 - x1)) ** 0.5

    k_local = E * A / L * np.array([[1, 0, -1, 0], [0, 0, 0, 0], [-1, 0, 1, 0], [0, 0, 0, 0]])
    m_local = rho * A * L / 6 * np.array([[2, 0, 1, 0], [0, 2, 0, 1], [1, 0, 2, 0], [0, 1, 0, 2]])

    e1 = (x2 - x1) / L
    e2 = np.cross(np.array([0, 0, 1]), np.append(e1, 0))
    e2 = e2[0:-1]

    T_glob2loc = np.vstack((e1, e2))
    T_glob2loc_element = sp.linalg.block_diag(T_glob2loc, T_glob2loc)

    k_global = T_glob2loc_element.T @ k_local @ T_glob2loc_element
    m_global = T_glob2loc_element.T @ m_local @ T_glob2loc_element

    return L, k_global, m_global


def beam_element(E, A, I, rho, x1, x2):

    L = ((x2 - x1) @ (x2 - x1)) ** 0.5

    EA = E * A
    EI = E * I

    k_local = np.array([[EA / L, 0, 0, -EA / L, 0, 0],
                        [0, 12 * EI / L ** 3, -6 * EI / L ** 2, 0, -12 * EI / L ** 3, -6 * EI / L ** 2],
                        [0, -6 * EI / L ** 2, 4 * EI / L, 0, 6 * EI / L ** 2, 2 * EI / L],
                        [-EA / L, 0, 0, EA / L, 0, 0],
                        [0, -12 * EI / L ** 3, 6 * EI / L ** 2, 0, 12 * EI / L ** 3, 6 * EI / L ** 2],
                        [0, -6 * EI / L ** 2, 2 * EI / L, 0, 6 * EI / L ** 2, 4 * EI / L]])

    m_local = rho * A * L / 420 * np.array([[140, 0, 0, 70, 0, 0],
                                            [0, 156, -22 * L, 0, 54, 13 * L],
                                            [0, -22 * L, 4 * L ** 2, 0, -13 * L, -3 * L ** 2],
                                            [70, 0, 0, 140, 0, 0],
                                            [0, 54, -13 * L, 0, 156, 22 * L],
                                            [0, 13 * L, -3 * L ** 2, 0, 22 * L, 4 * L ** 2]])

    e1 = (x2 - x1) / L
    e2 = np.cross(np.array([0, 0, 1]), np.append(e1, 0))
    e2 = e2[0:-1]

    T_glob2loc = np.vstack((e1, e2))
    T_glob2loc = sp.linalg.block_diag(T_glob2loc, 1.0)
    T_glob2loc_element = sp.linalg.block_diag(T_glob2loc, T_glob2loc)

    k_global = T_glob2loc_element.T @ k_local @ T_glob2loc_element
    m_global = T_glob2loc_element.T @ m_local @ T_glob2loc_element

    return L, k_global, m_global


def refine_mesh(nodes, elements, T_BC, element_refine_factor, dofs_in_nodes):
    nodes_add = np.zeros(((element_refine_factor - 1) * elements.shape[0], 3))
    nodes_add[:, 0] = np.arange(0, nodes_add.shape[0]) + 10000
    new_nodes_element = element_refine_factor - 1

    refined_T_BC = sp.linalg.block_diag(T_BC, np.eye(nodes_add.shape[0] * dofs_in_nodes))

    refined_elements = np.zeros((element_refine_factor * elements.shape[0], elements.shape[1]))
    refined_elements[:, 0] = np.arange(1, elements.shape[0] * element_refine_factor + 1)

    for k in range(elements.shape[0]):
        refined_elements[k * element_refine_factor:(k + 1) * element_refine_factor, 1] = np.hstack(
            (elements[k, 1], np.arange(k * new_nodes_element, (k + 1) * new_nodes_element) + 10000))
        refined_elements[k * element_refine_factor:(k + 1) * element_refine_factor, 2] = np.hstack(
            (np.arange(k * new_nodes_element, (k + 1) * new_nodes_element) + 10000, elements[k, 2]))
        refined_elements[k * element_refine_factor:(k + 1) * element_refine_factor, 3] = np.full(
            (1,element_refine_factor), elements[k,3])

        refined_elements[k * element_refine_factor:(k + 1) * element_refine_factor, 4:] = elements[k, 4:]


        node_index1 = np.where(nodes[:, 0] == elements[k, 1])[0][0]
        node_index2 = np.where(nodes[:, 0] == elements[k, 2])[0][0]

        x1 = nodes[node_index1, 1:]
        x2 = nodes[node_index2, 1:]

        nodes_add[k * new_nodes_element:(k + 1) * new_nodes_element, 1] = np.linspace(x1[0], x2[0],
                                                                                      element_refine_factor + 1)[1:-1]
        nodes_add[k * new_nodes_element:(k + 1) * new_nodes_element, 2] = np.linspace(x1[1], x2[1],
                                                                                      element_refine_factor + 1)[1:-1]

    refined_nodes = np.vstack((nodes, nodes_add))

    return refined_nodes, refined_elements, refined_T_BC


def assembly(nodes, elements, dofs_in_nodes):
    mass_matrix = np.zeros((nodes.shape[0] * dofs_in_nodes, nodes.shape[0] * dofs_in_nodes))
    stiffness_matrix = np.zeros((nodes.shape[0] * dofs_in_nodes, nodes.shape[0] * dofs_in_nodes))

    for k in range(elements.shape[0]):
        node_index1 = np.where(nodes[:, 0] == elements[k, 1])[0][0]
        node_index2 = np.where(nodes[:, 0] == elements[k, 2])[0][0]

        x1 = nodes[node_index1, 1:]
        x2 = nodes[node_index2, 1:]

        if elements[k][3] == 1:
            _, k_global, m_global = beam_element(elements[k, 4], elements[k, 5], elements[k, 6], elements[k, 7], x1, x2)

        elif elements[k][3] == 2:
            _, k_global, m_global = truss_element(elements[k, 4], elements[k, 5], elements[k, 7], x1, x2)

        stiffness_matrix[dofs_in_nodes * node_index1:dofs_in_nodes * (node_index1 + 1), dofs_in_nodes * node_index1:dofs_in_nodes * (node_index1 + 1)] = \
            stiffness_matrix[dofs_in_nodes * node_index1:dofs_in_nodes * (node_index1 + 1), dofs_in_nodes * node_index1:dofs_in_nodes * (node_index1 + 1)] + k_global[0:dofs_in_nodes,0:dofs_in_nodes]
        stiffness_matrix[dofs_in_nodes * node_index1:dofs_in_nodes * (node_index1 + 1), dofs_in_nodes * node_index2:dofs_in_nodes * (node_index2 + 1)] = \
            stiffness_matrix[dofs_in_nodes * node_index1:dofs_in_nodes * (node_index1 + 1), dofs_in_nodes * node_index2:dofs_in_nodes * (node_index2 + 1)] + k_global[0:dofs_in_nodes, dofs_in_nodes:]
        stiffness_matrix[dofs_in_nodes * node_index2:dofs_in_nodes * (node_index2 + 1), dofs_in_nodes * node_index1:dofs_in_nodes * (node_index1 + 1)] = \
            stiffness_matrix[dofs_in_nodes * node_index2:dofs_in_nodes * (node_index2 + 1), dofs_in_nodes * node_index1:dofs_in_nodes * (node_index1 + 1)] + k_global[dofs_in_nodes:, 0:dofs_in_nodes]
        stiffness_matrix[dofs_in_nodes * node_index2:dofs_in_nodes * (node_index2 + 1), dofs_in_nodes * node_index2:dofs_in_nodes * (node_index2 + 1)] = \
            stiffness_matrix[dofs_in_nodes * node_index2:dofs_in_nodes * (node_index2 + 1), dofs_in_nodes * node_index2:dofs_in_nodes * (node_index2 + 1)] + k_global[dofs_in_nodes:, dofs_in_nodes:]

        mass_matrix[dofs_in_nodes * node_index1:dofs_in_nodes * (node_index1 + 1), dofs_in_nodes * node_index1:dofs_in_nodes * (node_index1 + 1)] = \
            mass_matrix[dofs_in_nodes * node_index1:dofs_in_nodes * (node_index1 + 1), dofs_in_nodes * node_index1:dofs_in_nodes * (node_index1 + 1)] + m_global[0:dofs_in_nodes, 0:dofs_in_nodes]
        mass_matrix[dofs_in_nodes * node_index1:dofs_in_nodes * (node_index1 + 1), dofs_in_nodes * node_index2:dofs_in_nodes * (node_index2 + 1)] = \
            mass_matrix[dofs_in_nodes * node_index1:dofs_in_nodes * (node_index1 + 1), dofs_in_nodes * node_index2:dofs_in_nodes * (node_index2 + 1)] + m_global[0:dofs_in_nodes, dofs_in_nodes:]
        mass_matrix[dofs_in_nodes * node_index2:dofs_in_nodes * (node_index2 + 1), dofs_in_nodes * node_index1:dofs_in_nodes * (node_index1 + 1)] = \
            mass_matrix[dofs_in_nodes * node_index2:dofs_in_nodes * (node_index2 + 1), dofs_in_nodes * node_index1:dofs_in_nodes * (node_index1 + 1)] + m_global[dofs_in_nodes:, 0:dofs_in_nodes]
        mass_matrix[dofs_in_nodes * node_index2:dofs_in_nodes * (node_index2 + 1), dofs_in_nodes * node_index2:dofs_in_nodes * (node_index2 + 1)] = \
            mass_matrix[dofs_in_nodes * node_index2:dofs_in_nodes * (node_index2 + 1), dofs_in_nodes * node_index2:dofs_in_nodes * (node_index2 + 1)] + m_global[dofs_in_nodes:, dofs_in_nodes:]

    if not check_symmetric(stiffness_matrix):
        if not check_symmetric(stiffness_matrix, rtol=1e-3, atol=1e-6):
            raise ValueError('Assembled stiffness matrix in not symmetric. Aborting...')
        else:
            print('ATTENTION!!!! Assembled stiffness matrix close to be not symmetric.')
    if not check_symmetric(mass_matrix):
        if not check_symmetric(mass_matrix, rtol=1e-3, atol=1e-6):
            raise ValueError('Assembled mass matrix in not symmetric. Aborting...')
        else:
            print('ATTENTION!!!! Assembled mass matrix close to be not symmetric.')

    return stiffness_matrix, mass_matrix


def rayleigh_damping_matrix(stiffness, mass, n1=1, n2=2, damp_ratio=(0.05,0.05), nat_freq=None, vec=None):
    # Computes the damping matrix in the form of: c1*M + c2*S (Rayleigh damping)
    # given the damping ratios at 2 nat. freq.

    if nat_freq is None or vec is None:
        lam, vec = sp.linalg.eig(stiffness, mass)
        indx = np.argsort(lam)
        lam = lam[indx]
        nat_freq = np.real(lam ** 0.5)
        vec = vec[:, indx]

    vec = np.array(vec)
    ray_matrix = np.array([[1/nat_freq[n1-1], nat_freq[n1-1]],
                           [1/nat_freq[n2-1], nat_freq[n2-1]]])/2
    ray_coeffs = np.linalg.solve(ray_matrix, damp_ratio)

    damping_matrix = ray_coeffs[0] * mass + ray_coeffs[1] * stiffness
    C = vec.T @ damping_matrix @ vec

    return damping_matrix, C


def forced_damping_matrix(stiffness, mass, damp_ratio=0.01, nat_freq=None, vec=None):

    if nat_freq is None or vec is None:
        lam, vec = sp.linalg.eig(stiffness, mass)
        indx = np.argsort(lam)
        lam = lam[indx]
        nat_freq = np.real(lam ** 0.5)
        vec = vec[:, indx]

    vec = np.array(vec)
    M = vec.T @ mass @ vec
    C = 2 * damp_ratio * M * np.diag(nat_freq)

    damping_matrix = np.linalg.inv(vec.T) @ C @ np.linalg.inv(vec)

    return damping_matrix, C


def ___get_state_space(n_dof, m, k, c):                 # DEPRECATED !!!!
    a1 = np.zeros((n_dof, n_dof))
    a2 = np.eye(n_dof)
    A1 = np.hstack((a1, a2))
    a3 = -np.linalg.inv(m) @ k  # M^-1 @ K (ndof x ndof)
    a4 = -np.linalg.inv(m) @ c  # M^-1 @ C (ndof x ndof)
    A2 = np.hstack((a3, a4))
    Ac = np.vstack((A1, A2))  # State Matrix A (2*ndof x 2*ndof)
    b2 = -np.linalg.inv(m)
    Bc = np.vstack((a1, b2))  # Input Influence Matrix B (2*ndof x n°input=ndof)

    # n°output is 1, 2 or 3 (displacements, velocities, accelerations)
    # the Cc matrix is defined accordingly
    c1 = np.hstack((a2, a1))  # displacements row
    c2 = np.hstack((a1, a2))  # velocities row
    c3 = np.hstack((a3, a4))  # accelerations row
    Cc = np.vstack((c1, c2, c3))  # Output Influence Matrix C (n°output*ndof x 2*ndof)

    Dc = np.vstack((a1, a1, b2)) # Direct Transmission Matrix D (n°output*ndof x n°input=ndof)

    # Defining the system in State Space
    system_stsp = sp.signal.lti(Ac, Bc, Cc, Dc)

    return system_stsp


def solve_state_space(system, load, t, n_dof, T_BC=None):
    _, y_out, _ = sp.signal.lsim(system, U=load.T, T=t)
    y = np.transpose(y_out[:,:n_dof])            # displacement
    ydot = np.transpose(y_out[:,n_dof:2*n_dof])  # velocity
    y2dot = -np.transpose(y_out[:, 2*n_dof:])   # acceleration

    if T_BC is not None:
        y = T_BC @ y
        ydot = T_BC @ ydot
        y2dot = T_BC @ y2dot

    return y, ydot, y2dot, y_out



######################### Plotting Functions ###########################################################################

def plot_fe_model(nodes, elements):
    hor_size = np.max(nodes[:, 1]) - np.min(nodes[:, 1])
    hor_mid = 1 / 2 * (np.max(nodes[:, 1]) + np.min(nodes[:, 1]))
    vert_size = np.max(nodes[:, 2]) - np.min(nodes[:, 2])
    vert_mid = 1 / 2 * (np.max(nodes[:, 2]) + np.min(nodes[:, 2]))
    max_dim = np.max([hor_size, vert_size]) * 1.1

    plt.figure()
    plt.plot(nodes[:, 1], nodes[:, 2], "o")

    for k in range(elements.shape[0]):
        x1 = [nodes[nodes[:, 0] == elements[k, 1], 1], nodes[nodes[:, 0] == elements[k, 2], 1]]
        x2 = [nodes[nodes[:, 0] == elements[k, 1], 2], nodes[nodes[:, 0] == elements[k, 2], 2]]

        plt.plot(x1, x2)

    for k in range(nodes.shape[0]):
        plt.annotate(str(int(nodes[k, 0])), (nodes[k, 1], nodes[k, 2]), textcoords="offset points", xytext=(0,5), ha='center')

    plt.xlim([hor_mid - max_dim / 2, hor_mid + max_dim / 2])
    plt.ylim([vert_mid - max_dim / 2, vert_mid + max_dim / 2])
    plt.grid()
    plt.show()

    # for k in range(nodes.shape[0]):
    # plt.text(nodes[k,1],nodes[k,2], str(nodes[k,0]))


def plot_deformed_model(nodes, elements, u, dofs_in_nodes, skd):
    hor_size = np.max(nodes[:, 1]) - np.min(nodes[:, 1])
    hor_mid = 1 / 2 * (np.max(nodes[:, 1]) + np.min(nodes[:, 1]))

    vert_size = np.max(nodes[:, 2]) - np.min(nodes[:, 2])
    vert_mid = 1 / 2 * (np.max(nodes[:, 2]) + np.min(nodes[:, 2]))

    max_dim = np.max([hor_size, vert_size]) * 1.3

    nodes_deformed = np.copy(nodes)
    nodes_deformed[:, 1] = nodes_deformed[:, 1] + skd * u[0::dofs_in_nodes]
    nodes_deformed[:, 2] = nodes_deformed[:, 2] + skd * u[1::dofs_in_nodes]

    #fig = plt.figure(figsize=(12, 4))

    for k in range(elements.shape[0]):
        x1 = [nodes_deformed[nodes_deformed[:, 0] == elements[k, 1], 1],
              nodes_deformed[nodes_deformed[:, 0] == elements[k, 2], 1]]
        x2 = [nodes_deformed[nodes_deformed[:, 0] == elements[k, 1], 2],
              nodes_deformed[nodes_deformed[:, 0] == elements[k, 2], 2]]
        plt.plot(x1, x2)

    #plt.plot(nodes[:, 1] + skd * u[0::dofs_in_nodes], nodes[:, 2] + skd * u[1::dofs_in_nodes], "o")
    plt.plot(nodes_deformed[:, 1], nodes_deformed[:, 2], marker='o', color='r', linestyle='None')
    plt.xlim([hor_mid - hor_size*1.2 / 2, hor_mid + hor_size*1.2 / 2])
    plt.ylim([vert_mid - vert_size*2 / 2, vert_mid + vert_size*2 / 2])
    plt.grid()
    plt.gca().set_aspect('equal')
    #plt.tight_layout()

    #return fig


def framed_mode(mode, res=20, periods=4):
    mod_frames1 = np.tile(np.append(np.linspace(-1, 1, int(res+1)), np.linspace(1, -1, int(res+1))), periods * 2)
    mod_frames2 = np.append(mod_frames1[1:], 0)
    mod_frames = mod_frames1[mod_frames1 != mod_frames2]
    framed_mode = np.outer(mod_frames, mode[:,1:])
    framed_mode = np.hstack((modes[:,0],framed_mode))
    return framed_mode


def plot_deformed_model_anim(nodes, elements, u, dofs_in_nodes, skd, filename=None, speed=20, plottitle=None):

    def animate(i):
        ax.clear()
        nodes_deformed = np.copy(nodes)
        nodes_deformed[:, 1] = nodes_deformed[:, 1] + skd * u[i][0::dofs_in_nodes]
        nodes_deformed[:, 2] = nodes_deformed[:, 2] + skd * u[i][1::dofs_in_nodes]

        for k in range(elements.shape[0]):
            x1 = [nodes_deformed[nodes_deformed[:, 0] == elements[k, 1], 1],
                  nodes_deformed[nodes_deformed[:, 0] == elements[k, 2], 1]]
            x2 = [nodes_deformed[nodes_deformed[:, 0] == elements[k, 1], 2],
                  nodes_deformed[nodes_deformed[:, 0] == elements[k, 2], 2]]
            ax.plot(x1, x2)

        ax.plot(nodes_deformed[:, 1], nodes_deformed[:, 2], "o")
        ax.set_xlim([hor_mid - hor_size * 1.2 / 2, hor_mid + hor_size * 1.2 / 2])
        ax.set_ylim([vert_mid - vert_size * 2 / 2, vert_mid + vert_size * 2 / 2])
        ax.grid()
        ax.set_title(plottitle)
        plt.tight_layout()
        plt.gca().set_aspect('equal')


    hor_size = np.max(nodes[:, 1]) - np.min(nodes[:, 1])
    hor_mid = 1 / 2 * (np.max(nodes[:, 1]) + np.min(nodes[:, 1]))
    vert_size = np.max(nodes[:, 2]) - np.min(nodes[:, 2])
    vert_mid = 1 / 2 * (np.max(nodes[:, 2]) + np.min(nodes[:, 2]))
    max_dim = np.max([hor_size, vert_size]) * 1.3

    fig, ax = plt.subplots(figsize=(20, 5))
    plt.tight_layout()
    plt.gca().set_aspect('equal')
    ax.set_xlim([hor_mid - hor_size * 1.2 / 2, hor_mid + hor_size * 1.2 / 2])
    ax.set_ylim([vert_mid - vert_size * 2 / 2, vert_mid + vert_size * 2 / 2])
    ax.grid()

    # Create the animation
    animation = animplt.FuncAnimation(fig, animate, frames=len(u), interval=speed)

    # Save the animation (optional)
    if filename is None:
        filename = 'deformed_frame_animation'

    animation.save(f'{filename}.mp4', writer='ffmpeg')


def anim_def_model(nodes, elements, u, skd, res, filename=None, speed=20, plottitle=None):
    def animate(frame):
        ax.clear()
        nodes_deformed = np.copy(nodes)
        deformation_factor = frame / (res - 1)  # Calculate the deformation factor based on the frame
        for id, el in enumerate(u[:, 0]):
            el = int(np.real(el))
            nodes_deformed[el - 1, 1] = nodes_deformed[el - 1, 1] + skd * deformation_factor * u[id, 1]
            nodes_deformed[el - 1, 2] = nodes_deformed[el - 1, 2] + skd * deformation_factor * u[id, 2]
            nodes_deformed[el - 1, 3] = nodes_deformed[el - 1, 3] + skd * deformation_factor * u[id, 3]

        for k in range(elements.shape[0]):
            x1 = [nodes_deformed[nodes_deformed[:, 0] == elements[k, 1], 1],
                  nodes_deformed[nodes_deformed[:, 0] == elements[k, 2], 1]]
            x2 = [nodes_deformed[nodes_deformed[:, 0] == elements[k, 1], 2],
                  nodes_deformed[nodes_deformed[:, 0] == elements[k, 2], 2]]
            x3 = [nodes_deformed[nodes_deformed[:, 0] == elements[k, 1], 3],
                  nodes_deformed[nodes_deformed[:, 0] == elements[k, 2], 3]]
            ax.plot(x1, x2, x3)

        ax.scatter(nodes_deformed[:, 1], nodes_deformed[:, 2], nodes_deformed[:, 3])
        ax.set_xlim([x_mid - x_size * 1.2 / 2, x_mid + x_size * 1.2 / 2])
        ax.set_ylim([y_mid - y_size * 2 / 2, y_mid + y_size * 2 / 2])
        ax.set_zlim([z_mid - z_size * 2 / 2, z_mid + z_size * 2 / 2])
        ax.grid()
        ax.set_title(plottitle)

    x_size = np.max(nodes[:, 1]) - np.min(nodes[:, 1])
    x_mid = 1 / 2 * (np.max(nodes[:, 1]) + np.min(nodes[:, 1]))
    y_size = np.max(nodes[:, 2]) - np.min(nodes[:, 2])
    y_mid = 1 / 2 * (np.max(nodes[:, 2]) + np.min(nodes[:, 2]))
    z_size = np.max(nodes[:, 3]) - np.min(nodes[:, 3])
    z_mid = 1 / 2 * (np.max(nodes[:, 3]) + np.min(nodes[:, 3]))
    max_dim = np.max([x_size, y_size, z_size]) * 1.3

    fig = plt.figure(figsize=(20, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(nodes[:, 1], nodes[:, 2], nodes[:, 3], c='b', marker='o')
    ax.set_xlim([x_mid - x_size * 1.2 / 2, x_mid + x_size * 1.2 / 2])
    ax.set_ylim([y_mid - y_size * 2 / 2, y_mid + y_size * 2 / 2])
    ax.set_zlim([z_mid - z_size * 2 / 2, z_mid + z_size * 2 / 2])
    ax.grid()
    plt.tight_layout()
    plt.gca().set_aspect('equal')

    # Create the animation
    animation = animplt.FuncAnimation(fig, animate, frames=res, interval=speed)

    # Save the animation (optional)
    if filename is None:
        filename = 'deformed_frame_animation'

    animation.save(f'{filename}.mp4', writer='ffmpeg')



## new!!
# def anim_def_model(nodes, elements, u, skd, res, filename=None, speed=20, plottitle=None):
#     
#     def animate(frames):
#         ax.clear()
#         nodes_deformed = np.copy(nodes)
#         for id, el in enumerate(u[:,0]):
#             nodes_deformed[el-1, 1] = nodes_deformed[el-1, 1] + skd * u[id, 1]
#             nodes_deformed[el-1, 2] = nodes_deformed[el-1, 2] + skd * u[id, 2]
#             nodes_deformed[el-1, 3] = nodes_deformed[el-1, 3] + skd * u[id, 3]
# 
#         #fig = plt.figure()
#         for k in range(elements.shape[0]):
#             x1 = [nodes_deformed[nodes_deformed[:, 0] == elements[k, 1], 1],
#                   nodes_deformed[nodes_deformed[:, 0] == elements[k, 2], 1]]
#             x2 = [nodes_deformed[nodes_deformed[:, 0] == elements[k, 1], 2],
#                   nodes_deformed[nodes_deformed[:, 0] == elements[k, 2], 2]]
#             x3 = [nodes_deformed[nodes_deformed[:, 0] == elements[k, 1], 3],
#                   nodes_deformed[nodes_deformed[:, 0] == elements[k, 2], 3]]
#             ax.plot(x1, x2, x3)
# 
#         ax.scatter(nodes_deformed[:, 1], nodes_deformed[:, 2], nodes_deformed[:, 3])
#         # ax.plot(nodes_deformed[:, 1], nodes_deformed[:, 2], nodes_deformed[:, 3], "o")
#         ax.set_xlim([x_mid - x_size * 1.2 / 2, x_mid + x_size * 1.2 / 2])
#         ax.set_ylim([y_mid - y_size * 2 / 2, y_mid + y_size * 2 / 2])
#         ax.set_zlim([z_mid - z_size * 2 / 2, z_mid + z_size * 2 / 2])
#         ax.grid()
#         ax.set_title(plottitle)
# 
#     x_size = np.max(nodes[:, 1]) - np.min(nodes[:, 1])
#     x_mid = 1 / 2 * (np.max(nodes[:, 1]) + np.min(nodes[:, 1]))
#     y_size = np.max(nodes[:, 2]) - np.min(nodes[:, 2])
#     y_mid = 1 / 2 * (np.max(nodes[:, 2]) + np.min(nodes[:, 2]))
#     z_size = np.max(nodes[:, 3]) - np.min(nodes[:, 3])
#     z_mid = 1 / 2 * (np.max(nodes[:, 3]) + np.min(nodes[:, 3]))
#     max_dim = np.max([x_size, y_size, z_size]) * 1.3
#     
#     fig, ax = plt.subplots(figsize=(20, 5), projection='3d')
#     plt.tight_layout()
#     plt.gca().set_aspect('equal')
#     ax.set_xlim([hor_mid - hor_size * 1.2 / 2, hor_mid + hor_size * 1.2 / 2])
#     ax.set_ylim([vert_mid - vert_size * 2 / 2, vert_mid + vert_size * 2 / 2])
#     ax.grid()
#     
#     # Create the animation
#     frames = np.linspace(-1,1, res+1)
#     animation = animplt.FuncAnimation(fig, animate, frames, interval=speed)
# 
#     # Save the animation (optional)
#     if filename is None:
#         filename = 'deformed_frame_animation'
# 
#     animation.save(f'{filename}.mp4', writer='ffmpeg')




