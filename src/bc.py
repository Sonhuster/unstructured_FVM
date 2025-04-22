import numpy as np
import solver as sol
import utils as ufs
from input import *

# --------------------------Boundary condition------------------------------ #
lid_driven_cavity = {
        'top_face':   {'u': ['dirichlet', u_inf], 'v': ['dirichlet', v_inf], 'p': ['neumann', 0]},
        'bot_face':   {'u': ['dirichlet', 0], 'v': ['dirichlet', 0], 'p': ['neumann', 0]},
        'left_face':  {'u': ['dirichlet', 0], 'v': ['dirichlet', 0], 'p': ['neumann', 0]},
        'right_face': {'u': ['dirichlet', 0], 'v': ['dirichlet', 0], 'p': ['neumann', 0]},
    }

airfoil_flow = {
        'mainwing_face':   {'u': ['dirichlet', 0], 'v': ['dirichlet', 0], 'p': ['neumann', 0]},
        'slatwing_face':   {'u': ['dirichlet', 0], 'v': ['dirichlet', 0], 'p': ['neumann', 0]},
        'flapwing_face':   {'u': ['dirichlet', 0], 'v': ['dirichlet', 0], 'p': ['neumann', 0]},

        'top_face':   {'u': ['neumann', 0], 'v': ['neumann', 0], 'p': ['neumann', 0]},
        'bot_face':   {'u': ['neumann', 0], 'v': ['neumann', 0], 'p': ['neumann', 0]},
        'left_face':  {'u': ['dirichlet', u_inf], 'v': ['dirichlet', v_inf], 'p': ['neumann', 0]},
        'right_face': {'u': ['neumann', 0],         'v': ['neumann', 0],        'p': ['dirichlet', 0.0]},
    }

channel_flow = {
        'top_face':   {'u': ['dirichlet', 0],       'v': ['dirichlet', 0],      'p': ['neumann', 0]},
        'bot_face':   {'u': ['dirichlet', 0],       'v': ['dirichlet', 0],      'p': ['neumann', 0]},
        'left_face':  {'u': ['dirichlet', u_inf],   'v': ['dirichlet', v_inf],  'p': ['neumann', 0]},
        'right_face': {'u': ['neumann', 0],         'v': ['neumann', 0],        'p': ['dirichlet', 0.0]},
    }

heat_conduction = {
        'top_face':   {'u': ['dirichlet', u_inf],   'v': ['dirichlet', v_inf],      'p': ['neumann', 0]},
        'bot_face':   {'u': ['dirichlet', 0],       'v': ['dirichlet', 0],          'p': ['neumann', 0]},
        'left_face':  {'u': ['dirichlet', 0],       'v': ['dirichlet', 0],          'p': ['neumann', 0]},
        'right_face': {'u': ['dirichlet', 0],       'v': ['dirichlet', 0],          'p': ['neumann', 0]},
    }


def set_bc(mesh, var, SIMULATION_MODE):
    global lid_driven_cavity, airfoil_flow, channel_flow, heat_conduction

    if SIMULATION_MODE == 0:
        face_patches = lid_driven_cavity
    elif SIMULATION_MODE == 1:
        face_patches = airfoil_flow
    elif SIMULATION_MODE == 2:
        face_patches = channel_flow
    elif SIMULATION_MODE == 3:
        face_patches = heat_conduction
    else:
        "---Check simulation mode---"

    var_uv_old, var_vv_old, var_pv_old = var.uv.copy(), var.vv.copy(), var.pv.copy()
    var_ubc_old, var_vbc_old, var_pbc_old = var.ubc.copy(), var.vbc.copy(), var.pbc.copy()
    for iter_ in range(100):
        for face_patch in face_patches.keys():
            assert face_patch in mesh.boundary_info.face_patches.keys(), 'Re-check name of the bc face patches'
            for var_str, [bc_type, bc_value] in face_patches[face_patch].items():
                var_c, var_bc, var_v, var_f = ufs.shorted_name(var, var_str + 'c', var_str + 'bc', var_str + 'v', var_str + 'f')
                if bc_type == 'dirichlet':
                    var_bc, var_v, var_f = dirichlet_bc(mesh, face_patch, var_bc, var_v, var_f, bc_value)
                elif bc_type == 'dirichlet_one':
                    var_bc, var_v, var_f = dirichlet_bc(mesh, face_patch, var_bc, var_v, var_f, bc_value, one_value=True)
                elif bc_type == 'neumann':
                    var_bc, var_v, var_f = neumann_bc_new(mesh, face_patch, var_c, var_bc, var_v, var_f, bc_value)
                elif bc_type == 'none':
                    continue
        [error_uv, error_vv, error_pv] = sol.cal_outer_res((var.uv, var_uv_old), (var.vv, var_vv_old), (var.pv, var_pv_old))
        [error_ubc, error_vbc, error_pbc] = sol.cal_outer_res((var.ubc, var_ubc_old), (var.vbc, var_vbc_old), (var.pbc, var_pbc_old))
        var_uv_old, var_vv_old, var_pv_old = var.uv.copy(), var.vv.copy(), var.pv.copy()
        var_ubc_old, var_vbc_old, var_pbc_old = var.ubc.copy(), var.vbc.copy(), var.pbc.copy()

        # print(f"Neumann iter: {iter_} - residual = {error_pv}, {error_pbc}")

        if error_uv < 1e-09 and error_vv < 1e-09 and error_pv < 1e-09 and error_ubc < 1e-09 and error_vbc < 1e-09 and error_pbc < 1e-09:
            break

    return var


def dirichlet_bc(mesh, face_patch, var_bc, var_v, var_f, value, one_value=False):
    ifc = mesh.boundary_info.face_patches[face_patch]
    ifc = ifc if one_value==False else ifc[int(len(ifc)/2)]
    ifb = mesh.link.f_2_bf[ifc]
    node_idx = np.unique(mesh.global_faces.define[ifc])

    var_bc[ifb] = value
    var_v[node_idx] = value
    var_f[ifc] = var_bc[ifb]

    return var_bc, var_v, var_f

def neumann_bc(mesh, face_patch, var_c, var_bc, var_v, var_f, grad):
    (k_nbc, iff) = ufs.shorted_name(mesh.boundary_info, 'intersect_point', 'faces')

    ifc_patch = mesh.boundary_info.face_patches[face_patch]
    ifb = np.where(np.isin(iff, ifc_patch) == True)[0]

    # Pre-stored for step 1
    z_idx = k_nbc[1][ifb, 0]
    c_idx = k_nbc[1][ifb, 1]
    R = k_nbc[0][ifb]
    se_nodes = mesh.coords[k_nbc[1][ifb]]  # start to end node: start node lies on BC, end node is interior node
    m = mesh.global_faces.centroid[ifc_patch]  # get face center coordinate
    k = se_nodes[:, 1] + R.reshape(-1, 1) * (se_nodes[:, 0] - se_nodes[:, 1])  # get k coordinate
    L_km = np.linalg.norm(k - m, axis=1)

    # Pre-stored for step 2
    bc_nodes_idx = mesh.global_faces.define[ifc_patch]
    (left_p, right_p) = find_bc_face_centroid_next_to(mesh, bc_nodes_idx, ifb)
    weighted_left = mesh.global_faces.area[ifc_patch] / (
                mesh.global_faces.area[iff[left_p]] + mesh.global_faces.area[ifc_patch])
    weighted_right = mesh.global_faces.area[ifc_patch] / (
                mesh.global_faces.area[iff[right_p]] + mesh.global_faces.area[ifc_patch])

    # step 1 (updated centroid values from gradients, thinks of neumann -> dirichlet)
    var_k = var_v[c_idx] + R * (var_v[z_idx] - var_v[c_idx])
    var_bc[ifb] = var_k + grad * L_km

    # step 2 (get bc node values from updated bc centroid values from step 1)
    # TODO: left and right centroids are not actually visibly left and right, it actually is how the nodes of bc faces are reordered.
    var_v[bc_nodes_idx[:, 0]] = (1 - weighted_left) * var_bc[ifb] + weighted_left * var_bc[left_p]
    var_v[bc_nodes_idx[:, 1]] = (1 - weighted_right) * var_bc[ifb] + weighted_right * var_bc[right_p]

    var_f[ifc_patch] = var_bc[ifb]
    return var_bc, var_v, var_f


def neumann_bc_new(mesh, face_patch, var_c, var_bc, var_v, var_f, grad):
    def cal_weighted_coeff(a, center, b):
        # d_a = np.linalg.norm(center - a, axis=1)
        # d_b = np.linalg.norm(center - b, axis=1)
        d_a = np.linalg.norm(center - a)
        d_b = np.linalg.norm(center - b)
        d_sum = d_a + d_b

        wa = d_b / d_sum
        wb = d_a / d_sum
        return wa, wb

    (m, iff) = ufs.shorted_name(mesh.boundary_info, 'neumann_points', 'faces')

    ifc_patch = mesh.boundary_info.face_patches[face_patch]
    ifb = np.where(np.isin(iff, ifc_patch) == True)[0]

    c = mesh.elems.centroid[mesh.boundary_info.elems]
    L_cm = np.linalg.norm(c - m, axis=1)
    var_m = var_c[mesh.boundary_info.elems[ifb]] + grad * L_cm[ifb]
    bc_nodes_idx = mesh.global_faces.define[ifc_patch]

    for face_idx, ifc in enumerate(ifc_patch):
        for node_idx in range(2):
            common_node_idx = mesh.global_faces.define[ifc][node_idx]
            mask_common_node = np.isin(mesh.global_faces.define[ifc_patch], common_node_idx)
            face_share_node = np.where(mask_common_node[:, 1-node_idx] == True)[0]
            if len(face_share_node) == 0:
                face_share_node = face_idx
            left_node = m[ifb][face_idx]
            center_node = mesh.coords[common_node_idx]
            right_node = m[ifb][face_share_node]
            w_left, w_right = cal_weighted_coeff(left_node, center_node, right_node)
            var_v[common_node_idx] = w_left * var_m[face_idx] + w_right * var_m[face_share_node]

    var_bc[ifb] = np.mean(var_v[bc_nodes_idx], axis=1)
    var_f[ifc_patch] = var_bc[ifb]

    return var_bc, var_v, var_f


def find_bc_face_centroid_next_to(mesh, bc_nodes_idx, bf_idx):
    # TODO: This function should find the point just in current patches, not all the bc patches.
    list_fc = []

    # Find the bc faces contain left/right node in common.
    for idx, bc_nodes in enumerate(bc_nodes_idx):
        for local_node in range(2):
            mask = np.any(bc_nodes_idx == bc_nodes[local_node], axis=1)
            if np.sum(mask) == 1:  # If there's no face next to, add current face
                list_fc.append(bf_idx[idx])
            else:
                mask[idx] = False
                list_fc.append(bf_idx[np.where(mask)[0]][0])

    left_nodes =  np.array(list_fc).reshape(-1, 2).T[0]
    right_nodes = np.array(list_fc).reshape(-1, 2).T[1]
    return (left_nodes, right_nodes)


def get_fix_vel_bc_mask(mesh, SIMULATION_MODE):
    global lid_driven_cavity, airfoil_flow, channel_flow, heat_conduction

    if SIMULATION_MODE == 0:
        face_patches = lid_driven_cavity
    elif SIMULATION_MODE == 1:
        face_patches = airfoil_flow
    elif SIMULATION_MODE == 2:
        face_patches = channel_flow
    elif SIMULATION_MODE == 3:
        face_patches = heat_conduction
    else:
        "---Check simulation mode---"

    mask_bc = np.zeros(mesh.no_faces(), dtype=bool)
    for face_patch, bc_type in face_patches.items():
            if bc_type['u'][0] in ('dirichlet', 'dirichlet_one'):
                mask_bc[mesh.boundary_info.face_patches[face_patch]] = True
            else:
                mask_bc[mesh.boundary_info.face_patches[face_patch]] = False

    return mask_bc


airfoil_flow_out = {
        'right_face': 'out_flow',
    }

channel_flow_out = {
        'right_face': 'out_flow',
    }

def get_outflow_bc_mask(mesh, SIMULATION_MODE):
    global lid_driven_cavity, airfoil_flow, channel_flow, heat_conduction

    if SIMULATION_MODE == 0:
        face_patches = lid_driven_cavity
    elif SIMULATION_MODE == 1:
        face_patches = airfoil_flow_out
    elif SIMULATION_MODE == 2:
        face_patches = channel_flow_out
    elif SIMULATION_MODE == 3:
        face_patches = heat_conduction
    else:
        "---Check simulation mode---"

    mask_bc = np.zeros(mesh.no_faces(), dtype=bool)
    for face_patch, bc_type in face_patches.items():
        if bc_type == 'out_flow':
            mask_bc[mesh.boundary_info.face_patches[face_patch]] = True


    return mask_bc