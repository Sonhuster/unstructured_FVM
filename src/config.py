import numpy as np
import utils as ufs
from input import *

def cell_to_face_interpolation(mesh):
    ifc = np.arange(mesh.no_faces())
    ic0 = mesh.link.f2c[ifc, 0]
    ic1 = mesh.link.f2c[ifc, 1]
    d0 = np.linalg.norm(mesh.elems.centroid[ic0] - mesh.global_faces.centroid[ifc], axis=1)
    d1 = np.linalg.norm(mesh.elems.centroid[ic1] - mesh.global_faces.centroid[ifc], axis=1)
    wf = (1 / d0) / ((1 / d0) + (1 / d1))
    return wf


def cell_to_node_interpolation(mesh):
    wv = np.zeros((mesh.no_nodes(), mesh.no_elems()))
    for inc in range(mesh.no_nodes()):
        contain_elems = np.where(np.isin(mesh.elems.define, inc) == True)[0]
        node_coord = mesh.coords[inc]
        elem_coords = mesh.elems.centroid[contain_elems]
        o_distance = 1 / np.linalg.norm(node_coord - elem_coords, axis = 1)
        wv_ = o_distance / (np.sum(o_distance))
        wv[inc, contain_elems] = wv_

    return wv

class Face:
    def __init__(self, number_of_faces):
        self.define = np.array([None])
        self.centroid = np.zeros(number_of_faces)
        self.area = np.array([None])
        self.sn = np.array([None])
        self.st = np.array([None])
        self.snsign = np.array([None])
        self.delta = np.array([None])

class Element:
    def __init__(self, elems):
        self.define = elems
        self.centroid = np.array([None])
        self.volume = np.array([None])

class BoundaryInfo:
    def __init__(self):
        self.elems = np.array([None])
        self.faces = np.array([None])
        self.nodes = np.array([None])
        self.intersect_point = []
        self.face_patches = {}
        self.node_patches = {}
        self.neumann_points = np.array([None])

class Connectivity:
    def __init__(self, no_cells, no_faces, no_local_faces):
        self.c2f = np.zeros((no_cells, no_local_faces), dtype = int)    # [Global cell, local face] =  Global face
        self.c2v = np.zeros((no_cells, no_local_faces), dtype = int)     # [Global cell, local node] =  Global node
        self.f2c = np.zeros((no_faces, 2), dtype = int)     # [Global face, local cell] =  Global cell
        self.f2v = np.zeros((no_cells, 2), dtype = int)     # [Global face, local node] =  Global node
        self.bf_2_f = np.array([None])
        self.f_2_bf = np.array([None])

class BlockData2D:
    def __init__(self, coords, elems, neighbors, no_faces):
        self.coords = coords
        self.elems = Element(elems)
        self.global_faces = Face(no_faces)
        self.cell_mapping = np.array([None])
        self.neighbors = neighbors
        self.boundary_info = BoundaryInfo()
        self.link = Connectivity(self.no_elems(), self.no_faces(), self.no_local_faces())

    def no_elems(self):
        return len(self.elems.define)

    def no_nodes(self):
        return len(self.coords)

    def no_faces(self):
        return len(self.global_faces.centroid)

    def no_local_faces(self):
        return self.elems.define.shape[1]

    def elems_centroid(self):
        centroid_coords = self.coords[self.elems.define]
        self.elems.centroid = np.mean(centroid_coords, axis = 1)

    def face_centroid(self):
        face_coords = self.coords[self.global_faces.define]
        self.global_faces.centroid = np.mean(face_coords, axis=1)

    def get_face_area(self):
        node_coords = self.coords[self.global_faces.define]
        self.global_faces.area =  np.linalg.norm(node_coords[:, 0] - node_coords[:, 1], axis = 1)

    def get_elem_volume(self):
        x = self.coords[self.elems.define][:, :, 0]
        y = self.coords[self.elems.define][:, :, 1]

        if self.no_local_faces() == 3:
            self.elems.volume = 0.5 * np.abs(
                x[:, 0] * (y[:, 1] - y[:, 2]) +
                x[:, 1] * (y[:, 2] - y[:, 0]) +
                x[:, 2] * (y[:, 0] - y[:, 1])
            )
        elif self.no_local_faces() == 4:
            self.elems.volume = (0.5 * np.abs(
                x[:, 0] * (y[:, 1] - y[:, 2]) +
                x[:, 1] * (y[:, 2] - y[:, 0]) +
                x[:, 2] * (y[:, 0] - y[:, 1])
            ) +
                                 0.5 * np.abs(
                x[:, 0] * (y[:, 2] - y[:, 3]) +
                x[:, 2] * (y[:, 3] - y[:, 0]) +
                x[:, 3] * (y[:, 0] - y[:, 2])
            ))


    # Get global face index + face_to_cell_connectivity + face_to_node_connectivity
    def global_face_numbering_tri(self):
        """
        Algorithm: Face-th is the opposite to the deleted node-th.
        :param : Simplices matrix of Delaunay triangles.
        :return: Global Face index, Face-to-Cell connectivity
        """
        global_faces_temp = set()

        global_faces = {}
        cell_mapping = {}
        global_face_id = 0
        cell_id = 0

        for simplex in self.elems.define:
            for i in range(self.no_local_faces()):
                face = (simplex[(i + 1) % 3], simplex[(i + 2) % 3])
                sorted_face = tuple(sorted(face))

                if sorted_face not in global_faces_temp:
                    global_faces_temp.add(sorted_face)
                    global_faces[face] = global_face_id
                    cell_mapping[tuple([global_face_id, 0])] = cell_id

                    global_face_id += 1
                else:
                    if face in global_faces:
                        cell_mapping[tuple([global_faces[face], 1])] = cell_id
                    else:
                        cell_mapping[tuple([global_faces[face[::-1]], 1])] = cell_id
            cell_id += 1

        # From dict to array
        cell_mapping_ = np.full((self.no_faces(), 2), -1)
        for (i, j), value in cell_mapping.items():
            cell_mapping_[i, j] = value

        for i in range(self.no_faces()):
            if cell_mapping_[i, 1] == -1:
                cell_mapping_[i, 1] = cell_mapping_[i, 0]

        global_face_ = np.zeros((self.no_faces(), 2), dtype = int)
        dem = 0
        for i, j in global_faces.keys():
            global_face_[dem, 0] = i
            global_face_[dem, 1] = j
            dem += 1

        self.global_faces.define = global_face_
        self.link.f2c = cell_mapping_

    def global_face_numbering_quad(self, global_face):
        """
        Algorithm: Face-th is the opposite to the deleted node-th.
        :param : Simplices matrix of Delaunay triangles.
        :return: Global Face index, Face-to-Cell connectivity
        """
        self.global_faces.define = global_face

        cell_mapping = {}

        for idx in range(self.no_faces()):
            face = self.global_faces.define[idx]

            cell_idx = tuple(sorted(np.where(np.sum(np.isin(self.elems.define, face), axis=1) == 2)[0]))
            cell_idx = cell_idx if len(cell_idx) > 1 else (cell_idx[0], -1)
            cell_mapping[idx] = cell_idx

        # From dict to array
        cell_mapping_ = np.full((self.no_faces(), 2), -1)
        for value, (i, j) in cell_mapping.items():
            cell_mapping_[value] = np.array([i, j])

        for i in range(self.no_faces()):
            if cell_mapping_[i, 1] == -1:
                cell_mapping_[i, 1] = cell_mapping_[i, 0]

        self.link.f2c = cell_mapping_


    # From global cell + local face, points out global face
    def cal_l_cell_to_face(self):
        simplex = self.elems.define
        local_face_index = np.arange(self.no_local_faces())
        if self.no_local_faces() == 3:
            for ic in range(self.no_elems()):
                for face_order in local_face_index:
                    face = np.delete(simplex[ic], face_order)
                    global_face_idx = np.where(np.sum(np.isin(self.global_faces.define, face), axis=1) == 2)[0][0]
                    self.link.c2f[ic, face_order] = global_face_idx
        else:
            for ic in range(self.no_elems()):
                for face_order in local_face_index:
                    face = np.array([simplex[ic, face_order % self.no_local_faces()],
                                     simplex[ic, (face_order + 1) % self.no_local_faces()]])
                    global_face_idx = np.where(np.sum(np.isin(self.global_faces.define, face), axis=1) == 2)[0][0]
                    self.link.c2f[ic, face_order] = global_face_idx

    # Return local boundary face if bface, or -1 if interior face
    def cal_link_boundary_face(self):
        f_to_bf = np.full(self.no_faces(), -1)
        global_face_idx = np.arange(self.no_faces())
        indices_in_b = np.where(np.isin(global_face_idx, self.boundary_info.faces))[0]
        for idx in indices_in_b:
            f_to_bf[idx] = np.where(self.boundary_info.faces == global_face_idx[idx])[0][0]

        self.link.f_2_bf = f_to_bf
        self.link.bf_2_f = self.boundary_info.faces

    # Get the indices where faces are on boundary.
    def domain_boundary(self, convex_hull):
        self.boundary_info.nodes = np.unique(convex_hull)
        for i in range(100000):
            if i > 500:
                print("WARNING: Inefficiently method for searching boundary face")
            temp1 = convex_hull + i
            check1 = temp1[:, 0] * temp1[:, 1]
            temp2 = self.global_faces.define + i
            check2 = temp2[:, 0] * temp2[:, 1]
            if len(check1) == len(np.unique(check1)) and len(check2) == len(np.unique(check2)):
                break
            else:
                continue

        self.boundary_info.faces = np.where(np.isin(check2, check1) == True)[0]
        bcells = []
        for ifb in self.boundary_info.faces:
            bc_cell = np.where(self.link.c2f == ifb)[0][0]
            bcells.append(bc_cell)

        self.boundary_info.elems = np.array(bcells)

    def point_intersect_normal_bc(self):
        def find_next_local_nodes(local_nodes, current_node):
            idx_20 = np.where(local_nodes == current_node)[0][0]

            left = local_nodes[(idx_20 - 1) % len(local_nodes)]
            right = local_nodes[(idx_20 + 1) % len(local_nodes)]

            return [left, right]

        R = []
        se_nodes = [] # start end nodes
        epsilon = 1e-010
        for idx, ifc in enumerate(self.boundary_info.faces):
            bc_nodes = self.global_faces.define[ifc]
            bc_cells = self.elems.define[self.boundary_info.elems[idx]]

            a = self.coords[bc_nodes[0]]
            b = self.coords[bc_nodes[1]]
            c_idx = np.setdiff1d(bc_cells, bc_nodes)[0]
            c = self.coords[c_idx]
            m = (a + b) / 2
            numerator = (b[0] - a[0]) * (c[0] - m[0]) + (b[1] - a[1]) * (c[1] - m[1])
            Ra = (a[0] - c[0]) * (a[0] - b[0]) + (a[1] - c[1]) * (a[1] - b[1])
            Rb = (b[0] - c[0]) * (a[0] - b[0]) + (b[1] - c[1]) * (a[1] - b[1])

            if 0 <= numerator / (Ra + epsilon) < 1:
                R.append(numerator / Ra)
                se_nodes.append((bc_nodes[0], c_idx))
            else:
                R.append(numerator / Rb)
                se_nodes.append((bc_nodes[1], c_idx))

        self.boundary_info.intersect_point = [np.asarray(R)] + [np.asarray(se_nodes)]


    def projected_cell_to_bc(self):
        bc_nodes = self.global_faces.define[self.boundary_info.faces]
        centroids = self.elems.centroid[self.boundary_info.elems]
        b, a = self.coords[bc_nodes[:, 1]], self.coords[bc_nodes[:, 0]]
        edge_vec = b - a  # shape (N, 2)
        edge_len_sq = np.sum(edge_vec ** 2, axis=1)  # (N, )

        to_centroid = centroids - a  # shape (N, 2)

        alpha = np.sum(to_centroid * edge_vec, axis=1) / edge_len_sq
        alpha = np.clip(alpha, 0.0, 1.0)

        projected = a + (edge_vec.T * alpha).T  # shape (N, 2)

        self.boundary_info.neumann_points = projected


    # Get local outward normal vector defined from cell 1 to 2
    def get_face_sn_st(self):
        normal = np.zeros((self.no_faces(), 2))
        tangent = np.zeros((self.no_faces(), 2))
        for ifc in range(self.no_faces()):
            coords = self.coords[self.global_faces.define[ifc]]
            vector = coords[1] - coords[0]
            centroid = self.elems.centroid[self.link.f2c[ifc, 0]]
            vector_centroid = coords[0] - centroid
            normal_ = np.array([vector[1], -vector[0]])

            if np.dot(vector_centroid, normal_) > 0:
                normal[ifc] = ufs.normalized_vector(normal_)
                tangent[ifc] = ufs.normalized_vector(vector)
            else:
                normal[ifc] = - ufs.normalized_vector(normal_)
                tangent[ifc] = - ufs.normalized_vector(vector)

        self.global_faces.sn = normal
        self.global_faces.st = tangent

    def get_normal_face_sign(self):
        local_normal_sign = np.zeros((self.no_elems(), self.no_local_faces()))
        for ic in range(self.no_elems()):
            for ifc in range(self.no_local_faces()):
                ifc_global = self.link.c2f[ic, ifc]
                ic_temp = self.link.f2c[ifc_global, 0]
                if ic == ic_temp:
                    local_normal_sign[ic, ifc] = 1
                else:
                    local_normal_sign[ic, ifc] = -1

        self.global_faces.snsign = local_normal_sign

    # Delta = distance between two cell in normal direction
    def delta_distance_cal(self):
        vec_l = np.zeros((self.no_faces(), 2))
        for ifc in range(self.no_faces()):
            ic0 = self.link.f2c[ifc, 0]
            ic1 = self.link.f2c[ifc, 1]
            if self.link.f_2_bf[ifc] == -1:
                vec_l[ifc] = self.elems.centroid[ic1] - self.elems.centroid[ic0]
            else:
                vec_l[ifc] = self.global_faces.centroid[ifc] - self.elems.centroid[ic0]

        self.global_faces.delta = np.abs(np.sum(self.global_faces.sn * vec_l, axis=1))

    def set_patches(self, bc_info=None):
        if SIMULATION_MODE in (0, 2, 3):
            boundary_face_centroids = self.global_faces.centroid[self.boundary_info.faces]
            boundary_nodes = self.coords[self.boundary_info.nodes]
            # Set face patches
            mask_bot_faces = boundary_face_centroids[:, 1] < 1e-03
            mask_top_faces = boundary_face_centroids[:, 1] + 1e-03 > np.max(boundary_face_centroids[:, 1])
            mask_left_faces = boundary_face_centroids[:, 0] < 1e-03
            mask_right_faces = boundary_face_centroids[:, 0] + 1e-03 > np.max(boundary_face_centroids[:, 0])
            # Set node patches
            mask_bot_nodes = boundary_nodes[:, 1] < 1e-03
            mask_top_nodes = boundary_nodes[:, 1] + 1e-03 > np.max(boundary_nodes[:, 1])
            mask_left_nodes = boundary_nodes[:, 0] < 1e-03
            mask_right_nodes = boundary_nodes[:, 0] + 1e-03 > np.max(boundary_nodes[:, 0])

            self.boundary_info.face_patches["bot_face"] = self.boundary_info.faces[mask_bot_faces]
            self.boundary_info.face_patches["top_face"] = self.boundary_info.faces[mask_top_faces]
            self.boundary_info.face_patches["left_face"] = self.boundary_info.faces[mask_left_faces]
            self.boundary_info.face_patches["right_face"] = self.boundary_info.faces[mask_right_faces]

            self.boundary_info.node_patches["bot_node"] = self.boundary_info.nodes[mask_bot_nodes]
            self.boundary_info.node_patches["top_node"] = self.boundary_info.nodes[mask_top_nodes]
            self.boundary_info.node_patches["left_node"] = self.boundary_info.nodes[mask_left_nodes]
            self.boundary_info.node_patches["right_node"] = self.boundary_info.nodes[mask_right_nodes]
        elif SIMULATION_MODE == 1:
            if bc_info is None:
                raise "Error: Don't find attached boundary patches!!!"
            else:
                mainwing_mask = bc_info[:, 3] == 1
                slatwing_mask = bc_info[:, 3] == 2
                flapwing_mask = bc_info[:, 3] == 3
                farfield_mask = ~ (mainwing_mask + slatwing_mask + flapwing_mask)

                self.boundary_info.face_patches["mainwing_face"] = self.link.c2f[bc_info[mainwing_mask, 2], bc_info[mainwing_mask, 4]]
                self.boundary_info.face_patches["slatwing_face"] = self.link.c2f[bc_info[slatwing_mask, 2], bc_info[slatwing_mask, 4]]
                self.boundary_info.face_patches["flapwing_face"] = self.link.c2f[bc_info[flapwing_mask, 2], bc_info[flapwing_mask, 4]]
                farfield_faces = self.link.c2f[bc_info[farfield_mask, 2], bc_info[farfield_mask, 4]]

                self.boundary_info.node_patches["mainwing_node"] = np.unique(self.link.f2v[self.boundary_info.face_patches["mainwing_face"]])
                self.boundary_info.node_patches["slatwing_node"] = np.unique(self.link.f2v[self.boundary_info.face_patches["slatwing_face"]])
                self.boundary_info.node_patches["flapwing_node"] = np.unique(self.link.f2v[self.boundary_info.face_patches["flapwing_face"]])
                farfield_nodes = np.unique(self.link.f2v[farfield_faces])

                boundary_face_centroids = self.global_faces.centroid[farfield_faces]    # oke
                boundary_nodes = self.coords[farfield_nodes]
                domain_sizex = np.max(self.coords[:, 0]) - np.min(self.coords[:, 0])
                # Set face patches
                mask_bot_faces = (boundary_face_centroids[:, 1] < 1e-03) & (boundary_face_centroids[:, 0] >= inflow_length*domain_sizex)
                mask_top_faces = ((boundary_face_centroids[:, 1] + 1e-03 > np.max(boundary_face_centroids[:, 1]))
                                  & (boundary_face_centroids[:, 0] >= inflow_length*domain_sizex))
                mask_left_faces = boundary_face_centroids[:, 0] < inflow_length*domain_sizex
                mask_right_faces = boundary_face_centroids[:, 0] + 1e-03 > np.max(boundary_face_centroids[:, 0])
                # Set node patches
                mask_bot_nodes = (boundary_nodes[:, 1] < 1e-03) & (boundary_nodes[:, 0] >= inflow_length*domain_sizex)
                mask_top_nodes = ((boundary_nodes[:, 1] + 1e-03 > np.max(boundary_nodes[:, 1]))
                                  & (boundary_nodes[:, 0] >= inflow_length*domain_sizex))
                mask_left_nodes = boundary_nodes[:, 0] < inflow_length*domain_sizex
                mask_right_nodes = boundary_nodes[:, 0] + 1e-03 > np.max(boundary_nodes[:, 0])

                self.boundary_info.face_patches["bot_face"] = farfield_faces[mask_bot_faces]
                self.boundary_info.face_patches["top_face"] = farfield_faces[mask_top_faces]
                self.boundary_info.face_patches["left_face"] = farfield_faces[mask_left_faces]
                self.boundary_info.face_patches["right_face"] = farfield_faces[mask_right_faces]

                self.boundary_info.node_patches["bot_node"] = farfield_nodes[mask_bot_nodes]
                self.boundary_info.node_patches["top_node"] = farfield_nodes[mask_top_nodes]
                self.boundary_info.node_patches["left_node"] = farfield_nodes[mask_left_nodes]
                self.boundary_info.node_patches["right_node"] = farfield_nodes[mask_right_nodes]
        else:
            raise "---Check simulation mode---"



    def get_face_mask_element_wise(self):
        basis_bc_mask = np.where(self.link.f_2_bf != -1)[0]
        filter_ = self.link.f_2_bf.copy()
        filter_[basis_bc_mask] = self.link.bf_2_f.copy()
        mask_bc = (filter_ >= 0)[self.link.c2f]
        mask_in = ~ mask_bc
        return mask_bc, mask_in


    def var_elem_wise(self, *vars):
        result = []
        for var in vars:
            if len(var) == self.no_faces():
                result.append(var[self.link.c2f])
            elif len(var) == self.no_nodes():
                result.append(var[self.link.c2v])
            elif len(var) == len(self.boundary_info.faces):
                arr_temp = np.zeros_like(self.link.f_2_bf)
                filter_ = self.link.f_2_bf != -1
                arr_temp[filter_] = var.copy()
                result.append(self.var_elem_wise(arr_temp)[0])
            else:
                raise ValueError("Re-check variable passing")

        return tuple(result)

    def var_face_wise(self, *vars):
        result = []
        for var in vars:
            if len(var) == self.no_elems():
                result.append(var[self.link.f2c])
            else:
                raise ValueError("Re-check variable passing")

        return tuple(result)

    def call_configuration(self, convex_hull, global_face, bc_info):
        self.elems_centroid()
        self.get_elem_volume()
        self.link.c2v = self.elems.define
        if self.no_local_faces() == 3:
            self.global_face_numbering_tri()
        else:
            self.global_face_numbering_quad(global_face)
        self.face_centroid()
        self.get_face_area()
        self.cal_l_cell_to_face()
        self.link.f2v = self.global_faces.define
        self.domain_boundary(convex_hull)
        self.point_intersect_normal_bc()        # For neumann boundary condition (Method 1) - Cover "tri" elements
        self.projected_cell_to_bc()             # For neumann boundary condition (Method 2) - Cover "tri" and "quad" elements
        self.cal_link_boundary_face()
        self.get_face_sn_st()
        self.get_normal_face_sign()
        self.delta_distance_cal()
        self.set_patches(bc_info)


# ---------------------------------------------------------------------------- #
def from_coords_to_mesh(coordinates, noise, element_type : str):
    assert element_type == "TRI" or element_type == "QUAD", "Element type must be TRI or QUAD"
    if element_type == "QUAD":
        elem_coords = get_elems(coordinates)
        nx = len(np.unique(coordinates[:, 0])) - 1  # Number of cell along each dir
        ny = len(np.unique(coordinates[:, 1])) - 1  # Number of cell along each dir
        n_v = len(coordinates)
        n_c = len(elem_coords)
        n_f = nx * (ny + 1) + (nx + 1) * ny
        simplices = get_simplices(coordinates, n_c, nx, ny)
        global_face = get_global_face(n_v, nx, ny)
        neighbors = get_neighbor_quad(n_c, nx, ny)
        convex_hull = get_convex_hull(coordinates)
        coordinates = get_noise(coordinates, noise)
        mesh = BlockData2D(coordinates, simplices, neighbors, n_f)
        mesh.call_configuration(convex_hull, global_face, bc_info=None)
    elif element_type == "TRI":
        coordinates = get_noise(coordinates, noise)
        tri = ufs.create_tetrahedral_faces(coordinates)
        tri.no_faces = ufs.number_face_tri(tri.neighbors)
        mesh = BlockData2D(tri.points, tri.simplices, tri.neighbors, tri.no_faces)
        global_face = np.array([None])
        mesh.call_configuration(tri.convex_hull, global_face, bc_info=None)


    print(f"_______________________FINISH GENERATING {element_type} UNSTRUCTURED MESH_______________________")
    print(f"Number of elements: {mesh.no_elems()}")
    print(f"Number of faces: {mesh.no_faces()}")
    print(f"Number of nodes: {mesh.no_nodes()}")
    return mesh


def loadfile(refined_level):
    def normalize_coords(coords):
        unit_coords = coords.copy()
        range_x = np.max(coords[:, 0]) - np.min(coords[:, 0])
        unit_coords[:, 0] = (unit_coords[:, 0] - np.min(coords[:, 0])) / range_x
        unit_coords[:, 1] = (unit_coords[:, 1] - np.min(coords[:, 1])) / range_x
        return unit_coords

    def build_neighbors_from_ie(data, simplicies):
        # n1, n2, eL, eR, fL, fR for each interior edge
        # n1, n2 = two nodes that define the edge
        # eL, eR = element on left/right as we walk from n1 to n2
        # fL, fR = local edge numbers of this edge in each element      # note: local edge f is opposite local node f
        n_elems = len(simplicies)
        neighbors = -np.ones((n_elems, 3), dtype=int)
        for row in data:
            _, _, eL, eR, fL, fR = row
            neighbors[eL, fL] = eR
            neighbors[eR, fR] = eL

        return neighbors

    def get_convex_hull(data):
        # n1, n2, bc_elems, bc_type, _  for each interior edge
        convex_hull = data[:, :2]

        return convex_hull

    if refined_level in (0, 1, 2, 3):
        coords          = np.genfromtxt(f'../mesh_files/mesh_{refined_level}/{refined_level}V.txt', dtype=float)
        simplices       = np.genfromtxt(f'../mesh_files/mesh_{refined_level}/{refined_level}E.txt', dtype=int) - 1
        interior_edge   = np.genfromtxt(f'../mesh_files/mesh_{refined_level}/{refined_level}IE.txt', dtype=int) - 1
        bc_info         = np.genfromtxt(f'../mesh_files/mesh_{refined_level}/{refined_level}BE.txt', dtype=int) - 1

        coords = normalize_coords(coords)
        neighbors = build_neighbors_from_ie(interior_edge, simplices)
        convex_hull = get_convex_hull(bc_info)
        no_faces = ufs.number_face_tri(neighbors)

        mesh = BlockData2D(coords, simplices, neighbors, no_faces)
        global_face = np.array([None])
        mesh.call_configuration(convex_hull, global_face, bc_info)

        print(f"_______________________FINISH READING UNSTRUCTURED MESH_______________________")
        print(f"Number of elements: {mesh.no_elems()}")
        print(f"Number of faces: {mesh.no_faces()}")
        print(f"Number of nodes: {mesh.no_nodes()}")

        return mesh
    else:
        raise 'no mesh file found'


def get_elems(coords):
    mean_x = 0.5 * (np.unique(coords[:, 0])[0:-1] + np.unique(coords[:, 0])[1:])
    mean_y = 0.5 * (np.unique(coords[:, 1])[0:-1] + np.unique(coords[:, 1])[1:])
    X, Y = np.meshgrid(mean_x, mean_y)
    elem_coords = np.dstack((X, Y)).reshape(-1, 2)
    return elem_coords

def get_simplices(coords, n_c, nx, ny):
    arr = np.tile(np.arange(n_c), (4, 1)).T
    arr_ = np.tile(np.array([0, 1, nx + 2, nx + 1]), (n_c, 1))
    arr__ = np.tile(np.arange(ny), (nx, 1)).T.ravel().reshape(-1, 1)
    return arr + arr_ + arr__

def get_global_face(n_v, nx, ny):
    node_idx = np.arange(n_v).reshape(-1, nx + 1)
    row_face = np.zeros((ny + 1, nx, 2), dtype = int)
    for i in range(len(row_face)):
        row_face[i] = np.column_stack((node_idx[i, :-1], node_idx[i, 1:]))
    row_face = row_face.reshape(-1, 2)

    col_face = np.zeros((ny, nx + 1, 2), dtype=int)
    for i in range(len(col_face)):
        col_face[i] = np.column_stack((node_idx[i], node_idx[i + 1]))
    col_face = col_face.reshape(-1, 2)

    return np.vstack((row_face, col_face))

def get_neighbor_quad(n_c, nx, ny):
    neighbor = np.zeros((n_c, 4), dtype = int)
    for i in range(n_c):
        neighbor[i] = np.array([i - nx, i + 1, i + nx, i - 1])
        if i % nx == 0:
            neighbor[i, 3] = -1
        if (i + 1) % nx == 0:
            neighbor[i, 1] = -1
        if (i + nx) >= n_c:
            neighbor[i, 2] = -1
        if (i - nx) < 0:
            neighbor[i, 0] = -1

    return neighbor

def get_convex_hull(coords):
    # Western
    west = np.where(coords[:, 0] < 1e-03)[0][::-1]
    southern = np.where(coords[:, 1] < 1e-03)[0]
    eastern = np.where(coords[:, 0] > np.max(coords[:, 0]) - 1e-03)[0]
    northen = np.where(coords[:, 1] > np.max(coords[:, 1]) - 1e-03)[0][::-1]

    convex_hull = np.concatenate((west, southern[1:], eastern[1:], northen[1:]))
    return np.column_stack((convex_hull[:-1], convex_hull[1:]))

def get_noise(coords, noise):
    # Identify boundary nodes
    west = coords[coords[:, 0] < 1e-06]
    east = coords[np.abs(coords[:, 0] - np.max(coords[:, 0])) < 1e-06]
    north = coords[np.abs(coords[:, 1] - np.max(coords[:, 1])) < 1e-06]
    south = coords[coords[:, 1] < 1e-06]

    boundary_nodes = np.vstack([west, east, north, south])

    interior_mask = np.ones(len(coords), dtype=bool)
    for boundary in boundary_nodes:
        interior_mask = interior_mask & ~np.all(coords == boundary, axis=1)

    interior_nodes = coords[interior_mask]

    interior_nodes_noised = interior_nodes + np.random.uniform(-noise, noise, size=interior_nodes.shape)

    coords[interior_mask] = interior_nodes_noised

    return coords
