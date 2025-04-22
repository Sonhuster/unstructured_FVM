import numpy as np
from input import *

# --------------------------Initialization------------------------------ #
class Fluid:
    def __init__(self, mesh, init_u, init_v, init_p):
        n_cells = mesh.no_elems()
        n_faces = mesh.no_faces()
        n_nodes = mesh.no_nodes()

        self.uc = np.zeros(n_cells) + 0.0
        self.vc = np.zeros(n_cells) + init_v
        self.pc = np.zeros(n_cells) + init_p

        self.uf = np.zeros(n_faces)
        self.vf = np.zeros(n_faces)
        self.pf = np.zeros(n_faces)
        self.mdotf = np.zeros(n_faces)

        self.uv = np.zeros(n_nodes)
        self.vv = np.zeros(n_nodes)
        self.pv = np.zeros(n_nodes)

        self.ubc = np.zeros(len(mesh.boundary_info.faces))
        self.vbc = np.zeros(len(mesh.boundary_info.faces))
        self.pbc = np.zeros(len(mesh.boundary_info.faces))

    @classmethod
    def init_fluid(cls, mesh, init_u, init_v, init_p):
        obj = Fluid(mesh, init_u, init_v, init_p)
        area = mesh.global_faces.area[mesh.boundary_info.face_patches['left_face']]
        snsign_inlet = mesh.global_faces.snsign[np.isin(mesh.link.c2f, mesh.boundary_info.face_patches['left_face'])]
        obj.mdotf[mesh.boundary_info.face_patches['left_face']] = rho * init_u * area * snsign_inlet
        return obj


def momentum_equation_arg(mesh):
    n_cells = mesh.no_elems()
    n_local_faces = mesh.no_local_faces()

    scx = np.zeros(n_cells)
    scy = np.zeros(n_cells)
    skewx = np.zeros(n_cells)
    skewy = np.zeros(n_cells)
    ap = np.zeros(n_cells)
    res = np.zeros(n_cells)
    anb = np.zeros((n_cells, n_local_faces))
    return scx, scy, skewx, skewy, ap, res, anb

def possion_equation_arg(mesh):
    n_cells = mesh.no_elems()
    n_local_faces = mesh.no_local_faces()

    sc_p = np.zeros(n_cells)
    ap_p = np.zeros(n_cells)
    res_p = np.zeros(n_cells)
    anb_p = np.zeros((n_cells, n_local_faces))

    return sc_p, ap_p, res_p, anb_p

def vel_correction_arg(mesh):
    n_cells = mesh.no_elems()
    n_faces = mesh.no_faces()

    ucor = np.zeros(n_cells)
    vcor = np.zeros(n_cells)
    pcor = np.zeros(n_cells)
    mdotfcor = np.zeros(n_faces)
    pfcor = np.zeros(n_faces)

    return ucor, vcor, pcor, mdotfcor, pfcor