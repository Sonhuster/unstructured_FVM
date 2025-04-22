import numpy as np
import config as cf
import bc
import utils as ufs
from input import *

"""
Symbol explanation:
    - In order to vectorize formulations, I used the combination symbols to mark array shape, it this function,
    they include three types:
        + _var  implies this array's head is shaped (number of elements) x (number of local faces).
        +  var_ implies this array's head is shaped (number of global faces) x (2).
        + _var_ implies this array's head is shaped (number of elements) x (2) x (number of local faces).
    The tail (number of elements along the latest axis) of the var array depends on its original tail, for example:
        + sn has tail == 2, so _sn = (number of elements) x (number of local faces) x (2)
        + area has tail == 1, so _area = (number of elements) x (number of local faces) x (1)
"""


def cal_face_value(mesh, fw, var, varf, bc_face=False):
    f2c, f_2_bf = ufs.shorted_name(mesh.link, 'f2c', 'f_2_bf')
    (var_,) = mesh.var_face_wise(var,)
    weighted_coeff = np.vstack((fw, 1 - fw)).T
    if not bc_face:
        maskf_bc = f_2_bf >= 0
        maskf_in = ~ maskf_bc

        # Keep face values on BCs
        varf_bc = varf * maskf_bc
        varf_in = np.sum(var_ * weighted_coeff, axis=1) * maskf_in

        return varf_bc + varf_in
    else:
        return np.sum(var_ * weighted_coeff, axis=1)


def cal_node_value(mesh, var, cw):
    var.uv = np.sum(var.uc * cw, axis=1)
    var.vv = np.sum(var.vc * cw, axis=1)
    var.pv = np.sum(var.pc * cw, axis=1)

    # Cal boundary conditions (update var_bc, var_v, var_f <- from var_v (Neumann))
    var = bc.set_bc(mesh, var, SIMULATION_MODE)

    return var


def cal_momemtum_link_coeff(mesh, ap, scx, scy, var):
    global mu
    area, delta, snsign = ufs.shorted_name(mesh.global_faces, 'area', 'delta', 'snsign')

    mask_bc_dirichlet = bc.get_fix_vel_bc_mask(mesh, SIMULATION_MODE)
    mask_bc_neumann = bc.get_outflow_bc_mask(mesh, SIMULATION_MODE)
    mask_bc, mask_in = mesh.get_face_mask_element_wise()
    (_mask_bc_dirichlet, _mask_bc_neumann) = mesh.var_elem_wise(mask_bc_dirichlet, mask_bc_neumann)
    _mask_in_dirichlet, _mask_in_neumann = ~_mask_bc_dirichlet, ~_mask_bc_neumann
    (_area, _delta, _ubc, _vbc, mf) = mesh.var_elem_wise(area, delta, var.ubc, var.vbc, var.mdotf)
    ap.fill(0.0), scx.fill(0.0), scy.fill(0.0)

    # Diagonal coefficient
    mf *= snsign
    ap = (mu * _area / _delta) * _mask_in_neumann + (0.5 * (np.abs(mf) + mf)) * _mask_in_dirichlet
    # Off-diagonal coefficient
    anb_in = (- mu * _area / _delta - 0.5 * (np.abs(mf) - mf)) * mask_in
    anb_bc = (- 0.0) * (~_mask_in_dirichlet)
    anb = anb_in + anb_bc
    # Source terms
    # scx = (_ubc * mu * _area / _delta) * _mask_bc_dirichlet + (0.5 * (np.abs(mf) + mf) * _ubc) * mask_bc
    # scy = (_vbc * mu * _area / _delta) * _mask_bc_dirichlet + (0.5 * (np.abs(mf) + mf) * _vbc) * mask_bc

    scx = (_ubc * mu * _area / _delta) * _mask_bc_dirichlet - (mf * _ubc) * mask_bc
    scy = (_vbc * mu * _area / _delta) * _mask_bc_dirichlet - (mf * _vbc) * mask_bc

    (ap, scx, scy) = ufs.get_total_flux(ap, scx, scy)
    ap = np.round(ap, 2)
    anb = np.round(anb, 2)
    scx = np.round(scx, 2)
    scy = np.round(scy, 2)

    return ap, anb, scx, scy        # TODO: Think of dirichlet BC


def cal_momentume_pressure_source(mesh, fw, scx, scy, pc, pf):
    area, sn, snsign = ufs.shorted_name(mesh.global_faces, 'area', 'sn', 'snsign')

    # Calculate pressure face value
    pf = cal_face_value(mesh, fw, pc, pf)
    (_pf, _area, _sn) = mesh.var_elem_wise(pf, area, sn)

    scx -= ufs.get_total_flux(_pf * _sn[:,:,0] * snsign * _area, )[0]
    scy -= ufs.get_total_flux(_pf * _sn[:,:,1] * snsign * _area, )[0]

    return scx, scy, pf


def cal_momentum_skew_term(uv, vv, mesh):
    # When the uniform Dirichlet bc is applied, difference of vertex value is zero, while in Neumann, it is not. Then should calculate the skew term for bc faces.
    centroid = ufs.shorted_name(mesh.elems, 'centroid')[0]
    st, snsign, delta = ufs.shorted_name(mesh.global_faces, 'st', 'snsign', 'delta')
    c2f, f2c, f2v, f_2_bf = ufs.shorted_name(mesh.link, 'c2f', 'f2c', 'f2v', 'f_2_bf')

    mask_bc_dirichlet = bc.get_fix_vel_bc_mask(mesh, SIMULATION_MODE)
    (_mask_bc_dirichlet, ) = mesh.var_elem_wise(mask_bc_dirichlet, )

    (_delta, _st, _uv, _vv, _f2c, _f2v) = mesh.var_elem_wise(delta, st, uv, vv, f2c, f2v)
    d1 = centroid[_f2c][:,:,1,:] - centroid[_f2c][:,:,0,:]
    tdotl = np.sum(_st * d1, axis = 2)

    sumfx = (tdotl * (uv[_f2v][:, :, 1] - uv[_f2v][:, :, 0]) * snsign / _delta) * _mask_bc_dirichlet
    sumfy = (tdotl * (vv[_f2v][:, :, 1] - vv[_f2v][:, :, 0]) * snsign / _delta) * _mask_bc_dirichlet

    skewx = ufs.get_total_flux(sumfx)[0] * mu
    skewy = ufs.get_total_flux(sumfy)[0] * mu

    return skewx, skewy


def solve_mom_eq(mesh, varc_, ap_, anb_, sc_, skew_, res_, tag):
    snsign = ufs.shorted_name(mesh.global_faces, 'snsign')[0]
    c2f, f2c = ufs.shorted_name(mesh.link, 'c2f', 'f2c')
    def cal_residual(mesh, varc, ap, anb, sc, skew, res, res2_init, iter_):
        (_f2c,) = mesh.var_elem_wise(f2c, )
        icn = np.where(snsign == 1, _f2c[:, :, 1], _f2c[:, :, 0])   # Neighbor cells
        sumf = ufs.get_total_flux(anb * varc[icn])[0]
        source_term = sc + skew
        res = source_term - ap * varc - sumf

        res2 = np.linalg.norm(res)

        if iter_ == 0:
            res2_init = res2

        return res, res2, res2_init

    def cal_jacobi_loop(varc, ap, res, rin_uv):
        var_tilda = res / (ap * (1.0 + rin_uv))

        assert not np.any(np.isnan(varc + var_tilda)), f"{tag} momentum - NAN value"
        assert not np.any(np.isinf(varc + var_tilda)), f"{tag} momentum - INF value"
        return varc + var_tilda

    res2_init = 0.0
    for iter_ in range(iter_mom):
        res_, res2, res2_init = cal_residual(mesh, varc_, ap_, anb_, sc_, skew_, res_, res2_init, iter_)

        varc_ = cal_jacobi_loop(varc_, ap_, res_, rin_uv = 0.1)

        if res2_init == 0.0:
            # print(f"\t{tag}-mom converged at {iter_} iteration")
            break
        if res2/res2_init < tol_inner:
            print(f"\t{tag}-mom converged at {iter_} iteration")
            break

    return varc_


def cal_massflow_face(mesh, uc, vc, pc, uf, vf, pf, ap, fw):
    """
    Calculate face mass flow using PWIM method.
    Ref: https://www.youtube.com/watch?v=4jQxtz29UQw&list=PLVuuXJfoPgT4gJcBAAFPW7uMwjFKB9aqT&t=1243s

    For boundary condition: pressure and velocity bc face value are taken from cell value (1st upwind)
    ref: https://www.youtube.com/watch?v=EnK6PkdkW4E&list=PLVuuXJfoPgT4gJcBAAFPW7uMwjFKB9aqT&index=9
    """
    # TODO: Because of the confliction between boundary conditions at conner, the mass flow will be computed incorrectly. The problem is unsolved yet. Now multiply bby the mask_in to solve for zero value of Dirichlet.

    def grad_2nd(var, delta__):
        assert len(var) == len(delta), "TypeError: Segmentation fault"
        return (var[..., 1] - var[..., 0]) / delta__

    sn, snsign, area, delta = ufs.shorted_name(mesh.global_faces, 'sn', 'snsign', 'area', 'delta')
    volume = ufs.shorted_name(mesh.elems, 'volume')[0]
    mask_in_neumann = ~ bc.get_outflow_bc_mask(mesh, SIMULATION_MODE)
    mask_bc_dirichlet = bc.get_fix_vel_bc_mask(mesh, SIMULATION_MODE)
    mask_in_dirichlet = ~ mask_bc_dirichlet

    # Weighted cell components
    (uc_, vc_) = mesh.var_face_wise(uc, vc)
    weighted_coeff = np.array([fw, 1 - fw]).T
    velf_x = np.sum(weighted_coeff * uc_, axis=1)
    velf_y = np.sum(weighted_coeff * vc_, axis=1)
    velf_ic = (np.einsum('ij,ij->i', np.array([velf_x, velf_y]).T, sn) * mask_in_dirichlet
               + np.einsum('ij,ij->i', np.array([uf, vf]).T, sn) * mask_bc_dirichlet)
    # * Should calculate the dirichlet boundary condition separately

    # Weighted cell pressure components.
    (_pf, _area, _sn, _mask_bc_dirichlet) = mesh.var_elem_wise(pf, area, sn, mask_bc_dirichlet)
    _mask_in_dirichlet = ~ _mask_bc_dirichlet
    first_order_pressure_dirichlet = _mask_bc_dirichlet * pc.reshape(-1, 1)
    _pf = _pf * _mask_in_dirichlet + first_order_pressure_dirichlet
    (_pf_, _area_, _sn_, snsign_, ap_) = mesh.var_face_wise(_pf, _area, _sn, snsign, ap) # ap is already corrected by mask_outflow
    Vdp_x = np.sum(_pf_ * _area_ * _sn_[:,:,:, 0] * snsign_, axis=2)
    Vdp_y = np.sum(_pf_ * _area_ * _sn_[:,:,:, 1] * snsign_, axis=2)
    velf_x = np.sum(weighted_coeff * Vdp_x / ap_, axis=1)
    velf_y = np.sum(weighted_coeff * Vdp_y / ap_, axis=1)
    velf_pc = np.einsum('ij,ij->i', np.array([velf_x,velf_y]).T, sn) * mask_in_neumann

    # Weighted face pressure components
    (ap_, vol_, pc_) = mesh.var_face_wise(ap, volume, pc)
    velf_pf = np.sum(weighted_coeff * vol_ / ap_, axis=1) * grad_2nd(pc_, delta) * mask_in_neumann

    vdotn = velf_ic + velf_pc - velf_pf
    mdotf = vdotn * rho * area

    return mdotf


def cal_source_mass_imbalance(mesh, sc_p, mdotf):    # Calculate source due to mass imbalance
    snsign = ufs.shorted_name(mesh.global_faces, 'snsign')[0]

    (_mdotf, ) = mesh.var_elem_wise(mdotf, )
    sc_p = - np.sum(_mdotf * snsign, axis=1)

    return sc_p

def cal_pressure_link_coeff(mesh, ap, ap_p, anb_p, fw):
    """
        For boundary condition, link coeff should be changed, because of disappearance of neighbor cells.
    """
    def find_opposite_face_neumann_cell(_mask_bc_neumann):   # only works for structured mesh
        opposite_neumann_face = np.zeros_like(_mask_bc_neumann, dtype=bool)
        opposite_faces = [2, 3, 0, 1]
        for i in range(4):
            opposite_neumann_face[:, opposite_faces[i]] = _mask_bc_neumann[:, i]
        return opposite_neumann_face


    def calculate_neumann_normal_coeff(mesh, _sn, snsign, _mask_bc_neumann):
        neumann_cells = np.where(_mask_bc_neumann == True)
        rest_faces_of_neumann = _mask_bc_neumann.copy()
        rest_faces_of_neumann[neumann_cells[0]] = ~_mask_bc_neumann[neumann_cells[0]]

        # Reference: https://www.youtube.com/watch?v=4jQxtz29UQw&list=PLVuuXJfoPgT4gJcBAAFPW7uMwjFKB9aqT&index=14
        sn_outer = _sn * _mask_bc_neumann[:,:, np.newaxis]
        sn_outer[neumann_cells[0]] = np.tile(sn_outer[neumann_cells], (mesh.no_local_faces(), 1)).reshape(-1, mesh.no_local_faces(), 2)
        sn_outer = sn_outer * rest_faces_of_neumann[:, :, np.newaxis]
        sn_inner = _sn * rest_faces_of_neumann[:, :, np.newaxis]

        snsign_outer = snsign * _mask_bc_neumann
        snsign_outer[neumann_cells[0]] = np.tile(snsign_outer[neumann_cells], (mesh.no_local_faces(), 1)).reshape(-1, mesh.no_local_faces())
        snsign_outer = snsign_outer * rest_faces_of_neumann

        return sn_outer, sn_inner, snsign_outer

    (sn, snsign, area, delta) = ufs.shorted_name(mesh.global_faces, 'sn', 'snsign', 'area', 'delta')
    c2f, f2c, f_2_bf = ufs.shorted_name(mesh.link, 'c2f', 'f2c', 'f_2_bf')
    volume = ufs.shorted_name(mesh.elems, 'volume')[0]
    ap_p.fill(0.0), anb_p.fill(0.0)

    (_f2c, _area, _delta, _sn, _fw) = mesh.var_elem_wise(f2c, area, delta, sn, fw)
    ic = np.tile(np.arange(mesh.no_elems()), (mesh.no_local_faces(), 1)).T
    icn = np.where(snsign == 1, _f2c[:, :, 1], _f2c[:, :, 0])

    mask_bc_neumann = bc.get_outflow_bc_mask(mesh, SIMULATION_MODE)
    _, mask_in = mesh.get_face_mask_element_wise()
    (_mask_bc_neumann, ) = mesh.var_elem_wise(mask_bc_neumann, )

    # # link coeff in dirichlet bc = 0
    # ap_p_dirichlet  = (  (_fw * volume[ic] / ap[ic] + (1.0 - _fw) * volume[icn] / ap[icn]) * rho * _area / _delta) * mask_in
    # anb_p_dirichlet = (- (_fw * volume[ic] / ap[ic] + (1.0 - _fw) * volume[icn] / ap[icn]) * rho * _area / _delta) * mask_in
    #
    # # # Neumann for structured mesh
    # # opposite_neumann_bc = find_opposite_face_neumann_cell(_mask_bc_neumann)
    # # ap_p_neumann  = (       _fw  * (volume / ap).reshape(-1, 1)) * rho * _area / _delta * opposite_neumann_bc
    # # anb_p_neumann = ((1.0 - _fw) * (volume / ap).reshape(-1, 1)) * rho * _area / _delta * opposite_neumann_bc
    #
    # # Neumann for arbitrary mesh
    # _sn_outer, _sn_inner, snsign_outer = calculate_neumann_normal_coeff(mesh, _sn, snsign, _mask_bc_neumann)
    # ap_p_neumann  = - (rho * _area / ap[ic]) * _sn_outer[:,:,0] * snsign_outer * (_fw * _area * _sn_inner[:, :, 0] * snsign)
    # ap_p_neumann += - (rho * _area / ap[ic]) * _sn_outer[:,:,1] * snsign_outer * (_fw * _area * _sn_inner[:, :, 1] * snsign)
    #
    # anb_p_neumann  = - (rho * _area / ap[ic]) * _sn_outer[:,:,0] * snsign_outer * ((1 - _fw) * _area * _sn_inner[:, :, 0] * snsign)
    # anb_p_neumann += - (rho * _area / ap[ic]) * _sn_outer[:,:,1] * snsign_outer * ((1 - _fw) * _area * _sn_inner[:, :, 1] * snsign)
    #
    # ap_p = np.sum(ap_p_dirichlet + ap_p_neumann, axis=1)
    # anb_p = anb_p_dirichlet + anb_p_neumann

    ap_p  = (  (_fw * volume[ic] / ap[ic] + (1.0 - _fw) * volume[icn] / ap[icn]) * rho * _area / _delta)
    anb_p = (- (_fw * volume[ic] / ap[ic] + (1.0 - _fw) * volume[icn] / ap[icn]) * rho * _area / _delta)
    ap_p = np.sum(ap_p, axis=1)

    return ap_p, anb_p


def solve_poison_eq(mesh, pcor_, ap_, anb_p_, ap_p_, sc_p_, res_p_, mdotf_, fw):
    snsign = ufs.shorted_name(mesh.global_faces, 'snsign')[0]
    c2f, f2c = ufs.shorted_name(mesh.link, 'c2f', 'f2c')

    def cal_jacobi_loop(mesh, pcor, anb_p, ap_p, sc_p):
        (_f2c,) = mesh.var_elem_wise(f2c, )
        icn = np.where(snsign == 1, _f2c[:, :, 1], _f2c[:, :, 0])  # Neighbor cells
        sumf = ufs.get_total_flux(anb_p * pcor[icn])[0]
        pcor = relax_p * (sc_p - sumf) / ap_p

        return pcor

    def cal_residual(pcor, pcor_old, res2p, iter_):
        res = np.linalg.norm(pcor - pcor_old)
        res2 = np.sqrt(np.maximum(0.0, res))
        if iter_ == 0:
            res2p = res2

        return res2, res2p

    pcor_.fill(0.0)
    sc_p_ = cal_source_mass_imbalance(mesh, sc_p_, mdotf_)
    ap_p_, anb_p_ = cal_pressure_link_coeff(mesh, ap_, ap_p_, anb_p_, fw)

    res2p = 0.0
    for iter_ in range(iter_pp):
        pcor_old = pcor_.copy()
        pcor_ = cal_jacobi_loop(mesh, pcor_, anb_p_, ap_p_, sc_p_)
        res2, res2p = cal_residual(pcor_, pcor_old, res2p, iter_)

        # print(f"poison eq: iter {iter_} residual {res2}")
        if res2p == 0:
            # print(f"\tpoison eq converged at {iter_} iteration")
            break
        if res2 / res2p < tol_inner:
            # print(f"\tpoison eq converged at {iter_} iteration")
            break
    return pcor_


def corrected_cell_vel(mesh, ucor, vcor, pcor, uc, vc, pfcor, ap, fw):
    sn, snsign, area = ufs.shorted_name(mesh.global_faces, 'sn', 'snsign', 'area')

    # Calculate correction pressure face value
    pfcor = cal_face_value(mesh, fw, pcor, pfcor, bc_face=True)
    (_pfcor, _area, _sn) = mesh.var_elem_wise(pfcor, area, sn)

    ucor.fill(0.0), vcor.fill(0.0)

    (ucor,) = ufs.get_total_flux(_pfcor * _sn[:, :, 0] * snsign * _area)
    (vcor,) = ufs.get_total_flux(_pfcor * _sn[:, :, 1] * snsign * _area)

    ucor = - ucor / ap
    vcor = - vcor / ap
    uc += relax_uv * ucor
    vc += relax_uv * vcor

    return uc, vc


def corrected_massflux(mesh, ap, fw, mdotfcor, pcor, mdotf):
    area, delta = ufs.shorted_name(mesh.global_faces, 'area', 'delta')
    volume = ufs.shorted_name(mesh.elems, 'volume')[0]
    weighted_coeff = np.array([fw, 1 - fw]).T
    (vol_, ap_, pcor_) = mesh.var_face_wise(volume, ap, pcor)
    coeff = np.einsum('ij,ij->i', weighted_coeff, vol_ * ap_)       # ap is already corrected by mask_outflow
    mdotfcor = (rho * coeff * area * (pcor_[:, 0] - pcor_[:, 1]) / delta)
    mdotf += relax_uv * mdotfcor

    return mdotf



def corrected_pressure(pc, pf, pcor, pfcor):
    pc = pc + relax_p * pcor
    # pf = pf + relax_p * pfcor
    return pc, pf


def cal_outer_res(*var_pairs):
    errors = []
    for var, var_old in var_pairs:
        denom = np.sum(np.abs(var))
        if denom > 1e-9:
            error = np.sum(np.abs(var - var_old)) / denom
        else:
            error = 0.0
        errors.append(error)
    return errors

