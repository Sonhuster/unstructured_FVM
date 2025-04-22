import numpy as np
import config as cf
import utils as ufs
import solver as sol
import initialization as init
from input import *


if __name__ == '__main__':
    if READ_MESH is False and SIMULATION_MODE in (0, 2, 3):
        coordinates = ufs.get_coordinate(Lx=5.0, Ly=2.0, nx=51, ny=21)
        mesh = cf.from_coords_to_mesh(coordinates, noise, element_type = "QUAD")     # Element type "UPPER CASE"
        del coordinates
    elif READ_MESH is True and SIMULATION_MODE in (1, ):
        mesh = cf.loadfile(refined_level=2)
    else:
        raise "Incorrectly defined input file"

    # Node and face weighted coefficients
    fw = cf.cell_to_face_interpolation(mesh)
    cw = cf.cell_to_node_interpolation(mesh)

    # Allocate memory && initialization
    print("_______________________START INITIALIZATION_______________________")
    var = init.Fluid.init_fluid(mesh, u_inf, v_inf, 0.0)
    scx, scy, skewx, skewy, ap, res, anb = init.momentum_equation_arg(mesh)
    sc_p, ap_p, res_p, anb_p = init.possion_equation_arg(mesh)
    ucor, vcor, pcor, mdotfcor, pfcor = init.vel_correction_arg(mesh)
    print("_______________________FINISH INITIALIZATION_______________________")

    uc_old, vc_old = var.uc.copy(), var.vc.copy()

    print("_______________________START SOLVING_______________________")
    for iter_ in range(iter_outer):
        var = sol.cal_node_value(mesh, var, cw)
        # Cal momentum equation coefficients
        ap, anb, scx, scy = sol.cal_momemtum_link_coeff(mesh, ap, scx, scy, var)
        scx, scy, var.pf = sol.cal_momentume_pressure_source(mesh, fw, scx, scy, var.pc, var.pf)
        skewx, skewy = sol.cal_momentum_skew_term(var.uv, var.vv, mesh)

        # Solve mom-equation
        var.uc = sol.solve_mom_eq(mesh, var.uc, ap, anb, scx, skewx, res, tag = "X")
        var.vc = sol.solve_mom_eq(mesh, var.vc, ap, anb, scy, skewy, res, tag = "Y")

        # Solve poison equation
        var.mdotf = sol.cal_massflow_face(mesh, var.uc, var.vc,  var.pc, var.uf, var.vf, var.pf, ap, fw)
        pcor = sol.solve_poison_eq(mesh, pcor, ap, anb_p, ap_p, sc_p, res_p, var.mdotf, fw)

        # Velocity and pressure corrected
        var.uc, var.vc = sol.corrected_cell_vel(mesh, ucor, vcor, pcor, var.uc, var.vc, pfcor, ap, fw)
        var.mdotf = sol.corrected_massflux(mesh, ap, fw, mdotfcor, pcor, var.mdotf)
        var.pc, _ = sol.corrected_pressure(var.pc, var.pf, pcor, pfcor)  # TODO: should update pf?

        # Calculate residual
        [error_u, error_v] = sol.cal_outer_res((var.uc, uc_old), (var.vc, vc_old))
        uc_old, vc_old = var.uc.copy(), var.vc.copy()
        print(f"Outer iteration {iter_} - residual (u, v) = ({error_u}, {error_v})")
        if error_u < 1e-05 and error_v < 1e-05:
            break

        if iter_ % int(n_plot) == 0:
            filename = f"../paraview/para_{iter_}"
            var = sol.cal_node_value(mesh, var, cw)
            ufs.write_vtk_with_streamline([var.uv, var.vv, var.pv], mesh,
                                          ["Velocity U", "Velocity V", "Pressure"], filename)

    var = sol.cal_node_value(mesh, var, cw)
    ufs.plot_vtk(var.uv, mesh, "Velocity U")
    ufs.plot_vtk(var.vv, mesh, "Velocity V")
    ufs.plot_vtk(var.pv, mesh, "Pressure")
    print("_______________________FINISH CASE_______________________")

    # # TODO: Investigate the remaining parameter (such as pressure relaxation)
    # # TODO: Calculate the outer residual
