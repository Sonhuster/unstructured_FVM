import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pyvista as pv
import os
from input import *

def shorted_name(mesh, *attrs):
    return [getattr(mesh, attr) for attr in attrs]


def normalized_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("! Vector is zero")

    return vector / norm

def get_total_flux(*vars):
    result = []
    for var in vars:
        result.append(np.sum(var, axis=1))

    return tuple(result)


def get_coordinate(Lx, Ly, nx, ny):
    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    z = np.linspace(0, 0, int((nx + ny) / 2))

    if len(np.unique(z)) == 1:
        X, Y = np.meshgrid(x, y)
        coords = np.vstack([X.ravel(), Y.ravel()]).T
    else:
        X, Y, Z = np.meshgrid(x, y, z)
        coords = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T

    return coords


def create_tetrahedral_faces(coordinates):
    tri = Delaunay(coordinates)
    return tri

def number_face_tri(tri_neighbors):
    num_faces = 0
    for simplex_index, neighbors in enumerate(tri_neighbors):
        for neighbor in neighbors:
            if neighbor == -1:
                num_faces += 1
            elif neighbor > simplex_index:
                num_faces += 1

    return num_faces

def plot_tetrahedral_surface(coordinates, faces):
    fig = plt.figure()

    if coordinates.shape[1] == 2:
        ax = fig.add_subplot(111)
        plt.triplot(coordinates[:, 0], coordinates[:, 1], faces)
        plt.plot(coordinates[:, 0], coordinates[:, 1], 'o')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    else:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], color='r')
        for tetra in faces:
            for i in range(4):
                triangle = coordinates[np.delete(tetra, i)]  # Get triangle from 3 nodes of a tetrahedron
                ax.add_collection3d(Poly3DCollection([triangle], facecolors='cyan', linewidths=1, edgecolors='r', alpha=.25))

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.show()

def delete_redundant_vars(*var_names):
    for var_name in var_names:
        if var_name in globals():
            del globals()[var_name]
        elif var_name in locals():
            del locals()[var_name]

def plot_vtk(phi_node, mesh, field):
    nodes = np.hstack((mesh.coords, np.zeros(mesh.no_nodes()).reshape(-1, 1)))

    if mesh.no_local_faces() == 3:
        cells = np.hstack([[3] + list(e) for e in mesh.elems.define])
        celltypes = [pv.CellType.TRIANGLE] * mesh.no_elems()
    else:
        cells = np.hstack([[4] + list(e) for e in mesh.elems.define])
        celltypes = [pv.CellType.QUAD] * mesh.no_elems()

    grid = pv.UnstructuredGrid(cells, celltypes, nodes)
    grid.point_data[field] = phi_node

    plotter = pv.Plotter()
    plotter.add_mesh(grid, show_edges=True, scalars=field, cmap="jet", edge_color="black", line_width=2)
    plotter.show_axes()

    label_actor = None

    def callback(picked_point, picker):
        nonlocal label_actor
        # Đảm bảo rằng bạn lấy tọa độ từ picker, thay vì picked_point
        point = np.array(nodes[picker])  # Lấy tọa độ của điểm được chọn từ picker

        distances = np.linalg.norm(nodes - point, axis=1)
        nearest_idx = np.argmin(distances)
        value = phi_node[nearest_idx]
        point_coords = nodes[nearest_idx]

        print(f"[CLICK] Point: {point_coords}, Index: {nearest_idx}, Value = {value:.4f}")

        # Xóa label cũ nếu có
        if label_actor:
            plotter.remove_actor(label_actor)

        label_actor = plotter.add_point_labels(
            [point_coords],
            [f'{field} = {value:.4f}'],
            point_size=10,
            font_size=14,
            text_color="black",
            shape_opacity=0.3,
            always_visible=True
        )

    plotter.enable_point_picking(callback=callback, show_message=True, use_mesh=True, show_point=True)

    plotter.view_vector((0, 0, -1), viewup=(0, 1, 0))

    plotter.add_points(np.array([[0, 0, 0]]), color="black", point_size=10, render_points_as_spheres=True)
    plotter.add_point_labels([[0, 0, 0]], ['Origin'], point_size=20, text_color="black")

    plotter.show()


def write_vtk(node_value, mesh, fields, filename):
    nodes = np.hstack((mesh.coords, np.zeros(mesh.no_nodes()).reshape(-1, 1)))

    if mesh.no_local_faces() == 3:
        cells = np.hstack([[3] + list(e) for e in mesh.elems.define])
        celltypes = [pv.CellType.TRIANGLE] * mesh.no_elems()
    else:
        cells = np.hstack([[4] + list(e) for e in mesh.elems.define])
        celltypes = [pv.CellType.QUAD] * mesh.no_elems()

    grid = pv.UnstructuredGrid(cells, celltypes, nodes)

    for field_value, field_name in enumerate(fields):
        grid.point_data[field_name] = node_value[field_value]

    grid.save(filename)

    print(f"File saved as {filename}")

def write_vtk_with_streamline(node_value, mesh, fields, filename, y_range=(0.30, 0.36), n_seeds=1500, x_seed=0.05):
    global streamline_mode
    """
        fields: Should be in order: u, v, p
        y_range: Focused range where streamlines are plotted.
    """
    nodes = np.hstack((mesh.coords, np.zeros(mesh.no_nodes()).reshape(-1, 1)))

    if mesh.no_local_faces() == 3:
        cells = np.hstack([[3] + list(e) for e in mesh.elems.define])
        celltypes = [pv.CellType.TRIANGLE] * mesh.no_elems()
    else:
        cells = np.hstack([[4] + list(e) for e in mesh.elems.define])
        celltypes = [pv.CellType.QUAD] * mesh.no_elems()

    grid = pv.UnstructuredGrid(cells, celltypes, nodes)

    for field_value, field_name in enumerate(fields):
        grid.point_data[field_name] = node_value[field_value]

    if streamline_mode is False:
        filename += '.vtk'
        grid.save(filename)
        print(f"File saved as {filename}")
    else:
        if SIMULATION_MODE in (0, 2, 3):
            y_range = (np.min(mesh.coords[:, 1]), np.max(mesh.coords[:, 1]))

        u = node_value[0]
        v = node_value[1]
        vectors = np.stack((u, v, np.zeros_like(u)), axis=1)
        grid.point_data['Velocity'] = vectors

        y_min, y_max = y_range
        y_seeds = np.linspace(y_min, y_max, n_seeds)
        seeds = np.column_stack((np.full_like(y_seeds, x_seed), y_seeds, np.zeros_like(y_seeds)))
        seed_points = pv.PolyData(seeds)

        streamlines = grid.streamlines_from_source(
            seed_points,
            vectors='Velocity',
            integrator_type=45,
            max_steps=1000,
            initial_step_length=0.01,
            terminal_speed=1e-5
        )
        streamlines.clear_data()

        multi = pv.MultiBlock()
        multi["Grid"] = grid
        multi["Streamlines"] = streamlines

        head = os.path.join(os.path.split(filename)[0], 'streamline')
        tail = os.path.split(filename)[1] + '.vtm'

        filename = os.path.join(head, tail)
        multi.save(filename)

        print(f"Streamline saved as {filename}")