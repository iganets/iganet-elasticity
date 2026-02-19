import numpy as np
import splinepy
import gustaf as gus
import json


def generate_knot_vector(degree, nrCtrlPts):
    num_knots = nrCtrlPts + degree + 1 
    num_repeats = degree + 1 
    num_inner_knots = num_knots - 2 * num_repeats 

    # place the starting p+1 knots
    knot_vector = [0] * num_repeats

    # place the inner knots
    if num_inner_knots > 0:
        step = 1 / (nrCtrlPts - degree)
        inner_knots = [i * step for i in range(1, num_inner_knots + 1)]
        knot_vector.extend(inner_knots)

    # place the ending p+1 knots
    knot_vector.extend([1] * num_repeats)

    # return tuple of knot vectors
    return (knot_vector, knot_vector)

def has_key_nonempty(d, key):
    return (key in d) and (d[key] is not None) and (len(d[key]) > 0)

def as_ctrlpts(d, key):
    arr = np.array(d[key])
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"'{key}' has unexpected shape {arr.shape}. Expected (n, dim>=2).")
    return arr

def nr_ctrlpts_1d_from_flat(ctrlpts, key):
    n = ctrlpts.shape[0]
    r = int(np.sqrt(n))
    if r * r != n:
        raise ValueError(f"'{key}' has {n} control points; expected a perfect square for a 2D tensor grid.")
    return r

""" PREPARE DATA """

import json
import numpy as np
import splinepy

# read data
with open("result.json", "r") as file:
    data = json.load(file)

# degrees of the splines
deg = data["net_Degree"]


""" CREATE SPLINES """

show_items = []

# --- G+Smo original (optional)
if has_key_nonempty(data, "gsOriginCtrlPts"):
    control_points_gs_original = as_ctrlpts(data, "gsOriginCtrlPts")
    nr_ctrl_pts_gs_original = nr_ctrlpts_1d_from_flat(control_points_gs_original, "gsOriginCtrlPts")
    kv_gs_original = generate_knot_vector(deg, nr_ctrl_pts_gs_original)

    BSpline_gs_original = splinepy.BSpline(
        degrees=(deg, deg),
        knot_vectors=kv_gs_original,
        control_points=control_points_gs_original,
    )

    show_items.append(["G+Smo Original Control Points", BSpline_gs_original])

# --- G+Smo deformed/current (optional)
if has_key_nonempty(data, "gsCtrlPts"):
    control_points_gs = as_ctrlpts(data, "gsCtrlPts")
    nr_ctrl_pts_gs = nr_ctrlpts_1d_from_flat(control_points_gs, "gsCtrlPts")
    kv_gs = generate_knot_vector(deg, nr_ctrl_pts_gs)

    BSpline_gs = splinepy.BSpline(
        degrees=(deg, deg),
        knot_vectors=kv_gs,
        control_points=control_points_gs,
    )

    show_items.append(["G+Smo Control Points", BSpline_gs])

# --- Standard Collocation
control_points_stdColl = as_ctrlpts(data, "stdCollCtrlPts")
nr_ctrl_pts_stdColl = nr_ctrlpts_1d_from_flat(control_points_stdColl, "stdCollCtrlPts")
kv_coll = generate_knot_vector(deg, nr_ctrl_pts_stdColl)

BSpline_coll = splinepy.BSpline(
    degrees=(deg, deg),
    knot_vectors=kv_coll,
    control_points=control_points_stdColl,
)
show_items.append(["Collocation Control Points", BSpline_coll])

# --- Net
control_points_net = as_ctrlpts(data, "net_CtrlPts")
nr_ctrl_pts_net = nr_ctrlpts_1d_from_flat(control_points_net, "net_CtrlPts")
kv_net = generate_knot_vector(deg, nr_ctrl_pts_net)

BSpline_net = splinepy.BSpline(
    degrees=(deg, deg),
    knot_vectors=kv_net,
    control_points=control_points_net,
)
show_items.append(["IgANet Control Points", BSpline_net])

# show whatever exists
gus.show(
    *show_items,
    control_mesh=False,
    control_point_ids=False,
)

exit()

BSpline_gs_displacement_refined = BSpline_gs_displacement.copy()
BSpline_ref_gs_displacement_refined = BSpline_ref_gs_displacement.copy()
# BSpline_gs_displacement_refined.insert_knots(0, kv_insert)
# BSpline_gs_displacement_refined.insert_knots(1, kv_insert)
BSpline_gs_displacement_refined.insert_knots(0, kv_ref_gs[0])
BSpline_gs_displacement_refined.insert_knots(1, kv_ref_gs[0])
BSpline_ref_gs_displacement_refined.insert_knots(0, kv_gs[0])
BSpline_ref_gs_displacement_refined.insert_knots(1, kv_gs[0])
# print(len(BSpline_gs_displacement_refined.knot_vectors[0]))
# print(len(BSpline_ref_gs_displacement_refined.knot_vectors[0]))
# kv_gs[0] = np.round(kv_gs[0], 6)
# BSpline_gs_displacement_refined.remove_knots(0, np.round(kv_gs[0][5:-5]), 0.1)
# print(BSpline_gs_displacement_refined.knot_vectors[0])
# print(BSpline_ref_gs_displacement_refined.knot_vectors[0])
# print(len(BSpline_gs_displacement_refined.knot_vectors[0]))
# print(len(BSpline_gs_displacement_refined.control_points))
# print(len(BSpline_ref_gs_displacement_refined.control_points))


cp_array = BSpline_ref_gs_displacement_refined.control_points - BSpline_gs_displacement_refined.control_points


BSpline_difference = splinepy.BSpline(
    degrees=(deg_ref_gs, deg_ref_gs),
    knot_vectors=BSpline_ref_gs_displacement_refined.knot_vectors,
    control_points = cp_array,
)

# gus.show(
#     ["Difference Spline", BSpline_difference],
#     # ["G+Smo Reference Control Points", BSpline_ref_gs_original],
#     control_mesh=False,
#     control_point_ids=False,
# )

# # displacements
# gus.show(
#     ["G+Smo Displacement Control Points", BSpline_gs_displacement],
#     ["G+Smo Reference Displacement Control Points", BSpline_ref_gs_displacement],
# )
# create G+Smo reference spline with high resolution
BSpline_ref_gs = splinepy.BSpline(
    degrees=(deg_ref_gs, deg_ref_gs),
    knot_vectors=kv_ref_gs,
    control_points=control_points_ref_gs,
)
BSpline_ref_gs.show_options["knot_lw"] = 0.001
BSpline_ref_gs.show_options["knot_c"] = "black"
BSpline_ref_gs.show_options["control_points"] = True
BSpline_ref_gs.show_options["control_point_r"] = 0.003
BSpline_ref_gs.show_options["control_point_ids"] = False
BSpline_ref_gs.show_options["control_mesh"] = False
gus.show(
    ["Reference Solution Control Points", BSpline_ref_gs],
    control_point_ids=False,
    control_mesh=False,
)
# create Collocation reference spline with high resolution
BSpline_ref_coll = splinepy.BSpline(
    degrees=(deg_ref_coll, deg_ref_coll),
    knot_vectors=kv_ref_coll,
    control_points=control_points_ref_coll,)
BSpline_ref_coll.show_options["knot_lw"] = 0.005
BSpline_ref_coll.show_options["knot_c"] = "black"
BSpline_ref_coll.show_options["control_points"] = False
BSpline_ref_coll.show_options["control_point_ids"] = False

# create G+Smo solution spline
BSpline_gs = splinepy.BSpline(
    degrees=(deg_net, deg_net),
    knot_vectors=kv_gs,
    control_points=control_points_gs,
)
BSpline_gs.show_options["knot_lw"] = 0.005
BSpline_gs.show_options["knot_c"] = "black"
BSpline_gs.show_options["control_point_ids"] = False
BSpline_gs.show_options["control_mesh"] = False

# create Matlab solution spline
BSpline_matlab = splinepy.BSpline(
    degrees=(deg_matlab, deg_matlab),
    knot_vectors=kv_matlab,
    control_points=control_points_matlab,
)
BSpline_matlab.show_options["knot_lw"] = 0.005
BSpline_matlab.show_options["knot_c"] = "black"
BSpline_matlab.show_options["control_point_ids"] = False
BSpline_matlab.show_options["control_mesh"] = False

# create IgANet solution spline
BSpline_net = splinepy.BSpline(
    degrees=(deg_net, deg_net),
    knot_vectors=kv_net,
    control_points=control_points_net,
)
BSpline_net.show_options["knot_lw"] = 0.005
BSpline_net.show_options["knot_c"] = "black"
BSpline_net.show_options["control_point_ids"] = False
BSpline_net.show_options["control_mesh"] = False

exit()

# ' CREATING ORIGINAL WAVE SPLINE '
# deformed_cps = control_points_gs_original.copy()
# for lin_idx in range(deformed_cps.shape[0]):
#     deformed_cps[lin_idx, 0] *= 0.9
#     deformed_cps[lin_idx, 1] *= 1.1

# BSpline_wave_original = splinepy.BSpline(
#     degrees=(deg_gs, deg_gs),
#     knot_vectors=kv_gs,
#     control_points=deformed_cps,
# )
# BSpline_wave_original.show_options["knot_lw"] = 0.005
# BSpline_wave_original.show_options["knot_c"] = "black"
# BSpline_wave_original.show_options["control_point_ids"] = False
# BSpline_wave_original.show_options["control_mesh"] = False

# gus.show(
#     ["Wave Original Control Points", BSpline_wave_original],
#     control_point_ids=False,
#     control_mesh=False,
# )

# splinepy.io.svg.export("rect_original.svg", BSpline_wave_original)
# exit()

# gus.show(
#     # ["IgANet Solution", BSpline_net],
#     # ["matlab's Control Points", BSpline_matlab],
#     control_mesh=False,
#     control_point_ids=False,
#     size=(450, 300),
#     cam = dict(
#         position=(1.01819, 0.530040, 2.66613),
#         focal_point=(1.01819, 0.530040, 0),
#         viewup=(0, 1.00000, 0),
#         roll=0,
#         distance=2.66613,
#         clipping_range=(2.49821, 2.88625),
#     )
# )

""" STRESS PLOTS """

vm_stress_distribution_gs = np.array(data["gsStresses"])
vm_stress_distribution_ref_gs = np.array(data["gsRefStresses"]).reshape(-1, 1)
vm_stress_distribution_matlab = np.array(data["matlabStresses"]).reshape(-1, 1)

vm_stress_distribution_net = np.array(data["netVmStresses"])
x_stress_distribution_net = np.array(data["netXStresses"])
y_stress_distribution_net = np.array(data["netYStresses"])

# create stress distribution splines
vm_stress_spline_ref = splinepy.BSpline(
    degrees=(1,1),
    knot_vectors=kv_coll_ref,
    control_points=vm_stress_distribution_ref_gs,)

vm_stress_spline_gs = splinepy.BSpline(
    degrees=(1,1),
    knot_vectors=kv_coll_gs,
    control_points=vm_stress_distribution_gs,)

vm_stress_spline_matlab = splinepy.BSpline(
    degrees=(1,1),
    knot_vectors=kv_coll_matlab,
    control_points=vm_stress_distribution_matlab,)

vm_stress_spline_net = splinepy.BSpline(
    degrees=(1,1),
    knot_vectors=kv_coll_net,
    control_points=vm_stress_distribution_net,)

x_stress_spline_net = splinepy.BSpline(
    degrees=(1,1),
    knot_vectors=kv_coll_net,
    control_points=x_stress_distribution_net,)

y_stress_spline_net = splinepy.BSpline(
    degrees=(1,1),
    knot_vectors=kv_coll_net,
    control_points=y_stress_distribution_net,)

# assign stress splines to geometry splines
BSpline_vm_stress_ref_gs = BSpline_ref_gs.copy()
BSpline_vm_stress_ref_gs.spline_data["gs_ref_vm_stress"] = vm_stress_spline_ref
BSpline_vm_stress_ref_gs.show_options["data"] = "gs_ref_vm_stress"
BSpline_vm_stress_ref_gs.show_options["control_points"] = False
BSpline_vm_stress_ref_gs.show_options["scalarbar"] = True
BSpline_vm_stress_ref_gs.show_options["knot_lw"] = 0.001
BSpline_vm_stress_ref_gs.show_options["knot_c"] = "black"

BSpline_vm_stress_gs = BSpline_gs.copy()
BSpline_vm_stress_gs.spline_data["gs_vm_stress"] = vm_stress_spline_gs
BSpline_vm_stress_gs.show_options["data"] = "gs_vm_stress"
BSpline_vm_stress_gs.show_options["control_points"] = False
BSpline_vm_stress_gs.show_options["scalarbar"] = True
# BSpline_vm_stress_gs.show_options["vmin"] = 180
# BSpline_vm_stress_gs.show_options["vmax"] = 280
BSpline_vm_stress_gs.show_options["knot_lw"] = 0.005
BSpline_vm_stress_gs.show_options["knot_c"] = "black"

BSpline_vm_stress_matlab = BSpline_matlab.copy()
BSpline_vm_stress_matlab.spline_data["matlab_vm_stress"] = vm_stress_spline_matlab
BSpline_vm_stress_matlab.show_options["data"] = "matlab_vm_stress"
BSpline_vm_stress_matlab.show_options["control_points"] = False
BSpline_vm_stress_matlab.show_options["scalarbar"] = True
BSpline_vm_stress_matlab.show_options["knot_lw"] = 0.3
# BSpline_vm_stress_matlab.show_options["vmin"] = 180
# BSpline_vm_stress_matlab.show_options["vmax"] = 280
BSpline_vm_stress_matlab.show_options["knot_lw"] = 0.005
BSpline_vm_stress_matlab.show_options["knot_c"] = "black"

BSpline_vm_stress_net = BSpline_net.copy()
BSpline_vm_stress_net.spline_data["net_vm_stress"] = vm_stress_spline_net
BSpline_vm_stress_net.show_options["data"] = "net_vm_stress"
BSpline_vm_stress_net.show_options["control_points"] = False
BSpline_vm_stress_net.show_options["scalarbar"] = True
BSpline_vm_stress_net.show_options["knot_lw"] = 0.3
# BSpline_vm_stress_net.show_options["vmin"] = 200
# BSpline_vm_stress_net.show_options["vmax"] = 220
BSpline_vm_stress_net.show_options["knot_lw"] = 0.005
BSpline_vm_stress_net.show_options["knot_c"] = "black"

BSpline_x_stress_net = BSpline_net.copy()
BSpline_x_stress_net.spline_data["x_stress"] = x_stress_spline_net
BSpline_x_stress_net.show_options["data"] = "x_stress"
BSpline_x_stress_net.show_options["control_points"] = False
BSpline_x_stress_net.show_options["scalarbar"] = True
# BSpline_x_stress_net.show_options["vmin"] = 208
# BSpline_x_stress_net.show_options["vmax"] = 212

BSpline_y_stress_net = BSpline_net.copy()
BSpline_y_stress_net.spline_data["y_stress"] = y_stress_spline_net
BSpline_y_stress_net.show_options["data"] = "y_stress"
BSpline_y_stress_net.show_options["control_points"] = False
BSpline_y_stress_net.show_options["scalarbar"] = True

# show splines and stress corresponding to the splines
gus.show(
    ["Reference Solution", BSpline_ref_gs],    
    ["G+Smo Solution", BSpline_gs],
    ["Matlab Solution", BSpline_matlab],
    ["IgANet Solution", BSpline_net],
    ["Reference vM Stress", BSpline_vm_stress_ref_gs],
    ["G+Smo vM Stress", BSpline_vm_stress_gs],
    ["Matlab vM Stress", BSpline_vm_stress_matlab],
    ["IgANet vM Stress", BSpline_vm_stress_net],
    control_mesh=False,
    control_point_ids=False,
    size=(1800, 600),
    cam = dict(
        position=(1.08802, 0.557011, 2.89474),
        focal_point=(1.08802, 0.557011, 0),
        viewup=(0, 1.00000, 0),
        roll=0,
        distance=2.89474,
        clipping_range=(2.71144, 3.13246),
    ),
    )

# stress plots in x and y direction
# gus.show(
#     ["IgANet vM Stress", BSpline_vm_stress_net],
#     ["IgANet x Stress", BSpline_x_stress_net],
#     ["IgANet y Stress", BSpline_y_stress_net],
#     control_mesh=False,
#     control_point_ids=False,
#     size=(1350, 300),
#     cam = dict(
#         position=(1.11048, 0.526086, 2.93173),
#         focal_point=(1.11048, 0.526086, 0),
#         viewup=(0, 1.00000, 0),
#         roll=0,
#         distance=2.93173,
#         clipping_range=(2.74609, 3.17249),
#     )
# )

""" DIFFERENCE PLOTS """

mesh_resolution = 1000
ptu = np.linspace(0, 1, mesh_resolution)
ptv = np.linspace(0, 1, mesh_resolution)  

u_mesh, v_mesh =    np.meshgrid(ptu, ptv)
queries_mesh =      np.column_stack([u_mesh.ravel(), v_mesh.ravel()])

ref_points_gs =     BSpline_ref_gs.evaluate(queries_mesh)
ref_points_coll=    BSpline_ref_coll.evaluate(queries_mesh)
gs_points =         BSpline_gs.evaluate(queries_mesh)
matlab_points =     BSpline_matlab.evaluate(queries_mesh)
net_points =        BSpline_net.evaluate(queries_mesh)

# reference to G+Smo solution
ref_diff_gs =       ref_points_gs - gs_points
ref_diff_collToGs = ref_points_gs - matlab_points
# reference to collocation solution
ref_diff_matlab =   ref_points_coll - matlab_points
ref_diff_net =      ref_points_coll - net_points

abs_error_gs =      np.linalg.norm(ref_diff_gs, axis=1).reshape(-1,1)
abs_error_matlab =  np.linalg.norm(ref_diff_matlab, axis=1).reshape(-1,1)
abs_error_net =     np.linalg.norm(ref_diff_net, axis=1).reshape(-1,1)
abs_error_collToGs = np.linalg.norm(ref_diff_collToGs, axis=1).reshape(-1,1)

# # calculate MSE
mse_gs = np.mean((ref_diff_gs) ** 2)
mse_matlab = np.mean((ref_diff_matlab) ** 2)
mse_net = np.mean((ref_diff_net) ** 2)
mse_collToGs = np.mean((ref_diff_collToGs) ** 2)

print("MSE of G+Smo Solution:   ", mse_gs)
print("MSE of Matlab Solution:  ", mse_matlab)
print("MSE of IgANet Solution:  ", mse_net)
print("MSE of CollToGS Solution:", mse_collToGs)

# create absolute error splines
kv_mesh_diff = generate_knot_vector(1, mesh_resolution)

abs_error_spline_gs = splinepy.BSpline(
    degrees=(1, 1),
    knot_vectors=kv_mesh_diff,
    control_points=abs_error_gs,)

abs_error_spline_matlab = splinepy.BSpline(
    degrees=(1, 1),
    knot_vectors=kv_mesh_diff,
    control_points=abs_error_matlab,)

abs_error_spline_net = splinepy.BSpline(
    degrees=(1, 1),
    knot_vectors=kv_mesh_diff,
    control_points=abs_error_net,)

# assign stress splines to geometry splines
BSpline_queries_gs = BSpline_gs.copy()
BSpline_queries_gs.spline_data["ref_diff_gs"] = abs_error_spline_gs
BSpline_queries_gs.show_options["data"] = "ref_diff_gs"
BSpline_queries_gs.show_options["control_points"] = False
BSpline_queries_gs.show_options["scalarbar"] = True
BSpline_queries_gs.show_options["knot_lw"] = 0.005
BSpline_queries_gs.show_options["knot_c"] = "black"

BSpline_queries_matlab = BSpline_matlab.copy()
BSpline_queries_matlab.spline_data["ref_diff_matlab"] = abs_error_spline_matlab
BSpline_queries_matlab.show_options["data"] = "ref_diff_matlab"
BSpline_queries_matlab.show_options["control_points"] = False
BSpline_queries_matlab.show_options["scalarbar"] = True
BSpline_queries_matlab.show_options["knot_lw"] = 0.005
BSpline_queries_matlab.show_options["knot_c"] = "black"

BSpline_queries_net = BSpline_net.copy()
BSpline_queries_net.spline_data["ref_diff_net"] = abs_error_spline_net
BSpline_queries_net.show_options["data"] = "ref_diff_net"
BSpline_queries_net.show_options["control_points"] = False
BSpline_queries_net.show_options["scalarbar"] = True
BSpline_queries_net.show_options["knot_lw"] = 0.005
BSpline_queries_net.show_options["knot_c"] = "black"

# gus.show(
#          ["G+Smo Absolute Error", BSpline_queries_gs],
#          ["Matlab Absolute Error", BSpline_queries_matlab],
#          ["IgANet Absolute Error", BSpline_queries_net],
#          control_mesh=False,
#          control_point_ids=False,
#         cam = dict(
#             position=(1.13969, 0.490378, 2.96394),
#             focal_point=(1.13969, 0.490378, 0),
#             viewup=(0, 1.00000, 0),
#             roll=0,
#             distance=2.96394,
#             clipping_range=(2.77626, 3.20734),
#         ),
#          size = (1350, 300),
#          )


""" ELASTICITY ERROR PLOTS"""

# read net divergence
net_div_x = np.array(data["netDivergenceX"])
net_div_y = np.array(data["netDivergenceY"])

# add #nr_coll_pts_net zeros to net_div_x and net_div_y at the start
net_div_x = np.append(np.zeros(nr_coll_pts_net), net_div_x)
net_div_y = np.append(np.zeros(nr_coll_pts_net), net_div_y)
# add #nr_coll_pts_net zeros to net_div_x and net_div_y at the end
net_div_x = np.append(net_div_x, np.zeros(nr_coll_pts_net))
net_div_y = np.append(net_div_y, np.zeros(nr_coll_pts_net))
# add a zero at every #nr_coll_pts_net-2 position
net_div_x = np.insert(net_div_x, np.arange(nr_coll_pts_net, 
                             net_div_x.size-nr_coll_pts_net, nr_coll_pts_net-2), 0)
net_div_y = np.insert(net_div_y, np.arange(nr_coll_pts_net, 
                             net_div_y.size-nr_coll_pts_net, nr_coll_pts_net-2), 0)
# add a zero at every #nr_coll_pts_net-1 position
net_div_x = np.insert(net_div_x, np.arange(2*nr_coll_pts_net, 
                             net_div_x.size-nr_coll_pts_net, nr_coll_pts_net-1), 0)
net_div_y = np.insert(net_div_y, np.arange(2*nr_coll_pts_net, 
                             net_div_y.size-nr_coll_pts_net, nr_coll_pts_net-1), 0)
# add a zero at the end
net_div_x = np.append(net_div_x, 0)
net_div_y = np.append(net_div_y, 0)

# normalize divergence
# net_div_x = np.abs(net_div_x)
# net_div_y = np.abs(net_div_y)

# calculate total divergence
net_div = np.sqrt(net_div_x**2 + net_div_y**2)

# # calculate logarithm of divergence
# net_div_x = np.log10(net_div_x)
# net_div_y = np.log10(net_div_y)
# net_div = np.log10(net_div)

# reshape divergence
net_div_x = net_div_x.reshape(-1, 1)
net_div_y = net_div_y.reshape(-1, 1)
net_div = net_div.reshape(-1, 1)

# create divergence distribution splines
net_div_x_spline = splinepy.BSpline(
    degrees=(1,1),
    knot_vectors=kv_coll_net,
    control_points=net_div_x,)

net_div_y_spline = splinepy.BSpline(
    degrees=(1,1),
    knot_vectors=kv_coll_net,
    control_points=net_div_y,)

net_div_spline = splinepy.BSpline(
    degrees=(1,1),
    knot_vectors=kv_coll_net,
    control_points=net_div,)

# assign divergence splines to geometry splines
BSpline_net_div_x = BSpline_net.copy()
BSpline_net_div_x.spline_data["divergence_x"] = net_div_x_spline
BSpline_net_div_x.show_options["data"] = "divergence_x"
BSpline_net_div_x.show_options["control_points"] = False
BSpline_net_div_x.show_options["scalarbar"] = True
BSpline_net_div_x.show_options["knot_lw"] = 0.3
# BSpline_net_div_x.show_options["vmin"] = 0
# BSpline_net_div_x.show_options["vmax"] = 4

BSpline_net_div_y = BSpline_net.copy()
BSpline_net_div_y.spline_data["divergence_y"] = net_div_y_spline
BSpline_net_div_y.show_options["data"] = "divergence_y"
BSpline_net_div_y.show_options["control_points"] = False
BSpline_net_div_y.show_options["scalarbar"] = True
BSpline_net_div_y.show_options["knot_lw"] = 0.3

BSpline_net_div = BSpline_net.copy()
BSpline_net_div.spline_data["divergence"] = net_div_spline
BSpline_net_div.show_options["data"] = "divergence"
BSpline_net_div.show_options["control_points"] = False
BSpline_net_div.show_options["scalarbar"] = True
BSpline_net_div.show_options["knot_lw"] = 0.005
BSpline_net_div.show_options["knot_c"] = "black"


# # show splines
# gus.show(
#     # ["Elasticity Error in x", BSpline_net_div_x],
#     # ["Elasticity Error in y", BSpline_net_div_y],
#     ["Total Elasticity Error", BSpline_net_div],
#     control_mesh=False,
#     control_point_ids=False,
#     cam = dict(
#         position=(1.14990, 0.536767, 2.17484),
#         focal_point=(1.14990, 0.536767, 0),
#         viewup=(0, 1.00000, 0),
#         roll=0,
#         distance=2.17484,
#         clipping_range=(2.03713, 2.35344),
#     ),
#     size=(600, 300),
#     )

""" POISSON RATIO PLOTS """

# create poisson distribution spline
poisson_distribution_net = np.array(data["netPoisson"])
poisson_spline_net = splinepy.BSpline(
    degrees=(1,1),
    knot_vectors=kv_coll_net,
    control_points=poisson_distribution_net,)

BSpline_net_poisson = BSpline_net.copy()
BSpline_net_poisson.spline_data["poisson"] = poisson_spline_net
BSpline_net_poisson.show_options["data"] = "poisson"
BSpline_net_poisson.show_options["control_points"] = False
BSpline_net_poisson.show_options["scalarbar"] = True
BSpline_net_poisson.show_options["knot_lw"] = 0.005
BSpline_net_poisson.show_options["knot_c"] = "black"
# BSpline_net_poisson.show_options["vmin"] = 0.0
# BSpline_net_poisson.show_options["vmax"] = 0.5

# gus.show(
#     ["Poisson Ratio", BSpline_net_poisson],
#     control_mesh=False,
#     control_point_ids=False,
#     cam = dict(
#         position=(1.11882, 0.540652, 2.17484),
#         focal_point=(1.11882, 0.540652, 0),
#         viewup=(0, 1.00000, 0),
#         roll=0,
#         distance=2.17484,
#         clipping_range=(2.03712, 2.35344),
#     ),
#     size=(600, 300),
#     )

""" EXPORT SVGS """
def get_svg_kwargs(vmin, vmax):
    return {
        "scalarbar": True,
        "n_ticks": 2,
        "vmin": vmin,
        "vmax": vmax,
        "scalarbar_width": 0.4,
        "scalarbar_font_size": 0.08,
        "font_family": "Times New Roman",
        "cmap": "plasma",
    }

write = False
if write:
    # geometriesq
    # BSpline_ref_gs.reduce_degrees([0,1],1)
    # BSpline_ref_gs.reduce_degrees([0],1)    
    # BSpline_ref_gs.reduce_degrees([1],1)    

    splinepy.io.svg.export(f"11a_bspline_ref_gs_cp{nr_ctrl_pts_ref_gs}_p{deg_ref_gs}.svg",
                            BSpline_ref_gs,
                            )
    
    # BSpline_gs.reduce_degrees([0],1)
    splinepy.io.svg.export(f"11b_bspline_gs_cp{nr_ctrl_pts_gs}_p{deg_gs}.svg",
                            BSpline_gs,
                            )
    
    # BSpline_matlab.reduce_degrees([0,1],1)
    # BSpline_matlab.reduce_degrees([0],1)
    splinepy.io.svg.export(f"11c_bspline_matlab_cp{nr_ctrl_pts_matlab}_p{deg_matlab}.svg",
                            BSpline_matlab,
                            )
    
    # BSpline_net.reduce_degrees([0,1],1)
    # BSpline_net.reduce_degrees([0],1)
    splinepy.io.svg.export(f"11d_bspline_net_cp{nr_ctrl_pts_net}_p{deg_net}.svg",
                            BSpline_net,
                            )
    
    # stress distributions
    
    # BSpline_vm_stress_ref_gs.reduce_degrees([0,1],1)
    splinepy.io.svg.export(f"12a_bspline_vm_stress_ref_gs_cp{nr_ctrl_pts_ref_gs}_p{deg_ref_gs}.svg",
                            BSpline_vm_stress_ref_gs,
                            **get_svg_kwargs(150, 300)
                            )
    
    # BSpline_vm_stress_gs.reduce_degrees([0],1)
    splinepy.io.svg.export(f"12b_bspline_vm_stress_gs_cp{nr_ctrl_pts_gs}_p{deg_gs}.svg",
                            BSpline_vm_stress_gs,
                            **get_svg_kwargs(150, 300)
                            )
    
    # BSpline_vm_stress_matlab.reduce_degrees([0],1)
    splinepy.io.svg.export(f"12c_bspline_vm_stress_matlab_cp{nr_ctrl_pts_matlab}_p{deg_matlab}.svg",
                            BSpline_vm_stress_matlab,
                            **get_svg_kwargs(150, 300)
                            )
    
    # BSpline_vm_stress_net.reduce_degrees([0,1],1)
    splinepy.io.svg.export(f"12d_bspline_vm_stress_net_cp{nr_ctrl_pts_net}_p{deg_net}.svg",
                            BSpline_vm_stress_net,
                            **get_svg_kwargs(150, 300)
                            )
    
    # errors to reference

    vmax_gs_queries = np.round(np.max(abs_error_gs), 4)
    vmax_matlab_queries = np.round(np.max(abs_error_matlab), 4)
    vmax_net_queries = np.round(np.max(abs_error_net), 4)

    # BSpline_queries_gs.reduce_degrees([0],1)
    splinepy.io.svg.export(f"13a_bspline_query_gs_cp{nr_ctrl_pts_gs}_p{deg_gs}.svg",
                            BSpline_queries_gs,
                            **get_svg_kwargs(0.0, vmax_gs_queries)
                            )
    
    # BSpline_queries_matlab.reduce_degrees([0],1)
    splinepy.io.svg.export(f"13b_bspline_query_matlab_cp{nr_ctrl_pts_matlab}_p{deg_matlab}.svg",
                            BSpline_queries_matlab,
                            **get_svg_kwargs(0.0, vmax_matlab_queries)
                            )
    
    # BSpline_queries_net.reduce_degrees([0,1],1)
    splinepy.io.svg.export(f"13c_bspline_query_net_cp{nr_ctrl_pts_net}_p{deg_net}.svg",
                            BSpline_queries_net,
                            **get_svg_kwargs(0.0, vmax_net_queries)
                            )
    
    # elasticity error

    vmax_net_div = np.round(np.max(net_div), 6)

    # BSpline_net_div.reduce_degrees([0,1],1)
    splinepy.io.svg.export(f"14a_bspline_net_div_cp{nr_ctrl_pts_net}_p{deg_net}.svg",
                            BSpline_net_div,
                            **get_svg_kwargs(0.0, vmax_net_div)
                            )
    
    # poisson ratio

    # BSpline_net_poisson.reduce_degrees([0,1],1)
    splinepy.io.svg.export(f"15a_bspline_net_poisson_cp{nr_ctrl_pts_net}_p{deg_net}.svg",
                            BSpline_net_poisson,
                            **get_svg_kwargs(0.0, 0.25)
                            )

