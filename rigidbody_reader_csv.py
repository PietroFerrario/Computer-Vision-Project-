import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import open3d as o3d
import time
import numpy as np

# Load the Optitrack CSV file parser module.
import optitrack.csv_reader as csv
from optitrack.geometry import *

# Find the path to the test data file located alongside the script.

#Rigid Body 60fps 
filename = "D:\school\Master\Computer vision\Computer Vision - Project\data\ProvaRigidBody.csv"

# Read the file.
take = csv.Take().readCSV(filename)

# Print out some statistics
print("Found rigid bodies:", take.rigid_bodies.keys())
print("~~~~~~~~~~~~~~~")
print("Found markers:", take.markers.keys())

############################################
#BODIES 

# Process the first rigid body into a set of planes.
bodies = take.rigid_bodies

# for now:
xaxis = [1,0,0]
yaxis = [0,1,0]

body_edges = [[0,1],[0,2],[0,3]]

bones_pos = []
orientations = []
if len(bodies) > 0:
    for body in bodies: 
        bones = take.rigid_bodies[body]
        orientations.append(bones.rotations)
        bones_pos.append(bones.positions)
        #for pos,rot in zip(bones.positions, bones.rotations):
             #if pos is not None and rot is not None:
                # xaxis, yaxis = quaternion_to_xaxis_yaxis(rot)
                # plane = rs.PlaneFromFrame(pos, xaxis, yaxis)

                # # create a visible plane, assuming units are in meters
                # rs.AddPlaneSurface( plane, 0.1, 0.1 )
              
for i, arr in enumerate(bones_pos):
    for j, lis in enumerate(arr):
       if lis is None and j > 0:
           arr[j] = arr[j-1]
for i, arr in enumerate(orientations):
    for j, lis in enumerate(arr):
       if lis is None and j > 0:
           arr[j] = arr[j-1]
    
bones_pos = np.moveaxis(bones_pos, 1, 0)
orientations = np.moveaxis(orientations, 1, 0)
print(orientations[0][0])
rotationmatrix = quaternion_to_rotation_matrix(orientations[0][0])
origin = np.array([0,0,0])
vect_x = np.array(rotationmatrix) @ np.array([1,0,0])
vect_y = np.array(rotationmatrix) @ np.array([0,1,0])
vect_z = np.array(rotationmatrix) @ np.array([0,0,1])


colors = [[1, 0, 0] for i in range(len(body_edges))]

#Rigidbody
rigidbody = o3d.geometry.LineSet()
rigidbody.points = o3d.utility.Vector3dVector((bones_pos[0][0], vect_x + bones_pos[0][0], vect_y + bones_pos[0][0], vect_z + bones_pos[0][0]))
center_rigidbody = rigidbody.get_center()
rigidbody.lines = o3d.utility.Vector2iVector(body_edges)
rigidbody.colors = o3d.utility.Vector3dVector(colors)

############################################
#MARKERS
markers = take.markers

marker_pos = []

if len(markers) > 0:
    for body in markers: 
        bones_markers = take.markers[body]
        marker_pos.append(bones_markers.positions)
        
for i, arr in enumerate(marker_pos):
    for j, lis in enumerate(arr):
       if lis is None and j > 0:
           arr[j] = arr[j-1]
           
marker_pos = np.moveaxis(marker_pos, 1, 0)

black_colors = [[0, 0, 0] for i in range(len(markers))]
keypoints_m = o3d.geometry.PointCloud()
keypoints_m.points = o3d.utility.Vector3dVector(marker_pos[0])
keypoints_m.colors= o3d.utility.Vector3dVector(black_colors)

vis = o3d.visualization.Visualizer()
control = vis.get_view_control()
vis.create_window(window_name = "Stuffstuff", width = 1000, height = 1000)

# This plot the entire skeleton
vis.add_geometry(rigidbody)
#vis.add_geometry(keypoints)
vis.add_geometry(keypoints_m)

# move the camera backwards to get everything in frame
vis.get_view_control().camera_local_translate(-6,0,0)
vis.get_view_control().change_field_of_view(step=0.1)
vis.get_view_control().set_constant_z_far(10)

time.sleep(0)
for i in range(1,20000):
    #print(i)
    
    new_markers = marker_pos[i]
    rotationmatrix = quaternion_to_rotation_matrix(orientations[i][0])
    vect_x = np.array(rotationmatrix) @ np.array([1,0,0])
    vect_y = np.array(rotationmatrix) @ np.array([0,1,0])
    vect_z = np.array(rotationmatrix) @ np.array([0,0,1])
    new_joints = (bones_pos[i][0], vect_x + bones_pos[i][0], vect_y + bones_pos[i][0], vect_z + bones_pos[i][0])
    center_skel = rigidbody.get_center()
    rigidbody.points = o3d.utility.Vector3dVector(new_joints)
    #keypoints.points = o3d.utility.Vector3dVector(new_joints)
    keypoints_m.points = o3d.utility.Vector3dVector(new_markers)
    #print(new_joints)
    # This plot the entire skeleton
    vis.update_geometry(rigidbody)
    #vis.update_geometry(keypoints)
    vis.update_geometry(keypoints_m)
    vis.update_renderer()
    vis.poll_events()

    time.sleep(0)
    
    if not vis.poll_events():
        break

    
vis.run()


