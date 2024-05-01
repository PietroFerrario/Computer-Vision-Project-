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
filename = r"..\Material\360fps\rigidbody.csv"

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

body_edges = [[0,1]]


bones_pos = []
if len(bodies) > 0:
    for body in bodies: 
        bones = take.rigid_bodies[body]
        bones_pos.append(bones.positions)
        
# Creating a list of filtered marker pos
to_discard_frame = []
              
for rb_index, rb_data in enumerate(bones_pos):
    print("The n of frame to print BEFORE filtering is:", len(rb_data))
    #print(f"marker_Index: {marker_index}")
    
    for frame_index, frame_data in enumerate(rb_data):
        #print(f"frame_Index: {frame_index}, frame_data: {frame_data}")
        discard_frame = False
        
        if frame_data is None: 
            discard_frame = True
        
        if discard_frame:
            to_discard_frame.append(frame_index)
    



############################################
#MARKERS
markers = take.markers

marker_pos = []

if len(markers) > 0:
    for body in markers: 
        bones_markers = take.markers[body]
        marker_pos.append(bones_markers.positions)
        
        
print("The len of marker_pos is", len(marker_pos))


for marker_index, marker_data in enumerate(marker_pos):
    print("The n of frame to print BEFORE filtering is:", len(marker_data))
    #print(f"marker_Index: {marker_index}")
    
    for frame_index, frame_data in enumerate(marker_data):
        #print(f"frame_Index: {frame_index}, frame_data: {frame_data}")
        discard_frame = False
        
        if frame_data is None: 
            discard_frame = True
        
        if discard_frame:
            to_discard_frame.append(frame_index)
            
print(f"the frames to discard are {to_discard_frame}")


# Actual Filtering 

for marker_index, marker_data in enumerate(marker_pos):
    # filtered frame for the single marker
    filtered_frame_sm = []
    for frame_index, frame_data in enumerate(marker_data):
        if frame_index not in to_discard_frame: 
            filtered_frame_sm.append(frame_data)
            
    marker_pos[marker_index] = filtered_frame_sm
    
    print("The n of frame to print AFTER filtering is:", len(filtered_frame_sm))
    
for rb_index, rb_data in enumerate(bones_pos):
    # filtered frame for the single marker
    filtered_frame_sb = []
    for frame_index, frame_data in enumerate(rb_data):
        if frame_index not in to_discard_frame: 
            filtered_frame_sb.append(frame_data)
            
    bones_pos[rb_index] = filtered_frame_sb

       
 
# Praparing for printing   
bones_pos = np.moveaxis(bones_pos, 1, 0)


colors = [[1, 0, 0] for i in range(len(body_edges))]
keypoints = o3d.geometry.PointCloud()
keypoints.points = o3d.utility.Vector3dVector(bones_pos[0])
keypoints_center = keypoints.get_center()
keypoints.points = o3d.utility.Vector3dVector(bones_pos[0])
skeleton_joints = o3d.geometry.LineSet()
skeleton_joints.points = o3d.utility.Vector3dVector(bones_pos[0])
center_skel = skeleton_joints.get_center()
skeleton_joints.points = o3d.utility.Vector3dVector(bones_pos[0])
skeleton_joints.lines = o3d.utility.Vector2iVector(body_edges)
skeleton_joints.colors = o3d.utility.Vector3dVector(colors)
   
     
marker_pos = np.moveaxis(marker_pos, 1, 0)

black_colors = [[0, 0, 0] for i in range(len(markers))]
keypoints_m = o3d.geometry.PointCloud()
keypoints_m.points = o3d.utility.Vector3dVector(marker_pos[0])
keypoints_m.colors= o3d.utility.Vector3dVector(black_colors)


vis = o3d.visualization.Visualizer()

vis.create_window(window_name = "Stuffstuff", width = 1000, height = 1000)

# This plot the body
vis.add_geometry(skeleton_joints)
vis.add_geometry(keypoints)

vis.add_geometry(keypoints_m)

# Scaling factor 
view_control = vis.get_view_control()
view_control.scale(1000)

time.sleep(0)
for i in range(1,20000):
    #print(i)
    new_joints = bones_pos[i]
    new_markers = marker_pos[i]
    # center_skel = skeleton_joints.get_center()
    skeleton_joints.points = o3d.utility.Vector3dVector(new_joints)
    keypoints.points = o3d.utility.Vector3dVector(new_joints)
    keypoints_m.points = o3d.utility.Vector3dVector(new_markers)
    #print(new_joints)
    # This plot the entire skeleton
    vis.update_geometry(skeleton_joints)
    vis.update_geometry(keypoints)
    vis.update_geometry(keypoints_m)
    
    vis.update_renderer()
    vis.poll_events()

    time.sleep(0)
    
    if not vis.poll_events():
        break

    
vis.run()


