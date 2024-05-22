import sys, os
#sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import open3d as o3d
import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load the Optitrack CSV file parser module.
import optitrack.csv_reader as csv
from optitrack.geometry import *

# Find the path to the test data file located alongside the script.

#Rigid Body 60fps 
filename = r"..\Material\360fps\rigidbody.csv"

# Read the file.
take = csv.Take().readCSV(filename)

last_mes = current_mes = np.zeros((3,1), np.float32)
last_pre = current_pre = np.zeros((3,1), np.float32)

last_mes_rot = current_mes_rot = np.zeros((4,1), np.float32)
last_pre_rot = current_pre_rot = np.zeros((4,1), np.float32)

def kalman_fun_pos(x, y, z): 
    global current_mes, last_mes, current_pre, last_pre
    last_pre = current_pre 
    last_mes = current_mes 
    current_mes = np.array([[np.float32(x)],[np.float32(y)], [np.float32(z)]])

    kalman1.correct(current_mes)
    current_pre = kalman1.predict()
    
def kalman_fun_rot(x, y, z, w): 
    global current_mes_rot, last_mes_rot, current_pre_rot, last_pre_rot
    last_pre_rot = current_pre_rot 
    last_mes_rot = current_mes_rot 
    current_mes_rot = np.array([[np.float32(x)],[np.float32(y)], [np.float32(z)], [np.float32(w)]])

    kalman2.correct(current_mes_rot)
    current_pre_rot = kalman2.predict()

#Dynamic parameters (9) and measurement parameters (3)
kalman1 = cv2.KalmanFilter(9,3)
# H 
kalman1.measurementMatrix = np.array([[1,0,0,0,0,0,0,0,0],
                                     [0,1,0,0,0,0,0,0,0],
                                     [0,0,1,0,0,0,0,0,0]], np.float32)
kalman1.transitionMatrix = \
    np.array([[1,0,0,1,0,0,0.5,0,0],
              [0,1,0,0,1,0,0,0.5,0],
              [0,0,1,0,0,1,0,0,0.5],
              [0,0,0,1,0,0,1,0,0],
              [0,0,0,0,1,0,0,1,0],
              [0,0,0,0,0,1,0,0,1],
              [0,0,0,0,0,0,1,0,0],
              [0,0,0,0,0,0,0,1,0],
              [0,0,0,0,0,0,0,0,1]], np.float32)

kalman1.processNoiseCov =  np.array([[1,0,0,0,0,0,0,0,0],
                                    [0,1,0,0,0,0,0,0,0],
                                    [0,0,1,0,0,0,0,0,0],
                                    [0,0,0,1,0,0,0,0,0],
                                    [0,0,0,0,1,0,0,0,0],
                                    [0,0,0,0,0,1,0,0,0],
                                    [0,0,0,0,0,0,1,0,0],
                                    [0,0,0,0,0,0,0,1,0],
                                    [0,0,0,0,0,0,0,0,1]], np.float32) * 0.0000003

# v
kalman1.measurementNoiseCov =np.array([[1,0,0],
                                      [0,1,0],
                                      [0,0,1]], np.float32) * 0.00001


#Dynamic parameters (7) 4 quaternions q1, q2, q3,q4 = x,y,z,w + 3 angular velocities and measurement parameters (4) quaternions 
kalman2 = cv2.KalmanFilter(7,4)
# H 
kalman2.measurementMatrix = np.array([[1,0,0,0,0,0,0],
                                      [0,1,0,0,0,0,0],
                                      [0,0,1,0,0,0,0],
                                      [0,0,0,1,0,0,0]], np.float32)
kalman2.transitionMatrix = \
    np.array([[0.5,0,0,0,0,0,0],
              [0,0.5,0,0,1,0,0],
              [0,0,0.5,0,0,1,0],
              [0,0,0,0.5,0,0,1],
              [0,0,0,0,1,0,0],
              [0,0,0,0,0,1,0],
              [0,0,0,0,0,0,1]], np.float32)

kalman2.processNoiseCov =  np.array([[1,0,0,0,0,0,0],
                                     [0,1,0,0,0,0,0],
                                     [0,0,1,0,0,0,0],
                                     [0,0,0,1,0,0,0],
                                     [0,0,0,0,1,0,0],
                                     [0,0,0,0,0,1,0],
                                     [0,0,0,0,0,0,1]], np.float32) * 0.0000003

# v
kalman2.measurementNoiseCov =np.array([[1,0,0,0],
                                       [0,1,0,0],
                                       [0,0,1,0],
                                       [0,0,0,1]], np.float32) * 0.00001

# Both values for processNC and measNC can be adjusted accordingly 
    
    
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
        
# Creating a list of filtered frames
filtered_frames = []
              
for rb_index, rb_data in enumerate(bones_pos):
    #print(f"Rigid Body Index: {rb_index}, Rigid body data: {rb_data}")
    
    for frame_index, frame_data in enumerate(rb_data):
        #print(f"frame_Index: {frame_index}, frame_data: {frame_data}")
        
        
        if frame_data is None: 
            #print("frame index",frame_index)
            filtered_frames.append(frame_index)
            #print("first",frame_data)
            #print("previous", rb_data[frame_index-1])
            frame_data = rb_data[frame_index-1]
            #print("second", frame_data)
            x = frame_data[0]
            y = frame_data[1]
            z = frame_data[2]
            kalman_fun_pos(x,y,z)
           
            predicted_pos = [current_pre[0,0], current_pre[1,0], current_pre[2,0]]
            #print("third", predicted_pos)
            bones_pos[rb_index][frame_index] = predicted_pos
            #print("fourth", bones_pos[rb_index][frame_index])
           
           
for rb_index, rb_data in enumerate(orientations):
    #print(f"Rigid Body Index: {rb_index}, Rigid body data: {rb_data}")
    
    for frame_index, frame_data in enumerate(rb_data):
        #print(f"frame_Index: {frame_index}, frame_data: {frame_data}")
        
        
        if frame_data is None: 
            print("frame index",frame_index)
            filtered_frames.append(frame_index)
            print("first",frame_data)
            print("previous", rb_data[frame_index-1])
            frame_data = rb_data[frame_index-1]
            print("second", frame_data)
            x = frame_data[0]
            y = frame_data[1]
            z = frame_data[2]
            w = frame_data[3]
            kalman_fun_rot(x,y,z,w)
           
            predicted_rot = [current_pre_rot[0,0], current_pre_rot[1,0], current_pre_rot[2,0], current_pre_rot[3,0]]
            print("third", predicted_rot)
            orientations[rb_index][frame_index] = predicted_rot
            print("fourth", orientations[rb_index][frame_index])


           
           
# for i, arr in enumerate(bones_pos):
#     for j, lis in enumerate(arr):
#        if lis is None and j > 0:
#            arr[j] = arr[j-1]


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
        
        
print("The len of marker_pos is", len(marker_pos))


# for marker_index, marker_data in enumerate(marker_pos):
#     print("The n of frame to print BEFORE filtering is:", len(marker_data))
#     #print(f"marker_Index: {marker_index}")
    
#     for frame_index, frame_data in enumerate(marker_data):
#         #print(f"frame_Index: {frame_index}, frame_data: {frame_data}")
#         discard_frame = False
        
#         if frame_data is None: 
#             discard_frame = True
        
#         if discard_frame:
#             to_discard_frame.append(frame_index)
# for i, arr in enumerate(marker_pos):
#     for j, lis in enumerate(arr):
#        if lis is None and j > 0:
#            arr[j] = arr[j-1]
                    



for rb_index, rb_data in enumerate(marker_pos):
    #print(f"marker index : {rb_index}, marker data: {rb_data}")
    
    for frame_index, frame_data in enumerate(rb_data):
        #print(f"frame_Index: {frame_index}, frame_data: {frame_data}")
        
        
        if frame_data is None: 
            #print("frame index",frame_index)
            filtered_frames.append(frame_index)
            #print("first",frame_data)
            #print("previous", rb_data[frame_index-1])
            frame_data = rb_data[frame_index-1]
            #print("second", frame_data)
            x = frame_data[0]
            y = frame_data[1]
            z = frame_data[2]
            kalman_fun_pos(x,y,z)
           
            predicted_pos = [current_pre[0,0], current_pre[1,0], current_pre[2,0]]
            #print("third", predicted_pos)
            marker_pos[rb_index][frame_index] = predicted_pos
            #print("fourth", marker_pos[rb_index][frame_index])

print("indexes filtered", filtered_frames)
   
     
marker_pos = np.moveaxis(marker_pos, 1, 0)

black_colors = [[0, 0, 0] for i in range(len(markers))]
keypoints_m = o3d.geometry.PointCloud()
keypoints_m.points = o3d.utility.Vector3dVector(marker_pos[0])
keypoints_m.colors= o3d.utility.Vector3dVector(black_colors)


vis = o3d.visualization.Visualizer()
control = vis.get_view_control()
vis.create_window(window_name = "Stuffstuff", width = 1000, height = 1000)


# This plot the body
vis.add_geometry(rigidbody)
#vis.add_geometry(keypoints)
vis.add_geometry(keypoints_m)

# move the camera backwards to get everything in frame
vis.get_view_control().camera_local_translate(-6,0,0)
vis.get_view_control().change_field_of_view(step=0.1)
vis.get_view_control().set_constant_z_far(10)


time.sleep(0)
for i in range(1,3451):
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