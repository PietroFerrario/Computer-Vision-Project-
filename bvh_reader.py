import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import numpy as np 
import time
import open3d as o3d 
import bvh_reader.bvh_file as bvh


# Opening the 60p file 
#filename = r"..\Material\360fps\animation.bvh"

# Opening the 360p file 
filename = r"D:\school\Master\Computer vision\Computer Vision - Project\data\360fps\animation.bvh"


#body_edges = [[0,1],[1,2],[2,3],[3,4],[3,5],[5,6],[6,7],[7,8],[3,9],[9,10],[10,11],[11,12],[0,13],[13,14],[14,15],
#                [0,16],[16,17],[17,18],[18,20],[15,19]]

animation, joint_names, frame_time = bvh.read_bvh(filename)

animation_pos = animation.positions
animation_rot = animation.rotations
offsets = animation.offsets
parents = animation.parents

print("Number of Joints:", len(joint_names))
print("The names of the Joints are:", joint_names)
print("Number of Frames:", len(animation_pos))
#print("The shape of the positions is:", animation_pos.shape)
#print("The shape of the rotation is:", animation_rot.shape)
#print("The shape of the offsets is:", offsets.shape)
print("The joints Offsets are:", offsets)
#print("The shape of the parents is:", parents.shape)
print("The Parents are:", parents)

# Extracting the edges 
edges = []

for child_index, parent_index in enumerate(parents):
    if parent_index == 0:
        continue  # Skip the root joint
    edges.append([child_index, parent_index])  # Add edge connecting child to parent

print("The Edges (joints connected to eachother) are:", edges)
print("Number of Edges:", len(edges))

# Extracting the position informations
joint_pos = [[] for _ in range(len(joint_names))]

for frame_index, frame_data in enumerate(animation_pos):
#     print("Frame number", frame_index)
#     print("Frame data", frame_data[2])
    for joint_index, joint_data in enumerate(frame_data):
        #print("Joint number", joint_index)
        #Extracting the coordinates 
        #print("Joint data", joint_data)
        #print(len(joint_data))
        x, y, z = joint_data
        joint_pos[joint_index].append([x, y, z])
        #print(joint_pos[joint_index])
        
# Extracting the rotation informations 
joint_rot = [[] for _ in range(len(joint_names))]
for frame_index, frame_data in enumerate(animation_rot):
    #rint("Frame number", frame_index)
    #print("Frame data", frame_data[2])
    for joint_index, joint_data in enumerate(frame_data):
      x, y, z = joint_data
      joint_rot[joint_index].append([x, y, z])
     

print(joint_pos[0][0:5])
print(joint_rot[0][0:5])
# print(len(joint_pos))
joint_pos = np.moveaxis(joint_pos, 1, 0)
joint_rot = np.moveaxis(joint_rot, 1, 0)
# print(len(joint_pos))

# Preparing for printing 
colors = [[1, 0, 0] for i in range(len(joint_names))]
keypoints = o3d.geometry.PointCloud()
keypoints.points = o3d.utility.Vector3dVector(joint_pos[0])
#keypoints_center = keypoints.get_center()
keypoints.points = o3d.utility.Vector3dVector(joint_pos[0])
skeleton_joints = o3d.geometry.LineSet()
skeleton_joints.points = o3d.utility.Vector3dVector(joint_pos[0])
center_skel = skeleton_joints.get_center()
skeleton_joints.points = o3d.utility.Vector3dVector(joint_pos[0])
skeleton_joints.lines = o3d.utility.Vector2iVector(edges)
skeleton_joints.colors = o3d.utility.Vector3dVector(colors)



vis =o3d.visualization.Visualizer()

vis.create_window(window_name = "StuffStuff")

vis.add_geometry(skeleton_joints)
vis.add_geometry(keypoints)

view_control = vis.get_view_control()

#Scaling factor
view_control = vis.get_view_control()
view_control.scale(100)

time.sleep(0)
for i in range(1,20000):
  
    new_joints = joint_pos[i]
    
    skeleton_joints.points = o3d.utility.Vector3dVector(new_joints)
    keypoints.points = o3d.utility.Vector3dVector(new_joints)
    
    vis.update_geometry(skeleton_joints)
    vis.update_geometry(keypoints)
    
    vis.update_renderer()
    vis.poll_events()

    time.sleep(0)
    
    if not vis.poll_events():
        break

    
vis.run()

