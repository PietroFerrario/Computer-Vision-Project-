import numpy as np 
import time
import open3d as o3d 
import c3d

#Opening the .c3d file 60p
#c3d_stream = c3d.Reader(open(r"..\Material\60fps\marker.c3d", 'rb'))

#Opening the .c3d file 360p
c3d_stream = c3d.Reader(open(r"..\Material\360fps\markers.c3d", 'rb'))

header = c3d_stream.header

# Checking header and printing general info about the c3d file 
if header:
# Print specific header information
      print("C3D file has a header.")
      print("Number of points (markers):", header.point_count)
      print("Number of analog channels:", header.analog_count)
      print("Number of frames:", header.last_frame)
      print("Marker rate (frames per second):", header.frame_rate)
      
      # if hasattr(header, 'parameter_group_labels'):
      # # Print parameter header information
      #       print("Parameter Group Labels:", header.parameter_group_labels)
      #       print("Parameter Names:", header.parameter_names)
      
      # Check if there are any event labels (Not sure what an event is)
      if header.event_labels.size > 0:
            print("Event labels:", header.event_labels)
      else:
            print("No event present in the C3D file.")
else:
        print("C3D file does not have a header.")


#Creating a list to store the marker lables
marker_labels = []

#Frame counter to avoid reaching the end of the file
frame_counter = 0

#Creating a list of a list to store all the markers and the relative position for every frame
marker_pos = [[] for _ in range(header.point_count)]

for i, points, analog in c3d_stream.read_frames():
      # Adding the marker lables to the list
      if not marker_labels:
            marker_labels.extend(label.strip() for label in c3d_stream.point_labels)
   
      frame_counter += 1
   
      if frame_counter > header.last_frame:
            break
      #print("Frame number",i)
      for j, marker in enumerate(points):
      #print("Marker number", j)
      #Extracting the coordinates 
            x, y, z = marker[:3]
     
            marker_pos[j].append([x, y, z])
     #print(markers_pos[j])
     
#print(marker_pos)
#print(len(marker_pos))
marker_pos = np.moveaxis(marker_pos, 1, 0)

print("The number of marker labels is:", len(marker_labels))

if len(marker_labels) == header.point_count:
      print("The number of marker labels COINCIDE with the number of marker ")
else:
      print("The number of marker labels DOESN'T COINCIDE with the number of marker, check metadata")

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("The marker labels are:", marker_labels )
print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")



black_colors = [[0, 0, 0] for i in range(header.point_count)]
keypoints_m = o3d.geometry.PointCloud()
keypoints_m.points = o3d.utility.Vector3dVector(marker_pos[0])
keypoints_m.colors= o3d.utility.Vector3dVector(black_colors)


vis =o3d.visualization.Visualizer()

vis.create_window(window_name = "StuffStuff")
view_control = vis.get_view_control()


# This plot all the markers

vis.add_geometry(keypoints_m)

# Scaling factor
# view_control = vis.get_view_control()
# view_control.scale(100)

time.sleep(0)
for i in range(1,20000):
    #print(i)
    new_markers = marker_pos[i]
    # center_skel = skeleton_joints.get_center()
    keypoints_m.points = o3d.utility.Vector3dVector(new_markers)
    #print(new_markers)
    # This plot the entire skeleton
    vis.update_geometry(keypoints_m)
    
    vis.update_renderer()
    vis.poll_events()

    time.sleep(0)
    
    if not vis.poll_events():
        break

    
vis.run()


