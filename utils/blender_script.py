import bpy
import numpy as np
import os
import json
import random

# Settings
scene_file_path = '/Users/tzlilovadia/Downloads/datasets/blender_dataset/spec_teapot/spec_teapot.glb'
output_dir = '/Users/tzlilovadia/Downloads/datasets/blender_dataset/spec_teapot'
# output_dir = '/Users/tzlilovadia/Downloads/datasets/blender'  # Replace with your desired output directory

num_views = 360
train_ratio = 0.8

distance = 0.05  # Distance from the object
height = .05  # Height of the camera (relative to the original height)
object_name = 'Utah_teapot_(solid).001'  # The correct name of the object to focus on
os.makedirs(output_dir, exist_ok=True)
images_dir = os.path.join(output_dir, 'images')
os.makedirs(images_dir, exist_ok=True)
# Ensure the output directory exists

ply_output_path = os.path.join(output_dir, 'points3d.ply')  # Path to save the PLY file

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=True)

# Import the GLB file
bpy.ops.import_scene.gltf(filepath=scene_file_path)

# Get the object to focus on
focus_object = bpy.data.objects.get(object_name)
if focus_object is None:
    raise Exception(f"No object named '{object_name}' found in the scene.")

# Set the world background to a dark color
bpy.context.scene.world.use_nodes = False
world_nodes = bpy.context.scene.world.node_tree.nodes
bg_node = world_nodes.get('Background')
bg_node.inputs['Color'].default_value = (0, 0, 0, 1)  # Dark gray background

# Add a sun light
bpy.ops.object.light_add(type='SUN', radius=100, location=(5, 5, 0))
sun_light = bpy.context.object
sun_light.data.energy = 500  # Lower the energy to reduce brightness

bpy.ops.object.light_add(type='POINT', radius=100, location=(-1, -1, -1))
point_light2 = bpy.context.object
point_light2.data.energy = 100
# Get or create a camera
camera = bpy.data.objects.get('Camera')
if camera is None:
    bpy.ops.object.camera_add(location=(0, 0, height))
    camera = bpy.context.object
bpy.context.scene.camera = camera

# Ensure the correct material is applied
material_name = 'Material.004'  # Replace with the name of the modified material
material = bpy.data.materials.get(material_name)
if material is None:
    raise Exception(f"No material named '{material_name}' found in the scene.")

# Apply the material to the object
if len(focus_object.data.materials) == 0:
    focus_object.data.materials.append(material)
else:
    focus_object.data.materials[0] = material

# Set up the render settings
bpy.context.scene.render.image_settings.file_format = 'PNG'
# bpy.context.scene.render.resolution_x = 1024
# bpy.context.scene.render.resolution_y = 1024

# Adjust camera exposure settings
bpy.context.scene.view_settings.exposure = 1  # Lower exposure to reduce brightness

# Prepare to store camera poses
camera_poses = []
camera_intrinsics = []
frames = []
# Calculate angles for evenly spaced views
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)

for i, angle in enumerate(angles):
    # Calculate new camera position
    x = distance * np.cos(angle)
    y = distance * np.sin(angle)
    camera.location = (x, y, height)

    # Point the camera at the object
    direction = focus_object.location - camera.location
    #    direction *= -1
    rot_quat = direction.to_track_quat('Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()

    # Print debug information
    print(f"Camera position: {camera.location}, Camera rotation: {camera.rotation_euler}")

    # Render the scene
    bpy.context.scene.render.filepath = os.path.join(output_dir, f'view_{i:03d}.png')
    bpy.ops.render.render(write_still=True)
    # Save camera transformation matrix
    transformation_matrix = camera.matrix_world.inverted()
    frame_data = {
        'file_path': f'images/view_{i:03d}.png',
        'rotation': 0.5,  # Example value, adjust as necessary
        'transform_matrix': [list(row) for row in transformation_matrix]
    }
    frames.append(frame_data)

    # Save camera intrinsics
    scene = bpy.context.scene
    render = scene.render
    intrinsics = {
        'focal_length': camera.data.lens,
        'sensor_width': camera.data.sensor_width,
        'sensor_height': camera.data.sensor_height,
        'resolution_x': render.resolution_x,
        'resolution_y': render.resolution_y,
        'pixel_aspect_x': render.pixel_aspect_x,
        'pixel_aspect_y': render.pixel_aspect_y,
        'shift_x': camera.data.shift_x,
        'shift_y': camera.data.shift_y
    }
    camera_intrinsics.append(intrinsics)

# Split images into training and test sets
random.shuffle(frames)
train_count = int(train_ratio * len(frames))
train_frames = frames[:train_count]
test_frames = frames[train_count:]

# Save transforms to JSON
transforms_train = {
    'camera_angle_x': camera.data.angle_x,
    'images': train_frames
}
with open(os.path.join(output_dir, 'transforms_train.json'), 'w') as f:
    json.dump(transforms_train, f, indent=4)

transforms_test = {
    'camera_angle_x': camera.data.angle_x,
    'images': test_frames
}
with open(os.path.join(output_dir, 'transforms_test.json'), 'w') as f:
    json.dump(transforms_test, f, indent=4)

# Save camera intrinsics to JSON
with open(os.path.join(output_dir, 'camera_intrinsics.json'), 'w') as f:
    json.dump(camera_intrinsics, f, indent=4)


# Save point cloud to PLY file
def save_mesh_as_ply(obj, filepath):
    mesh = obj.data
    if not mesh.vertex_colors:
        # Add a new vertex color layer if it doesn't exist
        mesh.vertex_colors.new(name='Col')

    vertex_colors = mesh.vertex_colors.active.data
    mesh.calc_loop_triangles()  # Ensure the loop triangles are calculated
    #    mesh.calc_normals()  # Ensure normals are calculated

    vertices = mesh.vertices
    loops = mesh.loops
    loop_triangles = mesh.loop_triangles

    colors = np.zeros((len(vertices), 4))
    normals = np.zeros((len(vertices), 3))

    for loop_triangle in loop_triangles:
        for loop_index in loop_triangle.loops:
            vert_index = loops[loop_index].vertex_index
            color = vertex_colors[loop_index].color
            normal = loops[loop_index].normal
            colors[vert_index] = np.array(color)
            normals[vert_index] = np.array(normal)

    with open(filepath, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write(f'element vertex {len(vertices)}\n')
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property float nx\n')
        f.write('property float ny\n')
        f.write('property float nz\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('property uchar alpha\n')
        f.write('end_header\n')
        for i, v in enumerate(vertices):
            color = (colors[i] * 255).astype(int)
            f.write(
                f'{v.co.x} {v.co.y} {v.co.z} {normals[i][0]} {normals[i][1]} {normals[i][2]} {color[0]} {color[1]} {color[2]} {color[3]}\n')


save_mesh_as_ply(focus_object, ply_output_path)

print("Rendering complete. Images, camera parameters, and point cloud saved.")
#
# import bpy
# import numpy as np
# import os
# import json
# import random
# import math
#
#
# def prepare_scene():
#     # Clear existing objects
#     #    bpy.ops.object.select_all(action='SELECT')
#     #    bpy.ops.object.delete(use_global=True)
#     #    # Import the GLB file
#     #    bpy.ops.import_scene.gltf(filepath=scene_file_path)
#
#     # Get the object to focus on
#     focus_object = bpy.data.objects.get(object_name)
#     if focus_object is None:
#         raise Exception(f"No object named '{object_name}' found in the scene.")
#     # Set the world background to a dark color
#     bpy.context.scene.world.use_nodes = False
#     world_nodes = bpy.context.scene.world.node_tree.nodes
#     bg_node = world_nodes.get('Background')
#     #    bg_node.inputs['Color'].default_value = (-1, -1, -1, 1)  # Dark gray background
#     bg_node.inputs['Color'].default_value = (1, 1, 1, 1)  # Dark gray background
#
#     #    # Add a sun light
#     #    bpy.ops.object.light_add(type='SUN', radius=100, location=(5, 5, 0))
#     #    sun_light = bpy.context.object
#     #    sun_light.data.energy = 500  # Lower the energy to reduce brightness
#
#     #    bpy.ops.object.light_add(type='POINT', radius=100, location=(-1, -1, -1))
#     #    point_light2 = bpy.context.object
#     #    point_light2.data.energy = 100
#     # Get or create a camera
#     camera = bpy.data.objects.get('Camera')
#     #    if camera is None:
#     #        bpy.ops.object.camera_add(location=(0, 0, height))
#     #        camera = bpy.context.object
#     bpy.context.scene.camera = camera
#
#     # Ensure the correct material is applied
#     material_name = 'Material.003'  # Replace with the name of the modified material
#     material = bpy.data.materials.get(material_name)
#     if material is None:
#         raise Exception(f"No material named '{material_name}' found in the scene.")
#
#     # Apply the material to the object
#     if len(focus_object.data.materials) == 0:
#         focus_object.data.materials.append(material)
#     else:
#         focus_object.data.materials[0] = material
#
#     # Set up the render settings
#     bpy.context.scene.render.image_settings.file_format = 'PNG'
#
#     # Adjust camera exposure settings
#     bpy.context.scene.view_settings.exposure = 1  # Lower exposure to reduce brightness
#
#
# def save_camera_data(file_path, cameras, fovx):
#     data = {
#         "camera_angle_x": fovx,
#         "images": cameras
#     }
#     with open(file_path, 'w') as outfile:
#         json.dump(data, outfile, indent=4)
#
#
# def get_camera_transform_matrix(camera):
#     loc, rot, _ = camera.matrix_world.decompose()
#     transform_matrix = np.array(camera.matrix_world)
#     return transform_matrix
#
#
# # Settings
# # scene_file_path = '/Users/tzlilovadia/Downloads/datasets/blender/teapot.glb'  # Replace with the path to your GLB file
# scene_file_path = '/Users/tzlilovadia/Downloads/datasets/blender_dataset/spec_teapot/spec_teapot.glb'  # Replace with your desired output directory
# output_dir = '/Users/tzlilovadia/Downloads/datasets/blender_dataset/spec_teapot'  # Replace with your desired output directory
# images_dir = '/Users/tzlilovadia/Downloads/datasets/blender_dataset/spec_teapot/images'  # Replace with your desired output directory
#
# num_views = 360
# train_ratio = 0.8
# distance = 0.05  # Distance from the object
# height = .05  # Height of the camera (relative to the original height)
#
# object_name = 'Utah_teapot_(solid)'  # The correct name of the object to focus on
# os.makedirs(output_dir, exist_ok=True)
# # images_dir = os.path.join(output_dir, 'images')
# # Ensure the output directory exists
# os.makedirs(images_dir, exist_ok=True)
# ply_output_path = os.path.join(output_dir, 'points3d.ply')  # Path to save the PLY file
# camera = None
# focus_object = None
# material = None
#
# prepare_scene()
# object = bpy.data.objects.get(object_name)
# # Prepare to store camera poses
# camera_poses = []
# camera_intrinsics = []
# images = []
#
# # Calculate angles for evenly spaced views
# angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)
#
#
# def generate_360_camera_positions(num_frames, radius):
#     cameras = []
#     angle_step = 360 / num_frames
#
#     for frame in range(num_frames):
#         angle = math.radians(frame * angle_step)
#         x = radius * math.sin(angle)
#         y = radius * math.cos(angle)
#         z = bpy.context.scene.camera.location.z
#
#         bpy.context.scene.render.filepath = os.path.join(images_dir, f"frame_{frame:04d}.png")
#         bpy.ops.render.render(write_still=True)
#         bpy.context.scene.camera.location = (x, y, z)
#         bpy.context.scene.camera.rotation_euler = (
#             0, 0, angle + math.radians(90))  # Keep the camera looking at the origin
#
#         bpy.context.view_layer.update()  # Update the scene to reflect changes
#
#         transform_matrix = get_camera_transform_matrix(bpy.context.scene.camera)
#         transform_matrix_list = transform_matrix.tolist()
#
#         frame_data = {
#             "file_path": f"frame_{frame:04d}",
#             "transform_matrix": transform_matrix_list
#         }
#         cameras.append(frame_data)
#
#         # Render the frame (optional, can be removed if only JSON output is needed)
#     #        bpy.context.scene.render.filepath = os.path.join(images_dir, f"frame_{frame:04d}.png")
#     #        bpy.ops.render.render(write_still=True)
#
#     return cameras
#
#
# ####################################################################################
# ################################### Cutted original Loop############################
# ####################################################################################
# def save_camera_data(train_path, test_path, cameras, fovx):
#     #    random.shuffle(images)
#     num_train = int(0.8 * len(cameras))
#     train_cameras = cameras[:num_train]
#     test_cameras = cameras[num_train:]
#
#     data_train = {
#         "camera_angle_x": fovx,
#         "images": train_cameras
#     }
#     data_test = {
#         "camera_angle_x": fovx,
#         "images": test_cameras
#     }
#
#     with open(train_path, 'w') as outfile:
#         json.dump(data_train, outfile, indent=4)
#
#     with open(test_path, 'w') as outfile:
#         json.dump(data_test, outfile, indent=4)
#
#
# # Split images into training and test sets
#
#
# def save_mesh_as_ply(obj, filepath):
#     mesh = obj.data
#     if not mesh.vertex_colors:
#         # Add a new vertex color layer if it doesn't exist
#         mesh.vertex_colors.new(name='Col')
#
#     vertex_colors = mesh.vertex_colors.active.data
#     mesh.calc_loop_triangles()  # Ensure the loop triangles are calculated
#     #    mesh.calc_normals()  # Ensure normals are calculated
#
#     vertices = mesh.vertices
#     loops = mesh.loops
#     loop_triangles = mesh.loop_triangles
#
#     colors = np.zeros((len(vertices), 4))
#     normals = np.zeros((len(vertices), 3))
#
#     for loop_triangle in loop_triangles:
#         for loop_index in loop_triangle.loops:
#             vert_index = loops[loop_index].vertex_index
#             color = vertex_colors[loop_index].color
#             normal = loops[loop_index].normal
#             colors[vert_index] = np.array(color)
#             normals[vert_index] = np.array(normal)
#
#     with open(filepath, 'w') as f:
#         f.write('ply\n')
#         f.write('format ascii 1.0\n')
#         f.write(f'element vertex {len(vertices)}\n')
#         f.write('property float x\n')
#         f.write('property float y\n')
#         f.write('property float z\n')
#         f.write('property float nx\n')
#         f.write('property float ny\n')
#         f.write('property float nz\n')
#         f.write('property uchar red\n')
#         f.write('property uchar green\n')
#         f.write('property uchar blue\n')
#         f.write('property uchar alpha\n')
#         f.write('end_header\n')
#         for i, v in enumerate(vertices):
#             color = (colors[i] * 255).astype(int)
#             f.write(
#                 f'{v.co.x} {v.co.y} {v.co.z} {normals[i][0]} {normals[i][1]} {normals[i][2]} {color[0]} {color[1]} {color[2]} {color[3]}\n')
#
#
# # Main script
# num_frames = 36  # Number of images for 360° rotation
# radius = 5.0  # Radius of the circular path
# train_json_path = os.path.join(output_dir, "transforms_train.json")
# test_json_path = os.path.join(output_dir, "transforms_test.json")
# point_cloud_path = os.path.join(output_dir, "points3d.ply")
#
# # Get initial camera settings
# camera = bpy.context.scene.camera
# initial_loc = camera.location.copy()
# initial_rot = camera.rotation_euler.copy()
# fovx = camera.data.angle_x
#
# # Generate camera positions and transformations
# cameras = generate_360_camera_positions(num_frames, radius)
#
# # Restore initial camera settings
# camera.location = initial_loc
# camera.rotation_euler = initial_rot
#
# # Save camera data to JSON
# save_camera_data(train_json_path, test_json_path, cameras, fovx)
#
# # Save point cloud
# save_mesh_as_ply(object, point_cloud_path)
# print("Rendering complete. Images, camera parameters, and point cloud saved.")
