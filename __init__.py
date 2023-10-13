bl_info = {
    "name": "c2m",
    "blender": (3, 60, 4),
    "description": "Converts pointcloud to textured mesh.",
    "location": "Right side panel > \"c2m\" tab",
    "category": "c2m",
}

import sys
import os
import subprocess
import site
import importlib

libraries_to_install = ["open3d", "laspy", "numpy", "tqdm"]

auto_import = True

python_exe = sys.executable
target = site.getsitepackages()[0]

subprocess.call([python_exe, '-m', 'ensurepip'])
subprocess.call([python_exe, '-m', 'pip', 'install', '--upgrade', 'pip'])

if auto_import:

    for library in libraries_to_install:
        try:
            globals()[library] = __import__(library)
        except ImportError:
            import subprocess

            subprocess.call([python_exe, '-m', 'pip', 'install', library])
            globals()[library] = __import__(library)

else:
    import open3d
    import laspy
    import numpy
    import tqdm

o3d = open3d
np = numpy

import bpy
import bmesh
import math
from typing import Tuple

from bpy.types import Scene
from bpy.types import Panel
from bpy.types import Operator
from bpy.props import StringProperty, BoolProperty, IntProperty, FloatProperty

pointcloud = None


def read_pointcloud(self, context):
    global pointcloud

    # help python to free some memory
    if pointcloud is not None:
        del pointcloud
        pointcloud = None

    path = bpy.path.abspath(context.scene.pointcloud_path)
    print(path)

    cloud = o3d.geometry.PointCloud()

    if path.split(".")[-1] == "las" or path.split(".")[-1] == "laz":
        with laspy.open(path) as file:
            dims = []
            h = file.header
            for dimension in h.point_format.dimensions:
                dims.append(dimension.name)

            # load data chunk wise
            total_points = h.point_count
            for data_chunk in tqdm.tqdm(file.chunk_iterator(10_000), total=total_points // 10000,
                                        desc="Reading Pointcloud"):

                points = np.ascontiguousarray(
                    np.vstack((data_chunk.x, data_chunk.y, data_chunk.z)).transpose(),
                    dtype=np.float64)
                # scale cloud to unit cube
                # max_len = max(file.header.x_max, max(file.header.y_max, file.header.z_max))
                cloud.points.extend(o3d.utility.Vector3dVector(points))

                if 'red' in dims and 'green' in dims and 'blue' in dims:
                    colors = np.ascontiguousarray(
                        np.vstack((data_chunk.red, data_chunk.green, data_chunk.blue)).transpose(), dtype=np.float64)
                    cloud.colors.extend(o3d.utility.Vector3dVector(colors / 65535.0))

            cloud.translate(-np.array([h.x_offset, h.y_offset, h.z_offset]))
    else:
        cloud = o3d.io.read_point_cloud(path, print_progress=True)

    pointcloud = cloud
    path = os.path.basename(path)
    name = os.path.splitext(path)[0]

    # set some properties
    context.scene.texturing_pointcloud_size = len(pointcloud.points)
    context.scene.pointcloud_name = name

    # add new collection if it doesn't exist yet
    collection = bpy.data.collections.get(context.scene.collection_name)
    if collection is None:
        collection = bpy.data.collections.new(context.scene.collection_name)
        bpy.context.scene.collection.children.link(collection)

    return {'FINISHED'}


def triangulate_pointcloud(self, context):
    global pointcloud

    if pointcloud is None:
        self.report({'ERROR'}, "Pointcloud is None.")
        return {'CANCELLED'}

    # properties
    size = context.scene.pointcloud_downsampling_size
    depth = context.scene.triangulation_depth
    scale = context.scene.triangulation_scale
    removal_threshold = context.scene.triangulation_removal_threshold

    # downsampling pointcloud
    print("downsampling pointcloud...")
    point_count = int(len(np.asarray(pointcloud.points)))
    ratio = max(0.0, min(size / point_count, 1.0))
    pointcloud_down_sampled = pointcloud.random_down_sample(ratio)

    # calculate normals
    print("calculating normals...")
    pointcloud_down_sampled.estimate_normals(fast_normal_computation=True)
    pointcloud_down_sampled.orient_normals_consistent_tangent_plane(8)
    pointcloud_down_sampled.normals = o3d.utility.Vector3dVector(np.asarray(pointcloud_down_sampled.normals) * -1)

    # triangulation
    print("triangulating pointcloud...")
    mesh = o3d.geometry.TriangleMesh()
    mesh, densities = mesh.create_from_point_cloud_poisson(
        pcd=pointcloud_down_sampled,
        depth=depth,
        scale=scale,
        linear_fit=True
    )

    vertices = np.asarray(mesh.vertices)
    edges = []
    faces = np.asarray(mesh.triangles)

    blender_mesh = bpy.data.meshes.new('c2m_mesh')
    blender_mesh.from_pydata(vertices, edges, faces)
    blender_mesh.update()

    # add object to scene
    mesh_object = bpy.data.objects.new('c2m_mesh', blender_mesh)
    collection = bpy.data.collections.get(context.scene.collection_name)
    if collection is None:
        collection = bpy.data.collections.new(context.scene.collection_name)
    collection.objects.link(mesh_object)

    context.view_layer.objects.active = mesh_object

    bm = bmesh.new()
    bm.from_mesh(mesh=mesh_object.data)

    # add density property to my mesh
    density_layer = bm.verts.layers.float.new('c2m_density')
    bm.verts.ensure_lookup_table()
    for vert in bm.verts:
        vert.select_set(False)
        density = densities[vert.index]
        bm.verts[vert.index][density_layer] = density

    bm.to_mesh(mesh_object.data)
    bm.free()
    del mesh

    return {'FINISHED'}


def texture_mesh(self, context):
    global pointcloud

    if pointcloud is None:
        self.report({'ERROR'}, "Pointcloud is None.")
        return {'CANCELLED'}

    if context.view_layer.objects.active:

        # go to object mode
        bpy.ops.object.mode_set(mode="OBJECT")

        mesh_object = context.view_layer.objects.active
        if mesh_object.type != 'MESH':
            self.report({'ERROR'}, "Selected object is not of type 'MESH' ")
            return {'CANCELLED'}

        bm = bmesh.new()
        bm.from_mesh(mesh_object.data)

        # properties    
        output_name = context.scene.pointcloud_name + "_1"
        output_format = ".png"
        output_path = os.path.dirname(context.scene.texture_output_path)

        # if output_path is empty just use the path od the pointcloud
        if output_path == "":
            pointcloud_path = os.path.dirname(context.scene.pointcloud_path)
            output_path = pointcloud_path
        output_path = bpy.path.abspath(output_path)

        if not os.path.exists(output_path) and not os.path.isdir(output_path):
            print(f"Path does not exits: {output_path}")
            return {"CANCELLED"}

        output_path = os.path.join(output_path, output_name + output_format)

        color_search_radius = context.scene.color_search_radius
        color_max_neighbors = context.scene.color_max_neighbors
        tex_size = context.scene.texture_size
        sub_pixels = context.scene.texture_sub_pixels

        vertices = []
        triangles = []
        triangle_uvs = []

        bm = bmesh.new()
        bm.from_mesh(mesh_object.data)
        uv_layer = bm.loops.layers.uv.active

        bpy.ops.object.mode_set(mode="EDIT")

        # if no uv_layer was found calculate a new layer

        uv_layer = bm.loops.layers.uv.active
        if not uv_layer:
            print("Recalculating UVs ...")
            bpy.ops.mesh.select_all(action='SELECT')
            bpy.ops.uv.smart_project(scale_to_bounds=True)
            bm = bmesh.from_edit_mesh(mesh_object.data)
            uv_layer = bm.loops.layers.uv.active

        for vert in bm.verts:
            x, y, z = vert.co
            vertices.extend([x, y, z])

        for face in bm.faces:
            if len(face.verts) == 3:
                for loop in face.loops:
                    vertex_index = loop.vert.index
                    triangles.append(vertex_index)
                    uv = loop[uv_layer].uv
                    triangle_uvs.extend(uv)

        bm.free()
        bpy.ops.object.mode_set(mode="OBJECT")

        vert_count = int(len(vertices) / 3)
        vertices = np.asarray(vertices).reshape((vert_count, 3))
        triangle_count = int(len(triangles) / 3)
        triangles = np.asarray(triangles).reshape((triangle_count, 3))
        triangle_uvs = np.asarray(triangle_uvs).reshape((3 * triangle_count, 2))

        # Downsample Pointcloud
        size = context.scene.texturing_pointcloud_size
        point_count = int(len(pointcloud.points))
        ratio = max(0.0, min(size / point_count, 1.0))

        point_count = int(len(pointcloud.points))
        ratio = max(0.0, min(size / point_count, 1.0))

        pointcloud_downsampled = None
        if ratio < 0.99:
            print("Downsampling pointcloud...")
            pointcloud_down_sampled = pointcloud.random_down_sample(ratio)
        else:
            pointcloud_down_sampled = pointcloud

        print("Prepare texturing...")

        tree = o3d.geometry.KDTreeFlann(pointcloud_down_sampled)
        point_colors = np.asarray(pointcloud_down_sampled.colors)

        width = tex_size
        height = tex_size
        pixel_width = 1 / width
        pixel_height = 1 / height

        subpixel_width = 1 / math.sqrt(sub_pixels)
        subpixel_height = 1 / math.sqrt(sub_pixels)

        colors = np.zeros((width, height, 3))

        subpixel_hits = np.zeros((width, height), dtype=np.int32)

        print(f"Points in pointcloud: {len(pointcloud_down_sampled.points)}")
        print(f"Triangles in mesh   : {triangle_count}")

        # utility functions
        def barycentric(px: float, py: float, ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> \
        Tuple[float, float, float]:
            v0x = bx - ax
            v0y = by - ay
            v1x = cx - ax
            v1y = cy - ay
            v2x = px - ax
            v2y = py - ay
            den = v0x * v1y - v1x * v0y
            if den != 0.0:
                v = (v2x * v1y - v1x * v2y) / den
                w = (v0x * v2y - v2x * v0y) / den
                u = 1.0 - v - w
                return u, v, w
            else:
                return -1.0, -1.0, -1.0

        def map_to_range(value: float, value_min: float, value_max: float, range_min: float, range_max: float) -> float:
            return range_min + (float(value - value_min) / float(value_max - value_min) * (range_max - range_min))

        def calculate_intersections(x1, y1, x2, y2, x3, y3, y):
            intersections = []
            
            # Sortiere die Eckpunkte nach y-Koordinaten
            if y1 > y2:
                x1, y1, x2, y2 = x2, y2, x1, y1
            if y1 > y3:
                x1, y1, x3, y3 = x3, y3, x1, y1
            if y2 > y3:
                x2, y2, x3, y3 = x3, y3, x2, y2
                

            if y1 <= y <= y2 and y1 != y2:
                x = x1 + (x2 - x1) * (y - y1) / (y2 - y1)
                intersections.append(x)

            if y2 <= y <= y3 and y2 != y3:
                x = x2 + (x3 - x2) * (y - y2) / (y3 - y2)
                intersections.append(x)

            if y1 <= y <= y3 and y1 != y3:
                x = x1 + (x3 - x1) * (y - y1) / (y3 - y1)
                intersections.append(x)

            return intersections

        for i, triangle in tqdm.tqdm(enumerate(triangles), total=triangle_count, desc="Texturing"):

            # calculate bounding box
            min_u = np.min(triangle_uvs[3 * i:3 * i + 3, 0])
            max_u = np.max(triangle_uvs[3 * i:3 * i + 3, 0])
            min_v = np.min(triangle_uvs[3 * i:3 * i + 3, 1])
            max_v = np.max(triangle_uvs[3 * i:3 * i + 3, 1])

            min_u_pixel = max(0, math.floor(min_u * width) - 1)
            max_u_pixel = min(width, math.floor(max_u * width) + 1)
            min_v_pixel = max(0, math.floor(min_v * height) - 1)
            max_v_pixel = min(height, math.floor(max_v * height) + 1)

            # for each pixel
            for act_height in range(min_v_pixel, max_v_pixel):
                # get scanline height
                
                scanline = (min_v_pixel + act_height + 0.5) / height

                # calculate triangle intersections
                intersections = calculate_intersections(triangle_uvs[3 * i + 0][0], triangle_uvs[3 * i + 0][1],
                                                        triangle_uvs[3 * i + 1][0], triangle_uvs[3 * i + 1][1],
                                                        triangle_uvs[3 * i + 2][0], triangle_uvs[3 * i + 2][1],
                                                        scanline)
                intersections = [x for x in intersections if min_u <= x <= max_u]

                new_min = min_u_pixel
                new_max = max_u_pixel
                #if intersections:
                #    new_min = math.floor(min(intersections) * width)
                #    new_max = math.floor(max(intersections) * width) 

                for act_width in range(new_min, new_max):

                    color = np.array([0.0, 0.0, 0.0])
                    normal = np.array([0.0, 0.0, 0.0])

                    # for each pixel, sample points to check if pixel is inside a triangle
                    # mid point of pixel is used every time
                    # if 'texture_pixel_corners' is active, also the corner points of each pixel are sampled
                    # if 'sub_pixel' are active (> 1) more sample points across the pixel are sampled

                    p = [(act_width + 0.5) * pixel_width,
                         (act_height + 0.5) * pixel_height]

                    pixel_positions = [p]

                    # for each subpixel
                    if sub_pixels > 1:
                        for p_h in range(int(math.sqrt(sub_pixels))):
                            for p_w in range(int(math.sqrt(sub_pixels))):
                                # iterate over every corners of a pixel
                                p = [(act_width + (p_w + 0.5) * subpixel_width) * pixel_width,
                                     (act_height + (p_h + 0.5) * subpixel_height) * pixel_height]
                                pixel_positions.append(p)

                    if context.scene.texture_pixel_corners:
                       
                        #if act_width == new_min or act_width == new_max:
                        for corner in range(4):
                            p = [(act_width + (2.0 * (corner % 2) * 0.5)) * pixel_width,
                                 (act_height + (2.0 * (corner // 2) * 0.5)) * pixel_height]
                            pixel_positions.append(p)
                        


                    for pixel_pos in pixel_positions:

                        alpha, beta, gamma = barycentric(pixel_pos[0], pixel_pos[1],
                                                         triangle_uvs[3 * i + 0][0], triangle_uvs[3 * i + 0][1],
                                                         triangle_uvs[3 * i + 1][0], triangle_uvs[3 * i + 1][1],
                                                         triangle_uvs[3 * i + 2][0], triangle_uvs[3 * i + 2][1])

                        if 0.0 <= alpha  and 0.0 <= beta <= 1.0 and 0.0 <= gamma <= 1.0:

                            # if barycentric coordinates are positive the pixel position lays within the triangle
                            v_a = vertices[triangles[i][0]]
                            v_b = vertices[triangles[i][1]]
                            v_c = vertices[triangles[i][2]]

                            pos = alpha * v_a + beta * v_b + gamma * v_c

                            # colors
                            v, n_vertices, n_distances = tree.search_hybrid_vector_3d(query=pos,
                                                                                      radius=color_search_radius,
                                                                                      max_nn=5)

                            # just take the nearest vertex if no neighbours were found in search radius
                            if not len(n_vertices):
                                nearest = tree.search_knn_vector_3d(query=pos, knn=1)[1][0]
                                color += np.copy(point_colors[nearest])

                            else:
                                weights = np.array(
                                    [map_to_range(n_distances[j], 0.0, color_search_radius, 1.0, 0.0) for j in
                                     range(len(n_vertices))])
                                # The weights of all neighbors should sum up to 1, this way we keep the initial color brightness
                                weights_normalized = weights / np.sum(weights)
                                for j in range(len(n_vertices)):
                                    color += point_colors[n_vertices[j]] * weights_normalized[j]

                            subpixel_hits[act_height, act_width] += 1

                    colors[act_height, act_width] += color

        nonzero_indices = subpixel_hits != 0

        colors[nonzero_indices] /= subpixel_hits[nonzero_indices][:, None]
        colors[nonzero_indices] *= 255
        colors = colors.astype(np.uint8)

        # export texture
        color_texture = o3d.geometry.Image(colors)
        o3d.io.write_image(str(output_path), color_texture.flip_vertical())

        # create material with new texture
        mat = bpy.data.materials.new(name="c2m_Material")
        mesh_object.data.materials.append(mat)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        output_node = nodes.get("Material Output")
        texture_node = nodes.new(type='ShaderNodeTexImage')
        texture_node.location = (-200, 0)
        texture_node.image = bpy.data.images.load(output_path)
        mat.node_tree.links.new(texture_node.outputs["Color"], output_node.inputs["Surface"])

        # clean up
        del tree
        if ratio < 0.99:
            del pointcloud_downsampled

        return {'FINISHED'}
    else:
        self.report({'ERROR'}, "No active mesh.")
        return {'CANCELLED'}


def remove_vertices(self, context):
    if context.view_layer.objects.active:
        mesh_object = context.view_layer.objects.active
        if mesh_object.type != 'MESH':
            print("Selected object is not of type 'MESH'")
            return

        bpy.ops.object.mode_set(mode="EDIT")

        bm = bmesh.from_edit_mesh(mesh_object.data)

        density_layer = bm.verts.layers.float.get('c2m_density')

        removal_threshold = context.scene.triangulation_removal_threshold

        if density_layer:
            bpy.ops.mesh.select_mode(type="VERT")
            bpy.ops.mesh.select_all(action='DESELECT')

            # fetch density values
            densities = []
            bm.verts.ensure_lookup_table()
            densities = [bm.verts[vert.index][density_layer] for vert in bm.verts]

            # quantile (removing a certain percentage)
            vertices_to_remove = densities < np.quantile(densities, removal_threshold)
            for vert in bm.verts:
                if vertices_to_remove[vert.index] != 0:
                    bm.verts[vert.index].select_set(True)

            bmesh.update_edit_mesh(mesh_object.data)

            bm.select_flush(True)
            bpy.ops.mesh.select_mode(type="FACE")
            bm.free()
        else:
            bm.free()
            print("Mesh has no density layer.")
            return

    else:
        print("No active mesh.")
        return


class ReadPointCloud(Operator):
    bl_idname = "c2m.read_point_cloud"
    bl_label = "Read Pointcloud"

    @classmethod
    def poll(cls, context):
        return True
        # return (pointcloud is not None and not context.scene.is_triangulating_pointcloud and not context.scene.is_texturing_mesh)

    def execute(self, context):

        result = {'FINISHED'}
        print("\nREADING POINTCLOUD ...")
        try:
            result = read_pointcloud(self, context)
        except Exception as e:
            print(e)
        finally:
            print("READING POINTCLOUD DONE")
            return result


class TriangulatePointCloud(Operator):
    bl_idname = "c2m.triangulate_point_cloud"
    bl_label = "Triangulate Pointcloud"

    def execute(self, context):

        result = {'FINISHED'}
        print("\nTRIANGULATING POINTCLOUD ...")
        try:
            result = triangulate_pointcloud(self, context)
        except Exception as e:
            print(e)
        finally:
            print("TRIANGULATING POINTCLOUD DONE")
            return result


class TextureMesh(Operator):
    bl_idname = "c2m.texturing_mesh"
    bl_label = "Texture Mesh"

    def execute(self, context):

        result = {'FINISHED'}
        print("\nTEXTURING MESH ...")

        try:
            result = texture_mesh(self, context)
        except Exception as e:
            print(e)
        finally:
            print("TEXTURING MESH DONE")
            return result


class DecimateGeometry(Operator):
    bl_idname = "c2m.decimate_geometry"
    bl_label = "Decimate geometry"

    def execute(self, context):
        if context.view_layer.objects.active:
            mesh_object = context.view_layer.objects.active
            if mesh_object.type != 'MESH':
                return {'CANCELLED'}
            bpy.ops.object.mode_set(mode="EDIT")

        return {'FINISHED'}


class Cloud2MeshPanel(bpy.types.Panel):
    """ Settings for conversion """
    bl_label = "Cloud2Mesh"
    bl_idname = "cloud_to_mesh"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "c2m"

    def draw(self, context):
        global pointcloud

        layout = self.layout
        col = layout.column()

        # read pointcloud
        col.prop(context.scene, "pointcloud_path", text="pointcloud path")
        col.operator("c2m.read_point_cloud")

        if pointcloud is not None:
            name = context.scene.pointcloud_name

            box1 = col.box()
            box1.label(text=f"Name: {name}")
            box1.label(text=f"Points: {len(np.asarray(pointcloud.points))}")
            col.separator()

            # triangulate pointcloud
            col.operator("c2m.triangulate_point_cloud")

            # texuting mesh
            col.operator("c2m.texturing_mesh")


class UtilityPanel(bpy.types.Panel):
    """ Settings for conversion """
    bl_label = "Utility"
    bl_idname = "utility"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "c2m"

    def draw(self, context):
        row = self.layout.row()
        row.label(text="Vertex removal threshold:")
        row.prop(context.scene, "triangulation_removal_threshold", text="")

        # row2 = self.layout.row()
        # row2.label(text="Decimate geometry:")
        # row2.operator("c2m.decimate_geometry")


class SettingsPanel(bpy.types.Panel):
    """ Settings for conversion """
    bl_label = "Settings"
    bl_idname = "settings"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "c2m"
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout

        # box1 = layout.box()
        # box1.label(text="Pointcloud")
        # box1.prop(context.scene, "visualize_pointcloud", text="Visualize Pointcloud on load")

        box2 = layout.box()
        box2.label(text="Triangulation")
        box2.prop(context.scene, "pointcloud_downsampling_size", text="Pointcloud downsampling size")
        box2.prop(context.scene, "triangulation_depth", text="Triangulation depth")
        box2.prop(context.scene, "triangulation_scale", text="Triangulation scale")

        box3 = layout.box()
        box3.label(text="Texturing")
        box3.prop(context.scene, "texture_output_path", text="     Texture output path")
        box3.prop(context.scene, "texture_size", text="Texture size")
        box3.prop(context.scene, "texture_sub_pixels", text="Texture sub pixels")
        box3.prop(context.scene, "texturing_pointcloud_size", text="Texturing pointcloud size")
        box3.prop(context.scene, "color_search_radius", text="Color search radius")
        box3.prop(context.scene, "color_max_neighbors", text="Color max neighbors")
        box3.prop(context.scene, "texture_pixel_corners", text="pixel corners")


classes = (
    Cloud2MeshPanel,
    UtilityPanel,
    SettingsPanel,
    ReadPointCloud,
    TriangulatePointCloud,
    TextureMesh,
    DecimateGeometry
)


def register():
    from bpy.utils import register_class
    for cls in classes:
        register_class(cls)
    Scene.collection_name = StringProperty(name="collection_name", default="C2M Collection")
    Scene.pointcloud_path = StringProperty(name="pointcloud_path", subtype="FILE_PATH")
    Scene.pointcloud_name = StringProperty(name="pointcloud_name", default="")

    # Pointcloud properties
    Scene.visualize_pointcloud = BoolProperty(name="visualize_pointcloud", default=False)

    # Triangulation properties
    Scene.pointcloud_downsampling_size = IntProperty(name="pointcloud_downsampling_size", default=100_000)
    Scene.triangulation_depth = IntProperty(name="triangulation_depth", default=11, min=0)
    Scene.triangulation_scale = FloatProperty(name="triangulation_scale", default=1.1, soft_min=0.0)
    Scene.triangulation_removal_threshold = FloatProperty(name="vertex_removal_threshold", default=0.05, soft_min=0.0,
                                                          soft_max=1.0, description="Removal Slider", step=0.5,
                                                          precision=3, update=remove_vertices)

    # Texturing
    Scene.texture_output_path = StringProperty(name="texture_output_path", subtype="DIR_PATH", default="")
    Scene.color_search_radius = IntProperty(name="color_search_radius", default=1, min=0)
    Scene.color_max_neighbors = IntProperty(name="color_max_neighbors", default=1, min=0)
    Scene.texture_size = IntProperty(name="texture_size", default=1024, min=0)
    Scene.texture_sub_pixels = IntProperty(name="texture_sub_pixels", default=1)
    Scene.texture_pixel_corners = BoolProperty(name="texture_pixel_corners", default=False)
    Scene.texturing_pointcloud_size = IntProperty(name="texturing_pointcloud_size", default=1_000_000, min=0)


def unregister():
    from bpy.utils import unregister_class
    for cls in classes:
        unregister_class(cls)
    del Scene.pointcloud_path
    del Scene.collection_name
    del Scene.pointcloud_name

    del Scene.visualize_pointcloud

    del Scene.triangulation_depth
    del Scene.triangulation_scale
    del Scene.triangulation_removal_threshold

    del Scene.texture_output_path
    del Scene.color_search_radius
    del Scene.color_max_neighbors
    del Scene.texture_size
    del Scene.texture_sub_pixels
    del Scene.texture_pixel_corners
    del Scene.texturing_pointcloud_size

    global pointcloud
    if pointcloud:
        del pointcloud


if __name__ == "__main__":
    register()
