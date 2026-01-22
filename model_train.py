from os import path as osp
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage import correlate
import scipy.ndimage as ndimage
from scipy import interpolate
import cv2
import argparse

import sys
sys.path.append("..")
from Basics.RawData import RawData
from Basics.CalibData import CalibData
import Basics.params as pr
import Basics.sensorParams as psp
from tqdm import tqdm
import os
import open3d as o3d
import warnings
from matplotlib import colormaps
import trimesh
import pyrender

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm, trange
from taxim_render import TaximRender
from PIL import Image
import TouchNet_utils
import TouchNet_model
from taxim_render import TaximRender

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="train", help='Choose train mode')
parser.add_argument("-obj", nargs='?', default='model',
                    help="Name of Object to be tested, supported_objects_list = [square, cylinder6]")
parser.add_argument('-obj_path', default = './demo', type=str, help='Directory containing object point cloud')
parser.add_argument('-depth', default = 3.0, type=float, help='Indetation depth into the gelpad.')
parser.add_argument('-obj_scale_factor', default = 700.0, type=float, help='Scale factor to multiply to object before simulation.')
parser.add_argument('-depth_range_info', default = [0.1, 1.5, 100.], type=float, help='Indetation depth range information (min_depth, max_depth, num_range) into the gelpad.', nargs=3)
parser.add_argument('-slide_range_info', default = [-100., 100., -100., 100., 100., 1.0], type=float, help='Sliding range information (min_x, max_x, min_y, max_y, num_range, press_depth) into the gelpad.', nargs=6)
parser.add_argument('-rot_range_info', default = [0.3, 0.3, 0.3, 100., 1.0], type=float, help='Rotating range information (yaw_amplitude, pitch_amplitude, roll_amplitude, num_range, press_depth) into the gelpad.', nargs=5)
parser.add_argument('-contact_point', default = None, type=float, help='Contact point location', nargs=3)
parser.add_argument('-contact_theta', default = None, type=float, help='Contact point rotation angle')

parser.add_argument("--sample_ply", type=str, default="model.ply", help='ply file for dataset')
parser.add_argument("--object_model", type=str, default="ObjectFile.pth", help='model path')
parser.add_argument("--object_file_path", type=str, default="demo/ObjectFile.pth", help='ObjectFile path')
parser.add_argument('--touch_vertices_file_path', default='./demo/touch_vertices.npy', help='The path of the testing vertices file for touch, which should be a npy file.')
parser.add_argument('--touch_gelinfo_file_path', default='./demo/touch_gelinfo.npy', help='The path of the gel configurations for touch, which should be a npy file.')
parser.add_argument('--touch_results_dir', type=str, default='./results/touch/', help='The path of the touch results directory to save rendered tactile RGB images.')

parser.add_argument('--initialization', type=str, default="random", help="Initialization method for touchnet training")
args = parser.parse_args()


def rot_from_ypr(ypr_array):
    def _ypr2mtx(ypr):
        # ypr is assumed to have a shape of [3, ]
        yaw, pitch, roll = ypr
        yaw = yaw.reshape(1)
        pitch = pitch.reshape(1)
        roll = roll.reshape(1)

        tensor_0 = np.zeros(1, device=yaw.device)
        tensor_1 = np.ones(1, device=yaw.device)

        RX = np.stack([
                        np.stack([tensor_1, tensor_0, tensor_0]),
                        np.stack([tensor_0, np.cos(roll), -np.sin(roll)]),
                        np.stack([tensor_0, np.sin(roll), np.cos(roll)])]).reshape(3, 3)

        RY = np.stack([
                        np.stack([np.cos(pitch), tensor_0, np.sin(pitch)]),
                        np.stack([tensor_0, tensor_1, tensor_0]),
                        np.stack([-np.sin(pitch), tensor_0, np.cos(pitch)])]).reshape(3, 3)

        RZ = np.stack([
                        np.stack([np.cos(yaw), -np.sin(yaw), tensor_0]),
                        np.stack([np.sin(yaw), np.cos(yaw), tensor_0]),
                        np.stack([tensor_0, tensor_0, tensor_1])]).reshape(3, 3)

        R = RZ @ RY
        R = R @ RX

        return R

    if len(ypr_array.shape) == 1:
        return _ypr2mtx(ypr_array)
    else:
        tot_mtx = []
        for ypr in ypr_array:
            tot_mtx.append(_ypr2mtx(ypr))
        return np.stack(tot_mtx)


def skew(v):
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def rotation_from_axis_angle(axis, angle):
    axis = axis / np.linalg.norm(axis)
    K = skew(axis)
    I = np.eye(3)
    return I + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def align_normals(n1, n2):
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)

    v = np.cross(n1, n2)
    c = np.dot(n1, n2)

    if np.linalg.norm(v) < 1e-8:
        # n1 and n2 are parallel or antiparallel
        if c > 0:
            return np.eye(3)
        else:
            # 180-degree rotation around any axis orthogonal to n1
            a = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(a, n1)) > 0.9:
                a = np.array([0.0, 1.0, 0.0])
            axis = np.cross(n1, a)
            axis /= np.linalg.norm(axis)
            return rotation_from_axis_angle(axis, np.pi)

    axis = v / np.linalg.norm(v)
    angle = np.arccos(np.clip(c, -1.0, 1.0))
    return rotation_from_axis_angle(axis, angle)


def rotation_family(n1, n2, theta):
    """
    Returns a rotation R(theta) such that R(theta) @ n1 = n2
    """
    n1 = n1 / np.linalg.norm(n1)

    R0 = align_normals(n1, n2)
    R_spin = rotation_from_axis_angle(n1, theta)

    return R0 @ R_spin


class simulator(object):
    def __init__(self, data_folder, filePath, obj, obj_scale_factor=1.):
        """
        Initialize the simulator.
        1) load the object,
        2) load the calibration files,
        3) generate shadow table from shadow masks
        """
        # read in object's ply file
        # object facing positive direction of z axis
        objPath = osp.join(filePath,obj)
        self.obj_name = obj.split('.')[0]
        print("load object: " + self.obj_name)

        pcd = o3d.io.read_point_cloud(objPath)
        self.obj_scale_factor = obj_scale_factor
        self.vertices = np.asarray(pcd.points) * obj_scale_factor

        # Paint with uniform color if no colors exist
        if len(pcd.colors) == 0:
            pcd = pcd.paint_uniform_color([0.5, 0.5, 0.5])
        self.colors = np.asarray(pcd.colors)

        if len(pcd.normals) == 0:  # Esimate normals if none exist
            warnings.warn("No normals exist, resorting to Open3D normal estimation which is noisy")
            pcd.estimate_normals()
        self.vert_normals = np.asarray(pcd.normals)

        # polytable
        calib_data = osp.join(data_folder, "polycalib.npz")
        self.calib_data = CalibData(calib_data)

        # raw calibration data
        rawData = osp.join(data_folder, "dataPack.npz")
        data_file = np.load(rawData,allow_pickle=True)
        self.f0 = data_file['f0']
        self.bg_proc = self.processInitialFrame()

        #shadow calibration
        self.shadow_depth = [0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]
        shadowData = np.load(osp.join(data_folder, "shadowTable.npz"),allow_pickle=True)
        self.direction = shadowData['shadowDirections']
        self.shadowTable = shadowData['shadowTable']

    def processInitialFrame(self):
        """
        Smooth the initial frame
        """
        # gaussian filtering with square kernel with
        # filterSize : kscale*2+1
        # sigma      : kscale
        kscale = pr.kscale

        img_d = self.f0.astype('float')
        convEachDim = lambda in_img :  gaussian_filter(in_img, kscale)

        f0 = self.f0.copy()
        for ch in range(img_d.shape[2]):
            f0[:,:, ch] = convEachDim(img_d[:,:,ch])

        frame_ = img_d

        # Checking the difference between original and filtered image
        diff_threshold = pr.diffThreshold
        dI = np.mean(f0-frame_, axis=2)
        idx =  np.nonzero(dI<diff_threshold)

        # Mixing image based on the difference between original and filtered image
        frame_mixing_per = pr.frameMixingPercentage
        h,w,ch = f0.shape
        pixcount = h*w

        for ch in range(f0.shape[2]):
            f0[:,:,ch][idx] = frame_mixing_per*f0[:,:,ch][idx] + (1-frame_mixing_per)*frame_[:,:,ch][idx]

        return f0

    def simulating(self, heightMap, contact_mask, contact_height, shadow=False):
        """
        Simulate the tactile image from the height map
        heightMap: heightMap of the contact
        contact_mask: indicate the contact area
        contact_height: the height of each pix
        shadow: whether add the shadow

        return:
        sim_img: simulated tactile image w/o shadow
        shadow_sim_img: simluated tactile image w/ shadow
        """

        # generate gradients of the height map
        grad_mag, grad_dir = self.generate_normals(heightMap)

        # generate raw simulated image without background
        sim_img_r = np.zeros((psp.h,psp.w,3))
        bins = psp.numBins

        [xx, yy] = np.meshgrid(range(psp.w), range(psp.h))
        xf = xx.flatten()
        yf = yy.flatten()
        A = np.array([xf*xf,yf*yf,xf*yf,xf,yf,np.ones(psp.h*psp.w)]).T
        binm = bins - 1

        # discritize grids
        x_binr = 0.5*np.pi/binm # x [0,pi/2]
        y_binr = 2*np.pi/binm # y [-pi, pi]

        idx_x = np.floor(grad_mag/x_binr).astype('int')
        idx_y = np.floor((grad_dir+np.pi)/y_binr).astype('int')

        # look up polynomial table and assign intensity
        params_r = self.calib_data.grad_r[idx_x,idx_y,:]
        params_r = params_r.reshape((psp.h*psp.w), params_r.shape[2])
        params_g = self.calib_data.grad_g[idx_x,idx_y,:]
        params_g = params_g.reshape((psp.h*psp.w), params_g.shape[2])
        params_b = self.calib_data.grad_b[idx_x,idx_y,:]
        params_b = params_b.reshape((psp.h*psp.w), params_b.shape[2])

        est_r = np.sum(A * params_r,axis = 1)
        est_g = np.sum(A * params_g,axis = 1)
        est_b = np.sum(A * params_b,axis = 1)

        sim_img_r[:,:,0] = est_r.reshape((psp.h,psp.w))
        sim_img_r[:,:,1] = est_g.reshape((psp.h,psp.w))
        sim_img_r[:,:,2] = est_b.reshape((psp.h,psp.w))

        # attach background to simulated image
        sim_img = sim_img_r + self.bg_proc

        if not shadow:
            return sim_img, sim_img

        # add shadow
        cx = psp.w//2
        cy = psp.h//2

        # find shadow attachment area
        kernel = np.ones((5, 5), np.uint8)
        dialate_mask = cv2.dilate(np.float32(contact_mask),kernel,iterations = 2)
        enlarged_mask = dialate_mask == 1
        boundary_contact_mask = 1*enlarged_mask - 1*contact_mask
        contact_mask = boundary_contact_mask == 1

        # (x,y) coordinates of all pixels to attach shadow
        x_coord = xx[contact_mask]
        y_coord = yy[contact_mask]

        # get normal index to shadow table
        normMap = grad_dir[contact_mask] + np.pi
        norm_idx = normMap // pr.discritize_precision
        # get height index to shadow table
        contact_map = contact_height[contact_mask]
        height_idx = (contact_map * psp.pixmm - self.shadow_depth[0]) // pr.height_precision
        total_height_idx = self.shadowTable.shape[2]

        shadowSim = np.zeros((psp.h,psp.w,3))

        # all 3 channels
        for c in range(3):
            frame = sim_img_r[:,:,c].copy()
            frame_back = sim_img_r[:,:,c].copy()
            for i in range(len(x_coord)):
                # get the coordinates (x,y) of a certain pixel
                cy_origin = y_coord[i]
                cx_origin = x_coord[i]
                # get the normal of the pixel
                n = int(norm_idx[i])
                # get height of the pixel
                h = int(height_idx[i]) + 6
                if h < 0 or h >= total_height_idx:
                    continue
                # get the shadow list for the pixel
                v = self.shadowTable[c,n,h]

                # number of steps
                num_step = len(v)

                # get the shadow direction
                theta = self.direction[n]
                d_theta = theta
                ct = np.cos(d_theta)
                st = np.sin(d_theta)
                # use a fan of angles around the direction
                theta_list = np.arange(d_theta-pr.fan_angle, d_theta+pr.fan_angle, pr.fan_precision)
                ct_list = np.cos(theta_list)
                st_list = np.sin(theta_list)
                for theta_idx in range(len(theta_list)):
                    ct = ct_list[theta_idx]
                    st = st_list[theta_idx]

                    for s in range(1,num_step):
                        cur_x = int(cx_origin + pr.shadow_step * s * ct)
                        cur_y = int(cy_origin + pr.shadow_step * s * st)
                        # check boundary of the image and height's difference
                        if cur_x >= 0 and cur_x < psp.w and cur_y >= 0 and cur_y < psp.h and heightMap[cy_origin,cx_origin] > heightMap[cur_y,cur_x]:
                            frame[cur_y,cur_x] = np.minimum(frame[cur_y,cur_x],v[s])

            shadowSim[:,:,c] = frame
            shadowSim[:,:,c] = ndimage.gaussian_filter(shadowSim[:,:,c], sigma=(pr.sigma, pr.sigma), order=0)

        shadow_sim_img = shadowSim+ self.bg_proc
        shadow_sim_img = cv2.GaussianBlur(shadow_sim_img.astype(np.float32),(pr.kernel_size,pr.kernel_size),0)
        return sim_img, shadow_sim_img

    def generateHeightMap(self, gelpad_model_path, pressing_height_mm, dx, dy, contact_jitter_rot_mtx=None, contact_point=None, contact_theta=0.):
        """
        Generate the height map by interacting the object with the gelpad model.
        pressing_height_mm: pressing depth in millimeter
        dx, dy: shift of the object
        return:
        zq: the interacted height map
        gel_map: gelpad height map
        contact_mask: indicate contact area
        """
        # NOTE 1: Tactile sensor is placed at the x,y location of object center with z location at maximum object height, and object points with height over a threshold (0.2) are all considered
        # NOTE 2: Tactile sensor is placed oppositely facing the object placed on top of a virtual plane with z=0, although the object can be "floating"
        # NOTE 3: Currently normals are stored in "raw" xyz coordinates. To transform between two normals, we represent rotation as a 1-parameter family of transformations, controlled by contact_theta.
        # NOTE 4: This contact_theta controls the variation in contact rotations, or "rolling" on the contact point's tangent plane
        # NOTE 5: contact_jitter_rot_mtx additionally applies rotation to a contact point location to enable rotation other than tangent plane rolling

        assert(self.vertices.shape[1] == 3)
        # load dome-shape gelpad model
        gel_map = np.load(gelpad_model_path)
        gel_map = cv2.GaussianBlur(gel_map.astype(np.float32),(pr.kernel_size,pr.kernel_size),0)
        heightMap = np.zeros((psp.h,psp.w))

        # Raw color and normal maps
        rawcolorMap = np.zeros((psp.h,psp.w,3))
        rawnormalMap = np.zeros((psp.h,psp.w,3))

        # Identify original contact points
        orig_cx = np.mean(self.vertices[:,0])
        orig_cy = np.mean(self.vertices[:,1])
        xy_dist = np.linalg.norm(self.vertices[:, [0, 1]] - np.array([orig_cx, orig_cy]), axis=-1)
        kth = min(100, xy_dist.shape[0] // 2)  # NOTE: This is an arbitrarily chosen number
        topk_idx = np.argpartition(xy_dist, kth=kth)[:kth]
        orig_cz = self.vertices[topk_idx, 2].max()

        # Original concact point and normal direction
        orig_contact = np.array([orig_cx, orig_cy, orig_cz])
        orig_normal = self.vert_normals[np.linalg.norm(self.vertices - orig_contact, axis=-1).argmin()]

        # Copy original vertex set and normals
        sim_vertices = np.copy(self.vertices)
        sim_vert_normals = np.copy(self.vert_normals)

        # Set contact points given as array of shape (3, )
        if contact_point is not None:
            cx = contact_point[0] * self.obj_scale_factor
            cy = contact_point[1] * self.obj_scale_factor
            cz = contact_point[2] * self.obj_scale_factor

            # New contact point and normal direction
            new_contact = np.array([cx, cy, cz])
            new_normal = sim_vert_normals[np.linalg.norm(self.vertices - new_contact, axis=-1).argmin()]

            # Estimate rotation matrix that aligns new normal to the positive z direction
            contact_rot_mtx = rotation_family(new_normal, np.array([0, 0, 1]), contact_theta)

            # Fix contact point and rotate points
            sim_vertices = (sim_vertices - new_contact) @ contact_rot_mtx.T
            sim_vert_normals = sim_vert_normals @ contact_rot_mtx.T
        else:
            cx = orig_cx
            cy = orig_cy
            cz = orig_cz
            contact = np.array([cx, cy, cz])
            sim_vertices = (sim_vertices - contact) @ contact_rot_mtx.T

        if contact_jitter_rot_mtx is not None:
            sim_vertices = sim_vertices @ contact_jitter_rot_mtx.T
            sim_vert_normals = sim_vert_normals @ contact_jitter_rot_mtx.T
        else:
            sim_vert_normals = np.copy(sim_vert_normals)

        # Ensure minimum height is 0. during rendering
        sim_vertices[:, 2] -= sim_vertices[:, 2].min()
        cz = 0.

        # add the shifting and change to the pix coordinate
        uu = ((sim_vertices[:,0])/psp.pixmm + psp.w//2+dx).astype(int)
        vv = ((sim_vertices[:,1])/psp.pixmm + psp.h//2+dy).astype(int)
        vv = psp.h - vv  # NOTE: This is needed to ensure consistency with pyrender's orthographic rendering

        # check boundary of the image
        mask_u = np.logical_and(uu > 0, uu < psp.w)
        mask_v = np.logical_and(vv > 0, vv < psp.h)
        # check the depth
        mask_map = mask_u & mask_v
        heightMap[vv[mask_map],uu[mask_map]] = sim_vertices[mask_map][:,2]/psp.pixmm  # NOTE: We don't re-normalize with minimum value as point projections have holes, causing inaccurate minimum values

        # Fill in raw color and normals
        rawcolorMap[vv[mask_map],uu[mask_map]] = self.colors[mask_map]
        rawnormalMap[vv[mask_map],uu[mask_map]] = sim_vert_normals[mask_map]

        # Normal map for visualization
        vis_rawnormalMap = np.copy(rawnormalMap)
        vis_rawnormalMap[vv[mask_map],uu[mask_map]] = (rawnormalMap[vv[mask_map],uu[mask_map]] + 1.0) * 0.5
        vis_rawnormalMap = np.clip(vis_rawnormalMap, 0, 1)

        max_g = np.max(gel_map)
        min_g = np.min(gel_map)
        max_o = np.max(heightMap)
        # pressing depth in pixel
        pressing_height_pix = pressing_height_mm/psp.pixmm

        # shift the gelpad to interact with the object
        gel_map = -1 * gel_map + (max_g+max_o-pressing_height_pix)  # RHS is gel height map assuming object placed at z = 0

        # get the contact area
        contact_mask = heightMap > gel_map

        # combine contact area of object shape with non contact area of gelpad shape
        zq = np.zeros((psp.h,psp.w))

        zq[contact_mask]  = heightMap[contact_mask]
        zq[~contact_mask] = gel_map[~contact_mask]

        return zq, gel_map, contact_mask, rawcolorMap, rawnormalMap, vis_rawnormalMap, heightMap

    def deformApprox(self, pressing_height_mm, height_map, gel_map, contact_mask):
        zq = height_map.copy()
        zq_back = zq.copy()
        pressing_height_pix = pressing_height_mm/psp.pixmm
        # contact mask which is a little smaller than the real contact mask
        mask = (zq-(gel_map)) > pressing_height_pix * pr.contact_scale
        mask = mask & contact_mask

        # approximate soft body deformation with pyramid gaussian_filter
        for i in range(len(pr.pyramid_kernel_size)):
            zq = cv2.GaussianBlur(zq.astype(np.float32),(pr.pyramid_kernel_size[i],pr.pyramid_kernel_size[i]),0)
            zq[mask] = zq_back[mask]
        zq = cv2.GaussianBlur(zq.astype(np.float32),(pr.kernel_size,pr.kernel_size),0)

        contact_height = zq - gel_map

        return zq, mask, contact_height

    def interpolate(self,img):
        """
        fill the zero value holes with interpolation
        """
        x = np.arange(0, img.shape[1])
        y = np.arange(0, img.shape[0])
        # mask invalid values
        array = np.ma.masked_where(img == 0, img)
        xx, yy = np.meshgrid(x, y)
        # get the valid values
        x1 = xx[~array.mask]
        y1 = yy[~array.mask]
        newarr = img[~array.mask]

        GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                                  (xx, yy),
                                     method='linear', fill_value = 0) # cubic # nearest # linear
        return GD1

    def generate_normals(self,height_map):
        """
        get the gradient (magnitude & direction) map from the height map
        """
        [h,w] = height_map.shape
        top = height_map[0:h-2,1:w-1] # z(x-1,y)
        bot = height_map[2:h,1:w-1] # z(x+1,y)
        left = height_map[1:h-1,0:w-2] # z(x,y-1)
        right = height_map[1:h-1,2:w] # z(x,y+1)
        dzdx = (bot-top)/2.0
        dzdy = (right-left)/2.0

        mag_tan = np.sqrt(dzdx**2 + dzdy**2)
        grad_mag = np.arctan(mag_tan)
        invalid_mask = mag_tan == 0
        valid_mask = ~invalid_mask
        grad_dir = np.zeros((h-2,w-2))
        grad_dir[valid_mask] = np.arctan2(dzdx[valid_mask]/mag_tan[valid_mask], dzdy[valid_mask]/mag_tan[valid_mask])

        grad_mag = self.padding(grad_mag)
        grad_dir = self.padding(grad_dir)
        return grad_mag, grad_dir

    def padding(self,img):
        """ pad one row & one col on each side """
        return np.pad(img, ((1, 1), (1, 1)), 'symmetric')


class mesh_simulator(simulator):
    def __init__(self, data_folder, filePath, obj, obj_scale_factor=1.):
        """
        Initialize the simulator.
        1) load the object,
        2) load the calibration files,
        3) generate shadow table from shadow masks
        """
        # read in object's ply file
        # object facing positive direction of z axis
        objPath = osp.join(filePath,obj)
        self.obj_name = obj.split('.')[0]
        print("load object: " + self.obj_name)

        # Load assets for mesh-based rendering
        self.obj_scale_factor = obj_scale_factor
        self.tr_mesh = trimesh.load(objPath, force='mesh', process=False)
        self.tr_mesh.vertices = np.asarray(self.tr_mesh.vertices) * self.obj_scale_factor
        self.proximitry_query = trimesh.proximity.ProximityQuery(self.tr_mesh)  # Used for nearest neighbor queries
        if not isinstance(self.tr_mesh, trimesh.Trimesh):
            raise ValueError("OBJ did not load as a single mesh")
        self.vertices = self.tr_mesh.vertices
        self.vert_normals = self.tr_mesh.vertex_normals

        # Set orthographic camera (NOTE: we assume camera to be fixed and the object to be moving)
        self.znear = 0.01
        self.zfar = 1000.0

        self.cam = pyrender.camera.OrthographicCamera(
            xmag=psp.pixmm * psp.h / 2.,  # NOTE: This will be automatically re-scaled respecting designated height and width for rendering
            ymag=psp.pixmm * psp.h / 2.,  # NOTE: psp.h / 2. is multiplied to ensure identical orthographic scales as in Taxim
            znear=self.znear,
            zfar=self.zfar
        )

        # Camera pose in world frame
        self.cam_height = 1000.0  # Hard-coded camera height
        self.T_wc = np.eye(4)
        self.T_wc[:3, :3] = np.eye(3)
        self.T_wc[:3, 3] = np.array([0.0, 0.0, self.cam_height])

        # Lighting for rendering
        self.light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)

        # Set renderer for color and normals
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=psp.w,
            viewport_height=psp.h
        )
        self.normal_renderer = pyrender.OffscreenRenderer(
            viewport_width=psp.w,
            viewport_height=psp.h
        )

        # polytable
        calib_data = osp.join(data_folder, "polycalib.npz")
        self.calib_data = CalibData(calib_data)

        # raw calibration data
        rawData = osp.join(data_folder, "dataPack.npz")
        data_file = np.load(rawData,allow_pickle=True)
        self.f0 = data_file['f0']
        self.bg_proc = self.processInitialFrame()

        #shadow calibration
        self.shadow_depth = [0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2]
        shadowData = np.load(osp.join(data_folder, "shadowTable.npz"),allow_pickle=True)
        self.direction = shadowData['shadowDirections']
        self.shadowTable = shadowData['shadowTable']

    def generateHeightMap(self, gelpad_model_path, pressing_height_mm, dx, dy, contact_jitter_rot_mtx=None, contact_point=None, contact_theta=0.):
        """
        Generate the height map by interacting the object with the gelpad model.
        pressing_height_mm: pressing depth in millimeter
        dx, dy: shift of the object
        return:
        zq: the interacted height map
        gel_map: gelpad height map
        contact_mask: indicate contact area
        """
        # NOTE 1: Tactile sensor is placed at the x,y location of object center with z location at maximum object height, and object points with height over a threshold (0.2) are all considered
        # NOTE 2: Tactile sensor is placed oppositely facing the object placed on top of a virtual plane with z=0, although the object can be "floating"
        # NOTE 3: Currently normals are stored in "raw" xyz coordinates. To transform between two normals, we represent rotation as a 1-parameter family of transformations, controlled by contact_theta.
        # NOTE 4: This contact_theta controls the variation in contact rotations, or "rolling" on the contact point's tangent plane
        # NOTE 5: contact_jitter_rot_mtx additionally applies rotation to a contact point location to enable rotation other than tangent plane rolling

        assert(self.vertices.shape[1] == 3)
        # load dome-shape gelpad model
        gel_map = np.load(gelpad_model_path)
        gel_map = cv2.GaussianBlur(gel_map.astype(np.float32),(pr.kernel_size,pr.kernel_size),0)

        # Copy original vertex set and normals
        sim_vertices = np.copy(self.vertices)

        # Set contact points given as array of shape (3, )
        if contact_point is not None:
            cx = contact_point[0] * self.obj_scale_factor
            cy = contact_point[1] * self.obj_scale_factor
            cz = contact_point[2] * self.obj_scale_factor

            # Contact point array
            contact_arr = np.array([cx, cy, cz])

            # Find normal at contact point
            nn_point, _, nn_fid = self.proximitry_query.on_surface(contact_arr.reshape(1, 3))
            nn_bary = trimesh.triangles.points_to_barycentric(self.tr_mesh.triangles[nn_fid], points=contact_arr.reshape(1, 3))
            new_normal = trimesh.unitize((self.tr_mesh.vertex_normals[self.tr_mesh.faces[nn_fid]] * trimesh.unitize(nn_bary).reshape(-1, 3, 1)).sum(axis=1))
            new_normal = new_normal.reshape(-1)

            # Estimate rotation matrix that aligns new normal to the positive z direction
            contact_rot_mtx = rotation_family(new_normal, np.array([0, 0, 1]), contact_theta)

            # Fix contact point to origin and rotate points
            sim_vertices = (sim_vertices - contact_arr) @ contact_rot_mtx.T

        else:
            # Identify original contact points
            cx = np.mean(sim_vertices[:,0])
            cy = np.mean(sim_vertices[:,1])
            xy_dist = np.linalg.norm(sim_vertices[:, [0, 1]] - np.array([cx, cy]), axis=-1)
            kth = min(100, xy_dist.shape[0] // 2)  # NOTE: This is an arbitrarily chosen number
            topk_idx = np.argpartition(xy_dist, kth=kth)[:kth]
            cz = sim_vertices[topk_idx, 2].max()
            contact_arr = np.array([cx, cy, cz])

            sim_vertices = sim_vertices - contact_arr

        if contact_jitter_rot_mtx is not None:
            sim_vertices = sim_vertices @ contact_jitter_rot_mtx.T

        # Add sensor-plane shifts
        sim_vertices[:, 0] = sim_vertices[:, 0] + dx * psp.pixmm
        sim_vertices[:, 1] = sim_vertices[:, 1] + dy * psp.pixmm

        # Ensure minimum height is 0. during rendering
        sim_vertices[:, 2] -= sim_vertices[:, 2].min()

        # Render heightmap, color, and normals from mesh
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0],
                                ambient_light=[0.3, 0.3, 0.3])
        normal_scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0],
                                ambient_light=[0.3, 0.3, 0.3])  # Scene for normal map rendering

        # Set up mesh for rendering
        rgb_tr_mesh = self.tr_mesh.copy()
        rgb_tr_mesh.vertices = sim_vertices
        rgb_mesh = pyrender.Mesh.from_trimesh(rgb_tr_mesh, smooth=False)

        normal_tr_mesh = trimesh.Trimesh(vertices=sim_vertices, faces=self.tr_mesh.faces)
        normal_tr_mesh.visual.vertex_colors = (255 * (normal_tr_mesh.vertex_normals + 1.0) / 2.).astype(np.uint8)
        normal_tr_mesh.visual.face_colors = (255 * (normal_tr_mesh.face_normals + 1.0) / 2.).astype(np.uint8)
        normal_mesh = pyrender.Mesh.from_trimesh(normal_tr_mesh, smooth=False)

        # Set up scene
        scene.add(rgb_mesh)
        normal_scene.add(normal_mesh)
        scene.add(self.cam, pose=self.T_wc)
        normal_scene.add(self.cam, pose=self.T_wc)
        scene.add(self.light, pose=self.T_wc)
        normal_scene.add(self.light, pose=self.T_wc)

        # Render images
        rawcolorMap, depth = self.renderer.render(scene)
        rawcolorMap = rawcolorMap / 255.

        # Obtain depth
        depth_raw = depth.copy()
        depth_raw[depth_raw == 0] = np.nan

        # Normalize for visualization
        d_min = np.nanmin(depth_raw)
        d_max = np.nanmax(depth_raw)

        depth_norm = (depth_raw - d_min) / (d_max - d_min + 1e-8)
        depth_norm = np.nan_to_num(depth_norm)

        depth_img = (depth_norm * 255).astype(np.uint8)

        depth_raw = depth.copy()
        noninf = depth_raw > 0
        d_min = np.nanmin(depth_raw)
        d_max = np.nanmax(depth_raw)

        if (d_max - d_min) < (self.zfar - self.znear) / 10:  # CHOOSE A ROBUST CHECK FROM THRESHOLDING
            # A hack to fix depth maps for OrthographicCamera.
            # See: https://github.com/mmatl/pyrender/issues/72
            depth_raw[noninf] = self.zfar + self.znear - self.zfar * self.znear / depth_raw[noninf]

        # Obtain height map
        height_raw = self.T_wc[2, -1] - depth_raw
        height_raw[~noninf] = 0.

        # Ensure minimum height is 0. only considering regions within the view
        height_raw[noninf] = height_raw[noninf] - height_raw[noninf].min()
        heightMap = height_raw / psp.pixmm

        # Obtain normal map
        normal_rgb, _ = self.normal_renderer.render(normal_scene, flags=pyrender.RenderFlags.FLAT)
        normal = 2 * normal_rgb.astype(float) / 255 - 1.
        invalid_normal_loc = np.all(normal_rgb == 255, axis=-1)
        normal[invalid_normal_loc] = 0.
        normal[~invalid_normal_loc] = normal[~invalid_normal_loc] / np.linalg.norm(normal[~invalid_normal_loc], axis=-1, keepdims=True)

        rawnormalMap = np.copy(normal)
        vis_rawnormalMap = np.copy(normal_rgb)
        vis_rawnormalMap[invalid_normal_loc] = 0
        vis_rawnormalMap = vis_rawnormalMap / 255.

        max_g = np.max(gel_map)
        min_g = np.min(gel_map)
        max_o = np.max(heightMap)
        # pressing depth in pixel
        pressing_height_pix = pressing_height_mm/psp.pixmm

        # shift the gelpad to interact with the object
        gel_map = -1 * gel_map + (max_g+max_o-pressing_height_pix)  # RHS is gel height map assuming object placed at z = 0

        # get the contact area
        contact_mask = heightMap > gel_map

        # combine contact area of object shape with non contact area of gelpad shape
        zq = np.zeros((psp.h,psp.w))

        zq[contact_mask]  = heightMap[contact_mask]
        zq[~contact_mask] = gel_map[~contact_mask]

        return zq, gel_map, contact_mask, rawcolorMap, rawnormalMap, vis_rawnormalMap, heightMap


def TouchNet_train(args):

    checkpoint = torch.load(args.object_file_path)

    rotation_max = 15
    depth_max = 0.04
    depth_min = 0.0339
    displacement_min = 0.0005
    displacement_max = 0.0100
    depth_max = 0.04
    depth_min = 0.0339
    rgb_width = 120
    rgb_height = 160
    network_depth = 8

    #TODO load object...
    vertex_min = checkpoint['TouchNet']['xyz_min']
    vertex_max = checkpoint['TouchNet']['xyz_max']

    pcd_Path = os.path.join(args.obj_path, 'dataset', args.sample_ply)
    pcd = o3d.io.read_point_cloud(pcd_Path)
    vertex_coordinates_raw = np.asarray(pcd.points)
    print(vertex_coordinates_raw.shape)
    
    # vertex_coordinates = np.load(args.touch_vertices_file_path)
    N = vertex_coordinates_raw.shape[0]
    # gelinfo_data = np.load(args.touch_gelinfo_file_path)
    zero = np.zeros((N,3))
    zero[:, 2] = 0.001 * args.depth
    theta, phi, displacement = zero[:, 0], zero[:, 1], zero[:, 2]
    phi_x = np.cos(phi)
    phi_y = np.sin(phi)
    
    # normalize theta to [-1, 1]
    theta = (theta - np.radians(0)) / (np.radians(rotation_max) - np.radians(0))

    #normalize displacement to [-1,1]
    displacement_norm = (displacement - displacement_min) / (displacement_max - displacement_min)

    #normalize coordinates to [-1,1]
    vertex_coordinates = (vertex_coordinates_raw - vertex_min) / (vertex_max - vertex_min)
    
    #initialize horizontal and vertical features
    w_feats = np.repeat(np.repeat(np.arange(rgb_width).reshape((rgb_width, 1)), rgb_height, axis=1).reshape((1, 1, rgb_width, rgb_height)), N, axis=0)
    h_feats = np.repeat(np.repeat(np.arange(rgb_height).reshape((1, rgb_height)), rgb_width, axis=0).reshape((1, 1, rgb_width, rgb_height)), N, axis=0)
    #normalize horizontal and vertical features to [-1, 1]
    w_feats_min = w_feats.min()
    w_feats_max = w_feats.max()
    h_feats_min = h_feats.min()
    h_feats_max = h_feats.max()
    w_feats = (w_feats - w_feats_min) / (w_feats_max - w_feats_min)
    h_feats = (h_feats - h_feats_min) / (h_feats_max - h_feats_min)
    w_feats = torch.FloatTensor(w_feats)
    h_feats = torch.FloatTensor(h_feats)

    theta = np.repeat(theta.reshape((N, 1, 1)), rgb_width * rgb_height, axis=1)
    phi_x = np.repeat(phi_x.reshape((N, 1, 1)), rgb_width * rgb_height, axis=1)
    phi_y = np.repeat(phi_y.reshape((N, 1, 1)), rgb_width * rgb_height, axis=1)
    displacement_norm = np.repeat(displacement_norm.reshape((N, 1, 1)), rgb_width * rgb_height, axis=1)
    vertex_coordinates = np.repeat(vertex_coordinates.reshape((N, 1, 3)), rgb_width * rgb_height, axis=1)

    data_wh = np.concatenate((w_feats, h_feats), axis=1)
    data_wh = np.transpose(data_wh.reshape((N, 2, -1)), axes=[0, 2, 1])
    #Now get final feats matrix as [x, y, z, theta, phi_x, phi_y, displacement, w, h]
    data = np.concatenate((vertex_coordinates, theta, phi_x, phi_y, displacement_norm, data_wh), axis=2).reshape((-1, 9))

    #checkpoint = torch.load(args.object_file_path)
    embed_fn, input_ch = TouchNet_model.get_embedder(10, 0)
    model = TouchNet_model.NeRF(D = network_depth, input_ch = input_ch, output_ch = 1)
    state_dic = checkpoint['TouchNet']['model_state_dict']
    state_dic = TouchNet_utils.strip_prefix_if_present(state_dic, 'module.')

    if args.initialization == "random":
        pass
    elif args.initialization == "pretrained":
        model.load_state_dict(state_dic)
    else:
        raise NotImplementedError("Other intialization methods not supported")
    model = model.to(device)
    model.train()

    preds = np.empty((data.shape[0], 1))
    gt = np.empty((data.shape[0], 1))

    batch_size = 1024

    #testsavedir = args.touch_results_dir
    #os.makedirs(testsavedir, exist_ok=True)
    
    data_folder = osp.join("calibs")
    filePath = args.obj_path
    gelpad_model_path = osp.join( 'calibs', 'gelmap5.npy')


    obj = args.obj + '.obj'
    tac_sim = mesh_simulator

    sim = tac_sim(data_folder, args.obj_path, obj, args.obj_scale_factor)
    press_depth = args.depth
    dx = 0
    dy = 0
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, "min", patience=5)
    criterion = nn.L1Loss()

    pixels_per_image = rgb_width * rgb_height
    print(f">>> Start collecting data on {N} Samples...")
    pbar = tqdm(range(N))
    for i in pbar:         
        start_idx = i * pixels_per_image
        end_idx = (i + 1) * pixels_per_image

        # Ground Truth (GT) (Simulator)
        raw_vertices = vertex_coordinates_raw
        raw_height_map, _, _, _, _, _, _ = sim.generateHeightMap(gelpad_model_path, press_depth, dx, dy, contact_point=raw_vertices[i], contact_theta=0.)
        height_map = cv2.resize(raw_height_map, (rgb_height, rgb_width))
        height_map = (height_map - height_map.min()) / (height_map.max() - height_map.min() + 1e-6)
        height_map = 1 - height_map
        
        # GT tensor
        gt[start_idx:end_idx] = height_map.reshape(-1,1)
        
    
    gt_tensor = torch.tensor(gt).to(device).reshape(-1,1)
    inputs = torch.Tensor(data).to(device)
     
    train_dataset = TensorDataset(inputs, gt_tensor)
    dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    epoch = 50
    epoch_losses = []
    print(f">>> Start training on {epoch} Epochs...")
    for i in range(epoch):
        batch_loss = 0
        pbar = tqdm(dataloader)
        for index, (inputs, gt_tensor) in enumerate(pbar):
        
            embedded = embed_fn(inputs)
            preds = model(embedded)
            
            optimizer.zero_grad()
            
            loss = criterion(preds, gt_tensor)
            loss.backward()
            optimizer.step()

            batch_loss += loss.item()

            pbar.set_description(f"Epoch {i+1}/{epoch} Loss: {loss.item():.5f}")
        
        avg_loss = batch_loss / len(dataloader)
        scheduler.step(avg_loss)
        epoch_losses.append(avg_loss)
        if (i+1) % 5 ==0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Epoch {i+1} | Average Loss: {avg_loss:.5f}, LR: {lr:.6f}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epoch + 1), epoch_losses, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss')
    plt.title(f'Training Loss with {N} Samples')
    #plt.grid(True)
    plt.legend()
    
    ply_name = args.sample_ply.split('.')
    loss_plot_path = os.path.join(filePath, f"Training Loss {ply_name[0]} Samples.png")
    plt.savefig(loss_plot_path)
    #plt.show()

    new_weights = model.state_dict()

    checkpoint['TouchNet']['model_state_dict'] = new_weights
    print(">>> Saving the trained model...")
    save_path = os.path.join(filePath, f"finetuned_model_{ply_name[0]}.pth")
    torch.save(checkpoint, save_path)

    sim.renderer.delete()
    sim.normal_renderer.delete()


def TouchNet_eval(args):

    checkpoint = torch.load(args.object_file_path)      # Original model
    checkpoint_finetuned = torch.load(os.path.join(args.obj_path, args.object_model))       #Train model

    rotation_max = 15
    depth_max = 0.04
    depth_min = 0.0339
    displacement_min = 0.0005
    displacement_max = 0.0100
    depth_max = 0.04
    depth_min = 0.0339
    rgb_width = 120
    rgb_height = 160
    network_depth = 8

    #TODO load object...
    vertex_min = checkpoint['TouchNet']['xyz_min']
    vertex_max = checkpoint['TouchNet']['xyz_max']
    vertex_min_finetuned = checkpoint['TouchNet']['xyz_min']
    vertex_max_finetuned = checkpoint['TouchNet']['xyz_max']

    pcd_Path = os.path.join(args.obj_path, 'dataset', args.sample_ply)
    pcd = o3d.io.read_point_cloud(pcd_Path)
    pcd_vertices = np.asarray(pcd.points)
    if args.sample_ply == 'testcase.ply':
        vertex_coordinates_raw = pcd_vertices[:15]
    else:
        size = 15
        if pcd_vertices.shape[0] < 15:
            size = pcd_vertices.shape[0]
        random_index = np.random.choice(pcd_vertices.shape[0], size=size, replace=False)
        vertex_coordinates_raw = pcd_vertices[random_index]
    vertex_coordinates_raw = np.asarray([[-0.009997,0.009172,-0.000286]])
    N = vertex_coordinates_raw.shape[0]

    #gelinfo_data = np.load(args.touch_gelinfo_file_path)
    zero = np.zeros((N,3))
    zero[:, 2] = 0.001 * args.depth
    theta, phi, displacement = zero[:, 0], zero[:, 1], zero[:, 2]
    phi_x = np.cos(phi)
    phi_y = np.sin(phi)
    
    # normalize theta to [-1, 1]
    theta = (theta - np.radians(0)) / (np.radians(rotation_max) - np.radians(0))

    #normalize displacement to [-1,1]
    displacement_norm = (displacement - displacement_min) / (displacement_max - displacement_min)

    #normalize coordinates to [-1,1]
    vertex_coordinates = (vertex_coordinates_raw - vertex_min) / (vertex_max - vertex_min)
    vertex_coordinates_finetuned = (vertex_coordinates_raw - vertex_min_finetuned) / (vertex_max_finetuned - vertex_min_finetuned)
    #initialize horizontal and vertical features
    w_feats = np.repeat(np.repeat(np.arange(rgb_width).reshape((rgb_width, 1)), rgb_height, axis=1).reshape((1, 1, rgb_width, rgb_height)), N, axis=0)
    h_feats = np.repeat(np.repeat(np.arange(rgb_height).reshape((1, rgb_height)), rgb_width, axis=0).reshape((1, 1, rgb_width, rgb_height)), N, axis=0)
    #normalize horizontal and vertical features to [-1, 1]
    w_feats_min = w_feats.min()
    w_feats_max = w_feats.max()
    h_feats_min = h_feats.min()
    h_feats_max = h_feats.max()
    w_feats = (w_feats - w_feats_min) / (w_feats_max - w_feats_min)
    h_feats = (h_feats - h_feats_min) / (h_feats_max - h_feats_min)
    w_feats = torch.FloatTensor(w_feats)
    h_feats = torch.FloatTensor(h_feats)

    theta = np.repeat(theta.reshape((N, 1, 1)), rgb_width * rgb_height, axis=1)
    phi_x = np.repeat(phi_x.reshape((N, 1, 1)), rgb_width * rgb_height, axis=1)
    phi_y = np.repeat(phi_y.reshape((N, 1, 1)), rgb_width * rgb_height, axis=1)
    displacement_norm = np.repeat(displacement_norm.reshape((N, 1, 1)), rgb_width * rgb_height, axis=1)
    vertex_coordinates = np.repeat(vertex_coordinates.reshape((N, 1, 3)), rgb_width * rgb_height, axis=1)
    vertex_coordinates_finetuned = np.repeat(vertex_coordinates_finetuned.reshape((N, 1, 3)), rgb_width * rgb_height, axis=1)

    data_wh = np.concatenate((w_feats, h_feats), axis=1)
    data_wh = np.transpose(data_wh.reshape((N, 2, -1)), axes=[0, 2, 1])
    #Now get final feats matrix as [x, y, z, theta, phi_x, phi_y, displacement, w, h]
    data = np.concatenate((vertex_coordinates, theta, phi_x, phi_y, displacement_norm, data_wh), axis=2).reshape((-1, 9))
    data_finetuned = np.concatenate((vertex_coordinates_finetuned, theta, phi_x, phi_y, displacement_norm, data_wh), axis=2).reshape((-1, 9))

    #checkpoint = torch.load(args.object_file_path)
    embed_fn, input_ch = TouchNet_model.get_embedder(10, 0)
    model = TouchNet_model.NeRF(D = network_depth, input_ch = input_ch, output_ch = 1)
    state_dic = checkpoint['TouchNet']['model_state_dict']
    state_dic = TouchNet_utils.strip_prefix_if_present(state_dic, 'module.')
    model.load_state_dict(state_dic)
    model = model.to(device)
    model.eval()

    embed_fn_finetuned, input_ch_finetuned = TouchNet_model.get_embedder(10, 0)
    model_finetuned = TouchNet_model.NeRF(D = network_depth, input_ch = input_ch_finetuned, output_ch = 1)
    state_dic_finetuned = checkpoint_finetuned['TouchNet']['model_state_dict']
    state_dic_finetuned = TouchNet_utils.strip_prefix_if_present(state_dic_finetuned, 'module.')
    model_finetuned.load_state_dict(state_dic_finetuned)
    model_finetuned = model_finetuned.to(device)
    model_finetuned.eval()

    preds = np.empty((data.shape[0], 1))
    preds_finetuned = np.empty((data.shape[0], 1))
    gt = np.empty((data.shape[0], 1))

    batch_size = 1024

    #testsavedir = args.touch_results_dir
    #os.makedirs(testsavedir, exist_ok=True)
    
    data_folder = osp.join("calibs")
    filePath = args.obj_path
    gelpad_model_path = osp.join('calibs', 'gelmap5.npy')

    os.makedirs(os.path.join(args.obj_path, 'sim_img'), exist_ok=True)
    os.makedirs(os.path.join(args.obj_path, 'shadow_img'), exist_ok=True)
    os.makedirs(os.path.join(args.obj_path, 'rgb_height_img'), exist_ok=True)
    os.makedirs(os.path.join(args.obj_path, 'normal_img'), exist_ok=True)
    os.makedirs(os.path.join(args.obj_path, 'color_img'), exist_ok=True)
    os.makedirs(os.path.join(args.obj_path, 'height_img'), exist_ok=True)

    DeleteAllFiles(os.path.join(args.obj_path, 'sim_img'))
    DeleteAllFiles(os.path.join(args.obj_path, 'shadow_img'))
    DeleteAllFiles(os.path.join(args.obj_path, 'rgb_height_img'))
    DeleteAllFiles(os.path.join(args.obj_path, 'normal_img'))
    DeleteAllFiles(os.path.join(args.obj_path, 'color_img'))
    DeleteAllFiles(os.path.join(args.obj_path, 'height_img'))

    DeleteAllFiles(os.path.join('results', 'Pr'))
    DeleteAllFiles(os.path.join('results', 'Pr_finetuned'))
    DeleteAllFiles(os.path.join('results', 'finetuned'))

    obj = args.obj + '.obj'
    tac_sim = mesh_simulator

    sim = tac_sim(data_folder, args.obj_path, obj, args.obj_scale_factor)
    press_depth = args.depth
    dx = 0
    dy = 0
    taxim = TaximRender("./calibs/")

    criterion = nn.L1Loss()
    loss = []
    loss_finetuned = []
    #bg = np.load(os.path.join('calibs', 'real_bg.npy'))
    #plt.imshow(bg)
    #plt.show()
    #sleep(10)
    pixels_per_image = rgb_width * rgb_height
    print(f">>> Start compare on {N} Samples...")
    pbar = tqdm(range(N))
    for i in pbar:         
        start_idx = i * pixels_per_image
        end_idx = (i + 1) * pixels_per_image

        # Ground Truth (GT) (Simulator)
        raw_vertices = vertex_coordinates_raw
        height_map, gel_map, contact_mask, raw_color_map, _, vis_raw_normal_map, raw_height_map = sim.generateHeightMap(gelpad_model_path, press_depth, dx, dy, contact_point=raw_vertices[i], contact_theta=0.)
        #print(f"Min: {height_map.min()}, Max: {height_map.max()}")
        is_TDF = False
        if is_TDF:
            height_map = np.load(os.path.join(args.obj_path, '36_height_map_resize_7.npy'))
            height_map += 200
            print(f"Min: {height_map.min()}, Max: {height_map.max()}")
            contact_mask[:,:] = 1

        heightMap, contact_mask, contact_height = sim.deformApprox(press_depth, height_map, gel_map, contact_mask)
        sim_img, shadow_sim_img = sim.simulating(heightMap, contact_mask, contact_height, shadow=True)
        
        height_map_rgb = colormaps.get_cmap("viridis")((raw_height_map - raw_height_map.min()) / (raw_height_map.max() - raw_height_map.min() + 1e-6))
        height_map_rgb = (height_map_rgb * 255).astype(np.uint8)
        raw_normal_img = (vis_raw_normal_map * 255).astype(np.uint8)
        raw_color_img = (raw_color_map * 255).astype(np.uint8)

        height_savePath = os.path.join(args.obj_path, 'rgb_height_img', f'{i+1}.png')
        raw_normal_savePath = os.path.join(args.obj_path, 'normal_img', f'{i+1}.png')
        raw_color_savePath = os.path.join(args.obj_path, 'color_img', f'{i+1}.png')

        img_savePath = os.path.join(args.obj_path, 'sim_img', f'{i+1}.png')
        shadow_savePath = os.path.join(args.obj_path, 'shadow_img', f'{i+1}.png')

        cv2.imwrite(height_savePath, cv2.cvtColor(height_map_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(raw_normal_savePath, cv2.cvtColor(raw_normal_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(raw_color_savePath, cv2.cvtColor(raw_color_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(img_savePath, sim_img)
        cv2.imwrite(shadow_savePath, shadow_sim_img)


        height_map = cv2.resize(height_map, (rgb_height, rgb_width))
        height_map = (height_map - height_map.min()) / (height_map.max() - height_map.min() + 1e-6)
        height_map = 1 - height_map
        
        gt_path = os.path.join(args.obj_path, 'height_img', f"{i+1}.png")
        gt_height_map = (height_map * 255).astype(np.uint8)
        cv2.imwrite(gt_path, gt_height_map)

        
        # Groud Truth data
        gt = height_map.reshape(-1,1)
        gt_tensor = torch.tensor(gt).to(device).reshape(-1,1)

        inputs = torch.Tensor(data[start_idx:end_idx]).to(device)
        embedded = embed_fn(inputs)
        preds = model(embedded)

        inputs_finetuned = torch.Tensor(data_finetuned[start_idx:end_idx]).to(device)
        embedded_finetuned = embed_fn_finetuned(inputs_finetuned)
        preds_finetuned = model_finetuned(embedded_finetuned)
    
        # Original model prediction height image
        results = preds.detach().cpu().numpy()
        results = 255 * (results - results.min()) / (results.max() - results.min() + 1e-6)
        results = results.reshape((rgb_width, rgb_height)) 
        preds_savePath = os.path.join('results', 'Pr', obj + '_raw_height_' + str(i+1) + '.jpg')
        cv2.imwrite(preds_savePath, results)
        
        # Trained model prediction tactile & height image
        raw_results_finetuned = preds_finetuned.detach().cpu().numpy()
        results_finetuned = raw_results_finetuned * (depth_max - depth_min) + depth_min
        results_finetuned = results_finetuned.reshape((rgb_width, rgb_height))
        _, _, tactile_map = taxim.render(results_finetuned, displacement[i])
        cv2.imwrite(os.path.join('results', 'finetuned', f'{i+1}.png'), tactile_map)

        height_results_finetuned = 255 * (raw_results_finetuned - raw_results_finetuned.min()) / (raw_results_finetuned.max() - raw_results_finetuned.min() + 1e-6)
        height_results_finetuned = height_results_finetuned.reshape((rgb_width, rgb_height))
        
        preds_savePath = os.path.join('results', 'Pr_finetuned', obj + '_raw_height_' + str(i+1) + '.jpg')
        cv2.imwrite(preds_savePath, height_results_finetuned)

        loss.append(float(criterion(preds, gt_tensor).item()))
        loss_finetuned.append(float(criterion(preds_finetuned, gt_tensor).item()))


    # Loss compare Graph
    x = np.arange(1, N + 1)
    plt.figure(figsize=(10, 6))
    plt.bar(x - 0.2, loss, width=0.4, label='original Loss', color='blue')
    plt.bar(x + 0.2, loss_finetuned, width=0.4, label='finetuned Loss', color='orange')
    plt.xlabel(f'{N} Samples')
    plt.ylabel('L1 Loss')
    plt.title('Model Loss')
    #plt.grid(True)
    plt.legend()
    
    loss_plot_path = os.path.join('results/finetuned', "Loss Compare.png")
    plt.savefig(loss_plot_path)
    #plt.show()

    sim.renderer.delete()
    sim.normal_renderer.delete()


def DeleteAllFiles(filePath):
    if os.path.exists(filePath):
        for file in os.scandir(filePath):
            os.remove(file.path)
        return 'Remove All Files'
    else:
        return 'Directory Not Found'


if __name__ == "__main__":
    if(args.mode == 'train'):
        TouchNet_train(args=args)
    
    elif(args.mode == 'eval'):
        TouchNet_eval(args=args)
    
    elif(args.mode == 'comp'):
        filePath = './comp'
        '''
        object_number = 76
        
        height_0 = cv2.imread(os.path.join('./comp', object_number, '0.png'))
        sim_0 = cv2.imread(os.path.join('./comp', object_number, 'sim0.png'))
        height_10 = cv2.imread(os.path.join('./comp', object_number, '10.jpg'))
        sim_10 = cv2.imread(os.path.join('./comp', object_number, 'sim10.png'))
        height_50 = cv2.imread(os.path.join('./comp', object_number, '50.jpg'))
        sim_50 = cv2.imread(os.path.join('./comp', object_number, 'sim50.png'))
        height_250 = cv2.imread(os.path.join('./comp', object_number, '254.jpg'))
        sim_250 = cv2.imread(os.path.join('./comp', object_number, 'sim254.png'))
        height_1000 = cv2.imread(os.path.join('./comp', object_number, '1002.jpg'))
        sim_1000 = cv2.imread(os.path.join('./comp', object_number, 'sim1002.png'))
        #height_2000 = cv2.imread(os.path.join('./comp/76', '2008.jpg'))
        #sim_2000 = cv2.imread(os.path.join('./comp/76', 'sim2008.png'))

        fig = plt.figure(figsize=(10,6), dpi=100)
        fig.tight_layout()
        rows = 2
        cols = 6
        xlabel = ['GT', 'sim', '10 samples', 'sim', '50 samples', 'sim', '254 samples', 'sim', '1002 samples', 'sim']
        img_list = [height_0, sim_0, height_10, sim_10, height_50, sim_50, height_250, sim_250, height_1000, sim_1000]
        i = 0

        
        for n in range(rows):
            for m in range(cols):            
                if n == 0:
                    ax = fig.add_subplot(rows, cols, n*6 + m+1)
                    ax.imshow(cv2.cvtColor(img_list[i], cv2.COLOR_BGR2RGB))
                    ax.set_xlabel(xlabel[i])
                    ax.set_xticks([]), ax.set_yticks([])
                    i+=1
                elif n == 1 and m > 1:
                    ax = fig.add_subplot(rows, cols, n*6 + m+1)
                    ax.imshow(cv2.cvtColor(img_list[i], cv2.COLOR_BGR2RGB))
                    ax.set_xlabel(xlabel[i])
                    ax.set_xticks([]), ax.set_yticks([])
                    i+=1
                
        plt.suptitle(f'#{object_number} Testcase 13')
        plt.tight_layout()
        plt.savefig(f'./comp/#{object_number} Testcase 13 Compare.png')
        plt.show()
        '''



