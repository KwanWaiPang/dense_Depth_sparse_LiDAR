import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from depth_map import dense_map
import glob
import tqdm

# Class for the calibration matrices for KITTI data
class Calibration:
    def __init__(self, calib_filepath):

        # From Camera Coordinate system to Image frame (/media/lfl-data2/VECtor_h5/0_calib/left_regular_camera_intrinsic_results.yaml)
        self.P=np.array(
                    [737.85626,    0.     ,  609.6249 ,    0.     ,
                     0.     ,  784.1731 ,  515.77673,    0.     ,
                     0.     ,    0.     ,    1.     ,    0.     ]).reshape([3,4])
        
        # From Camera Coordinate system to Image frame
        # self.R0=np.array(
        #             [886.19107,    0.     ,  610.57891,
        #              0.     ,  886.59163,  514.59271,
        #              0.     ,    0.     ,    1.     ]).reshape([3,3])
        # self.R0初始化为单位矩阵
        self.R0=np.eye(3)
        
        # From LiDAR coordinate system to Camera Coordinate system
        self.L2C=np.array(
                    [0.0119197 , -0.999929  ,  0.0000523,  0.0853154,      
                    -0.00648951, -0.00012969, -0.999979 , -0.0684439,
                     0.999908  ,  0.0119191 , -0.0064906, -0.0958121]).reshape([3,4])

        # # transformation from LiDAR to left regular camera
        # left_regular_camera=np.array(
        #             [0.0119197 , -0.999929  ,  0.0000523,  0.0853154,      
        #              -0.00648951, -0.00012969, -0.999979 , -0.0684439,
        #               0.999908  ,  0.0119191 , -0.0064906, -0.0958121,
        #               0.0, 0.0, 0.0, 1.0]).reshape([4,4])

    @staticmethod
    def read_calib_file(filepath):
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line)==0: continue
                key, value = line.split(':', 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data
    
    # From LiDAR coordinate system to Camera Coordinate system
    def lidar2cam(self, pts_3d_lidar):
        n = pts_3d_lidar.shape[0] # Number of points,(n,3)
        pts_3d_hom = np.hstack((pts_3d_lidar, np.ones((n,1)))) #转换为齐次坐标(n,3)->(n,4)
        pts_3d_cam_ref = np.dot(pts_3d_hom, np.transpose(self.L2C))#(n,4)*(4,3)->(n,3)，转换到相机坐标系
        # pts_3d_cam_rec = np.transpose(np.dot(self.R0, np.transpose(pts_3d_cam_ref))) #转换到图像坐标系
        pts_3d_cam_rec=pts_3d_cam_ref
        return pts_3d_cam_rec
    
    # From Camera Coordinate system to Image frame
    def rect2Img(self, rect_pts, img_width, img_height):
        n = rect_pts.shape[0] #在相机坐标系下的点的个数，(n,3)
        points_hom = np.hstack((rect_pts, np.ones((n,1))))#转换为齐次坐标(n,3)->(n,4)
        points_2d = np.dot(points_hom, np.transpose(self.P)) # nx3

        # 进行归一化
        points_2d[:,0] /= points_2d[:,2]
        points_2d[:,1] /= points_2d[:,2]
        
        mask = (points_2d[:,0] >= 0) & (points_2d[:,0] <= img_width) & (points_2d[:,1] >= 0) & (points_2d[:,1] <= img_height)
        mask = mask & (rect_pts[:,2] > 2)
        return points_2d[mask,0:2], mask

if __name__ == "__main__":
    root = "/media/lfl-data2/VECtor_h5/school_dolly1/" #数据的根目录
    image_dir = os.path.join(root, "school_dolly1.synced.left_camera") #图像的目录
    velodyne_dir = os.path.join(root, "school_dolly1.synced.lidar") #LiDAR的目录

    imgdirout = os.path.join(root, f"lidar_dense_depth")
    if not os.path.exists(imgdirout):
        os.makedirs(imgdirout)
    else:
        print("The folder already exists")
        os.system(f"rm -rf {imgdirout}")
        os.makedirs(imgdirout)


    calib_dir = "camera_lidar_extrinsic_results.yaml" #os.path.join(root, "calib")
    # Data id
    cur_id = 21
    # Loading the image
    # img = cv2.imread(os.path.join(image_dir, "%06d.png" % cur_id))
    imgin_list = sorted(glob.glob(os.path.join(image_dir, "*.png")))#获取所有的png文件
    # imgin_list，stride=3
    stride=3
    imgin_list=imgin_list[::stride]


    # Loading the LiDAR data
    lidarin_list = sorted(glob.glob(os.path.join(velodyne_dir, "*.pcd")))#获取所有的pcd文件
    # lidar = np.fromfile(os.path.join(velodyne_dir, "%06d.bin" % cur_id), dtype=np.float32).reshape(-1, 4)

    num_imgs = len(imgin_list)
    assert len(lidarin_list) == num_imgs

    # 一张一张的读图像并且处理
    pbar = tqdm.tqdm(total=num_imgs-1)
    for i in range(num_imgs):
        img = cv2.imread(imgin_list[i])

        lidar = np.fromfile(os.path.join(lidarin_list[i]), dtype=np.float32).reshape(-1, 4)

        # Loading Calibration
        calib = Calibration(calib_dir)

        # From LiDAR coordinate system to Camera Coordinate system （将雷达转换到相机坐标系）
        lidar_rect = calib.lidar2cam(lidar[:,0:3])

        # From Camera Coordinate system to Image frame （从相机坐标系到图像坐标系）
        lidarOnImage, mask = calib.rect2Img(lidar_rect, img.shape[1], img.shape[0])
        
        # Concatenate LiDAR position with the intesity (3), with (2) we would have the depth
        lidarOnImage = np.concatenate((lidarOnImage, lidar_rect[mask,2].reshape(-1,1)), 1)

        out = dense_map(lidarOnImage.T, img.shape[1], img.shape[0], 1)
        # plt.figure(figsize=(20,40))
        # plt.imsave("depth_map_%06d.png" % cur_id, out)

        # # 步骤 1: 归一化深度值到 0-255
        # out_normalized = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX)

        # # 步骤 2: 转换为8位无符号整型
        # out_normalized = np.uint8(out_normalized)

        # # 步骤 3: 应用伪彩色映射
        # out_colored = cv2.applyColorMap(out_normalized, cv2.COLORMAP_JET)

        cv2.imwrite(os.path.join(imgdirout, f"{i:06d}.png"), out)
        pbar.update(1)

        gwp_debug=666;

