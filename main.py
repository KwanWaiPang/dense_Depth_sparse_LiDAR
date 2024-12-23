import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from depth_map import dense_map
import glob
import tqdm
import open3d as o3d #pip install open3d
from scipy.interpolate import griddata


# Class for the calibration matrices for KITTI data
class Calibration:
    def __init__(self):
        # From Camera Coordinate system to Image frame. Intrinsic (used the projection_matrix)
        self.P = np.array([ 737.85626,    0.     ,  609.6249 ,    0.     ,
                              0.     ,  784.1731 ,  515.77673,    0.     ,
                              0.     ,    0.     ,    1.     ,    0.     ]).reshape([3,4])
        
        # From LiDAR coordinate system to Camera Coordinate system. Extrinsic
        self.L2C=np.array(
                    [ 0.0119197 , -0.999929  ,  0.0000523,  0.0853154,      
                     -0.00648951, -0.00012969, -0.999979 , -0.0684439,
                      0.999908  ,  0.0119191 , -0.0064906, -0.0958121,
                      0.0       ,  0.0       ,  0.0      ,  1.0        ]).reshape([4,4])

        self.R0=np.eye(3)
    
    # From LiDAR coordinate system to Camera Coordinate system
    def lidar2cam(self, pts_3d_lidar):
        n = pts_3d_lidar.shape[0]
        pts_3d_hom = np.hstack((pts_3d_lidar, np.ones((n,1))))  # (n,4)
        pts_3d_cam_ref = np.dot(pts_3d_hom, np.transpose(self.L2C))  # (n,4)*(4,4)->(n,4)
        pts_3d_cam_rec = pts_3d_cam_ref[:,0:3]
        return pts_3d_cam_rec
    
    # From Camera Coordinate system to Image frame
    def rect2Img(self, rect_pts, img_width, img_height):
        n = rect_pts.shape[0]
        points_hom = np.hstack((rect_pts, np.ones((n,1))))  # (n,4)
        points_2d = np.dot(points_hom, np.transpose(self.P))  # (n,4)*(4,3)->(n,3)
        points_2d[:,0] /= points_2d[:,2]
        points_2d[:,1] /= points_2d[:,2]
        
        mask = (points_2d[:,0] >= 0) & (points_2d[:,0] < img_width) & (points_2d[:,1] >= 0) & (points_2d[:,1] < img_height)
        mask = mask & (rect_pts[:,2] >= 0)
        return points_2d[mask,0:2], mask


def plt_save(depth_img, save_path):
    plt.imshow(depth_img, cmap='plasma')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


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
    
    # Loading the image
    # img = cv2.imread(os.path.join(image_dir, "%06d.png" % cur_id))
    imgin_list = sorted(glob.glob(os.path.join(image_dir, "*.png")))#获取所有的png文件
    # imgin_list，stride=3
    
    # stride=3
    # imgin_list=imgin_list[::stride]

    # Loading the LiDAR data
    lidarin_list = sorted(glob.glob(os.path.join(velodyne_dir, "*.pcd")))#获取所有的pcd文件
    # lidar = np.fromfile(os.path.join(velodyne_dir, "%06d.bin" % cur_id), dtype=np.float32).reshape(-1, 4)

    num_imgs = len(lidarin_list)
    # assert len(lidarin_list) == num_imgs

    pbar = tqdm.tqdm(total=num_imgs-1)
    for i in range(num_imgs):
        # image = cv2.imread(imgin_list[i])
        image = cv2.imread(imgin_list[0]) #由于此处图像仅仅是提供长和宽的作用，为此只取第一张即可~

        # lidar = np.fromfile(os.path.join(lidarin_list[i]), dtype=np.float32).reshape(-1, 4)
        lidar = np.asarray(o3d.io.read_point_cloud(lidarin_list[i]).points)
        filename = os.path.basename(lidarin_list[i])  # 获取文件名，包括扩展名
        basename, extension = os.path.splitext(filename)  # 分离文件名和扩展名
        lidar_depth_name=basename

        # Loading Calibration
        calib = Calibration()

        # From LiDAR coordinate system to Camera Coordinate system （将雷达转换到相机坐标系）
        lidar_rect = calib.lidar2cam(lidar[:,0:3])

        # From Camera Coordinate system to Image frame （从相机坐标系到图像坐标系）
        lidarOnImage, mask = calib.rect2Img(lidar_rect, image.shape[1], image.shape[0])
        
        # Concatenate LiDAR position with the intesity (3), with (2) we would have the depth
        lidarOnImage = np.concatenate((lidarOnImage, lidar_rect[mask,2].reshape(-1,1)), 1)
        xs = np.int32(lidarOnImage[:,0])
        ys = np.int32(lidarOnImage[:,1])
        ds = lidar_rect[mask,2]

        # Generate the depth map from nx3 points
        depth_max = 10 #设置最远为10米
        depth_scale = 1000
        
        ### 1. simple project
        # not consider the z-buffer
        # depth_map = np.zeros((image.shape[0], image.shape[1]))
        # depth_map[ys, xs] = ds
        
        # condiser the z-buffer
        ids = ys * image.shape[1] + xs
        depth_map = np.ones((image.shape[0] * image.shape[1])) * np.inf
        np.minimum.at(depth_map, ids, ds)
        depth_map[depth_map == np.inf] = 0
        depth_map = depth_map.reshape(image.shape[0], image.shape[1])
        depth_map[depth_map > depth_max] = depth_max
        
        ### 2. use griddata to interpolate the depth map
        grid_x, grid_y = np.meshgrid(np.arange(image.shape[1]), np.arange(image.shape[0]))
        depth_map_grid = griddata((xs, ys), ds, (grid_x, grid_y), method='linear')
        depth_map_grid[depth_map_grid > depth_max] = depth_max
        
        ### 3. use dense_map to interpolate the depth map
        # points for dense map
        xs2, ys2 = lidarOnImage[:,0], lidarOnImage[:,1]
        pts = np.stack((xs2, ys2, ds), axis=1)
        grid_size = 10 #起码要设置为10才是稠密的~
        dense_depth_map = dense_map(pts.T, image.shape[1], image.shape[0], grid_size)
        dense_depth_map[dense_depth_map > depth_max] = depth_max
        
        # fig, axs = plt.subplots(2, 2)
        # axs[0, 0].imshow(image)
        # axs[0, 1].imshow(depth_map)
        # axs[1, 0].imshow(depth_map_grid)
        # axs[1, 1].imshow(dense_depth_map)
        # plt.show()
        
        # rescale the depth map to save. 16UC1 format
        depth_map = depth_map * depth_scale
        dense_depth_map = dense_depth_map * depth_scale

        cv2.imwrite(os.path.join(imgdirout, f"{lidar_depth_name}.png"), dense_depth_map.astype(np.uint16))

        # cv2.imwrite("depth_map.png", depth_map.astype(np.uint16))
        # plt_save(depth_map, "depth_map_vis.png")
        # cv2.imwrite(f"dense_depth_map_grid{grid_size}.png", dense_depth_map.astype(np.uint16))
        plt_save(dense_depth_map, f"color_dense_depth/dense_depth_map_{lidar_depth_name}.png") #保存深度图(彩色)
        
        pbar.update(1)

