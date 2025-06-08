import numpy as np
import pandas as pd
from math import cos, sin
import logging
from datetime import datetime
from config import CAMERA_PARAMS, CAMERA_EOP

class collinearity_solver:
    def __init__(self, focal_length, pixel_size, EOP1, EOP2, img_width, img_height, log=False):
        """
        initialize the collinearity solver with camera parameters and exterior orientation parameters
        
        args:
            focal_length (float): camera focal length in mm
            pixel_size (float): pixel size in mm
            EOP1 (np.array): exterior orientation parameters for first image [x, y, z, omega, phi, kappa]
            EOP2 (np.array): exterior orientation parameters for second image [x, y, z, omega, phi, kappa]
            img_width (int): image width in pixels
            img_height (int): image height in pixels
            log (bool): enable logging of intermediate calculations
        """
        self.focal_length = focal_length
        self.pixel_size = pixel_size
        self.EOP1 = EOP1
        self.EOP2 = EOP2
        self.c0 = img_width // 2    
        self.r0 = img_height // 2   

        # calculate rotation matrices
        self.m_L1 = self.create_rotation_matrix(EOP1)  
        self.m_L2 = self.create_rotation_matrix(EOP2)  

        # store camera positions
        self.x_L1, self.y_L1, self.z_L1 = EOP1[0], EOP1[1], EOP1[2]
        self.x_L2, self.y_L2, self.z_L2 = EOP2[0], EOP2[1], EOP2[2]

        # setup logging if enabled
        if log:
            logging.basicConfig(
                filename=f'collinearity_solver_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
                level=logging.INFO,
                format='%(asctime)s - %(message)s'
            )

            # log initialization parameters
            logging.info(f"""
            initialization parameters:
                focal length: {focal_length} mm
                pixel size: {pixel_size} mm
                image dimensions: {img_width}x{img_height} pixels
                
            camera 1 parameters:
                position: ({EOP1[0]:.3f}, {EOP1[1]:.3f}, {EOP1[2]:.3f})
                orientation (ω,φ,κ): ({EOP1[3]:.4f}°, {EOP1[4]:.4f}°, {EOP1[5]:.4f}°)
                rotation matrix:
                {self.m_L1[0]}
                {self.m_L1[1]}
                {self.m_L1[2]}
                
            camera 2 parameters:
                position: ({EOP2[0]:.3f}, {EOP2[1]:.3f}, {EOP2[2]:.3f})
                orientation (ω,φ,κ): ({EOP2[3]:.4f}°, {EOP2[4]:.4f}°, {EOP2[5]:.4f}°)
                rotation matrix:
                {self.m_L2[0]}
                {self.m_L2[1]}
                {self.m_L2[2]}
            """)

        self.log = log

    def create_rotation_matrix(self, EOP, unit="deg"):
        """
        create a 3x3 rotation matrix from omega, phi, kappa angles
        
        args:
            EOP (np.array): exterior orientation parameters [x, y, z, omega, phi, kappa]
            unit (str): angle unit, either 'deg' or 'rad'
            
        returns:
            np.array: 3x3 rotation matrix
        """
        if unit=="deg":
            omega = np.deg2rad(EOP[3])
            phi = np.deg2rad(EOP[4])
            kappa = np.deg2rad(EOP[5])
        else:
            omega, phi, kappa = EOP[3], EOP[4], EOP[5]
        
        # compute rotation matrix elements
        m11 = cos(phi) * cos(kappa)
        m12 = sin(omega) * sin(phi) * cos(kappa) + cos(omega) * sin(kappa)
        m13 = -cos(omega) * sin(phi) * cos(kappa) + sin(omega) * sin(kappa)
        
        m21 = -cos(phi) * sin(kappa)
        m22 = -sin(omega) * sin(phi) * sin(kappa) + cos(omega) * cos(kappa)
        m23 = cos(omega) * sin(phi) * sin(kappa) + sin(omega) * cos(kappa)
        
        m31 = sin(phi)
        m32 = -sin(omega) * cos(phi)
        m33 = cos(omega) * cos(phi)
        
        return np.array([[m11, m12, m13],
                        [m21, m22, m23],
                        [m31, m32, m33]])
    
    def _rotation_matrix_to_euler(self, R, unit="deg"):
        """
        convert rotation matrix back to euler angles
        
        args:
            R (np.array): 3x3 rotation matrix
            unit (str): output angle unit, either 'deg' or 'rad'
            
        returns:
            tuple: (omega, phi, kappa) angles
        """
        phi = np.arcsin(R[2, 0])
        
        # gimbal lock check
        if np.isclose(np.abs(R[2, 0]), 1):
            omega = 0
            kappa = np.arctan2(R[0, 1], R[0, 2])
        else:
            omega = np.arctan2(-R[2, 1], R[2, 2])
            kappa = np.arctan2(-R[1, 0], R[0, 0])

        if unit == "deg":
            omega = np.rad2deg(omega)
            phi = np.rad2deg(phi)
            kappa = np.rad2deg(kappa)
    
        return omega, phi, kappa
    
    def get_img_coords(self, x, y):
        """
        convert pixel coordinates to photo coordinates
        
        args:
            x (float): column coordinate in pixels
            y (float): row coordinate in pixels
            
        returns:
            tuple: (x_p, y_p) photo coordinates in mm
        """
        x_p = (x-self.c0)*self.pixel_size
        y_p = -(y-self.r0)*self.pixel_size
        return (x_p, y_p)
    
    def _calculate_pqr(self, xp, yp, rotation_matrix):
        """
        calculate p, q, r coefficients for collinearity equations
        
        args:
            xp (float): x photo coordinate in mm
            yp (float): y photo coordinate in mm
            rotation_matrix (np.array): 3x3 rotation matrix
            
        returns:
            tuple: (p, q, r) coefficients
        """
        p = (rotation_matrix[0,0]*xp + rotation_matrix[0,1]*yp + rotation_matrix[0,2]*-(self.focal_length))
        q = (rotation_matrix[1,0]*xp + rotation_matrix[1,1]*yp + rotation_matrix[1,2]*-(self.focal_length))
        r = (rotation_matrix[2,0]*xp + rotation_matrix[2,1]*yp + rotation_matrix[2,2]*-(self.focal_length))
        return p,q,r
    
    def compute_3D(self, cp1, rp1, cp2, rp2):
        """
        compute 3d coordinates from stereo pair image coordinates
        
        args:
            cp1 (float): column coordinate in first image
            rp1 (float): row coordinate in first image
            cp2 (float): column coordinate in second image
            rp2 (float): row coordinate in second image
            
        returns:
            list: [X, Y, Z] world coordinates
        """
        x_p1, y_p1 = self.get_img_coords(cp1, rp1)
        x_p2, y_p2 = self.get_img_coords(cp2, rp2)

        p1, q1, r1 = self._calculate_pqr(x_p1, y_p1, self.m_L1)
        p2, q2, r2 = self._calculate_pqr(x_p2, y_p2, self.m_L2)

        # calculate world coordinates
        Zp = (self.x_L1-self.x_L2 - (self.z_L1*p1/r1) + (self.z_L2*p2/r2)) / (p2/r2-p1/r1)
        Xp = self.x_L1 + (Zp - self.z_L1)*(p1/r1)
        Yp1 = self.y_L1 + (Zp - self.z_L1)*(q1/r1)
        Yp2 = self.y_L2 + (Zp - self.z_L2)*(q2/r2)
        Yp = (Yp1 + Yp2) / 2

        if self.log:
            logging.info(f"""
            Input coordinates: ({cp1}, {rp1}), ({cp2}, {rp2})
            Photo coordinates: ({x_p1:.4f}, {y_p1:.4f}), ({x_p2:.4f}, {y_p2:.4f})
            PQR coefficients: 
                Image 1: p={p1:.4f}, q={q1:.4f}, r={r1:.4f}
                Image 2: p={p2:.4f}, q={q2:.4f}, r={r2:.4f}
            Computed world coordinates: X={Xp:.4f}, Y={Yp:.4f}, Z={Zp:.4f}
            """)

        return [Xp, Yp, Zp]

def main():
    # create an instance of the collinearity solver
    stereo_solver = collinearity_solver(
        CAMERA_PARAMS['focal_length'],
        CAMERA_PARAMS['pixel_size'],
        CAMERA_EOP['425'],
        CAMERA_EOP['426'],
        CAMERA_PARAMS['img_width'],
        CAMERA_PARAMS['img_height'],
        log=True
    )

    # read points from csv file
    points_df = pd.read_csv('points.csv', header=0)
    results = []
    
    # process each point pair
    for _, point_pair in points_df.iterrows():
        xyz = stereo_solver.compute_3D(
            point_pair["425_x"],
            point_pair["425_y"],
            point_pair["426_x"],
            point_pair["426_y"],
        )
        print(xyz)
        results.append(xyz)
    
    # save results to csv
    results_df = pd.DataFrame(results, columns=['x', 'y', 'z'])
    results_df.to_csv('3d_coords.csv', index=False)
    print("3d coordinates have been saved to '3d_coords.csv'")

if __name__ == "__main__":
    main()
