import numpy as np

# constants

# camera parameters
CAMERA_PARAMS = {
    'focal_length': 14,  # focal length in mm
    'pixel_size': 0.0074,  # pixel size in mm
    'img_width': 1892,
    'img_height': 1060
}

# camera exterior orientation parameters
# format: [x, y, z, omega, phi, kappa]
CAMERA_EOP = {
    '425': np.array([473541.879, 3769434.262, 656.854, -0.7896, -0.1822, 156.7895]),
    '426': np.array([473584.873, 3769392.745, 657.933, -2.52437, -1.2326, 157.4567])
}
