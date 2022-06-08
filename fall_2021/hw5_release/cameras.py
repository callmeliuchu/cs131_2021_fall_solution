from typing import Tuple

import numpy as np


def camera_from_world_transform(d: float = 1.0) -> np.ndarray:
    """Define a transformation matrix in homogeneous coordinates that
    transforms coordinates from world space to camera space, according
    to the coordinate systems in Question 1.


    Args:
        d (float, optional): Total distance of displacement between world and camera
            origins. Will always be greater than or equal to zero. Defaults to 1.0.

    Returns:
        T (np.ndarray): Left-hand transformation matrix, such that c = Tw
            for world coordinate w and camera coordinate c as column vectors.
            Shape = (4,4) where 4 means 3D+1 for homogeneous.
    """
    T = np.eye(4)
    # YOUR CODE HERE
    t = np.array([
        [1,0,0,-1/np.sqrt(2)],
        [0,1,0,0],
        [0,0,1,-1/np.sqrt(2)],
        [0,0,0,1],
    ])
    r = np.array([
        [1/np.sqrt(2),0,-1/np.sqrt(2),0],
        [0,1,0,0],
        [-1/np.sqrt(2),0,-1/np.sqrt(2),0],
        [0,0,0,1],
    ])
    T = np.linalg.inv(r).dot(t)
    # END YOUR CODE
    assert T.shape == (4, 4)
    return T


def xx(T: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray]:
    """Apply a transformation matrix to a set of points.

    Hint: You'll want to first convert all of the points to homogeneous coordinates.
    Each point in the (3,N) shape edges is a length 3 vector for x, y, and z, so
    appending a 1 after z to each point will make this homogeneous coordinates.

    You shouldn't need any loops for this function.

    Args:
        T (np.ndarray):
            Left-hand transformation matrix, such that c = Tw
                for world coordinate w and camera coordinate c as column vectors.
            Shape = (4,4) where 4 means 3D+1 for homogeneous.
        points (np.ndarray):
            Shape = (3,N) where 3 means 3D and N is the number of points to transform.

    Returns:
        points_transformed (np.ndarray):
            Transformed points.
            Shape = (3,N) where 3 means 3D and N is the number of points.
    """
    N = points.shape[1]
    assert points.shape == (3, N)
    points = np.row_stack((points,[1]*N))
#     print(points)

    # You'll replace this!
    points_transformed = np.zeros((3, N))
    

    # YOUR CODE HERE
    t = T.dot(points)
#     print(t)
    t = t / t[3,:]
#     print(t)
    points_transformed = points_transformed[:3,:]
#     print(points_transformed)
    # END YOUR CODE

    assert points_transformed.shape == (3, N)
    return points_transformed


def intersection_from_lines(
    a_0: np.ndarray, a_1: np.ndarray, b_0: np.ndarray, b_1: np.ndarray
) -> np.ndarray:
    """Find the intersection of two lines (infinite length), each defined by a
    pair of points.

    Args:
        a_0 (np.ndarray): First point of first line; shape `(2,)`.
        a_1 (np.ndarray): Second point of first line; shape `(2,)`.
        b_0 (np.ndarray): First point of second line; shape `(2,)`.
        b_1 (np.ndarray): Second point of second line; shape `(2,)`.

    Returns:
        np.ndarray: the intersection of the two lines definied by (a0, a1)
                    and (b0, b1).
    """
    # Validate inputs
    assert a_0.shape == a_1.shape == b_0.shape == b_1.shape == (2,)
    assert a_0.dtype == a_1.dtype == b_0.dtype == b_1.dtype == np.float

    # Intersection point between lines
    out = np.zeros(2)

    # YOUR CODE HERE
    A = np.array([
        a_1 - a_0,
        b_0 - b_1
    ]).T
    b = b_0 - a_0
    t = np.linalg.inv(A).dot(b.T)
    out = a_0 + (a_1-a_0)*t[0]
    # END YOUR CODE

    assert out.shape == (2,)
    assert out.dtype == np.float

    return out


def optical_center_from_vanishing_points(
    v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> np.ndarray:
    """Compute the optical center of our camera intrinsics from three vanishing
    points corresponding to mutually orthogonal directions.

    Hints:
    - Your `intersection_from_lines()` implementation might be helpful here.
    - It might be worth reviewing vector projection with dot products.

    Args:
        v0 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v1 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v2 (np.ndarray): Vanishing point in image space; shape `(2,)`.

    Returns:
        np.ndarray: Optical center; shape `(2,)`.
    """
    assert v0.shape == v1.shape == v2.shape == (2,), "Wrong shape!"

    optical_center = np.zeros(2)

    # YOUR CODE HERE
    v21 = v2 - v1
    v21_ = np.array([v21[1],-v21[0]])
    v01 = v0 - v1
    v01_ = np.array([v01[1],-v01[0]])
    A = np.array([
        [v21_[0],-v01_[0]],
        [v21_[1],-v01_[1]]
    ])
    b = v2 - v0
    t = np.linalg.inv(A).dot(b)
    optical_center = v0 + v21_*t[0]
    # END YOUR CODE
    print(np.mean(optical_center))

    assert optical_center.shape == (2,)
    return optical_center


def focal_length_from_two_vanishing_points(
    v0: np.ndarray, v1: np.ndarray, optical_center: np.ndarray
) -> np.ndarray:
    """Compute focal length of camera, from two vanishing points and the
    calibrated optical center.

    Args:
        v0 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        v1 (np.ndarray): Vanishing point in image space; shape `(2,)`.
        optical_center (np.ndarray): Calibrated optical center; shape `(2,)`.

    Returns:
        float: Calibrated focal length.
    """
    assert v0.shape == v1.shape == optical_center.shape == (2,), "Wrong shape!"

    f = None

    # YOUR CODE HERE
    v10 = v1 - v0
    v10_ = np.array([v10[1],-v10[0]])
    A = np.array([
        [v10[0],-v10_[0]],
        [v10[1],-v10_[1]]
    ])
    b = optical_center - v0
    t = np.linalg.inv(A).dot(b)
    vv = v0 + v10 * t[0]
    d1 = np.sum((vv-v0)**2)**0.5
    d2 = np.sum((vv-v1)**2)**0.5
    d3 = np.sum((vv-optical_center)**2)**0.5
    f = np.sqrt(d1*d2-d3**2)
    # END YOUR CODE

    return float(f)


def physical_focal_length_from_calibration(
    f: float, sensor_diagonal_mm: float, image_diagonal_pixels: float
) -> float:
    """Compute the physical focal length of our camera, in millimeters.

    Args:
        f (float): Calibrated focal length, using pixel units.
        sensor_diagonal_mm (float): Length across the diagonal of our camera
            sensor, in millimeters.
        image_diagonal_pixels (float): Length across the diagonal of the
            calibration image, in pixels.

    Returns:
        float: Calibrated focal length, in millimeters.
    """
    f_mm = None

    # YOUR CODE HERE
    f_mm = sensor_diagonal_mm / image_diagonal_pixels * f
    # END YOUR CODE

    return f_mm
