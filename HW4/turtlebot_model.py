import numpy as np

EPSILON_OMEGA = 1e-3

def compute_dynamics(xvec, u, dt, compute_jacobians=True):
    """
    Compute Turtlebot dynamics (unicycle model).

    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
        compute_jacobians: bool         - compute Jacobians Gx, Gu if true.
    Outputs:
         g: np.array[3,]  - New state after applying u for dt seconds.
        Gx: np.array[3,3] - Jacobian of g with respect to xvec.
        Gu: np.array[3,2] - Jacobian of g with respect to u.
    """
    ########## Code starts here ##########
    # TODO: Compute g, Gx, Gu
    # HINT: To compute the new state g, you will need to integrate the dynamics of x, y, theta
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.
    x,y,θ = xvec
    V,ω = u
    if abs(ω) < EPSILON_OMEGA:
        g = np.array([
            x+V*dt*np.cos(θ),
            y+V*dt*np.sin(θ),
            θ
        ])
        Gx = np.array([
            [1,0,-V*dt*np.sin(θ)],
            [0,1,V*dt*np.cos(θ)],
            [0,0,1]
        ])
        Gu = np.array([
            [dt*np.cos(θ), -V*dt**2*np.sin(θ)],
            [dt*np.sin(θ), V*dt**2*np.cos(θ)],
            [0, dt]
        ])
    else:
        g = np.array([
            x+V/ω*(np.sin(θ+ω*dt)-np.sin(θ)),
            y+V/ω*(np.cos(θ)-np.cos(θ+ω*dt)),
            θ+ω*dt
        ])
        Gx = np.array([
            [1, 0, V/ω*(np.cos(θ+ω*dt)-np.cos(θ))],
            [0, 1, V/ω*(np.sin(θ+ω*dt)-np.sin(θ))],
            [0,0,1]
        ])
        Gu = np.array([
            [(np.sin(θ+ω*dt)-np.sin(θ))/ω, V*(np.sin(θ)-np.sin(θ+ω*dt)+ω*dt*np.cos(θ+ω*dt))/(ω**2)],
            [(np.cos(θ)-np.cos(θ+ω*dt))/ω, V*(np.cos(θ+ω*dt)-np.cos(θ)+ω*dt*np.sin(θ+ω*dt))/(ω**2)],
            [0, dt]
        ])

    ########## Code ends here ##########

    if not compute_jacobians:
        return g

    return g, Gx, Gu

def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True):
    """
    Given a single map line in the world frame, outputs the line parameters
    in the scanner frame so it can be associated with the lines extracted
    from the scanner measurements.

    Input:
                     line: np.array[2,] - map line (alpha, r) in world frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
         compute_jacobian: bool         - compute Jacobian Hx if true.
    Outputs:
         h: np.array[2,]  - line parameters in the scanner (camera) frame.
        Hx: np.array[2,3] - Jacobian of h with respect to x.
    """
    alpha, r = line

    ########## Code starts here ##########
    # TODO: Compute h, Hx
    # HINT: Calculate the pose of the camera in the world frame (x_cam, y_cam, th_cam), a rotation matrix may be useful.
    # HINT: To compute line parameters in the camera frame h = (alpha_in_cam, r_in_cam), 
    #       draw a diagram with a line parameterized by (alpha,r) in the world frame and 
    #       a camera frame with origin at x_cam, y_cam rotated by th_cam wrt to the world frame
    # HINT: What is the projection of the camera location (x_cam, y_cam) on the line r? 
    # HINT: To find Hx, write h in terms of the pose of the base in world frame (x_base, y_base, th_base)
    cam_x, cam_y, cam_th = tf_base_to_camera
    r_x, r_y, r_th = x
    alpha, r = line
    R = np.array([
        [np.cos(r_th), -np.sin(r_th), 0],
        [np.sin(r_th), np.cos(r_th), 0],
        [0, 0, 1]
    ])
    world_x, world_y, world_th = R@tf_base_to_camera+x
    r_beta = np.arctan2(world_y,world_x)
    r_dist = np.sqrt(world_x**2+world_y**2)
    h = np.array([alpha-world_th,r-r_dist*np.cos(alpha-r_beta)])
    term1 = -(((2*(-cam_x*np.sin(r_th)-cam_y*np.cos(r_th))*world_x+2*(cam_x*np.cos(r_th)-cam_y*np.sin(r_th))*world_y)*np.cos(alpha-np.arctan2(world_y,world_x)))/(2*np.sqrt(world_x**2+world_y**2)))
    term2 = ((cam_x*np.cos(r_th)-cam_y*np.sin(r_th))/world_x-(-cam_x*np.sin(r_th)-cam_y*np.cos(r_th))*world_y/world_x**2)*np.sqrt(world_x**2+world_y**2)*np.sin(alpha-np.arctan2(world_y,world_x))/(world_y**2/world_x**2+1)
    Hx = np.array([
        [0, 0, -1],
        [(world_x**2+world_y**2)**(-1/2)*(world_y*np.sin(alpha-np.arctan2(world_y,world_x))-world_x*np.cos(alpha-np.arctan2(world_y,world_x))),
         (world_x**2+world_y**2)**(-1/2)*(-world_y*np.cos(alpha-np.arctan2(world_y,world_x))-world_x*np.sin(alpha-np.arctan2(world_y,world_x))),
         term1-term2]
    ])

    ########## Code ends here ##########

    if not compute_jacobian:
        return h

    return h, Hx


def normalize_line_parameters(h, Hx=None):
    """
    Ensures that r is positive and alpha is in the range [-pi, pi].

    Inputs:
         h: np.array[2,]  - line parameters (alpha, r).
        Hx: np.array[2,n] - Jacobian of line parameters with respect to x.
    Outputs:
         h: np.array[2,]  - normalized parameters.
        Hx: np.array[2,n] - Jacobian of normalized line parameters. Edited in place.
    """
    alpha, r = h
    if r < 0:
        alpha += np.pi
        r *= -1
        if Hx is not None:
            Hx[1,:] *= -1
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    h = np.array([alpha, r])

    if Hx is not None:
        return h, Hx
    return h
