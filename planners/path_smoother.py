import numpy as np
import scipy.interpolate

def compute_smoothed_traj(path, V_des, alpha, dt):
    """
    Fit cubic spline to a path and generate a resulting trajectory for our
    wheeled robot.

    Inputs:
        path (np.array [N,2]): Initial path
        V_des (float): Desired nominal velocity, used as a heuristic to assign nominal
            times to points in the initial path
        alpha (float): Smoothing parameter (see documentation for
            scipy.interpolate.splrep)
        dt (float): Timestep used in final smooth trajectory
    Outputs:
        traj_smoothed (np.array [N,7]): Smoothed trajectory
        t_smoothed (np.array [N]): Associated trajectory times
    Hint: Use splrep and splev from scipy.interpolate
    """
    ########## Code starts here ##########
    path = np.array(path)    
    t_chunked = [0]
    for i in range(1, path.shape[0]):
        time_i = t_chunked[i - 1] + np.linalg.norm(path[i] - path[i - 1])/V_des
        t_chunked.append(time_i)

    tf = t_chunked[-1]
    t_smoothed = np.arange(0, tf, dt)
    
    spl = scipy.interpolate.splrep(t_chunked, path[:, 0], s = alpha)
    x_d = scipy.interpolate.splev(t_smoothed, spl, der = 0)

    spl = scipy.interpolate.splrep(t_chunked, path[:, 1], s = alpha)
    y_d = scipy.interpolate.splev(t_smoothed, spl, der = 0)

    spl = scipy.interpolate.splrep(t_chunked, path[:, 0], s = alpha)
    xd_d = scipy.interpolate.splev(t_smoothed, spl, der = 1)

    spl = scipy.interpolate.splrep(t_chunked, path[:, 1], s = alpha)
    yd_d = scipy.interpolate.splev(t_smoothed, spl, der = 1)

    spl = scipy.interpolate.splrep(t_chunked, path[:, 0], s = alpha)
    xdd_d = scipy.interpolate.splev(t_smoothed, spl, der = 2)

    spl = scipy.interpolate.splrep(t_chunked, path[:, 1], s = alpha)
    ydd_d = scipy.interpolate.splev(t_smoothed, spl, der = 2)

    theta_d = np.arctan2(yd_d, xd_d)

    ########## Code ends here ##########
    traj_smoothed = np.stack([x_d, y_d, theta_d, xd_d, yd_d, xdd_d, ydd_d]).transpose()

    return traj_smoothed, t_smoothed
