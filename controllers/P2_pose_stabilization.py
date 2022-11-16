#!/usr/bin/env python3
import rospy
import numpy as np
from std_msgs.msg import Float32
from utils.utils import wrapToPi

# command zero velocities once we are this close to the goal
RHO_THRES = 0.05
ALPHA_THRES = 0.1
DELTA_THRES = 0.1

class PoseController:
    """ Pose stabilization controller """
    def __init__(self, k1, k2, k3, V_max=0.5, om_max=1):
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

        self.V_max = V_max
        self.om_max = om_max

        self.alpha = 0
        self.delta = 0
        self.rho = 0

        self.pub_alpha = rospy.Publisher('/controller/alpha', Float32, queue_size=10)
        self.pub_delta = rospy.Publisher('/controller/delta', Float32, queue_size=10)
        self.pub_rho = rospy.Publisher('/controller/rho', Float32, queue_size=10)

    def load_goal(self, x_g, y_g, th_g):
        """ Loads in a new goal position """
        self.x_g = x_g
        self.y_g = y_g
        self.th_g = th_g

    def compute_control(self, x, y, th, t):
        """
        Inputs:
            x,y,th: Current state
            t: Current time (you shouldn't need to use this)
        Outputs: 
            V, om: Control actions

        Hints: You'll need to use the wrapToPi function. The np.sinc function
        may also be useful, look up its documentation
        """
        ########## Code starts here ##########
        dx = self.x_g-x
        dy = self.y_g-y
        self.alpha = wrapToPi(np.arctan2(dy,dx)-wrapToPi(th))
        self.delta = wrapToPi(np.arctan2(dy,dx)-self.th_g)
        self.rho = np.sqrt(dx**2+dy**2)
        V = self.k1*self.rho*np.cos(self.alpha)
        om = self.k2*self.alpha + self.k1*np.sinc(self.alpha/np.pi)*np.cos(self.alpha)*(self.alpha+self.k3*self.delta)
        self.pub_alpha.publish(self.alpha)
        self.pub_delta.publish(self.delta)
        self.pub_rho.publish(self.rho)
        ########## Code ends here ##########

        # apply control limits
        V = np.clip(V, -self.V_max, self.V_max)
        om = np.clip(om, -self.om_max, self.om_max)

        return V, om