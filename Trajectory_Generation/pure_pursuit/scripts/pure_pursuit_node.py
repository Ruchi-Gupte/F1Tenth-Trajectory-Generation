#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped, AckermannDrive

# TODO CHECK: include needed ROS msg type headers and libraries

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point,PoseStamped
from builtin_interfaces.msg import Duration
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R

import scipy
from scipy import interpolate
from scipy.interpolate import splprep, splev



# ADD A FEW POINTS HERE TO MAKE INTO SPLINE
pts = np.array([
[ 7.35  ,0.221 ],
[  4.61 , -1.35],
[  2.55 , -2.7],
[ 0.741  , -3.85 ],
[  -0.665 , -4.4],
[  -2.12 , -3.45 ],
[ -2.96  , -2.16 ],
[ -2.56  , -0.745],
[ -1.33  ,-0.0143 ],
[ 0.38  ,-0.0848 ],
[ 2.08  ,-0.318 ],
[  3.53 ,0.525 ],
[ 4.86  , 2.43],
[ 6.23  , 5.0 ],
[ 6.91  ,7.21 ],
[  8.86 , 9.11],
[ 10.1  , 9.82],
[ 10.9  , 9.92],
[ 11.3  , 8.9],
[ 10.3, 7.63 ],
[ 9.08  ,6.67 ],
[ 8.38  , 5.92],
[ 8.5  , 5.19],
[  9.2 , 4.61],
[ 10.4  ,5.4 ],
[ 12.1  , 6.41],
[ 13.5  , 7.49],
[  13.9 , 8.44 ],
[ 13.1   , 10.1],
[ 13.7  , 11.2 ],
[ 15  , 10.4],
[ 16.1  , 8.71 ],
[  16.9 , 6.91 ],
[  15.4 , 5.01 ],
[  13.4 , 4.1],
[  11.2 , 2.55 ],
[ 7.35  ,0.221 ]
])


#COMMENT THIS PART OUT IF TRYING TO DO 1 or 2 BELOW
pts = pts.T
x,y = pts
tck, u = interpolate.splprep([x, y], s=0, per=True)

xi, yi = interpolate.splev(np.linspace(0, 1, 2000), tck)

xi = xi.reshape((-1,1))
yi = yi.reshape((-1,1))

new_pts = np.hstack((xi,yi)).T

# 1:TO VISUALIZE JUST THE POINTS
# pts = pts.T
# new_pts = pts

# 2: TO VISUALIZE THE TUMN RACELINE
my_data = np.genfromtxt('/sim_ws/src/pure_pursuit/scripts/traj_race_cl.csv', delimiter=';')
my_array  = np.vstack((my_data[:,1], my_data[:,2], my_data[:,5], my_data[:,3], my_data[:,4])).T
np.save("/sim_ws/src/pure_pursuit/scripts/trajectory.npy",my_array)
new_pts2= my_array[:,0:2].T
np.savetxt("/sim_ws/src/pure_pursuit/scripts/avoidance_trajectory.csv", new_pts2, delimiter=",")



class PurePursuit(Node):
    """ 
    Implement Pure Pursuit on the car
    This is just a template, you are free to implement your own node!
    """
    def __init__(self):
        super().__init__('pure_pursuit_node')
        """
        self.waypoints: Load the waypoint saved 

            Type: numpy array -> Shape : [2,1000] where 2 corresponds to x and y and 1000 are the number of points
        """
        self.waypoints = new_pts
        # self.waypoints= np.load("/sim_ws/src/pure_pursuit/scripts/trajectory.npy")[:,:2].T
        vis_topic = "visualization_marker"
        self.visualize_pub              =       self.create_publisher(Marker, vis_topic, 10)
        self.vis_msg                         =       Marker()
        self.vis_msg.header.frame_id         =       "/map"
        self.vis_msg.type                    =       Marker.POINTS
        self.vis_msg.action                  =       Marker.ADD
        self.vis_msg.scale.x                 =       0.1
        self.vis_msg.scale.y                 =       0.1
        self.vis_msg.scale.z                 =       0.1
        self.vis_msg.color.g                 =       1.0
        self.vis_msg.color.a                 =       1.0
        self.vis_msg.pose.orientation.w      =       1.0
        self.vis_msg.lifetime                =       Duration()
        self.count=100000000
        self.count2=100000000
        self.position = [0,0]
        self.ldist= 0
        self.rdist= 0
        self.start_rec= None
        self.ifrecord= False
        for i in range(self.waypoints.shape[1]):
            p       =       Point()
            p.x     =       self.waypoints[0,i]
            p.y     =       self.waypoints[1,i]
            p.z     =       0.0
            self.vis_msg.points.append(p)

        for i in range(new_pts2.shape[1]):
            p       =       Point()
            p.x     =       new_pts2[0,i]
            p.y     =       new_pts2[1,i]
            p.z     =       0.0
            self.vis_msg.points.append(p)

        
        self.data= []
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.visualize_pub.publish(self.vis_msg)
        lidarscan_topic = '/scan'
        odomTopic = "/ego_racecar/odom"
        self.drivePub = self.create_publisher(AckermannDriveStamped,"drive",0)
        self.newscan = self.create_publisher(LaserScan,'/revised_scan', 0)
        self.odomSub = self.create_subscription(Odometry,odomTopic,self.pose_callback,0)
        self.scan_sub_ = self.create_subscription(LaserScan,lidarscan_topic,self.scan_callback,1)
        self.ld = 0.7 #lookahead distance constant to 0.5m


    def scan_callback(self, scan_msg):
        """
        LaserScan callback, you should update your occupancy grid here
        Args: 
            scan_msg (LaserScan): incoming message from subscribed topic
        Returns:
        """
        angle_increment = scan_msg.angle_increment 
        ranges = np.array(scan_msg.ranges)
        if self.ifrecord==True:
            self.ldist = ranges[int(np.radians(225)/angle_increment)]
            self.rdist = ranges[int(np.radians(45)/angle_increment)]
            self.count= self.count-1
            if self.count>self.count2-999 or self.count<=0:
                print("Not recording", self.count)
                self.start_rec = [self.position[0], self.position[1]]

            if self.count>0 and self.count<=self.count2-1004:
                print("recording", self.count)
                self.data.append([self.position[0], self.position[1], self.ldist, self.rdist])

            if self.count < self.count2-4000 and (np.sqrt(((np.array(self.start_rec) - np.array([self.position[0], self.position[1]])) ** 2).sum())) < 0.03:
                self.count=0

            if self.count==0:
                np.save("/sim_ws/src/pure_pursuit/scripts/Trajectory_unclipped.npy", np.array(self.data))

            print(np.sqrt(((np.array(self.start_rec) - np.array([self.position[0], self.position[1]])) ** 2).sum()))
            # print(len(self.data))
            # print(self.position, self.ldist, self.rdist)
            rev_scan = LaserScan()
            msg = np.zeros((ranges.shape))
            msg[int(np.radians(225)/angle_increment)]= ranges[int(np.radians(225)/angle_increment)]
            msg[int(np.radians(45)/angle_increment)]= ranges[int(np.radians(45)/angle_increment)]
            rev_scan=scan_msg
            rev_scan.ranges= list(msg.astype(np.float))
            self.newscan.publish(rev_scan)



    def timer_callback(self):
        self.visualize_pub.publish(self.vis_msg)


    def pose_callback(self, pose_msg):
        currPosex = pose_msg.pose.pose.position.x
        currPosey = pose_msg.pose.pose.position.y
        self.position= [currPosex, currPosey]
        currPose = np.array([currPosex,currPosey,0]).reshape((3,-1))
        qx = pose_msg.pose.pose.orientation.x
        qy = pose_msg.pose.pose.orientation.y
        qz = pose_msg.pose.pose.orientation.z
        qw = pose_msg.pose.pose.orientation.w
        rot_car_world = R.from_quat([qx,qy,qz,qw])       
        roll,pitch,yaw = rot_car_world.as_euler('xyz',degrees=False)
        wPts = self.waypoints
        wPts = np.vstack((wPts,np.zeros((1,wPts.shape[1]))))
        gPts = rot_car_world.apply((wPts - currPose).T,inverse=True) 
        gPts = gPts[:,:2].T #This is expected to be of shape (2xN) -> sanity check. If not, check the conversion wrt robot frame CHECKPT
        gPts = gPts[:, gPts[0,:]>0 ]
        distArray = np.linalg.norm(gPts,axis = 0) 
        bool_in_circle = distArray <= self.ld
        bool_out_circle = distArray >= self.ld
        gPts_out = gPts[:,bool_out_circle]
        out_pt = gPts_out[:, np.argmin(distArray[bool_out_circle])]
        if(bool_in_circle.sum() != 0):
            gPts_in = gPts[:,bool_in_circle]
            in_pt = gPts_in[:, np.argmax(distArray[bool_in_circle])]
        else:
            in_pt = out_pt
        goalPt = (out_pt + in_pt)/2
        x = goalPt[0]
        y = goalPt[1]
        curvature = 2*goalPt[1]/(self.ld**2)
        msg = AckermannDriveStamped()
        msg.drive.speed = float(1.0)
        msg.drive.steering_angle = curvature
        self.drivePub.publish(msg)
        
def main(args=None):
    rclpy.init(args=args)
    print("PurePursuit Initialized")
    pure_pursuit_node = PurePursuit()
    rclpy.spin(pure_pursuit_node)

    pure_pursuit_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
