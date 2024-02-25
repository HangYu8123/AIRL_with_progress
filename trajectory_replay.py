import rospy
import rosbag
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Header

rospy.init_node('publish_joint_trajectory')
pub = rospy.Publisher('/my_gen3_lite/gen3_lite_joint_trajectory_controller/command', JointTrajectory, queue_size=10)
# pub = rospy.Publisher('/my_gen3/gen3_joint_trajectory_controller/command', JointTrajectory, queue_size=10)
# Waiting for connection
rospy.sleep(1)

# Import xxx.Bag
bag_file_name = '0.bag'
bag = rosbag.Bag(bag_file_name, 'r')

# /my_gen3/joint_states: frequency of 50Hz (0.02s per point)
point_time_interval = rospy.Duration(0.02)  # 20ms or 0.02s per point

for index, (topic, msg, t) in enumerate(bag.read_messages(topics=['/my_gen3_lite/joint_states'])):
# for index, (topic, msg, t) in enumerate(bag.read_messages(topics=['/my_gen3/joint_states'])):
    traj = JointTrajectory()
    traj.header.stamp = rospy.Time.now()  # Use current time
    traj.joint_names = msg.name[:7]
    point = JointTrajectoryPoint()
    point.positions = list(msg.position)[:7]
    # point.velocities = list(msg.velocity)[:7]
    # point.effort = list(msg.effort)[:7]
    point.time_from_start = point_time_interval * index
    traj.points.append(point)

    # Publish the trajectory
    pub.publish(traj)

    # Wait for the duration of the point interval before sending the next point
    rospy.sleep(point_time_interval)
bag.close()
# command = "rosbag record -O {} {}".format(bag_name, topic_name)
# process = subprocess.Popen(command, shell=True)