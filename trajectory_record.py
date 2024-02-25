#!/usr/bin python
import subprocess
import rospy

def record_rosbag(topic_name, bag_name):
    try:
         # Construct the rosbag record command
        command = "rosbag record -O {} {}".format(bag_name, topic_name)
        # Run the command using subprocess
        process = subprocess.Popen(command, shell=True)
        rospy.loginfo("Recording rosbag. Topic: {}. Bag: {}".format(topic_name, bag_name))
        return process
    except Exception as e:
        rospy.logerr("Failed to start rosbag record: {}".format(str(e)))

if __name__ == '__main__':
    rospy.init_node('rosbag_record_script', anonymous=True)
    topic_to_record = "/my_gen3_lite/joint_states"
    bag_file_name = "recorded_data.bag"
    process = record_rosbag(topic_to_record, bag_file_name)

     # Wait until the node is shut down or Ctrl+C is pressed
    rospy.spin()

    # Stop recording
    if process:
        process.terminate()
        rospy.loginfo("Stopped recording rosbag.")