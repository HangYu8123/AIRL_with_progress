from arm_env.src.armpy import kortex_arm
import numpy as np
import rospy
from sensor_msgs.msg import JointState
class arm_sim:
    def __init__(self, observation_space, action_space, seed=0):
        self.observation_space = observation_space
        self.action_space = action_space
        self.seed = seed
        np.random.seed(seed)
        self.arm = kortex_arm.Arm()

        rospy.init_node('arm_sim_for_RL', anonymous=True)

    def get_state(self):
        return self.arm.get_joint_angles()
    
    def seed(self, seed):
        self.seed = seed
        np.random.seed(seed)
    
    def reset(self):
        self.arm.home_arm()
        return self.arm.get_joint_angles()
    
    def step(self, action):
        self.arm.goto_joint_pose_sim(action)
        state = self.arm.get_joint_angles()
        reward = 0
        done = 1
        return state, reward, done, None
    
arm = arm_sim(observation_space=7, action_space=7)
print(arm.get_state())