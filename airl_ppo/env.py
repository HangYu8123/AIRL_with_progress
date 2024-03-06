from arm_env.src.armpy import kortex_arm
from sensor_msgs.msg import JointState
import numpy as np
import rospy

class arm_sim:
    def __init__(self, observation_space, action_space, max_episode_steps=1000, seed=0):
        self.observation_space = observation_space
        self.action_space = action_space
        self.seed = seed
        self._max_episode_steps = max_episode_steps
        np.random.seed(seed)
        self.arm = kortex_arm.Arm()

        rospy.init_node('arm_sim_for_RL', anonymous=True)

    def get_state(self):
        state = rospy.wait_for_message(
            f"{self.arm.robot_name}/joint_states", JointState)
        return np.array(state.position[:self.observation_space[0]])
    
    
    def seed(self, seed):
        self.seed = seed
        np.random.seed(seed)
    
    def reset(self):
        self.arm.home_arm()
        return self.get_state()
    
    def step(self, action):
        self.arm.goto_joint_pose_sim(action)
        # print(action)
        # self.arm.send_gripper_command(action[:-1])
        state = self.get_state()
        reward = 0
        rand_int = np.random.randint(0, 1000)
        if rand_int == 1:
            done = True
        else:
            done = False
        return state, reward, done, None
    