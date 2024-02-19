from tqdm import tqdm
import numpy as np
import torch
from .buffer import Buffer
import rospy
import rosbag
from sensor_msgs.msg import JointState
import os
import csv
import random
cup_pos = [[0.5516026616096497,0.3308942914009094,0.1391326606273651],[0.41698557138442993,0.34413501620292664,0.1396018534898758],
           [0.29829156398773193,0.35091647505760193,0.1672777384519577],[0.17075064778327942,0.36124327778816223,0.15736795961856842]]

icecream_pos  = [0.34541743993759155,-0.13331931829452515,0.08949728310108185]

def read_bag(bagdir):
    bag = rosbag.Bag(bagdir,'r')
    messages = []
    for _,msg,_ in bag.read_messages(topics=['/my_gen3_lite/joint_states']):
        temp = JointState()
        temp.header = msg.header
        temp.position = msg.position
        temp.velocity = msg.velocity
        temp.name = msg.name
        temp.effort = msg.effort
        messages.append(temp)
    return messages
def get_all_bag_files(file_path:str=None):
    if file_path is not None:
        file_dir = file_path
        bag_file_list = [] 
    else:
        # use default bag file location
        bag_file_list = []
        file_dir = os.path.dirname(
            os.path.dirname(
                os.path.dirname(
                    os.path.abspath(__file__)
                )))
        file_dir = os.path.join(file_dir,"bags/")
    # look through directory to find all bag files
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if file.endswith(".bag"):
                bag_file_list.append(os.path.join(root,file))
    return bag_file_list
def read_cup_index_from_csv(file_path:str):
    with open(file_path, 'r') as f:
        cup_index = []
        reader = csv.reader(f)
        cup_index = [int(row[2]) for row in reader]
    return cup_index

def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)

def whole_bag_to_messages_with_cup_idx(bag_file_list, cup_idx_list):
    messages_dic= {0:[], 1:[], 2:[], 3:[]}
    for i in range(len(bag_file_list)):
        messages = read_bag(bag_file_list[i])
        messages_dic[cup_idx_list[i]].append(messages)
    return messages_dic

def disable_gradient(network):
    for param in network.parameters():
        param.requires_grad = False

def demo_to_buffer(file_path, cup_idx, cup_idx_list, device, buffer_size=10000, seed=0, observation_space=(7,), action_space=(7,)):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    buffer = Buffer(
        buffer_size=buffer_size,
        state_shape=observation_space,
        action_shape=action_space,
        device=device
    )
    bag_file_list = get_all_bag_files(file_path) 
    msg_dic = whole_bag_to_messages_with_cup_idx(bag_file_list, cup_idx_list)
    
    for traj in msg_dic[cup_idx]:
        for i in range(len(traj) - 1):
            dim = observation_space[0]
            state = np.array(traj[i].position[:dim])
            action = np.array(traj[i].position[:dim])
            next_state = np.array(traj[i+1].position[:dim])
            progress = 0.
            # 判断是否是最后一个轨迹
            if i == len(traj) - 1:
                done = True
            else:
                done = False
            buffer.append(state, action, progress, done, next_state)
    return buffer

if __name__ == '__main__':
    observation_space = (7,)
    action_space = (7,)
    seed = 0
    buffer_size = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cup_idx_list = [2] * 26
    cup_idx = 2
    file_path = "/Users/noahf/Documents/LAB/AABL/data/bags/"

    buffer = demo_to_buffer(file_path, cup_idx, cup_idx_list, device, buffer_size)
    print(buffer)