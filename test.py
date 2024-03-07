import os
import argparse
import torch
from airl_ppo.buffer import SerializedBuffer
from airl_ppo.utils import read_cup_index_from_csv, get_all_bag_files, whole_bag_to_messages_with_cup_idx
from airl_ppo.algo.airl import AIRL
from airl_ppo.env import arm_sim

import numpy as np
def run():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    observation_space = (7,)
    action_space = (7,)
    file_path = "/home/noahfang/Downloads/LfD_with_porgress-master/bags"
    cup_idx_list = [2] * 40
    # cup_idx_list = read_cup_index_from_csv("/home/hang/catkin_ws/src/ldf_with_progress/BC/participant_sheet.csv")
    cup_idx = 2
    seed = 0

    env = arm_sim(observation_space=observation_space, action_space=action_space, seed=0)
    buffer_exp = SerializedBuffer(
        path="/home/noahfang/Documents/Lab/AIRL_with_progress/arm_env/buffers/size10000_.pth",
        device=device
    )

    algo = AIRL(
        buffer_exp=buffer_exp,
        state_shape=observation_space,
        action_shape=action_space,
        device=device,
        seed=0,
        rollout_length=50000
    )
    algo.disc.load_state_dict(torch.load("/home/noahfang/Documents/Lab/AIRL_with_progress/log/seed0-20240305-1939/model/step39000/disc.pth"))
    algo.actor.load_state_dict(torch.load("/home/noahfang/Documents/Lab/AIRL_with_progress/log/seed0-20240305-1939/model/step39000/actor.pth"))  
    algo.critic.load_state_dict(torch.load("/home/noahfang/Documents/Lab/AIRL_with_progress/log/seed0-20240305-1939/model/step39000/critic.pth"))



    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    reward_memo = {}
    bag_file_list = get_all_bag_files(file_path) 
    msg_dic = whole_bag_to_messages_with_cup_idx(bag_file_list, cup_idx_list)
    for user_i, traj in enumerate(msg_dic[cup_idx]):
        reward_memo[user_i] = []
        for i in range(len(traj) - 1):
            dim = observation_space[0]
            state = np.array(traj[i].position[:dim])
            action = np.array(traj[i].position[:dim])
            next_state = np.array(traj[i+1].position[:dim])
            progress = 0.
            # 判断是否是最后一个轨迹
            if i == len(traj) - 1:
                done = True
                state = env.reset()
            else:
                done = False
            reward_memo[user_i].append(algo.critic.forward(torch.from_numpy(action).float()).item())
    torch.save(reward_memo, '/home/noahfang/Documents/Lab/AIRL_with_progress/replayed_traj/test.pt')
    # torch.save(state_memo, '/home/noahfang/Documents/Lab/AIRL_with_progress/replayed_traj/state_memo.pt')

if __name__ == '__main__':
    run()
