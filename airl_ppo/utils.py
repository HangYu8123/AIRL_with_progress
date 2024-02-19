from tqdm import tqdm
import numpy as np
import torch
from .buffer import Buffer
from .demo_preprocess import *

def soft_update(target, source, tau):
    for t, s in zip(target.parameters(), source.parameters()):
        t.data.mul_(1.0 - tau)
        t.data.add_(tau * s.data)


def disable_gradient(network):
    for param in network.parameters():
        param.requires_grad = False

def demo_to_buffer(env, algo, buffer_size, device, std, p_rand, seed=0, observation_space=(7,), action_space=(7,)):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    buffer = Buffer(
        buffer_size=buffer_size,
        state_shape=observation_space,
        action_shape=action_space,
        device=device
    )

    
    num_episodes = 0

    state = env.reset()
    t = 0
    episode_return = 0.0

    for _ in tqdm(range(1, buffer_size + 1)):
        t += 1

        if np.random.rand() < p_rand:
            action = env.action_space.sample()
        else:
            action = algo.exploit(state)
            action = add_random_noise(action, std)

        next_state, reward, done, _ = env.step(action)
        mask = False if t == env._max_episode_steps else done
        buffer.append(state, action, reward, mask, next_state)
        episode_return += reward

        if done:
            num_episodes += 1
            state = env.reset()
            t = 0
            episode_return = 0.0

        state = next_state
    return buffer

if __name__ == '__main__':
    observation_space = 7
    action_space = 7

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    buffer = Buffer(
        buffer_size=buffer_size,
        state_shape=observation_space,
        action_shape=action_space,
        device=device
    )
    # cup_idx_list = read_cup_index_from_csv("/home/hang/catkin_ws/src/ldf_with_progress/BC/participant_sheet.csv")
    cup_idx_list = [2] * 26
    file_path = "/Users/noahf/Documents/LAB/AABL/data/bags/"
    bag_file_list = get_all_bag_files(file_path) 
    msg_list = whole_bag_to_messages(bag_file_list)
    print(msg_list)
    #buffer.append(state, action, reward, mask, next_state)
    