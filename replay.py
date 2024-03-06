from datetime import datetime
import torch
from airl_ppo.buffer import SerializedBuffer
from airl_ppo.algo.airl import AIRL
from airl_ppo.env import arm_sim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
observation_space = (7,)
action_space = (7,)

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


# print(algo.disc.state_dict())
state = env.reset()
t = 0
epoch = 10
reward_memo = {epoch:[]}
state_memo = {epoch:[]}
step = 0
while epoch > 0:
    action, log_pi = algo.explore(state)
    next_state, reward, done, _ = env.step(action)
    reward = algo.disc.calculate_reward(torch.from_numpy(state).float(), float(done), float(log_pi), torch.from_numpy(next_state).float())
    state = next_state
    reward_memo[epoch].append(reward.item()) 
    step += 1
    if done:
        epoch -= 1
        step = 0
        state = env.reset()
        reward_memo[epoch] = []
        state_memo[epoch] = []
    print(step, reward.item())
torch.save(reward_memo, '/home/noahfang/Documents/Lab/AIRL_with_progress/replayed_traj/reward_memo.pt')
torch.save(state_memo, '/home/noahfang/Documents/Lab/AIRL_with_progress/replayed_traj/state_memo.pt')