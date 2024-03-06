import os
import argparse
from datetime import datetime
import torch

# from airl_ppo.env import make_env
from airl_ppo.buffer import SerializedBuffer
from airl_ppo.trainer import Trainer
from airl_ppo.algo.airl import AIRL
from airl_ppo.env import arm_sim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
observation_space = (7,)
action_space = (7,)

def run(args):
    env = arm_sim(observation_space=observation_space, action_space=action_space, seed=args.seed)
    buffer_exp = SerializedBuffer(
        path=args.buffer,
        device=device
    )

    algo = AIRL(
        buffer_exp=buffer_exp,
        state_shape=observation_space,
        action_shape=action_space,
        device=device,
        seed=args.seed,
        rollout_length=args.rollout_length
    )

    time = datetime.now().strftime("%Y%m%d-%H%M")
    # log_dir = os.path.join(
    #     './logs', args.buffer, 'AIRL', f'seed{args.seed}-{time}')
    log_dir = os.path.join(
    'logs', '/home/noahfang/Documents/Lab/AIRL_with_progress/log', f'seed{args.seed}-{time}')
    # print(log_dir)
    trainer = Trainer(
        env=env,
        # env_test=env_test,
        algo=algo,
        log_dir=log_dir,
        num_steps=args.num_steps,
        # eval_interval=args.eval_interval,
        seed=args.seed
    )
    trainer.train()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--buffer', type=str, required=True)
    p.add_argument('--rollout_length', type=int, default=50000)
    p.add_argument('--num_steps', type=int, default=10**7)
    # p.add_argument('--eval_interval', type=int, default=10**5)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    run(args)