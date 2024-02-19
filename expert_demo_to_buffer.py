import os
import argparse
import torch
from airl_ppo.buffer import Buffer
from airl_ppo.utils import demo_to_buffer, read_cup_index_from_csv


def run():
    buffer_size = 10000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    observation_space = (7,)
    action_space = (7,)
    file_path = "/Users/noahf/Documents/LAB/AABL/data/bags/"
    cup_idx_list = [2] * 26
    # cup_idx_list = read_cup_index_from_csv("/home/hang/catkin_ws/src/ldf_with_progress/BC/participant_sheet.csv")
    cup_idx = 2
    seed = 0

    buffer = demo_to_buffer(
        buffer_size=buffer_size,
        device=device,
        observation_space=observation_space,
        action_space=action_space,
        file_path=file_path,
        cup_idx_list=cup_idx_list,
        cup_idx=cup_idx,
        seed=seed,
    )

    buffer.save(
        os.path.join(
            'buffers',
            str(cup_idx),
            f'size{buffer_size}_.pth'
        )
    )


if __name__ == '__main__':
    run()
