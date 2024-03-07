this repo is in progress and built based on https://github.com/toshikwa/gail-airl-ppo.pytorch, https://github.com/Kinovarobotics/ros_kortex/blob/noetic-devel/kortex_gazebo/readme.md
this readme is not up-to-date, it will be updated after the paper been submitted.
### Tranfer raw expert demo to buffer

```bash
python expert_demo_to_buffer.py
```

### Train Imitation Learning

```bash
python train_imitation.py /home/noahfang/Documents/Lab/AIRL_with_progress/train_imitation.py --buffer /home/noahfang/Documents/Lab/AIRL_with_progress/arm_env/buffers/size10000_.pth

python train_imitation.py \
    --algo gail --cuda --env_id InvertedPendulum-v2 \
    --buffer buffers/InvertedPendulum-v2/size1000000_std0.01_prand0.0.pth \
    --num_steps 100000 --eval_interval 5000 --rollout_length 2000 --seed 0
```


## Launch the Simulation Environment

Execute the following command to launch the simulation environment:
```bash
roslaunch kortex_gazebo spawn_kortex_robot.launch arm:='gen3_lite'
```
additional arguments can be found at: [ros_kortex on GitHub](https://github.com/Kinovarobotics/ros_kortex/blob/noetic-devel/)	
		
## Recording and Replay

### Recording:

1. Edit the `bag_file_name` in `trajectory_record.py` to the desired path (e.g., `"xx/xx/xx.bag"`).
   - This will soon be updated to use `argparse` for command-line arguments like `python trajectory_record.py -filename 123123.bag`.

2. Run `trajectory_record.py` to start recording the trajectory data.

### Replay:

1. Edit the `bag_file_name` in `trajectory_replay.py` to the path where your `.bag` file is stored (e.g., `"xx/xx/xx.bag"`).
   - This will soon be updated to use `argparse` for command-line arguments like `python trajectory_replay.py -filename 123123.bag`.

2. Run `trajectory_replay.py` to replay the recorded trajectory.
