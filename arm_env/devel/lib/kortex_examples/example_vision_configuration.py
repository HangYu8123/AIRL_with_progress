#!/home/noahfang/miniconda3/envs/RL_Lab/bin/python
# -*- coding: utf-8 -*-
# generated from catkin/cmake/template/script.py.in
# creates a relay to a python script source file, acting as that file.
# The purpose is that of a symlink
python_script = '/home/noahfang/Documents/Lab/AIRL_with_progress/arm_env/src/ros_kortex/kortex_examples/src/vision_config/example_vision_configuration.py'
with open(python_script, 'r') as fh:
    context = {
        '__builtins__': __builtins__,
        '__doc__': None,
        '__file__': python_script,
        '__name__': __name__,
        '__package__': None,
    }
    exec(compile(fh.read(), python_script, 'exec'), context)
