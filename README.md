this repo is in progress
the AIRL algo is built based on https://github.com/toshikwa/gail-airl-ppo.pytorch

### Tranfer raw expert demo to buffer

```bash
python expert_demo_to_buffer.py
```

### Train Imitation Learning

```bash
python train_imitation.py \
    --algo gail --cuda --env_id InvertedPendulum-v2 \
    --buffer buffers/InvertedPendulum-v2/size1000000_std0.01_prand0.0.pth \
    --num_steps 100000 --eval_interval 5000 --rollout_length 2000 --seed 0
```
