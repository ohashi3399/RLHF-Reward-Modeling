# RLHF-Reward-Modeling

- This is a fork of [RLHF-Reward-Modeling](https://github.com/RLHFlow/RLHF-Reward-Modeling)
  - support models which can handle Japanese
  - support [Unsloth](https://github.com/unslothai/unsloth), which reduce VRAM when training and accelerte training efficiency
  - support wandb

# Environment setup

```sh
git clone https://github.com/ohashi3399/RLHF-Reward-Modeling.git && cd RLHF-Reward-Modeling
```

## Bradley-Terry-RM

```sh
source setup.sh && cd bradley-terry-rm
```

```sh
source tune.sh
```
