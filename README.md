# RLHF-Reward-Modeling

- This is a fork of [RLHF-Reward-Modeling](https://github.com/RLHFlow/RLHF-Reward-Modeling)
  - support models which can handle Japanese
  - support [Unsloth](https://github.com/unslothai/unsloth), which reduce VRAM when training and accelerte training efficiency
  - support wandb

# Support

|model|support|
|:---|:---:|
|[google/gemma-2b-it](https://huggingface.co/google/gemma-2b-it)|✅|
|[llm-jp/llm-jp-3-1.8b-instruct](https://huggingface.co/llm-jp/llm-jp-3-1.8b-instruct)|✅|

|dataset|support|
|:---|:---:|
|[hendrydong/preference_700K](https://huggingface.co/datasets/hendrydong/preference_700K)|✅|
|xxxx|-|

# Environment setup

```sh
git clone https://github.com/ohashi3399/RLHF-Reward-Modeling.git && cd RLHF-Reward-Modeling
```

## Bradley-Terry-RM

```sh
export HUGGINGFACE_API_KEY=<Your HUGGINGFACE_API token>
export WANDB_API_KEY=<Your WANDB_API token>
source setup.sh && cd bradley-terry-rm
source tune.sh
```
