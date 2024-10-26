import os
import wandb
import numpy as np
import torch, torch.nn as nn
from huggingface_hub import login
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from unsloth import FastLanguageModel, is_bfloat16_supported

# from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    # AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy
from utils import count_parameters


# 外部引数の受取
@dataclass
class ScriptArguments:
    # マルチGPUの設定
    local_rank: Optional[int] = field(default=-1)
    deepspeed: Optional[str] = field(default=None)
    # 学習対象のモデルとデータセット
    model_name: Optional[str] = field(default="llm-jp/llm-jp-3-1.8b-instruct")
    dataset_name: Optional[str] = field(default="ryota39/open-preference-en-ja")
    num_eval: Optional[int] = field(default=1000)
    # ハイパーパラメータ
    random_state: Optional[int] = field(default=42)
    per_device_train_batch_size: Optional[int] = field(default=1)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=256)
    num_train_epochs: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=3e-5)
    warmup_steps: Optional[int] = field(default=50)
    weight_decay: Optional[float] = field(default=0.0)
    # optim: Optional[str] = field(default="paged_adamw_32bit")
    # 学習戦略
    num_labels: Optional[int] = field(default=2)
    dataset_size: Optional[float] = field(default=0.25)
    max_seq_length: Optional[int] = field(default=4096)
    lr_scheduler_type: Optional[str] = field(default="cosine")
    gradient_checkpointing: Optional[bool] = field(default=True)
    eval_strategy: Optional[str] = field(default="steps")
    eval_steps: Optional[int] = field(default=50)
    logging_strategy: Optional[str] = field(default="steps")
    logging_steps: Optional[int] = field(default=10)
    save_strategy: Optional[str] = field(default="steps")
    save_total_limit: Optional[int] = field(default=3)
    load_best_model_at_end: Optional[bool] = field(default=True)
    log_dir: Optional[str] = field(default="./log")
    remove_unused_columns: Optional[bool] = field(default=False)
    report_to: Optional[str] = field(default="wandb")
    # wandb設定
    project: Optional[str] = field(default="YOU FORGOT PROJECT!!!")
    name: Optional[str] = field(default="YOU FORGOT NAME!!!")


parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

out_dir = f"./{args.name}"
wandb.init(project=args.project, name=args.name)
wandb.login(os.environ.get("WANDB_API_KEY"))
login(os.environ.get("HUGGINGFACE_API_KEY"), add_to_git_credential=False)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model_name,
    max_seq_length=args.max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)
hidden_size = model.lm_head.in_features
model.lm_head = nn.Linear(hidden_size, args.num_labels)
model.config.use_cache = not args.gradient_checkpointing
print(model)
count_parameters(model)


# 右URL先のフォーマットのみに対応(https://huggingface.co/datasets/hendrydong/preference_700K)
def build_dataset(tokenizer, dataset_name):

    def tokenize(sample):

        sample["positive"] = tokenizer.apply_chat_template(
            sample["chosen"], tokenize=False, add_generation_prompt=False
        ).replace(tokenizer.bos_token, "")
        sample["negative"] = tokenizer.apply_chat_template(
            sample["rejected"], tokenize=False, add_generation_prompt=False
        ).replace(tokenizer.bos_token, "")

        tokenized_pos = tokenizer(sample["positive"], truncation=True)
        tokenized_neg = tokenizer(sample["negative"], truncation=True)
        sample["input_ids_j"] = tokenized_pos["input_ids"]
        sample["attention_mask_j"] = tokenized_pos["attention_mask"]
        sample["input_ids_k"] = tokenized_neg["input_ids"]
        sample["attention_mask_k"] = tokenized_neg["attention_mask"]
        return sample

    train_valid_dataset = load_dataset(dataset_name)
    train_valid_dataset = train_valid_dataset.map(tokenize, num_proc=8)
    num_ds = int(args.dataset_size * len(train_valid_dataset["train"]))

    train_dataset = train_valid_dataset["train"].select(
        range(0, num_ds - args.num_eval)
    )
    valid_dataset = train_valid_dataset["train"].select(
        range(num_ds - args.num_eval, num_ds)
    )

    return train_dataset, valid_dataset


train_dataset, valid_dataset = build_dataset(tokenizer, args.dataset_name)
print(f"Training set: {len(train_dataset)} | Validation set: {len(valid_dataset)}")

# Trainerの定義(学習時の引数の指定)
training_args = TrainingArguments(
    local_rank=args.local_rank,
    deepspeed=args.deepspeed,
    seed=args.random_state,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_eval_batch_size,
    num_train_epochs=args.num_train_epochs,
    learning_rate=args.learning_rate,
    warmup_steps=args.warmup_steps,
    weight_decay=args.weight_decay,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    output_dir=out_dir,
    logging_dir=args.log_dir,
    fp16=not is_bfloat16_supported,
    bf16=is_bfloat16_supported,
    eval_strategy=args.eval_strategy,
    eval_steps=args.eval_steps,
    logging_strategy=args.logging_strategy,
    logging_steps=args.logging_steps,
    save_strategy=args.save_strategy,
    save_total_limit=args.save_total_limit,
    lr_scheduler_type=args.lr_scheduler_type,
    load_best_model_at_end=args.load_best_model_at_end,
    gradient_checkpointing=args.gradient_checkpointing,
    remove_unused_columns=args.remove_unused_columns,
    report_to=args.report_to,
)

# enable if you want to train with lora
# peft_config = LoraConfig(
#     task_type=TaskType.SEQ_CLS,
#     inference_mode=False,
#     r=8,
#     lora_alpha=32,
#     lora_dropout=0.1,
# )


# chosen vs. rejected形式でデータを取り出す(照合する=collate)クラスの定義
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: AutoTokenizer
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        merged_features = []
        for feature in features:
            merged_features.append(
                {
                    "input_ids": feature["input_ids_j"],
                    "attention_mask": feature["attention_mask_j"],
                }
            )
            merged_features.append(
                {
                    "input_ids": feature["input_ids_k"],
                    "attention_mask": feature["attention_mask_k"],
                }
            )
        batch = self.tokenizer.pad(
            merged_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "return_loss": True,
        }
        return batch


# 検証データの評価指標の定義
def compute_metrics(eval_pred):
    result = {}
    pos_predictions_scores = eval_pred.predictions[0]
    neg_predictions_scores = eval_pred.predictions[1]
    # We assume that the first sample is preferred by default in groundtruth
    result["accuracy"] = np.sum(pos_predictions_scores > neg_predictions_scores) / len(
        pos_predictions_scores
    )
    return result


# 報酬モデルの損失の定義
class RewardTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards = model(
            input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )[0]
        bsz = rewards.size(0)
        jidx = torch.arange(0, bsz, 2)
        kidx = jidx + 1
        rewards_j = rewards[jidx]
        rewards_k = rewards[kidx]
        loss = -nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


# 報酬モデルの学習器の定義
trainer = RewardTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    compute_metrics=compute_metrics,
    data_collator=RewardDataCollatorWithPadding(
        tokenizer=tokenizer, max_length=args.max_seq_length
    ),
)

print("*" * 50)
print(trainer.train_dataset)
print(tokenizer.decode(trainer.train_dataset[60]["input_ids_j"]))
print("*" * 50)

# 報酬モデルの学習
trainer.train()
trainer.save_model(out_dir)

# HuggingFace にアップロード
trainer.push_to_hub(out_dir)
print(f"Model successfully merged and uploaded to {out_dir}")
