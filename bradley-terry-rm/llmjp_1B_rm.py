import os
import wandb
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import login
from datasets import load_dataset
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union, Tuple
from unsloth import FastLanguageModel, is_bfloat16_supported

from transformers import (
    PreTrainedModel,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.utils import PaddingStrategy
from transformers.trainer_pt_utils import nested_detach
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
    label_names: Optional[str] = field(default="idx")
    report_to: Optional[str] = field(default="wandb")
    # wandb設定
    project: Optional[str] = field(default="YOU FORGOT PROJECT!!!")
    name: Optional[str] = field(default="YOU FORGOT NAME!!!")


parser = HfArgumentParser(ScriptArguments)
args = parser.parse_args_into_dataclasses()[0]

out_dir = f"./{args.name}"
wandb.login(os.environ.get("WANDB_API_KEY"))
wandb.init(project=args.project, name=args.name)
login(os.environ.get("HUGGINGFACE_API_KEY"), add_to_git_credential=False)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=args.model_name,
    max_seq_length=args.max_seq_length,
    dtype=torch.bfloat16,
    load_in_4bit=False,
)
tokenizer.padding_side = "right"
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

    train_dataset = load_dataset(dataset_name)["train"].map(tokenize, num_proc=16)
    valid_dataset = load_dataset(dataset_name)["valid"].map(tokenize, num_proc=16)
    # train_valid_dataset = load_dataset(dataset_name)
    # train_valid_dataset = train_valid_dataset.map(tokenize, num_proc=16)
    # num_ds = int(args.dataset_size * len(train_valid_dataset["train"]))

    # train_dataset = train_valid_dataset["train"].select(
    #     range(0, num_ds - args.num_eval)
    # )
    # valid_dataset = train_valid_dataset["train"].select(
    #     range(num_ds - args.num_eval, num_ds)
    # )

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
    label_names=args.label_names,
    report_to=args.report_to,
)


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


def compute_metrics(eval_pred):
    result = {}
    pos_predictions_scores = eval_pred.predictions[0]
    neg_predictions_scores = eval_pred.predictions[1]
    # We assume that the first sample is preferred by default in groundtruth
    result["accuracy"] = np.sum(pos_predictions_scores > neg_predictions_scores) / len(
        pos_predictions_scores
    )
    return result


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

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        予測ステップの実装
        数値の安定性を確保
        """
        # 1. 入力の準備
        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        if torch.any(torch.isnan(inputs["input_ids"])) or torch.any(
            torch.isnan(inputs["attention_mask"])
        ):
            print("0 | NaN detected in inputs")

        # 2. 予測の実行
        with torch.no_grad():
            try:
                loss, logits_dict = self.compute_loss(
                    model, inputs, return_outputs=True
                )
            except Exception as e:
                print(f"Error during prediction: {e}")
                return None, None, None

        # 3. 損失のみの場合の処理
        if prediction_loss_only:
            return (loss, None, None)

        # 4. 損失値のデタッチと検証
        loss = loss.detach()
        if torch.isnan(loss):
            print("3 | NaN detected in prediction loss")
            loss = torch.tensor(0.0, device=loss.device)

        # 5. ロジットの処理
        logits = tuple(v for k, v in logits_dict.items() if k not in ignore_keys)
        logits = nested_detach(logits)

        # 6. ロジットのスタック
        try:
            # スタック前のチェック
            if any(torch.isnan(l).any() for l in logits):
                print("4 | NaN detected in logits before stacking")
                # NaNを0で置換
                logits = tuple(torch.nan_to_num(l, 0.0) for l in logits)

            stacked = torch.stack(logits)

            # mean計算前のチェック
            if torch.isnan(stacked).any():
                print("5 | NaN detected  in logits after stacking")
                stacked = torch.nan_to_num(stacked, 0.0)

            mean_logits = stacked.mean(dim=2)

            # softmax計算前のチェック
            if torch.isnan(mean_logits).any():
                print("6 | NaN detected in logits before softmax")
                mean_logits = torch.nan_to_num(mean_logits, 0.0)

            # 数値安定性のためのsoftmax
            logits = F.softmax(mean_logits, dim=0).mT

        except Exception as e:
            print(f"Error during logits processing: {e}")
            return loss, None, None

        return loss, logits, None


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
tokenizer.save_pretrained(out_dir)

# HuggingFace にアップロード
trainer.push_to_hub(out_dir)
print(f"Model successfully merged and uploaded to {out_dir}")
