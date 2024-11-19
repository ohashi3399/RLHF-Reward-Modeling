import os, json, torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from huggingface_hub import login
import evaluate


def prediction(
    model: AutoModelForSequenceClassification, data_point: dict, column: str
):

    input_ids = tokenizer.apply_chat_template(
        data_point[column], tokenize=False, add_generation_prompt=False
    ).replace(tokenizer.bos_token, "")

    input_ids = tokenizer(
        input_ids, truncation=True, max_length=max_length, return_tensors="pt"
    )

    input_ids = {k: v.to(device) for k, v in input_ids.items()}
    print(input_ids)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids["input_ids"],
            attention_mask=input_ids["attention_mask"],
        )
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)

    label = 0 if column == "chosen" else 1
    pred = torch.argmax(probabilities, dim=1).item()
    prob = probabilities.cpu().numpy()[0]
    # print({"label": label, "pred": pred, "prob": prob})

    return {"label": label, "pred": pred, "prob": prob}


def plot_double_histograms(list1, list2, titles=["Chosen score", "Rejected score"]):
    """
    二つのリストのヒストグラムを横に並べて描画する関数

    Args:
        list1 (list): 1つ目のデータリスト
        list2 (list): 2つ目のデータリスト
        titles (list): 各ヒストグラムのタイトル
    """
    # フィギュアとサブプロットの設定
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 中央値の計算
    median1 = np.median(list1)
    median2 = np.median(list2)

    # 1つ目のヒストグラム
    sns.histplot(data=list1, bins=30, ax=ax1, color="skyblue")
    ax1.axvline(x=median1, color="black", linestyle="--", linewidth=2)
    ax1.text(
        median1,
        ax1.get_ylim()[1] * 0.95,
        f"Median: {median1:.3f}",
        rotation=90,
        verticalalignment="top",
        horizontalalignment="right",
    )

    # 2つ目のヒストグラム
    sns.histplot(data=list2, bins=30, ax=ax2, color="lightgreen")
    ax2.axvline(x=median2, color="red", linestyle="--", linewidth=2)
    ax2.text(
        median2,
        ax2.get_ylim()[1] * 0.95,
        f"Median: {median2:.3f}",
        rotation=90,
        verticalalignment="top",
        horizontalalignment="right",
    )

    # グラフの設定
    for ax, title in zip([ax1, ax2], titles):
        ax.set_xlim(0.0, 1.0)
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")

    # レイアウトの調整
    plt.tight_layout()

    return fig


def eval_prediction(preds: list, refs: list):
    metrics = dict()
    metrics.update(metric_accuracy.compute(predictions=preds, references=refs))
    metrics.update(metric_recall.compute(predictions=preds, references=refs))
    metrics.update(metric_precision.compute(predictions=preds, references=refs))
    metrics.update(metric_f1.compute(predictions=preds, references=refs))
    return metrics


def to_markdown(metrics_all, metrics_chosen, metrics_rejected):
    markdown = list()
    markdown.append("|metric|accuracy|recall|precision|f1|")
    markdown.append("|:---|:---|:---|:---|:---|")
    markdown.append(
        f"|all|{metrics_all['accuracy']}|{metrics_all['recall']}|{metrics_all['precision']}|{metrics_all['f1']}|"
    )
    markdown.append(
        f"|chosen-only|{metrics_chosen['accuracy']}|{metrics_chosen['recall']}|{metrics_chosen['precision']}|{metrics_chosen['f1']}|"
    )
    markdown.append(
        f"|rejected-only|{metrics_rejected['accuracy']}|{metrics_rejected['recall']}|{metrics_rejected['precision']}|{metrics_rejected['f1']}|"
    )
    markdown = "\n".join(markdown)

    with open("summary.md", mode="w", encoding="utf-8") as f:
        f.write(markdown)
    return


login(os.environ.get("HUGGINGFACE_API_KEY"), add_to_git_credential=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "ryota39/RM-waka-2B-100k"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model = model.to(device)
model.eval()
max_length = 4096

ds = load_dataset("ryota39/preference-en-ja-100k")["test"]

num_correct = 0
num_correct_chosen = 0
num_correct_rejected = 0
chosen_probs = list()
rejected_probs = list()
chosen_preds = list()
chosen_refs = list()
rejected_preds = list()
rejected_refs = list()
all_preds = list()
all_refs = list()
tbar = tqdm(enumerate(ds), total=len(ds))
for idx, data_point in tbar:

    chosen = prediction(model=model, data_point=data_point, column="chosen")
    rejected = prediction(model=model, data_point=data_point, column="rejected")

    if chosen["label"] == chosen["pred"]:
        num_correct += 1
        num_correct_chosen += 1

    if rejected["label"] == rejected["pred"]:
        num_correct += 1
        num_correct_rejected += 1

    chosen_preds.append(chosen["pred"])
    chosen_refs.append(chosen["label"])
    chosen_probs.append(chosen["prob"][0])

    rejected_preds.append(rejected["pred"])
    rejected_refs.append(rejected["label"])
    rejected_probs.append(rejected["prob"][1])

all_preds = chosen_preds + rejected_preds
all_refs = chosen_refs + rejected_refs

fig = plot_double_histograms(
    chosen_probs, rejected_probs, titles=["Chosen score", "Rejected score"]
)
plt.savefig("chosen-rejected distribution.png")

metric_accuracy = evaluate.load("accuracy")
metric_recall = evaluate.load("recall")
metric_precision = evaluate.load("precision")
metric_f1 = evaluate.load("f1")

metrics_all = eval_prediction(all_preds, all_refs)
metrics_chosen = eval_prediction(chosen_preds, chosen_refs)
metrics_rejected = eval_prediction(rejected_preds, rejected_refs)
print(metrics_all)
print(metrics_chosen)
print(metrics_rejected)
to_markdown(metrics_all, metrics_chosen, metrics_rejected)
metrics = [metrics_all, metrics_chosen, metrics_rejected]

with open("summary.jsonl", "w") as f:
    f.writelines([json.dumps(l, ensure_ascii=False) for l in metrics])
