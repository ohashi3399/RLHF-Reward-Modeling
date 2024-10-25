def count_parameters(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params_comma = f"{num_params:,}"

    # カンマで文字列を分割
    parts = num_params_comma.split(",")

    # 最初の部分（左から1番目のカンマまで）を取得
    left_over_comma = parts[0]
    right_over_comma = parts[1][0]

    # カンマの数に基づいて単位を決定
    if len(parts) == 2:
        unit = "k"
    elif len(parts) == 3:
        unit = "M"
    elif len(parts) == 4:
        unit = "B"
    elif len(parts) >= 5:
        unit = "T"
    else:
        return left_over_comma  # カンマがない場合はそのまま返す

    # 結果を文字列として返す
    return print(
        f"Base model params: {left_over_comma}.{right_over_comma}{unit}({num_params:,})"
    )
