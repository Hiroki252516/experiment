# Experiment Specification

## タスク定義

2 体のローカル LLM エージェント `agent_a` と `agent_b` が、共有 5x5 gridworld の中で 2 つのアイテムから高価値な方を共同回収する ScoreG 風タスクです。

- アイテムは `left_item` と `right_item`
- 各アイテム価値は 1〜9 の整数
- `agent_a` は left value のみ観測
- `agent_b` は right value のみ観測
- 通信は 7x7 binary glyph のみ
- `comm / silent / random` の比較を維持

## 観測・行動

各エージェント観測には以下を含めます。

- 自分の位置
- 相手の位置
- 2 アイテムの位置
- 自分に見える private value ベクトル
- 相手の前ターン glyph
- 総 step 数
- `phase`
- `phase_turn_index`
- `act_step_count`
- `comm_only_turns`

各ターンで各エージェントは次を返します。

- 移動アクション 1 つ: `UP`, `DOWN`, `LEFT`, `RIGHT`, `STAY`, `PICK`
- 7x7 binary glyph 1 つ
- target 仮説: `LEFT`, `RIGHT`, `UNKNOWN`

## Phase 設計

各 episode は 2 phase で進みます。

- `comm_only`: 既定 2 turn。位置は更新せず glyph のみ交換する
- `act`: 既定 `max_steps=10`。移動と `PICK` を行う

`comm_only` は act horizon を削りません。`max_steps` は act phase だけに適用します。

## Layout 設計

- start 位置は episode ごとにランダム化
- item 位置も episode ごとにランダム化
- `LEFT` / `RIGHT` は価値ラベルであり、物理位置の左右とは独立
- agent 開始位置は別マスで、既定では Manhattan 距離 2 以上
- start 位置と item 位置は重ならない

この最小ランダム化で、通信なしでは高価値アイテムへの合流が起きにくい状況を増やします。

## 報酬

- 高価値アイテムを共同回収: `+1.0`
- 同じアイテムを共同回収したが低価値側: `+0.2`
- 別々のアイテムを回収: `-0.2`
- act phase の毎ステップ時間罰: `-0.01`
- `comm_only` phase では時間罰なし

共同回収は、両エージェントが同じアイテムのマスにいて、同じターンに `PICK` したときに成立します。

## Prompt と Decision Guard

prompt では以下を強めます。

- 目標は team reward 最大化
- glyph は hidden information を伝える唯一のチャネル
- `comm_only` phase では、まず glyph で target 仮説を揃える
- 似た状況で成功した glyph は再利用してよい
- target disagreement は失敗シグナル

runner 側では hidden information leakage を起こさない範囲で最小限の action guard をかけます。

- `step >= 1` の `UNKNOWN` は、その agent 自身の直前 target、または自分側アイテム仮説へ補正
- target 未到達時の不要な `STAY` は target への greedy move に補正
- target cell 上で両者が揃っているのに `STAY` した場合は `PICK` に補正
- `comm_only` phase では move はそのまま保持し、env 側が位置更新を無視

`raw JSON` にはモデル生出力を残し、trace 上の `move` / `target` は実際に env に渡した値を記録します。

## Private Memory

各エージェントは private notebook を run 内だけ保持します。cross-agent 共有はしません。

1 episode ごとに最低限保持する項目:

- `episode_id`
- `condition`
- `my_known_value`
- `comm_sent_glyph`
- `comm_received_glyph`
- `final_target`
- `agreement`
- `outcome`
- `team_reward`
- `glyph_helped_note`

retention は直近 12 件、prompt 投入は直近 6 件を基本とし、成功 episode を優先します。

## Log / Trace

`logs/results.csv` は episode 単位の要約を保存します。`logs/traces/<run_id>.jsonl` は step trace を保存します。

trace には既存キーに加えて以下を含めます。

- `phase`, `phase_turn_index`, `act_step`, `comm_only_turns`
- `target_a_before`, `target_b_before`
- `target_a_after`, `target_b_after`
- `target_a_changed`, `target_b_changed`
- `glyph_a_reused_from_success`, `glyph_b_reused_from_success`

## 3 条件比較の意図

- `comm`: LLM が glyph を自分で生成する通常条件
- `silent`: glyph を常にゼロにして通信なしベースラインを測る
- `random`: glyph をランダム置換して、意味のある通信とノイズを切り分ける

最低限の期待は、`comm` が `silent` と `random` より高い平均報酬や target agreement を示すことです。さらに frozen LLM のまま glyph reuse、same-context consistency、convention persistence の兆候が見えるかを観察します。

## 限界

- 本格的な multi-agent RL ではありません
- LLM の重み更新は行いません
- glyph が安定した人工言語になる保証はありません
- 今回の目的は frozen LLM 上の局所的な convention / proto-language の芽を観察しやすくすることです
