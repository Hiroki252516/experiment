# Experiment Specification

## タスク定義

2 体のローカル LLM エージェント `agent_a` と `agent_b` が、共有された 5x5 gridworld の中で 2 つのアイテムから高価値な方を共同回収する ScoreG 風タスクです。

- アイテムは `left_item` と `right_item`
- 各アイテム価値は 1〜9 の整数
- `agent_a` は left value のみ観測
- `agent_b` は right value のみ観測
- 公開情報として位置情報と前ターン glyph を共有
- 通信は 7x7 binary glyph のみ

共同回収は、両エージェントが同じアイテムのマスにいて、同じターンに `PICK` したときに成立します。

この改善版では、固定 LLM 間で「弱いが再利用される glyph 慣習の芽」を観察しやすくするため、communication-only phase、ランダム配置、hard split episode を導入します。

## 観測・行動・報酬

### 観測

各エージェント観測には以下を含めます。

- 自分の位置
- 相手の位置
- 2 アイテムの位置
- 自分に見える private value ベクトル
- 相手の前ターン glyph
- 現在ステップ数
- movement が有効かどうか

### 行動

各ターンで各エージェントは同時に次を出します。

- 移動アクション 1 つ: `UP`, `DOWN`, `LEFT`, `RIGHT`, `STAY`, `PICK`
- 7x7 binary glyph 1 つ

### 報酬

- 高価値アイテムを共同回収: `+1.0`
- 同じアイテムを共同回収したが低価値側: `+0.2`
- 別々のアイテムを回収: `-0.2`
- 毎ステップ時間罰: `-0.01`

## communication-only phase とランダム配置

- episode 冒頭の `comm_phase_steps` では移動せず glyph だけ交換する
- 通常 phase に入ると移動と `PICK` が有効になる
- agent と item の開始位置は episode ごとにランダム化できる
- hard split sampling により左右価値差の大きい episode を増やせる

## 3 条件比較の意図

- `comm`: LLM が glyph を自分で生成する通常条件
- `silent`: glyph を常にゼロにして、通信なしベースラインを測る
- `random`: glyph をランダム置換して、意味のある通信とノイズを切り分ける

最低限の期待は、`comm` が `silent` と `random` より高い平均報酬や target agreement を示すことです。

## セッションメモの扱い

各エージェントは private memory を run 内だけ保持します。

- `agent_a` の memory は `agent_a` 専用
- `agent_b` の memory は `agent_b` 専用
- run 終了時に破棄
- hidden value を cross-agent に共有しない

memory は重み更新ではなく、固定 LLM 上のセッション内慣習形成を観察するための補助です。

memory summary は少なくとも次を含みます。

- `episode`
- `condition`
- `my_known_value`
- `my_sent_glyph`
- `my_received_glyph`
- `my_target`
- `agreement`
- `team_reward`
- `outcome`

## Prompt と action guard

各エージェントは毎ターン `LEFT` または `RIGHT` の仮説 target を持って行動することを基本方針とします。

- prompt では team reward 最大化を明示する
- glyph は唯一の hidden-info 通信チャネルだと明示する
- 成功した glyph の再利用を促す
- `UNKNOWN` は step 0 か本当に判断不能な場合だけに寄せる
- `STAY` は target 上での待機など合理的理由がある場合だけに寄せる

また、実験 runner では hidden information leakage を起こさない範囲で、最小限の action guard をかけます。

- `step >= 1` の `UNKNOWN` は、その agent 自身の直前 target、または自分側アイテム仮説へ補正する
- target 未到達時の不要な `STAY` は target への greedy move に補正する
- target cell 上で両者が揃っているのに `STAY` した場合は `PICK` に補正する

この guard は相手の hidden value や環境の真の最適解を使わず、各 agent の公開観測と自分の直前 target だけを使います。`raw JSON` にはモデル生出力を残し、trace 上の `move` / `target` は実際に env に渡した値を記録します。

## 限界

- 本格的な multi-agent RL ではありません
- LLM の重み更新は行いません
- hand-coded codebook は与えません
- 実験の主目的は、固定ローカル LLM による最小協調基盤と glyph 慣習の芽の観察です
- glyph が強い compositional 言語になる保証はありません
