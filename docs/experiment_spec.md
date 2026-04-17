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

## 観測・行動・報酬

### 観測

各エージェント観測には以下を含めます。

- 自分の位置
- 相手の位置
- 2 アイテムの位置
- 自分に見える private value ベクトル
- 相手の前ターン glyph
- 現在ステップ数

### 行動

各ターンで各エージェントは同時に次を出します。

- 移動アクション 1 つ: `UP`, `DOWN`, `LEFT`, `RIGHT`, `STAY`, `PICK`
- 7x7 binary glyph 1 つ

### 報酬

- 高価値アイテムを共同回収: `+1.0`
- 同じアイテムを共同回収したが低価値側: `+0.2`
- 別々のアイテムを回収: `-0.2`
- 毎ステップ時間罰: `-0.01`

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

## 限界

- 本格的な multi-agent RL ではありません
- LLM の重み更新は行いません
- 実験の主目的は、固定ローカル LLM による最小協調基盤の構築です
- glyph が有意味な記号体系になる保証はありません
