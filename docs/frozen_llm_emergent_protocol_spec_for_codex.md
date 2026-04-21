# Frozen-LLM Emergent Protocol Spec for Codex

## 目的

LLM 本体を再学習せず、ローカル Ollama 上の固定 LLM 同士の反復相互作用だけで、glyph ベースの局所的な convention / proto-language の芽が見えやすくなる条件を整える。

## 維持する条件

- LLM 本体の重みは更新しない
- Ollama を使う
- 2 agent / 5x5 grid / 2 item / left-right private value 分割を維持
- 通信路は 7x7 binary glyph のまま維持
- `comm / silent / random` を維持
- hidden information leakage を起こさない
- Docker と cloud API は使わない

## 今回の最小改修

### 1. 通信圧力を高める環境

- start 位置を episode ごとにランダム化
- item 位置を episode ごとにランダム化
- `LEFT` / `RIGHT` は価値ラベルであり、物理位置とは切り離す
- agent 開始位置には最低距離制約を入れる

### 2. communication-only phase

- 各 episode の先頭に `comm_only` phase を 2 turn 入れる
- この phase では env 側が move を無視し、glyph だけを交換する
- act phase の `max_steps=10` は維持し、comm-only turn は別枠にする

### 3. richer private memory

各 agent notebook には最低限以下を残す。

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

retention は直近 12 件、prompt 投入は直近 6 件を基本とし、成功 episode を優先する。

### 4. prompt の圧力

codebook は与えない。その代わり、以下だけを強く伝える。

- 目標は team reward 最大化
- glyph は hidden information を伝える唯一のチャネル
- `comm_only` phase では target 仮説をまず揃える
- 似た状況で成功した glyph は再利用してよい
- target disagreement は失敗シグナル

### 5. trace / viewer / metrics

trace には以下を追加する。

- `phase`, `phase_turn_index`, `act_step`
- `target_before`, `target_after`, `target_changed`
- `glyph_*_reused_from_success`

viewer では以下を確認できるようにする。

- current phase
- target update
- recent successful glyph
- same-context glyph consistency

分析では以下を扱う。

- `glyph_reuse_rate`
- `same_context_glyph_consistency`
- `success_failure_glyph_divergence`
- `convention_persistence`
- `target_switch_after_glyph_rate`
- `post_comm_agreement_rate`

context は最小実装として `(agent_name, my_known_value, final_target)` を使う。

## hidden information leakage 防止

- `agent_a` memory に right value を入れない
- `agent_b` memory に left value を入れない
- guard は公開観測と同一 agent の直前 target だけを使う
- viewer は read-only consumer に留める

## 非目標

- RL / LoRA / fine-tuning
- 障害物や一方通行のような大規模環境複雑化
- 人手で glyph の意味辞書を埋め込むこと

## 完了の見方

最低限、以下のいずれかが trace / replay / metrics で観察できる状態を目指す。

- `comm` が `silent` / `random` より高い
- successful episode で glyph reuse が見える
- same-context で glyph consistency が上がる
- successful episode 間で convention persistence が見える
