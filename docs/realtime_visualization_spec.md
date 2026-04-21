# Realtime Visualization Implementation Notes

## 目的

既存の CLI 実験を壊さず、step-level trace を使って live / replay / launch を提供する Streamlit viewer を維持する。viewer は trace と manifest の read-only consumer とし、agent prompt への逆流経路を持たせない。

## Run 出力

runner は以下を生成する。

- `logs/results.csv`
- `logs/episodes.jsonl`
- `logs/traces/<run_id>.jsonl`
- `logs/runs/<run_id>/manifest.json`
- `logs/runs/<run_id>/launcher.log`

manifest は run 状態の単一ソースであり、runner は起動直後に生成し、`starting` / `running` / `completed` / `failed` を更新する。

## Trace Schema

各 trace 行には最低限以下を含める。

- `run_id`, `timestamp`, `condition`, `episode`, `step`
- `phase`, `phase_turn_index`, `act_step`, `comm_only_turns`
- `agent_a_pos`, `agent_b_pos`
- `left_item_pos`, `right_item_pos`
- `value_left`, `value_right`, `best_item`
- `glyph_a_sent`, `glyph_b_sent`
- `glyph_a_received`, `glyph_b_received`
- `move_a`, `move_b`
- `target_a`, `target_b`
- `target_a_before`, `target_b_before`
- `target_a_after`, `target_b_after`
- `target_a_changed`, `target_b_changed`
- `glyph_a_reused_from_success`, `glyph_b_reused_from_success`
- `guard_a_applied`, `guard_b_applied`
- `guard_a_reason`, `guard_b_reason`
- `raw_a`, `raw_b`
- `reward_a`, `reward_b`, `team_reward`, `cumulative_team_reward`
- `done`, `outcome`, `error_message`

glyph は viewer と runner の間で 7 行の `0/1` 文字列配列に統一する。

## Viewer モード

- `live`: 実行中 run の最新 frame を 1 秒ごとに追跡する
- `replay`: run / condition / episode / step を選び再生する
- `launch`: GUI から `scripts/run_experiment.py` を subprocess 起動する

Launch 後は pending run id を維持し、manifest 出現前でも新規 run への追跡を失わない。失敗時は manifest の `last_error_message` または `launcher.log` 末尾を表示する。

## 表示項目

- Gridworld
- current phase
- episode / step / cumulative reward / outcome
- agent ごとの position / move / target
- `target_before` / `target_after` / `target_changed`
- sent glyph / received glyph
- guard 適用状況
- raw JSON
- Timeline
- Metrics Summary
- Convention Hints

## Convention Hints

viewer は trace から以下を表示できる。

- 最近の successful glyph
- same-context で頻出した glyph
- `glyph_reuse_rate`
- `same_context_glyph_consistency`
- `convention_persistence`
- `post_comm_agreement_rate`

context は最小実装として `(agent_name, my_known_value, final_target)` に固定する。

## hidden information leakage 防止

- viewer は trace と manifest だけを読む
- agent notebook は viewer に流さない
- private value の表示は人間観察用のみ
- viewer 側の convention hints は分析結果であり、agent prompt に接続しない
