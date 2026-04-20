# Realtime Visualization Implementation Notes

## 目的

既存の ScoreG 実験を壊さず、step-level trace を使って live/replay/launch を提供する Streamlit viewer を追加する。

## 追加された run 出力

- `logs/traces/<run_id>.jsonl`
- `logs/runs/<run_id>/manifest.json`
- `logs/runs/<run_id>/launcher.log`

trace は 1 step 1 行の append-only JSONL で、viewer はこのファイルを read-only で監視する。
manifest は run 状態の単一ソースであり、runner は起動直後に生成し、`starting` / `running` / `completed` / `failed` を更新する。

## trace schema

各行には最低限以下を含める。

- `run_id`, `timestamp`, `condition`, `episode`, `step`
- `phase`, `movement_enabled`
- `agent_a_pos`, `agent_b_pos`
- `left_item_pos`, `right_item_pos`
- `value_left`, `value_right`, `best_item`
- `glyph_a_sent`, `glyph_b_sent`
- `glyph_a_received`, `glyph_b_received`
- `move_a`, `move_b`
- `target_a`, `target_b`
- `initial_target_a`, `initial_target_b`
- `final_target_a`, `final_target_b`
- `target_changed_a`, `target_changed_b`
- `raw_a`, `raw_b`
- `reward_a`, `reward_b`, `team_reward`, `cumulative_team_reward`
- `done`, `outcome`, `error_message`

glyph は viewer と runner の間で 7 行の `0/1` 文字列配列に統一する。

manifest には最低限以下を含める。

- `run_id`, `model`, `conditions`, `episodes_per_condition`
- `base_seed`, `grid_size`, `max_steps`, `comm_phase_steps`
- `randomize_positions`, `hard_split_prob`, `memory_budget`, `prompt_version`
- `status`, `started_at`, `completed_at`
- `trace_path`, `results_csv_path`, `episodes_jsonl_path`
- `launcher_log_path`, `pid`
- `current_condition`, `current_episode`, `last_step`
- `last_error_message`

## Viewer モード

- `live`: trace を 1 秒ごとにポーリングして最新 step を表示
- `replay`: run / condition / episode / step を選び再生
- `launch`: GUI から `scripts/run_experiment.py` を subprocess 起動

Launch 後は pending run id を維持し、manifest 出現前でも UI 上の追跡対象を失わない。
失敗時は manifest の `last_error_message` または `launcher.log` 末尾を表示する。

改善版 viewer では次も表示する。

- communication-only phase と movement phase の区別
- initial target / final target / target changed
- 直近数 step の sent / received glyph history
- convention hints panel

## hidden information leakage 防止

- viewer は trace と manifest の read-only consumer
- agent prompt への逆流経路は持たない
- private value の表示は人間観察用のみ

## 起動

```bash
make viewer
```

または:

```bash
streamlit run viewer/app.py
```
