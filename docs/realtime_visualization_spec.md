# Realtime Visualization Implementation Notes

## 目的

既存の ScoreG 実験を壊さず、step-level trace を使って live/replay/launch を提供する Streamlit viewer を追加する。

## 追加された run 出力

- `logs/traces/<run_id>.jsonl`
- `logs/runs/<run_id>/manifest.json`

trace は 1 step 1 行の append-only JSONL で、viewer はこのファイルを read-only で監視する。

## trace schema

各行には最低限以下を含める。

- `run_id`, `timestamp`, `condition`, `episode`, `step`
- `agent_a_pos`, `agent_b_pos`
- `left_item_pos`, `right_item_pos`
- `value_left`, `value_right`, `best_item`
- `glyph_a_sent`, `glyph_b_sent`
- `glyph_a_received`, `glyph_b_received`
- `move_a`, `move_b`
- `target_a`, `target_b`
- `raw_a`, `raw_b`
- `reward_a`, `reward_b`, `team_reward`, `cumulative_team_reward`
- `done`, `outcome`, `error_message`

glyph は viewer と runner の間で 7 行の `0/1` 文字列配列に統一する。

## Viewer モード

- `live`: trace を 1 秒ごとにポーリングして最新 step を表示
- `replay`: run / condition / episode / step を選び再生
- `launch`: GUI から `scripts/run_experiment.py` を subprocess 起動

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
