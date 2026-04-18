# Realtime Visualization Spec for Codex

この文書は、既存の ScoreG 風ローカル LLM 実験環境に対して、**本格版のリアルタイム可視化 GUI** を追加するための実装仕様書です。

この文書は、既存の以下の文書を補完します。
- `AGENTS.md`
- `docs/CODEX_SETUP_PLAYBOOK.md`
- `docs/codex_task.md`
- `docs/experiment_context_for_codex.md`

この文書で追加するテーマは、**実験を回しながら、2体のローカル LLM がどのように動き、どのような 7x7 グリフを送り合い、どのように行動合意しているかを GUI でリアルタイム表示すること**です。

---

## 1. 目的

現在の実験基盤は、CLI 実行後に `logs/results.csv` や `logs/episodes.jsonl` を保存し、あとから解析するバッチ型です。

本仕様の目的は、この基盤に対して **リアルタイムで観察可能な GUI viewer** を追加し、以下を同時に可視化できるようにすることです。

1. 5x5 gridworld 上での agent / item の位置変化
2. 各ターンで agent_a / agent_b が送信した 7x7 binary glyph
3. 各ターンの LLM 出力 JSON（move, target, glyph）
4. 各ターンの reward / outcome / current condition
5. episode 全体の進行状況
6. 過去ログの replay

この GUI は「見て楽しい」だけでなく、**通信の内容と行動結果の対応を人間が追跡できる観察装置**である。

---

## 2. 採用方針

### 2.1 GUI 技術スタック

今回の GUI は **Streamlit ベースのローカル Web アプリ**として実装すること。

理由:
- Python コードだけでローカル GUI を作りやすい
- `streamlit run ...` でローカル起動できる
- `st.image` で NumPy 配列から 7x7 グリフをそのまま表示できる
- `st.session_state` で viewer 状態を保持できる
- `st.fragment(run_every=...)` を使えば、実験ログをポーリングしてリアルタイム更新しやすい

### 2.2 アーキテクチャ方針

GUI は **実験本体と分離**すること。

- 実験本体: `scripts/run_experiment.py`
- 可視化: `viewer/` 以下の Streamlit app
- 両者の接続: ログファイル（JSONL / manifest / state file）

つまり、**GUI が agent prompt や環境遷移に直接介入しない設計**にする。
GUI はあくまで「観察・再生・起動補助」を担当する。

---

## 3. 絶対に守ること

1. **hidden information leakage を絶対に起こさないこと**
   - GUI は人間観察用として真の値を表示してよい
   - ただし GUI 表示情報を agent prompt 側へ絶対に流してはならない
   - GUI 用コードは agent 入力経路と分離すること

2. **既存の実験意味論を変えないこと**
   - 2 エージェント
   - 5x5 gridworld
   - 2 アイテム
   - left/right private value 分割
   - 7x7 binary glyph 通信
   - `comm / silent / random`
   は固定仕様

3. **GUI が無くても既存 CLI 実験はそのまま動くこと**
   - GUI は追加機能であり、既存 runner を壊してはならない

4. **まずローカル単体で動く最小本格版を完成させること**
   - マルチユーザー対応やリモート配信は不要

---

## 4. 追加したい機能

### 4.1 Live View

実験実行中に、現在の episode / step をリアルタイム表示するモード。

表示要素:
- 実行中 condition (`comm`, `silent`, `random`)
- episode 番号
- step 番号
- 累積 reward
- current outcome（あれば）
- gridworld 表示
- agent_a / agent_b の現在位置
- left_item / right_item の位置
- 各アイテムの真の値（人間向け表示として可）
- agent_a の送信 glyph
- agent_b の送信 glyph
- 各 agent の received glyph
- 各 agent の `move`
- 各 agent の `target`
- 各 agent の raw JSON 出力
- 直近イベントログ

### 4.2 Replay View

実験後に保存された step-level ログを読み込み、episode 単位で再生できるモード。

必要機能:
- run 選択
- condition 選択
- episode 選択
- step slider
- play / pause
- next / prev
- 再生速度変更
- final outcome 表示

### 4.3 Launch View

GUI から実験を開始できるモード。

最小要件:
- モデル名入力
- 実行エピソード数入力
- 条件選択
- 「Run experiment」ボタン
- バックグラウンド subprocess で runner を起動
- ログ監視開始

Launch View が難しければ、初版では「別ターミナルで `make run` を実行し、GUI はそれを監視する」構成でもよい。ただし、その場合でも docs に明記すること。

---

## 5. UI レイアウト仕様

### 5.1 ページ構成

Streamlit app は最低限 1 ページでよいが、内部的には次のセクションを持つこと。

1. Header
2. Control Panel
3. Live / Replay Status
4. Gridworld Panel
5. Agent A Panel
6. Agent B Panel
7. Timeline / Event Log Panel
8. Metrics Summary Panel

### 5.2 Header

表示内容:
- アプリ名
- 現在の mode（live / replay）
- 監視対象 run id
- 現在の model 名

### 5.3 Control Panel

表示内容:
- mode 切り替え（live / replay）
- run 選択
- condition 選択
- episode 選択
- step slider
- 再生速度
- play / pause / reset
- 必要なら「実験開始」ボタン

### 5.4 Gridworld Panel

5x5 の world を視覚的に表示すること。

推奨表示:
- 空セル: 薄い背景
- `agent_a`: 青系
- `agent_b`: 赤系または橙系
- `left_item`: 緑系
- `right_item`: 紫系
- 高価値側 item: 枠線やハイライトで強調

セルには必要に応じて以下を重ねてよい。
- A / B
- L / R
- 値

### 5.5 Agent Panel

各 agent ごとに独立した panel を持つこと。

表示項目:
- agent 名
- self position
- current move
- current target
- private value
- sent glyph
- received glyph
- raw JSON output
- エラーがあればエラー表示

### 5.6 Glyph 表示

7x7 グリフは **拡大したピクセルアート**として表示すること。

実装要件:
- 7x7 の 0/1 データを NumPy 配列へ変換
- nearest-neighbor 的に拡大し、視認性を確保
- 可能なら「送信 glyph」「受信 glyph」を並べて比較表示
- 同じ turn の両 agent glyph を横並びで見られるようにする

### 5.7 Timeline / Event Log Panel

各ターンのイベントを時系列に表示すること。

例:
- `step 3: A sent glyph X, chose RIGHT, target LEFT`
- `step 3: B sent glyph Y, chose PICK, target RIGHT`
- `reward=-0.01`

### 5.8 Metrics Summary Panel

run 単位または condition 単位の簡易集計を表示すること。

最低限:
- success rate
- average reward
- target agreement rate
- outcome の内訳

---

## 6. 追加するデータ出力仕様

現在の `results.csv` と `episodes.jsonl` だけでは、**リアルタイム GUI に必要な step-by-step 表示情報が足りない可能性が高い**。

そのため、runner に **step-level trace 出力**を追加すること。

### 6.1 必須ファイル

以下のどちらか、または両方を実装すること。

#### Option A: 単一 live trace
- `logs/live_trace.jsonl`

#### Option B: run ごとの trace
- `logs/traces/<run_id>.jsonl`
- `logs/runs/<run_id>/manifest.json`

推奨は Option B。

### 6.2 Step trace 1 行に含める情報

各 step の JSON object に最低限以下を含めること。

- `run_id`
- `timestamp`
- `condition`
- `episode`
- `step`
- `agent_a_pos`
- `agent_b_pos`
- `left_item_pos`
- `right_item_pos`
- `value_left`
- `value_right`
- `best_item`
- `glyph_a_sent`
- `glyph_b_sent`
- `glyph_a_received`
- `glyph_b_received`
- `move_a`
- `move_b`
- `target_a`
- `target_b`
- `raw_a`
- `raw_b`
- `reward_a`
- `reward_b`
- `team_reward`
- `done`
- `outcome`
- `error_message`

### 6.3 Write policy

- step ごとに append する
- flush を意識して、viewer が tail しやすい形にする
- 中断時も壊れにくい JSONL を使う

---

## 7. 追加するコード構成

Codex は少なくとも次のファイルを追加・更新すること。

```text
viewer/
├── __init__.py
├── app.py
├── data.py
├── render.py
├── controls.py
└── utils.py

scripts/
├── run_experiment.py        # step trace 出力を追加
└── launch_viewer.py         # 必要なら補助スクリプト

tests/
├── test_trace_schema.py
├── test_viewer_data_loading.py
└── test_glyph_rendering.py

docs/
└── realtime_visualization_spec.md
```

必要なら以下を追加してよい。
- `viewer/assets/`
- `viewer/styles.py`

---

## 8. 追加依存関係

最低限、以下を追加してよい。

- `streamlit`

可能なら既存依存だけで進めること。追加依存は最小限にすること。

`Pillow` や `plotly` は、必要でなければ追加しない。グリフ描画は NumPy + Streamlit の `st.image` で十分である。

---

## 9. Makefile に追加してよいターゲット

最低限、以下のいずれかを追加すること。

- `make viewer`      : Streamlit viewer を起動
- `make live-view`   : 必要なら run + viewer を補助
- `make replay-view` : 既存ログの再生 viewer を起動

推奨例:
- `make viewer` → `streamlit run viewer/app.py`

---

## 10. 実装手順

Codex は以下の順番で進めること。

1. 既存 runner とログ形式を確認
2. step-level trace schema を設計
3. `scripts/run_experiment.py` に trace 出力を追加
4. trace schema 用テストを追加
5. `viewer/data.py` で trace 読み込み層を実装
6. `viewer/render.py` で grid / glyph 描画を実装
7. `viewer/app.py` に Streamlit UI を実装
8. live polling を追加
9. replay controls を追加
10. README と docs を更新
11. smoke test 的に viewer 起動と trace 読み込みを確認

---

## 11. Streamlit 実装要件

### 11.1 起動方法

以下のコマンドで起動できること。

```bash
streamlit run viewer/app.py
```

### 11.2 状態管理

- `st.session_state` を用いて mode, selected run, selected episode, selected step, playing state を保持すること

### 11.3 リアルタイム更新

- `st.fragment(run_every="1s")` または同等の公式 API を使い、ログ監視部分のみを定期更新すること
- app 全体を無駄に再描画しすぎないこと

### 11.4 glyph 描画

- 7x7 の二値配列から画像を生成
- 小さすぎて見えないので 10x〜30x 程度に拡大
- 送受信 glyph を見比べやすくする

### 11.5 grid 描画

- 5x5 world を視認性の高い形で描く
- 少なくとも positions は一目で分かること
- current step で何が起きているかが分かること

---

## 12. テスト要件

最低限、以下を追加すること。

### `tests/test_trace_schema.py`
- step trace が必要キーを持つ
- JSONL 1 行ごとに parse 可能

### `tests/test_viewer_data_loading.py`
- trace JSONL を読み込める
- partial log でも壊れにくい

### `tests/test_glyph_rendering.py`
- 7x7 glyph から描画用配列が生成できる
- サイズや shape が想定どおり

GUI の見た目そのものの snapshot test は必須ではない。

---

## 13. README 更新要件

README には最低限以下を追記すること。

1. Realtime Viewer の目的
2. 依存関係（Streamlit）
3. 起動方法
4. 実験を別ターミナルで走らせる手順
5. replay の使い方
6. よくある失敗
   - viewer は起動するが live trace が無い
   - Ollama が未起動
   - 実験がまだ logs を出していない

---

## 14. 非目標

今回は以下はやらなくてよい。

- WebSocket サーバ
- リモート共有ダッシュボード
- 認証
- DB 保存
- マルチユーザー編集
- GUI から agent prompt を書き換える機能
- GUI を経由した agent 入力注入

---

## 15. 完了条件

以下を満たしたら完了とする。

1. `streamlit run viewer/app.py` で viewer が起動する
2. 既存ログを replay 表示できる
3. 実験実行中ログを live 監視できる
4. 5x5 gridworld が表示される
5. 各 agent の送受信 glyph を表示できる
6. raw LLM 出力を表示できる
7. `comm / silent / random` の識別ができる
8. README に利用手順が追加されている
9. 既存 CLI 実験を壊していない
10. hidden information leakage を viewer が発生させない

---

## 16. 一文要約

この追加仕様の目的は、**Ollama 上の 2 体のローカル LLM が、部分観測の 5x5 world で 7x7 二値グリフを送り合いながら高価値アイテム共同回収を試みる様子を、リアルタイムおよび replay の両方で観察できる Streamlit ベースの本格 GUI viewer として実装すること**である。
