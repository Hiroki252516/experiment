# ScoreG-style Local LLM Coordination Experiment

このプロジェクトは、macOS Apple Silicon 上で Ollama を使い、2 体のローカル LLM エージェントが 5x5 gridworld で 7x7 binary glyph だけを使って高価値アイテムを共同回収できるかを試す実験基盤です。固定 LLM のまま、反復相互作用と private memory によって局所的な convention / proto-language の芽が見えるかを観察します。

## 前提

- macOS Apple Silicon
- Python 3.11〜3.12 互換コード
- Ollama ローカル実行
- クラウド LLM API 不使用
- Docker 不使用

## Ollama の公式インストール手順

Ollama が未導入なら、非公式な導入方法は使わず、公式手順でセットアップしてください。

1. 公式ページ [ollama.com/download](https://ollama.com/download) から macOS 版を取得してインストールする
2. Ollama アプリを起動する
3. ターミナルで `ollama --version` を確認する
4. 使うモデルを取得する。例: `ollama pull gemma3:1b`
5. `python scripts/check_ollama.py --model gemma3:1b` で状態確認する

`ollama` CLI が見つからない、または API に接続できない場合、このリポジトリの実験スクリプトは丁寧なメッセージを出して停止します。

## 仮想環境の作成

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Makefile を使う場合:

```bash
make setup
```

## モデル取得例

```bash
ollama pull gemma3:1b
```

モデル名は `OLLAMA_MODEL` 環境変数または CLI で上書きできます。

## 実験の実行方法

Ollama の準備確認:

```bash
python scripts/check_ollama.py --model gemma3:1b
```

ユニットテスト:

```bash
make test
```

短い smoke test:

```bash
make smoke MODEL=gemma3:1b
```

本番実験:

```bash
make run MODEL=gemma3:1b
```

長めの run:

```bash
make run MODEL=gemma3:1b EPISODES=100
make run MODEL=gemma3:1b EPISODES=300
```

解析:

```bash
make analyze
```

直接実行する場合:

```bash
python scripts/run_experiment.py --episodes 10 --conditions comm silent random --model gemma3:1b
python scripts/analyze_results.py --input logs/results.csv --figure logs/summary.png
```

## Realtime Viewer

Streamlit ベースの realtime viewer を追加しています。目的は、2 体の LLM がどの 7x7 glyph を送り、相手が何を受け取り、その直後に target / move / outcome がどう変わったかを live と replay の両方で観察することです。viewer は read-only で、agent prompt への情報注入経路にはなりません。

依存:

- `streamlit`

起動:

```bash
make viewer
```

または:

```bash
streamlit run viewer/app.py
```

viewer には次の 3 モードがあります。

- `live`: 実行中 run の trace を 1 秒ごとに監視
- `replay`: 保存済み trace を episode / step 単位で再生
- `launch`: GUI から experiment runner を起動

glyph-first UI では次を上から順に表示します。

- `Glyph Theater`: `agent_a sent / agent_b received / agent_b sent / agent_a received` を大きく並列表示し、step 間は `previous -> diff -> current` の疑似アニメで差分を強調
- `Glyph History Strip`: 直近 8 step の送信 glyph を film strip 風に表示し、`delta_pixels` と `same glyph xN` を確認
- `Communication Timeline`: glyph 変化、target switch、move、outcome の関係を時系列表示
- `Gridworld / Agent Detail / Convention Hints`: glyph の因果を見る補助表示

viewer では以下も確認できます。

- `comm_only` / `act` phase
- target の before / after と changed flag
- guard による補正の有無
- `Prev glyph event` / `Next glyph event` による replay ジャンプ
- `zero-signal collapse` の警告
- 最近の successful glyph と same-context glyph consistency

GUI から起動した run は `logs/traces/<run_id>.jsonl` と `logs/runs/<run_id>/manifest.json` を出力し、viewer はそれを監視します。
Launch 時の runner 標準出力と標準エラーは `logs/runs/<run_id>/launcher.log` に保存されます。

別ターミナルで手動実行する場合:

```bash
make run MODEL=gemma3:1b
make viewer
```

Replay の使い方:

1. viewer を起動する
2. `Run` を選ぶ
3. `Mode` を `replay` に切り替える
4. `Condition`, `Episode`, `Step` を選ぶ
5. `Play/Pause`, `Prev`, `Next` で確認する
6. glyph の変化点だけを追いたい場合は `Prev glyph event`, `Next glyph event` を使う
7. Glyph Theater は `previous -> diff -> current` の疑似アニメで 7x7 ピクセル差分を強調する

## 結果の見方

- `logs/results.csv`: episode 単位の要約
- `logs/episodes.jsonl`: final raw output を含む詳細ログ
- `logs/summary.png`: 条件別平均報酬、成功率、target agreement に加え、protocol metrics を含む図
- `logs/traces/<run_id>.jsonl`: realtime viewer 用 step trace
- `logs/runs/<run_id>/manifest.json`: run metadata

`results.csv` には最低限次を保存します。

- `seed`
- `condition`
- `episode_id`
- `value_left`
- `value_right`
- `best_item`
- `outcome`
- `team_reward`
- `target_a`
- `target_b`

step trace には以下も入ります。

- `phase`, `phase_turn_index`, `act_step`
- `target_a_before`, `target_a_after`, `target_a_changed`
- `target_b_before`, `target_b_after`, `target_b_changed`
- `glyph_a_reused_from_success`, `glyph_b_reused_from_success`
- `glyph_a_hash`, `glyph_b_hash`
- `glyph_a_received_hash`, `glyph_b_received_hash`
- `glyph_a_changed`, `glyph_b_changed`, `glyph_event`, `glyph_exchange_label`
- `glyph_a_zero`, `glyph_b_zero`
- `glyph_a_delta_pixels`, `glyph_b_delta_pixels`
- `glyph_a_same_streak`, `glyph_b_same_streak`

## よくある失敗

- `ollama` が見つからない  
  公式インストールが未完了です。`https://ollama.com/download` からセットアップしてください。

- `http://localhost:11434/api/tags` に接続できない  
  Ollama アプリが起動していない可能性があります。アプリを起動してから再実行してください。

- 指定モデルがない  
  `ollama pull <model>` を実行してください。

- `make test` は通るが `make smoke` が失敗する  
  unit test は Ollama なしでも通るようにしてありますが、実験 run は Ollama が必要です。

- viewer は起動するが live trace が無い  
  まだ run が始まっていないか、`logs/traces/` に対象 run の trace がありません。Launch View から起動するか、別ターミナルで runner を実行してください。

- viewer の Launch View で失敗する  
  Ollama が未導入またはモデル未取得の可能性があります。viewer 上の failure message、`logs/runs/<run_id>/manifest.json` の `last_error_message`、または `logs/runs/<run_id>/launcher.log` を確認してください。

- replay に何も出ない  
  対象 run の trace が空、または選択した condition / episode に該当フレームがありません。

- `comm` が `silent` を上回らない  
  小型モデルでは自然に有意味な glyph 協調が出ない場合があります。`comm_only` phase を含む replay、recent successful glyph、protocol metrics を見て、glyph reuse や target update の兆候を確認してください。viewer に `zero-signal collapse` が出ている場合は、`comm` でも実際には all-zero glyph に崩壊しています。

- `silent` が空白に見える  
  `silent` は意図的に全ゼログリフです。viewer では格子線と `zero glyph` バッジで表示され、異常ではありません。
