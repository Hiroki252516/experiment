# ScoreG-style Local LLM Coordination Experiment

このプロジェクトは、macOS Apple Silicon 上で Ollama を使い、2 体のローカル LLM エージェントが 5x5 gridworld で 7x7 binary glyph だけを使って高価値アイテムを共同回収できるかを試す最小実験基盤です。

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

Streamlit ベースの realtime viewer を追加しています。目的は、2 体の LLM がどの glyph を送り、どの行動を選び、gridworld がどう進んだかを live と replay の両方で観察することです。

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

GUI から起動した run は `logs/traces/<run_id>.jsonl` と `logs/runs/<run_id>/manifest.json` を出力し、viewer はそれを監視します。

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

## 結果の見方

- `logs/results.csv`: episode 単位の要約
- `logs/episodes.jsonl`: final raw output を含む詳細ログ
- `logs/summary.png`: 条件別平均報酬、成功率、target agreement rate の図
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
  Ollama が未導入またはモデル未取得の可能性があります。runner 側の manifest に失敗理由が出ます。

- replay に何も出ない  
  対象 run の trace が空、または選択した condition / episode に該当フレームがありません。

- `comm` が `silent` を上回らない  
  小型モデルでは自然に有意味な glyph 協調が出ない場合があります。まずはログを見て、モデルサイズや prompt の見直しを検討してください。
