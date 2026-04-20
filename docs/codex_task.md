# Codex task brief

## ゴール

このリポジトリに、macOS Apple Silicon + Ollama 前提の ScoreG 風 2 エージェント協調実験基盤を実装し、固定 LLM 間で再利用される glyph 慣習の芽を観察しやすい改善版にする。

- PettingZoo ParallelEnv の 5x5 gridworld
- Ollama ローカル API を使う 2 エージェント
- 7x7 binary glyph 通信
- `comm / silent / random` の 3 条件実験
- communication-only phase
- ランダム配置と hard split episode
- glyph reuse / glyph-target association 指標
- `logs/results.csv` と解析出力

## 変更してよい範囲

- このリポジトリ配下のファイル
- 依存追加は最小限

## 実行してよいコマンド

- `python3 -m venv .venv`
- `pip install -r requirements.txt`
- `pytest`
- `python scripts/check_ollama.py --model <model>`
- `python scripts/run_experiment.py --episodes 50 --conditions comm silent random --model <model> --comm-phase-steps 2 --randomize-positions --hard-split-prob 0.5 --memory-budget 50`
- `python scripts/analyze_results.py --input logs/results.csv --trace-dir logs/traces`

## 完了条件

- `pytest` が通る
- `make smoke` が通る
- `logs/results.csv` が出る
- `make analyze` が既存指標と glyph 指標の要約または図を出す
- README と docs が再現手順を含む
- Ollama 未導入時は公式手順を案内して丁寧に停止する
