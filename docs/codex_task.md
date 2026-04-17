# Codex task brief

このリポジトリに、macOS Apple Silicon + Ollama を前提とした、ローカル LLM 2 体による ScoreG 風協調実験環境を実装してください。

## ゴール
- PettingZoo ParallelEnv ベースの 5x5 gridworld 実装
- Ollama ローカル API を使う 2 エージェント
- 7x7 二値グリフ通信
- `comm / silent / random` の 3 条件実験
- `logs/results.csv` 出力
- pytest と smoke test が通る
- README と docs を整備

## 変更してよい範囲
- このリポジトリ配下のファイル全般
- 依存追加は最小限で可

## 実行してよいこと
- Python 仮想環境の作成
- `pip install -r requirements.txt`
- `pytest`
- `python scripts/check_ollama.py`
- `python scripts/run_experiment.py --episodes 10 --condition comm`
- `python scripts/analyze_results.py`

## 実行してはいけないこと
- Docker 導入
- クラウド LLM API の使用
- 非公式な Ollama インストール方法の採用
- 破壊的コマンド

## 完了条件
- `pytest` 成功
- `make smoke` 成功
- `logs/results.csv` 作成
- README に再現手順あり
- Ollama 未導入時は丁寧なエラーで停止
