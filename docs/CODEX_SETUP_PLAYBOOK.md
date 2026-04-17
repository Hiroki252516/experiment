# Codex setup playbook for the ScoreG local-LLM experiment

この文書は、Codex がこのリポジトリをセットアップし、最小実験を最後まで完成させるための実行計画です。

## 前提
- OS: macOS on Apple Silicon
- LLM server: Ollama
- App install: ユーザーが公式手順で導入
- Project implementation: Python + PettingZoo + requests

## Phase 1: bootstrap
1. `.gitignore`, `requirements.txt`, `Makefile`, `README.md` を作る
2. `agents/`, `envs/`, `scripts/`, `tests/`, `docs/`, `logs/` を作る
3. Python package として `__init__.py` を置く

## Phase 2: local prerequisites
1. `scripts/check_ollama.py` を実装
2. `ollama` コマンドの有無をチェック
3. `http://localhost:11434/api/tags` に到達できるか確認
4. 指定モデル名が存在するか確認
5. 失敗した場合は、ユーザーに公式 Ollama インストールを依頼して中断

## Phase 3: environment
1. `envs/scoreg_env.py` に ParallelEnv を実装
2. `parallel_api_test` を通す
3. seed 再現性を確認

## Phase 4: agents
1. `agents/ollama_agent.py` を実装
2. JSON schema による構造化出力を使う
3. hidden info が prompt に漏れないようガードする
4. 失敗時のリトライを入れる

## Phase 5: runner
1. `scripts/run_experiment.py` を実装
2. 条件 `comm`, `silent`, `random` を切り替え可能にする
3. 結果を CSV / JSONL に保存
4. まず 5 episode で smoke test を行う

## Phase 6: analysis
1. `scripts/analyze_results.py` を実装
2. 条件別平均報酬、成功率、target agreement を出す
3. 可能なら簡単な図を `logs/` に保存

## Phase 7: docs and polish
1. README を仕上げる
2. `docs/experiment_spec.md` を書く
3. 実装上の制約と今後の拡張（MLX-LM での LoRA など）を明記

## Acceptance checklist
- [ ] `pytest` が通る
- [ ] `make smoke` が通る
- [ ] `logs/results.csv` が出る
- [ ] README だけで再現できる
- [ ] hidden info leakage テストがある
