# AGENTS.md

## 目的
このリポジトリで行う作業は、macOS（Apple Silicon、想定: MacBook Air M4）上で動く、ローカル LLM ベースの 2 エージェント協調実験環境を構築することです。

題材は **高価値アイテム選択タスク（ScoreG 風）** です。2 体のエージェントは同じ 5x5 グリッド世界を共有しますが、各エージェントはアイテム価値の一部だけを知っています。通信は **7x7 の二値グリフ（49 bit）** に制限し、相手と合意しながら高価値アイテムを共同回収することを目標にします。

このリポジトリでは、以下を **Codex が自動で実装・整備** してください。

1. Python 仮想環境で動く実験コード一式
2. PettingZoo ParallelEnv ベースの環境
3. Ollama ローカル API を使う 2 エージェント実装
4. `comm / silent / random` の 3 条件を回す実験ランナー
5. 結果保存（CSV / JSONL）
6. 基本的な解析スクリプト
7. 環境テストと最小ユニットテスト
8. README と docs
9. Makefile

## 非目標
- クラウド LLM API の利用
- Docker ベースの実行環境
- 本格的な強化学習で LLM 本体重みを更新すること
- GUI アプリを操作する自動化

## 重要制約
- **ローカル LLM のみ**を使うこと。
- モデルサーバは **Ollama** を前提とすること。
- Ollama が未インストールなら、**非公式な導入方法（brew 等）を勝手に使わないこと**。
- `ollama` CLI が存在しない場合は、ユーザーに **公式 macOS インストール手順を案内して停止** すること。
- 実験コードは Python 3.11 〜 3.12 互換で書くこと。
- OS 固有処理は macOS Apple Silicon 前提でよいが、コード自体はできるだけ汎用的にすること。
- 外部依存は最小限にし、追加時は README に理由を書くこと。
- 破壊的なコマンド（`rm -rf`, git reset/hard, 大量削除）は使わないこと。

## 想定ディレクトリ構成
以下の構成を作成・維持してください。

```text
.
├── AGENTS.md
├── README.md
├── requirements.txt
├── Makefile
├── .gitignore
├── .codex/
│   └── config.toml
├── agents/
│   ├── __init__.py
│   └── ollama_agent.py
├── envs/
│   ├── __init__.py
│   └── scoreg_env.py
├── prompts/
│   ├── agent_a_system.txt
│   └── agent_b_system.txt
├── scripts/
│   ├── check_ollama.py
│   ├── run_experiment.py
│   ├── analyze_results.py
│   └── smoke_test.py
├── tests/
│   ├── test_env_api.py
│   ├── test_glyph_utils.py
│   └── test_prompt_safety.py
├── logs/
│   └── .gitkeep
└── docs/
    ├── experiment_spec.md
    └── codex_task.md
```

## 必須依存関係
以下を `requirements.txt` に含めてください。

- numpy
- pettingzoo
- gymnasium
- pandas
- matplotlib
- pydantic
- requests
- rich
- pytest

必要なら以下も検討可:
- python-dotenv
- seaborn は不要

## 実装要件

### 1. 環境
`envs/scoreg_env.py` に、PettingZoo の `ParallelEnv` を継承した環境を実装してください。

#### 固定仕様
- グリッドサイズ: デフォルト 5x5
- エージェント: `agent_a`, `agent_b`
- アイテム: `left_item`, `right_item` の 2 個
- 各アイテム価値: 1〜9 の整数ランダム
- `agent_a` は左アイテム価値のみ観測可
- `agent_b` は右アイテム価値のみ観測可
- 両エージェントは位置情報を観測可
- 最大ステップ数: デフォルト 10
- 各ターンで、各エージェントは同時に
  - 移動アクション 1 つ
  - 7x7 二値グリフ 1 つ
  を出す

#### 行動空間
以下 6 アクション:
- `UP`
- `DOWN`
- `LEFT`
- `RIGHT`
- `STAY`
- `PICK`

#### 観測空間
各エージェント観測に最低限含めること:
- 自分の位置
- 相手の位置
- 2 アイテムの位置
- 自分に見える private value ベクトル
- 相手の前ターングリフ（49 bit）
- 現在ステップ番号

#### 報酬
初期実装は以下でよい:
- 2 体が **高価値アイテム** を共同回収: `+1.0`
- 2 体が同じアイテムを共同回収したが低価値側: `+0.2`
- 2 体が別々のアイテムを回収: `-0.2`
- 毎ステップ時間罰: `-0.01`

### 2. Ollama エージェント
`agents/ollama_agent.py` を実装してください。

#### 必須要件
- Ollama ローカル API を使うこと
- ベース URL は `http://localhost:11434`
- `/api/generate` を使ってもよいし、OpenAI 互換 `/v1/responses` を使ってもよい
- ただし **Structured Outputs / JSON schema** により、出力を厳格 JSON に制限すること
- 各エージェントは独立した private memory を持つこと
- `agent_a` と `agent_b` の内部履歴を絶対に共有しないこと
- hidden value が漏れるような prompt を作らないこと

#### モデル名
- デフォルトは環境変数または CLI オプションで受け取る
- 既定値は軽量モデル名を使ってよい
- 実行前に `ollama list` か `/api/tags` でモデル存在確認をすること

#### 出力 JSON 仕様
最低限以下を返させること:
- `glyph`: 長さ 7 の 0/1 文字列 7 本、または 49 要素整数配列
- `move`: `UP/DOWN/LEFT/RIGHT/STAY/PICK`
- `target`: `LEFT/RIGHT/UNKNOWN`

#### 失敗時
- JSON が壊れた場合はリトライ戦略を入れること
- 連続失敗時はエラー内容をログへ残し、その episode を安全に失敗終了させること

### 3. 実験条件
`run_experiment.py` は最低限以下 3 条件をサポートしてください。

- `comm`: LLM が自分でグリフ生成
- `silent`: グリフを常にゼロベクトルに固定
- `random`: グリフをランダム 49 bit に置換

### 4. ログ
各 episode について少なくとも以下を保存してください。

- seed
- condition
- episode id
- value_left
- value_right
- best_item
- outcome
- team_reward
- target_a
- target_b
- 必要なら最終ターン raw 出力

保存先:
- `logs/results.csv`
- できれば `logs/episodes.jsonl`

### 5. 解析
`analyze_results.py` で最低限以下を出してください。
- 条件別の平均チーム報酬
- 条件別の成功率
- 条件別の `target agreement rate`
- 簡単な折れ線グラフまたは棒グラフ

### 6. テスト
最低限以下を実装してください。

#### `tests/test_env_api.py`
- PettingZoo `parallel_api_test` を通す

#### `tests/test_glyph_utils.py`
- 7x7 グリフ flatten/unflatten が可逆

#### `tests/test_prompt_safety.py`
- agent_a prompt に right value が入らない
- agent_b prompt に left value が入らない

## 実行順序
Codex は以下の順序で進めてください。

1. リポジトリ構成を作る
2. `requirements.txt`, `Makefile`, `.gitignore` を作る
3. `scripts/check_ollama.py` を作る
4. Ollama 利用可否を確認する
5. 環境本体を実装する
6. エージェント本体を実装する
7. 実験ランナーを実装する
8. テストを実装する
9. README / docs を書く
10. 依存インストール → テスト → スモークテスト → 解析まで実行する

## Ollama チェック手順
`scripts/check_ollama.py` を作り、以下を確認してください。

1. `ollama` コマンドが存在するか
2. `http://localhost:11434/api/tags` が応答するか
3. 指定モデルが存在するか

存在しない場合:
- 公式 macOS インストールが必要だと README と標準出力に明記
- その時点でセットアップを中断してよい
- 非公式導入は試さない

## Makefile 必須ターゲット
以下を作成してください。

- `make setup`  : 依存インストール
- `make test`   : pytest 実行
- `make smoke`  : 5〜10 episode の短い実験
- `make run`    : 本番実験
- `make analyze`: 結果解析

## README に必ず書くこと
- このプロジェクトの目的
- macOS Apple Silicon + Ollama 前提であること
- Ollama 未導入時の手動インストール手順
- 仮想環境作成手順
- モデル取得例
- 実験の実行方法
- 結果の見方
- よくある失敗

## docs/experiment_spec.md に必ず書くこと
- タスク定義
- 観測・行動・報酬定義
- 3 条件比較の意図
- セッションメモの扱い
- 限界（これは本格 RL 学習ではなく、まずは固定 LLM を用いた協調実験であること）

## docs/codex_task.md に必ず書くこと
Codex へ貼り付けるための短い作業指示を作成してください。短いが十分具体的で、以下を含めること。
- ゴール
- 変更してよい範囲
- 実行してよいコマンド
- 完了条件

## 実装スタイル
- 型ヒントを付ける
- 例外処理を入れる
- コメントは簡潔に
- マジックナンバーを避ける
- 設定値は CLI 引数または定数へ分離する

## 完了条件
以下をすべて満たしたら完了です。

1. `pytest` が通る
2. `make smoke` が通る
3. `logs/results.csv` が出力される
4. `make analyze` が図または要約を出す
5. README と docs が揃っている
6. Ollama 未導入時も、失敗理由が分かるように丁寧に停止する

## Codex への行動指針
- 不足している前提は README と docs で補う
- 勝手に仕様を広げすぎない
- まず動く最小版を完成させ、その後に改善する
- 変更後は必ずテストとスモークテストを回す
- ローカル環境で再現できることを最優先にする
