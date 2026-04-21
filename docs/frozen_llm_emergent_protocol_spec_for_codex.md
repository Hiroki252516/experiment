# Frozen-LLM Emergent Protocol Spec for Codex

この文書は、**LLM 本体の重みを更新しない**という条件を守ったまま、現在の ScoreG 風ローカル LLM 実験基盤を、**poetengineer 的な「やり取りを通じて記号的慣習が立ち上がる」挙動**に近づけるための Codex 向け修正仕様書である。

この文書は以下を前提にする。
- `AGENTS.md`
- `docs/codex_task.md`
- `docs/experiment_context_for_codex.md`
- 既存の `viewer/` を含む repo

この文書の目的は、**本格的な RL 再学習や LoRA 微調整を行わず、固定したローカル LLM 同士の反復相互作用・外部 memory・環境設計によって、glyph ベースの proto-language / convention の芽が生まれる条件を強化すること**である。

---

## 1. 研究目的の再定義

この改修の目的は、いきなり「安定した人工言語」を作ることではない。
まず目指すべき成果は以下である。

1. `comm` 条件が `silent` / `random` を安定して上回ること
2. 同じような状況で、同じまたは似た 7x7 glyph が再利用されること
3. replay 上で、glyph と target / move / 合流成功の対応が観察できること
4. 同一ペア・同一セッション内で、局所的な通信慣習が育つこと

この文書で狙うのは、**strong language** ではなく、**proto-protocol / emergent convention** である。

---

## 2. 絶対条件

以下は維持すること。

1. **LLM 本体の重みは更新しない**
2. ローカル LLM のみを使う
3. Ollama を使う
4. 2 エージェント構成を維持する
5. 通信路は 7x7 binary glyph のまま維持する
6. `comm / silent / random` の比較を維持する
7. hidden information leakage を起こさない
8. Docker やクラウド API は使わない

---

## 3. 重要な設計方針

### 3.1 「創発」の意味を明確化する

このプロジェクトでは、創発を次のように定義する。

- 事前に人間が glyph と意味の対応表を与えない
- エージェントが反復相互作用の中で、ある glyph を一貫して再利用し始める
- その再利用が協調成績の改善と結びつく

### 3.2 何をしないか

以下はこの改修では行わない。

- RL による policy 学習
- LoRA / SFT / 微調整
- 人手で glyph コード表を決めること
- `0000001 = RIGHT` のような固定辞書を prompt に埋め込むこと

### 3.3 何を増やすか

重み更新の代わりに、次を増やす。

- 通信が必要になる環境圧力
- 反復回数
- communication-only phase
- richer private memory
- glyph と結果の対応を追跡する評価指標

---

## 4. 最重要の改修: 環境を「通信必須」にする

現在の環境は、各 agent が自分の列を上に進めば自分側のアイテムへ自然に近づける構造になりやすく、通信圧力が弱い。

そのため、環境は以下のように修正すること。

### 4.1 開始位置の固定をやめる

- `agent_a`, `agent_b` の start position を episode ごとにランダム化する
- ただし、両 agent が完全に対称になりすぎないよう、最低距離制約を入れる

### 4.2 item 位置の固定をやめる

- left / right item の「意味」は private value の帰属だけにし、物理位置は毎回変える
- `left_item` が必ず左にある必要はない
- `right_item` が必ず右にある必要はない

### 4.3 communication-only phase を導入する

各 episode の最初に、**移動禁止・glyph 送信のみ可能**なフェーズを 2〜3 turn 入れる。

目的:
- 「まず動く」ではなく「まず伝える」を促す
- glyph が相手の target 更新に使われる余地を増やす
- 双方向プロトコル形成を助ける

実装案:
- `phase="comm_only"` を導入
- この phase では move は `STAY` に固定し、glyph のみ送信可能
- 2〜3 turn 経過後に normal phase へ移行

### 4.4 タスク成功条件を「合流」へ寄せる

現状の reward は維持してよいが、少なくとも以下を追加検討すること。

- 高価値アイテムへの**同時合流**を強く報酬化する
- 片方だけが着いても成功にしない
- 低価値側への合流は弱い正報酬、別々なら負報酬

### 4.5 ローカル最適を崩す

必要なら以下のどれか 1 つを導入してよい。

- 障害物
- 一方通行マス
- タイミング条件
- 一定 step 以内に同時に `PICK` しないと失敗

ただし、複雑化しすぎないこと。まずは communication-only phase + start/item randomization を優先する。

---

## 5. エージェント設計の改修

### 5.1 prompt を「協力圧力」寄りにする

system prompt は、今の安全制約を保ちつつ、以下を明示すること。

- あなたの目標は **チーム報酬最大化** である
- あなたの近くの item ではなく、**より高価値の共同目標**へ合流することが重要である
- glyph は hidden information を相手へ伝える唯一のチャネルである
- 同じような状況では、以前うまくいった glyph を再利用してよい
- 相手と target が一致しないのは失敗シグナルである

### 5.2 few-shot codebook は原則禁止

以下はしない。
- `0000001 = RIGHT`
- `0000010 = LOW_VALUE`

その代わり、必要なら **戦略レベルの few-shot** は許可する。
例:
- 「自分が low value を見たとき、相手の signal を待って target を更新したほうが高報酬になりやすい」
- 「成功した glyph は再利用してよい」

### 5.3 探索性を少し上げる

現状の deterministic setting が強すぎる場合、以下を検討してよい。

- glyph 生成時のみわずかに探索性を上げる
- move より glyph の多様性を優先する
- ただし JSON 構造の安定性を壊さない範囲にとどめる

この変更は控えめに行うこと。

---

## 6. private memory の改修

ここが最重要。

現在の memory が outcome 中心で薄すぎる場合、各 agent の private notebook に以下を残すこと。

### 6.1 1 episode summary に最低限含める情報

- episode id
- condition
- self known value
- self sent glyph（少なくとも最終 / 代表 glyph）
- partner received glyph
- self target history（簡略版でよい）
- final target agreement
- outcome
- team reward
- この episode で「glyph が効いた」と推定できる簡単な自然言語要約

### 6.2 memory retention policy

- 完全ログを無限に積まない
- 直近 N 件 + 成功例の代表サンプルを残す
- 成功例を優先保存する
- 失敗例も「何が噛み合わなかったか」を短く残す

### 6.3 memory の役割

memory は訓練ではない。役割は以下。

- same-pair convention の蓄積
- glyph と outcome の弱い対応付け
- 「以前これでうまくいった」を in-context で再利用

---

## 7. runner の改修

### 7.1 エピソード数を増やす

デフォルトの本番実験は少なすぎる。
以下のプリセットを用意すること。

- `smoke`: 5 episode / condition
- `pilot`: 30 episode / condition
- `study`: 100 episode / condition
- `long`: 300 episode / condition

### 7.2 同一ペア継続を明示する

同じ 2 体のエージェントが、同じ private memory を持ったまま連続 episode を経験することを README と docs に明記すること。

### 7.3 trace を強化する

step-level trace に以下を必須追加すること。

- phase (`comm_only` / `act`)
- sent glyph
- received glyph
- target before / after update
- whether target changed after partner glyph
- whether glyph matched prior successful glyphs

---

## 8. viewer の改修

viewer は現在の live/replay を維持しつつ、以下を強化すること。

### 8.1 target update を明示する

各 step で
- previous target
- current target
- target changed? yes/no
を表示する

### 8.2 communication-only phase を表示する

phase を UI 上に明示する

### 8.3 convention hints を表示する

可能なら以下を追加する
- 最近の成功 episode で再利用された glyph 一覧
- 同じ known value 状況で頻出した glyph
- glyph consistency score

### 8.4 replay comparison

同一 run の中で
- 成功 episode
- 失敗 episode
を並べて見比べられるようにすると望ましい

---

## 9. 新しい評価指標

平均報酬だけでは不十分。以下を追加すること。

### 9.1 既存
- success rate
- average reward
- target agreement rate

### 9.2 新規
- glyph reuse rate
- same-context glyph consistency
- target-switch-after-glyph rate
- comm-only phase 後の agreement improvement
- successful-vs-unsuccessful glyph divergence

### 9.3 余裕があれば
- pair 固定での安定性
- pair を変えたときの崩れ方（cross-play 的確認）

---

## 10. 実験計画

Codex は以下の 3 段階で実装すること。

### Phase A: protocol pressure upgrade

目的:
- 通信必須構造を作る

作業:
- 環境 randomization
- communication-only phase
- reward / success 条件の最小調整
- trace への phase 追加

### Phase B: memory and prompt upgrade

目的:
- same-pair convention を育ちやすくする

作業:
- prompt 改修
- private memory richer 化
- success-centered summary 保存

### Phase C: measurement and visualization upgrade

目的:
- 「芽が出たか」を観察可能にする

作業:
- 新指標追加
- viewer に target-switch / glyph reuse 可視化追加
- replay で成功/失敗比較を可能にする

---

## 11. Codex にやってほしい具体的変更

以下のファイルを中心に改修すること。

- `envs/scoreg_env.py`
- `agents/ollama_agent.py`
- `prompts/agent_a_system.txt`
- `prompts/agent_b_system.txt`
- `scripts/run_experiment.py`
- `scripts/analyze_results.py`
- `viewer/app.py`
- `tests/`
- `README.md`
- `docs/experiment_spec.md`

必要なら追加してよいファイル:
- `viewer/metrics.py`
- `viewer/conventions.py`
- `tests/test_memory_summary.py`
- `tests/test_phase_logic.py`

---

## 12. 新しいテスト要件

最低限、以下を追加すること。

### `tests/test_phase_logic.py`
- communication-only phase では移動しない
- phase が正しく切り替わる

### `tests/test_memory_summary.py`
- summary に glyph と outcome が入る
- hidden info leakage が起きない

### `tests/test_convention_metrics.py`
- glyph reuse / consistency 指標が計算できる

---

## 13. README に追記すべきこと

README には以下を追加すること。

1. 本プロジェクトは「固定 LLM による proto-protocol 観察」が目的であること
2. これは本格 RL 学習ではないこと
3. communication-only phase の意味
4. 長期 run (`pilot`, `study`, `long`) の使い方
5. glyph reuse / convention 指標の見方
6. 期待しすぎないこと
   - 強い人工言語の完成が保証されるわけではない
   - まずは局所的な慣習の芽を見るもの

---

## 14. 非目標

今回の改修では以下は行わない。

- LoRA / SFT / RL
- 人手による glyph 辞書埋め込み
- partner-independent universal language の実現保証
- 大規模 population 実験

---

## 15. 完了条件

今回の改修の done は以下。

1. 既存の CLI 実験が壊れていない
2. communication-only phase を含む環境が動く
3. same-pair long run が動く
4. richer memory summary が保存される
5. viewer で phase / glyph / target-switch を観察できる
6. 新しい評価指標が出る
7. `comm` が `silent` / `random` より改善する実験条件を少なくとも一部で確認できる
8. replay で glyph 再利用の兆しを人間が確認できる

---

## 16. 一文要約

この改修仕様の目的は、**LLM 本体を再学習せずに、環境圧力・反復相互作用・private memory・可視化を強化することで、7x7 glyph による局所的な emergent convention / proto-language の芽が観察できる ScoreG 風実験系へ現在の repo を進化させること**である。
