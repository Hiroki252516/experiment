# Frozen-LLM Glyph Emergence Spec for Codex

この文書は、既存の ScoreG 風ローカル LLM 実験基盤を、**LLM 本体の重み更新なし**で、より強く **創発的な glyph 慣習の芽** を観察しやすい実験系へ改善するための Codex 向け修正仕様書です。

この文書は、既存の以下の文書を前提にします。
- `AGENTS.md`
- `docs/CODEX_SETUP_PLAYBOOK.md`
- `docs/codex_task.md`
- `docs/experiment_context_for_codex.md`
- `docs/realtime_visualization_spec_for_codex.md`

---

## 1. 目的

この追加仕様の目的は、

- **LLM 本体を RL で再学習しない**
- **ローカル LLM を 2 体のエージェントとして使う**
- **7x7 binary glyph を唯一の外部通信路として維持する**
- そのうえで、**反復相互作用・環境設計・外部 memory・multi-step communication** によって、
  **pair-specific で局所的だが再利用される glyph 慣習の芽** が生まれるかを観察する

ことです。

本仕様は、「最初から強い compositional language を作ること」ではなく、
**固定 LLM 間で、通信が全く無意味なノイズではなくなる条件を作ること**を狙います。

---

## 2. 研究的な立ち位置

このプロジェクトで狙うのは、厳密な意味での MARL による emergent language 学習の完全再現ではありません。

むしろ、次の中間目標です。

1. 固定 LLM 同士の局所相互作用だけで、再利用される記号的慣習が生じうるかを見る
2. その慣習が `comm` 条件でのみ有効になり、`silent` や `random` より高い成功率につながるかを見る
3. 記号体系の「強い完成形」ではなく、「芽」や「痕跡」を評価対象にする

Codex はこの framing を守ること。

---

## 3. なぜ今の最小版では弱いのか

既存 repo は最小実験基盤としては正しいが、創発的通信を観察するには次の点が弱い。

1. **環境配置が局所ヒューリスティックを誘う**
   - それぞれが自分に近いアイテムへ進みやすい
   - 通信しなくても「何かしら行動できてしまう」

2. **prompt が弱い**
   - チーム報酬最大化や glyph 再利用の圧力が弱い

3. **memory が薄い**
   - glyph と成功・失敗の対応が保存されない

4. **反復が短い**
   - smoke 的な短いエピソード数では慣習形成が起きにくい

5. **通信フェーズが短い**
   - 1 ターン 1 glyph だけでは合意形成が不十分

この仕様書は、これらを改善する。

---

## 4. 絶対に守る制約

1. **LLM 本体の重み更新は禁止**
   - RL, SFT, LoRA, fine-tuning を今回のスコープに含めない

2. **ローカル LLM のみ**
   - Ollama を使うこと
   - クラウド API は使わない

3. **通信路は 7x7 binary glyph のみ**
   - 自然言語メッセージは agent 間通信に使わない

4. **2 agent / 5x5 / 2 item / partial value split** は維持

5. **hidden info leakage を絶対に起こさない**
   - viewer や logs を通じて agent prompt に漏れないこと

---

## 5. 期待する成果の再定義

成功の定義を、今後は次のように置く。

### 5.1 最低成功
- `comm` 条件の平均報酬が `silent` と `random` を上回る
- `comm` 条件の target agreement rate が上がる

### 5.2 中程度の成功
- 同じ hidden-value / task-state で、同一または類似 glyph が再利用される
- replay で「glyph を見て進路変更した」と解釈できる場面が観察される

### 5.3 強い成功
- 別 seed / 別 episode でも安定した glyph 慣習が維持される
- partner 固有の co-adaptation を超えて部分的に汎化する

今回の Codex 修正では、**最低成功〜中程度の成功** を狙う。

---

## 6. 変更の大方針

Codex は、次の 4 レイヤを修正すること。

1. **環境レイヤ**: 通信が必要になる task structure を強化
2. **prompt / policy レイヤ**: 協力・合流・glyph 再利用の圧力を追加
3. **memory レイヤ**: glyph と outcome の対応を保持
4. **評価レイヤ**: glyph 再利用や通信効果を測る指標を追加

---

## 7. 環境レイヤの修正要件

### 7.1 初期位置の固定をやめる

毎 episode で、agent の開始位置と item 位置をランダム化すること。

要件:
- 5x5 内に収まる
- agent と item が同じマスから始まらない
- trivial な初期配置ばかりにならないよう制約を入れる

目的:
- 「毎回まっすぐ上に進むだけ」の方策を壊す
- 通信しないと正しい合流先が決まりにくい状態を作る

### 7.2 communication-only phase を追加する

各 episode の冒頭に、**移動禁止で glyph だけ交換するフェーズ** を入れる。

推奨:
- `comm_phase_steps = 2` 〜 `3`
- この間、`move` は無視されるか `STAY` に固定
- glyph は通常どおり送信・受信される

目的:
- multi-step / bidirectional な合意形成を可能にする
- 「送る前に相手がもう動いてしまう」問題を避ける

### 7.3 hard split episode を増やす

通信の価値が明確に出る episode を意図的に増やすこと。

例:
- `value_left << value_right`
- `value_right << value_left`
- 左右差が大きい episode を一定割合でサンプリング

目的:
- glyph が target を反転させるべき局面を増やす

### 7.4 optional: corridor / obstacle variant

初版で余裕があれば、障害物または通行制約を入れてよい。
ただし 5x5 world の基本設定は崩さないこと。

---

## 8. Prompt / Policy レイヤの修正要件

### 8.1 system prompt を強化する

既存 prompt は安全だが弱い。次の内容を追加すること。

必須概念:
- あなたの目的は **チーム報酬最大化** である
- glyph は **hidden information を相手へ伝える唯一のチャネル** である
- 自分が低い価値を見ているなら、自分側へ固執せず相手側へ合流する可能性を考える
- 同じ状況では、以前うまくいった glyph を再利用するほうがよい
- 相手と target が一致しないと失敗しやすい

### 8.2 ただし codebook を固定しすぎない

禁止:
- `0000001 = RIGHT` のような明示的辞書を最初から与えること
- few-shot で glyph と意味を露骨に対応付けること

理由:
- それでは創発ではなく hand-coded protocol に近づく

### 8.3 reasoning は外に出さない

- 出力は従来どおり structured JSON
- rationale や chain-of-thought を返させない
- ただし内部的な target 選択や glyph 再利用は prompt 上で促してよい

### 8.4 探索性を少し上げる

現在の設定が完全決定論的なら、以下を検討してよい。

- `temperature = 0.2` 程度
- ただし JSON 安定性が壊れない範囲に限る

目的:
- glyph 空間の探索をわずかに許す

---

## 9. Memory レイヤの修正要件

### 9.1 private memory を richer にする

各 agent の memory に、少なくとも以下を残すこと。

- `episode`
- `condition`
- `my_known_value`
- `my_sent_glyph`
- `my_received_glyph`
- `my_target`
- `team_outcome`
- `team_reward`
- `agreement`

### 9.2 memory は agent ごとに閉じる

- `agent_a` memory は `agent_a` のみ参照
- `agent_b` memory は `agent_b` のみ参照
- memory 共有は禁止

### 9.3 memory summary format を一定にする

例:
```text
episode=17 condition=comm my_known_value=1 my_sent_glyph=000... my_received_glyph=111... my_target=RIGHT agreement=True team_reward=1.0 outcome=high_value
```

目的:
- LLM が成功時の glyph パターンを in-context で参照できるようにする

### 9.4 memory budget を持つ

- 直近 30〜100 件だけ保持
- 無制限に膨らませない

---

## 10. 実験 runner の修正要件

### 10.1 エピソード数を増やせる CLI にする

デフォルトは維持してよいが、README と docs では以下を推奨値として明記すること。

- exploratory run: `--episodes 50`
- convention run: `--episodes 200`
- deeper run: `--episodes 500`

### 10.2 warmup と evaluation を分ける

推奨フロー:
- warmup phase: memory 更新あり
- evaluation phase: memory を固定して held-out episodes を評価

可能なら次のオプションを追加してよい。
- `--eval-episodes`
- `--freeze-memory-before-eval`

### 10.3 run metadata を保存する

少なくとも次を run manifest に入れること。
- model
- temperature
- episodes
- conditions
- comm_phase_steps
- randomize_positions
- memory_budget
- prompt_version

---

## 11. 評価レイヤの修正要件

### 11.1 既存指標は維持

最低限以下は維持する。
- mean team reward
- success rate
- target agreement rate

### 11.2 追加指標

Codex は次の指標を追加すること。

#### glyph reuse consistency
同じ hidden-value / target 条件で、同じ agent が似た glyph を再利用している割合

#### glyph-target association
特定 glyph が LEFT/RIGHT の target とどれくらい結びついているか

#### target-flip rate
受信 glyph 後に、自分の当初 target から最終 target が変わった割合

#### communication gain
`comm - max(silent, random)` の差分

### 11.3 cross-seed stability
時間と余裕があれば、seed ごとの glyph 使用分布を比べる機能を追加してよい

---

## 12. Viewer の修正要件

viewer はすでにある前提で、次を追加・改善すること。

### 12.1 target change visibility
各 step で、
- initial target
n- final target
- glyph reception after effect
が分かるようにする

### 12.2 glyph history strip
直近数 step の glyph を横に並べて表示する

### 12.3 convention hints panel
簡易分析として、
- 最近よく出る glyph
- その glyph が多い target
- 成功時によく出る glyph
を表示する

### 12.4 episode segmentation
communication-only phase と movement phase を viewer 上で明示する

---

## 13. 新たに追加してよいファイル

```text
analysis/
├── glyph_metrics.py
└── convention_report.py

docs/
└── frozen_llm_glyph_emergence.md

tests/
├── test_memory_summary.py
├── test_comm_phase.py
└── test_glyph_metrics.py
```

必要なら `viewer/` 配下にも追加してよい。

---

## 14. テスト要件

最低限以下を追加すること。

### `tests/test_comm_phase.py`
- communication-only phase 中に位置が変わらない
- glyph の送受信だけが進む

### `tests/test_memory_summary.py`
- memory summary に sent/received glyph が入る
- hidden info leakage が無い

### `tests/test_glyph_metrics.py`
- glyph reuse consistency が計算できる
- glyph-target association が計算できる

---

## 15. README 更新要件

README に次を追記すること。

1. この実験は「固定 LLM 間の glyph 慣習の芽」を観察するものだと明記
2. 強い創発言語は保証しないこと
3. 推奨 run 設定（50 / 200 / 500 episodes）
4. `comm_phase_steps` の意味
5. memory の役割
6. 追加された glyph 指標の読み方

---

## 16. 非目標

今回も以下はやらない。

- LLM 本体の RL / SFT / LoRA
- hand-coded glyph dictionary の固定投入
- cloud API 利用
- Docker 導入
- partner-aware centralized training

---

## 17. 完了条件

以下を満たしたら完了とする。

1. 既存の `pytest` が壊れていない
2. communication-only phase が導入されている
3. 開始位置・アイテム位置のランダム化が入っている
4. prompt が team reward / glyph reuse / coordination を促す方向に改善されている
5. private memory に glyph と outcome の対応が残る
6. `comm` / `silent` / `random` の比較が引き続き可能
7. 追加指標（glyph reuse / glyph-target association など）が算出できる
8. viewer で target change や glyph history を追える
9. README に新しい実験目的と限界が明記されている

---

## 18. 実装順序

Codex は以下の順で進めること。

1. 既存 env / prompts / runner / viewer を読む
2. communication-only phase を env に追加
3. 初期位置・アイテム位置ランダム化を追加
4. prompts を強化
5. memory summary を拡張
6. runner に run metadata と長期 run オプションを追加
7. glyph 指標計算を追加
8. viewer に glyph history / target change 可視化を追加
9. テストを追加
10. README / docs を更新
11. smoke test と短い convention run を回す

---

## 19. 一文要約

この仕様の目的は、**LLM 本体を凍結したまま、環境設計・反復相互作用・外部 memory・multi-step glyph exchange を強化することで、ローカル LLM 間に pair-specific で再利用される創発的 glyph 慣習の芽が現れる条件を作ること**である。
