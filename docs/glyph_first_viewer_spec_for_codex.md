# Glyph-First Viewer Spec for Codex

この文書は、`experiment` リポジトリの `b` ブランチに対して、**7×7 binary glyph を最優先に可視化する viewer 改修**を依頼するための仕様書です。

既存の viewer は live / replay / launch を持ち、glyph も内部的には表示していますが、
**ユーザーが実際に観察したときに「7×7 ピクセルのやり取りが主役として見える GUI」になっていない**という問題があります。

この仕様の目的は、viewer を **glyph-first** な設計に改め、
「2 体のローカル LLM がどの glyph を送り、相手がどう受け取り、それによって target / move がどう変わったか」を
**リアルタイムでも replay でも一目で追える GUI** にすることです。

---

## 1. 背景

このリポジトリの実験目的は、固定したローカル LLM 2 体が、
5x5 gridworld 上で 7x7 binary glyph だけを使い、部分観測下で高価値アイテムへ共同合流できるかを試すことです。

現在の branch `b` では、README 上は realtime viewer があり、
viewer も `live / replay / launch` の 3 モードを持つと説明されています。
また `viewer/app.py` では `st.image(glyph_rows_to_array(...))` を使って sent / received glyph を表示しています。

しかし実際の UX としては、
- glyph が画面の主役になっていない
- 7x7 ピクセルで通信している感触が弱い
- glyph と move / target / reward の関係が追いにくい
- 「言語・記号が立ち上がる様子」を観察する viewer になっていない
という問題があります。

したがって、今回の改修では **viewer の中心を grid よりも glyph interaction に置く**。

---

## 2. ゴール

最終的に実現したいのは、次のような観察体験です。

1. 画面を開いた瞬間に、agent_a / agent_b の sent glyph と received glyph が大きく見える
2. どの step でどの glyph が送られたかが時間軸で追える
3. glyph を見た後で target が切り替わったかが分かる
4. glyph の繰り返し再利用が episode をまたいで見える
5. live 実験中に「今、通信だけしているのか」「今、移動しているのか」が明確に分かる
6. replay 時に 1 step ごとの glyph を“映像”として追える
7. 「7x7 ピクセルの小さな図形を用いて、記号体系・言語を創発している感じ」が viewer から伝わる

---

## 3. 非目標

今回の改修では以下は不要です。

- LLM 本体の重み更新
- LoRA / RL / fine-tuning
- Docker 導入
- クラウド LLM API
- viewer 経由で prompt を編集する機能
- DB やサーバを増やす大規模構成
- マルチユーザー共同閲覧

viewer はあくまで **観察・再生・比較** のためのものとする。

---

## 4. 最優先の設計原則

### 4.1 glyph-first
viewer の最上段、または最も目立つ領域は **Glyph Theater** にすること。
Gridworld は重要だが補助的位置づけとする。

### 4.2 送受信の因果が見えること
単に glyph を置くだけでなく、
- 誰が送ったか
- 相手は何を受け取ったか
- その後 move / target がどうなったか
が接続されて見えること。

### 4.3 live と replay の両方で同じ体験を提供すること
live だけ、または replay だけでは不十分。
両方で glyph 表示が viewer の主役であること。

### 4.4 hidden information leakage を防ぐこと
viewer は人間向け観察装置として真の値を表示してよいが、
その情報が agent prompt 側へ流れ込む経路を作ってはならない。

---

## 5. 追加・変更したい UI 要件

## 5.1 Glyph Theater（新規・最重要）

viewer の最上部に、常に表示される大きな glyph 表示領域を作ること。

最低限、以下 4 枚を同時表示する。

- A sent glyph
- B received glyph
- B sent glyph
- A received glyph

レイアウト例:

```text
A sent   ->   B received
B sent   ->   A received
```

要件:
- 7x7 の生ピクセル感が分かる表示にする
- 1 マスごとの境界線を表示する
- nearest-neighbor 的な拡大にする
- 112x112 程度ではなく、**最低でも 196x196 〜 280x280 程度**の十分大きい描画を使う
- 白黒だけでなく、必要なら薄い背景と濃いピクセルで視認性を上げる
- glyph の下に rows 文字列も表示してよい

### 5.2 Glyph History Strip（新規）

直近 N step 分の glyph を、film strip のように横並びに表示すること。

目的:
- glyph が繰り返し再利用されているかを一目で見る
- 同一エピソード内での変化を見る

表示例:
- step 0 の A/B glyph
- step 1 の A/B glyph
- step 2 の A/B glyph
- ...

最低要件:
- 直近 8 step 分を見られる
- replay では step slider と同期する

### 5.3 Communication Timeline（強化）

既存の event log を強化し、
各 step を次のような構造で表示すること。

- phase
- A glyph summary
- B glyph summary
- target_before / target_after
- move
- reward
- outcome (if any)

特に、**target が glyph の直後に変わった場合は強調表示**すること。

### 5.4 Agent Panel の glyph 表示を昇格

既存の agent panel にある sent / received glyph は残してよいが、
Glyph Theater の簡易コピーではなく、**agent 視点の補助表示**として扱うこと。

例:
- 「この agent が今見ている受信 glyph」
- 「この agent が今送った glyph」
- 「過去の成功時に再利用した glyph か」

### 5.5 Gridworld Panel は維持

gridworld は残す。
ただし viewer の主役は glyph なので、
Gridworld Panel は Glyph Theater の下段または横に置く。

---

## 6. 描画仕様

## 6.1 glyph の表示方式

既存の `glyph_rows_to_array()` は grayscale を返しているが、
以下の改善を行うこと。

1. 1 マスごとの境界線を視覚化
2. 余白を少し入れる
3. 画素拡大率を configurable にする
4. `st.image` に渡す前に、必要なら 3-channel にして見やすくする
5. 可能なら light / dark theme でも潰れにくい配色にする

### 6.2 sent と received を見分けやすくする

- sent 側は送信者カラー枠をつける
- received 側は受信者カラー枠をつける
- A 系と B 系でキャプション色や境界色を変える

### 6.3 ゼログリフも明確に見せる

`silent` 条件では全ゼログリフが出る。
これが「ただの空白」に見えると意味が伝わらないため、
全ゼロ時も 7x7 のグリッド線を表示し、
「ゼログリフ」であることが目で分かるようにする。

---

## 7. Live View の改修要件

### 7.1 live polling の対象を明確化

live モードでは、current frame を 1 秒ごとに更新するだけでなく、
Glyph Theater と Glyph History Strip も必ず同期更新すること。

### 7.2 communication-only phase の可視化

もし runner 側に communication-only phase がある場合、
viewer では phase を目立つバッジで表示すること。

例:
- `COMMUNICATION PHASE`
- `ACTION PHASE`

communication phase 中は「移動が起きないが glyph は動いている」ことを視覚的に強調する。

### 7.3 target switch の強調

live 中に target が切り替わった場合、
Glyph Theater または Timeline に
- `A target switched LEFT -> RIGHT`
- `B target switched RIGHT -> LEFT`
のようなバッジを出すこと。

---

## 8. Replay View の改修要件

### 8.1 step-by-step のアニメーション性を上げる

replay では、単に step slider があるだけでなく、
Glyph Theater が毎 step ごとに大きく更新されるようにする。

### 8.2 comparison replay

将来的な比較用として、最低限の土台を入れる。

必須ではないが推奨:
- `comm` と `silent` を並べて同じ episode index を比較
- `comm` と `random` を並べて比較

ただし、初版では単独 replay だけでもよい。
その場合でも docs に今後拡張として書くこと。

### 8.3 glyph-focused replay controls

replay コントロールの近くに、
- `Prev glyph event`
- `Next glyph event`
のようなボタンを追加してよい。

これは「次の step」ではなく、「glyph が変化した次の step」へ飛ぶ機能である。

---

## 9. trace / ログ要件の見直し

viewer が本当に glyph-first になるには、trace 側に十分な情報が必要である。

最低限、各 step trace に次を確実に含めること。

- `glyph_a_sent`
- `glyph_b_sent`
- `glyph_a_received`
- `glyph_b_received`
- `move_a`
- `move_b`
- `target_a_before`
- `target_b_before`
- `target_a`
- `target_b`
- `target_a_changed`
- `target_b_changed`
- `phase`
- `phase_turn_index`
- `reward_a`
- `reward_b`
- `team_reward`
- `done`
- `outcome`

加えて、viewer 用に次の補助フィールドがあると望ましい。

- `glyph_a_hash`
- `glyph_b_hash`
- `glyph_a_reused_from_success`
- `glyph_b_reused_from_success`
- `glyph_event`（glyph が変化した step かどうか）

---

## 10. 変更対象ファイル

Codex は最低限、以下のファイルを変更してよい。

```text
viewer/app.py
viewer/render.py
viewer/data.py
viewer/utils.py
scripts/run_experiment.py
README.md
Makefile
tests/test_glyph_rendering.py
tests/test_viewer_data_loading.py
tests/test_trace_schema.py
```

必要なら新規に追加してよい。

```text
viewer/glyphs.py
viewer/components.py
viewer/styles.py
docs/glyph_first_viewer_spec.md
```

---

## 11. 実装の優先順位

Codex は次の順で進めること。

1. 現状の glyph 表示の問題点を確認
2. `viewer/render.py` に glyph 専用の拡大表示関数を作る
3. Glyph Theater を `viewer/app.py` に追加
4. Glyph History Strip を追加
5. replay と live で両方動くように結線
6. trace に不足フィールドがあれば `run_experiment.py` を補強
7. README と Makefile を viewer 実態に合わせて更新
8. テスト追加
9. live / replay の手動確認

---

## 12. テスト要件

最低限、以下を追加すること。

### `tests/test_glyph_rendering.py`
- 7x7 rows から描画用配列が生成できる
- 拡大後サイズが想定どおり
- 全ゼログリフでもグリッドが視認できる描画関数がある

### `tests/test_viewer_data_loading.py`
- trace から glyph rows を正しく読み込める
- live trace / replay trace の両方で必要キーが揃う

### `tests/test_trace_schema.py`
- target_before / target_after / glyph sent/received など viewer 必須フィールドが存在する

GUI の screenshot snapshot test は必須ではない。

---

## 13. README / Makefile 更新要件

README は branch `b` の現状に viewer 記述があるが、
次を明確に追記または修正すること。

1. Glyph Theater が viewer の主機能であること
2. live で 7x7 glyph を大きく観察できること
3. replay で step ごとに glyph を追えること
4. `make viewer` の有無を README と Makefile で一致させること
5. viewer で見えるものの一覧

Makefile には `viewer` ターゲットが存在しない場合、追加すること。

```make
viewer:
	streamlit run viewer/app.py
```

---

## 14. 完了条件

以下をすべて満たしたら完了とする。

1. viewer を開くと、7x7 glyph が大きく目立つ位置に表示される
2. A/B の sent / received glyph が同時表示される
3. live 実験中に glyph がリアルタイムで更新される
4. replay で step ごとに glyph の変化を追える
5. 全ゼログリフも視認できる
6. glyph と target/move の関係を追いやすい
7. README に viewer の使い方が更新されている
8. Makefile と README の viewer 起動方法が一致している
9. 既存 CLI 実験と tests を壊していない

---

## 15. 一文要約

この改修の目的は、既存の branch `b` の viewer を **glyph-first** な GUI に再設計し、
2 体のローカル LLM が 7x7 binary glyph を送り合って proto-language / 記号的慣習を形成していく様子を、
**live と replay の両方でリアルタイムに観察できるようにすること**である。
