# Frozen LLM Glyph Emergence Notes

## 目的

この改善版の目的は、LLM 本体を再学習せずに、固定ローカル LLM の 2 体ペアが 7x7 binary glyph を反復的に再利用する条件を作ることです。狙いは強い言語体系ではなく、`comm` 条件でのみ成績改善や再利用傾向として現れる「glyph 慣習の芽」です。

## 変更点

- communication-only phase を追加し、冒頭数 step は移動せず glyph だけ交換する
- agent / item 開始位置をランダム化できる
- hard split episode を増やし、通信の必要性を高める
- prompt で team reward 最大化、glyph 再利用、target 一致を強調する
- private memory に sent/received glyph と outcome の対応を残す

## 評価指標

- mean team reward
- success rate
- target agreement rate
- glyph reuse consistency
- glyph-target association
- target flip rate
- communication gain

## 限界

- LLM 重み更新はしない
- hand-coded glyph codebook は与えない
- 強い compositional language は保証しない
- 観察される慣習は pair-specific で局所的な可能性が高い
