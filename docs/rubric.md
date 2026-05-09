# Manual scoring rubric

Used for axes where automated metrics either don't apply or where ground truth
isn't available (most notably the SEC 10-K pages we curated ourselves).

Each axis scored **1–5**, integer. Score one stack at a time per page; do not
score head-to-head — recency bias makes side-by-side scoring unreliable. The
parquet writer joins the manual scores with automated metrics afterward.

## Axes

### Layout fidelity
How well does the rendered output preserve the visual structure of the input?
- **5** — Reading order is correct, columns/sections are preserved, tables are
  rendered in-place with their captions, footnotes attached to the right body
  text. A reader could follow the output without seeing the original.
- **4** — One small structural slip (e.g. a footnote rendered before its
  reference, or one column boundary missed) but core flow is intact.
- **3** — Reading order is mostly right but at least one major structural
  element is wrong (table position, header hierarchy, etc.).
- **2** — Output is text-recoverable but structurally jumbled — would mislead a
  downstream consumer.
- **1** — Output is unusable for any structural purpose.

### Table quality
For pages with tables. If no table on the page, mark N/A.
- **5** — All cells present, all spans correct, all numbers correct, headers
  identified.
- **4** — Cells and content correct; one span error or one header mis-classification.
- **3** — Topology mostly right but ≥2 cell errors or ≥1 numeric error.
- **2** — Table is recognized but topology or numbers are wrong enough that a
  finance user wouldn't trust it.
- **1** — Table not recognized as a table, or topology fundamentally broken.

### KV quality
For pages with form fields, invoices, receipts.
- **5** — All gold KV pairs extracted, no spurious pairs.
- **4** — All gold KV pairs extracted, ≤1 spurious pair OR 1 missing minor field.
- **3** — Major fields (amount, date, vendor) all correct, ≥1 secondary field missing.
- **2** — At least one major field wrong or missing.
- **1** — Output unusable for KV consumption.

### Text accuracy
General OCR quality on body text. Use CER as the input — but the score reflects
how the text *reads*, not just edit distance.
- **5** — CER < 0.01, no semantic drift.
- **4** — CER < 0.03, occasional minor errors.
- **3** — CER < 0.08, readable but you'd want a human pass.
- **2** — CER < 0.20, partially garbled.
- **1** — Text is not reliably readable.

## Scoring discipline

1. Score one stack on one page, write notes, move on.
2. Re-randomize stack order between pages so you don't anchor on the first stack.
3. Notes are mandatory for any score of 2 or 1 — concrete failure mode, not "bad."
4. The workshop deck shows the rubric so the audience knows the scoring frame.
