# Golden Evaluation Set — Methodology

## File
`data/eval/golden_set.csv` — 200 hand-labeled (customer_message, intent, escalate, escalate_reason) rows.

## Sampling
1. Started from `amazon_pairs.csv` (40,861 pairs).
2. **Found a data quality issue in Step 1's English filter**: running `langdetect`
   over `customer_clean` showed 2,433 / 40,861 rows (~6%) are not actually English
   (mostly Italian/Portuguese/Spanish/German messages that happened to survive the
   ASCII-ratio + keyword-blocklist filter because they use plain Latin script with
   few accented characters, e.g. "non ho il kindle fire"). These rows were excluded
   from the sampling pool for the golden set. This is logged as a known limitation —
   worth carrying into the decision log / failure analysis in the report.
3. Each remaining message was scored against each intent's keyword list (from
   `agent/intents.py`) to get a **candidate intent** (highest keyword-overlap intent,
   or `other` if no keywords matched).
4. Took a **stratified random sample of 25 messages per intent** (8 intents x 25 = 200),
   `random_state=42` for reproducibility.

## Labeling
- Candidate intents from keyword scoring were reviewed by hand; **13 of 200** were
  corrected where the keyword match was clearly wrong (e.g. a message matching
  "fake" + "email" was keyword-tagged `product_issue` but is actually a phishing
  report -> relabeled `other`; a Kindle battery complaint matched the "charge"
  keyword and was mis-tagged `billing_payment` -> relabeled `device_tech_support`).
  Final distribution after correction:
  - delivery_issue: 29, other: 26, return_refund: 25, device_tech_support: 25,
    prime_subscription: 25, account_access: 24, billing_payment: 24, product_issue: 22
- **Escalation** was labeled using an explicit rubric (implemented as rules, then
  spot-checked by hand), not vibes. Escalate = True when the message shows:
  - Redacted account/payment identifiers (`__credit_card__`, `__email__`) needing
    secure manual verification
  - Account security language (hacked, stolen, compromised, phishing)
  - Legal/regulatory escalation language
  - An unrecognized or disputed charge needing investigation
  - Staff misconduct reports
  - Evidence the customer already tried multiple times / channels without resolution
  - Account lockout/suspension (needs identity verification, not a script)
  - Messages too short/vague to identify the actual issue
  - Otherwise: **auto-handle**, with the reason citing the standard historical
    resolution pattern for that intent (grounded in what AmazonHelp actually did
    in `brand_reply_clean` for similar messages).
- Overall escalation rate in the golden set: **12%** (ranges from 0% for
  `device_tech_support`/`prime_subscription` to 38% for `account_access`, which
  makes sense — login/lockout issues usually need identity verification a bot can't do).

## Known limitations of this golden set
- Escalation labels came from a rule-based rubric applied consistently, not
  independent human-only judgment — there is no inter-annotator agreement figure
  for this set (a caveat worth stating in the "what's misleading about my headline
  number" section of the report).
- `other` is a catch-all and slightly under-determined; boundary cases with
  `product_issue` vs `return_refund` (e.g. "damaged item, want replacement") were
  resolved by whichever action the customer explicitly asked for.
- Some `device_tech_support` messages are actually positive feedback or jokes
  rather than support requests (e.g. "binge watching on my fire stick") — these
  were reassigned to `other` during review but a production classifier should
  expect this kind of noise in the wild.

## Reproducing this set
The sampling + labeling logic lives conceptually in two steps:
1. Filter `amazon_pairs.csv` to English-only via `langdetect` (not just the
   Step 1 ASCII/keyword filter).
2. Score each message against `agent/intents.py` keywords, stratified-sample
   25/intent with `random_state=42`, then apply the escalation rubric above.
This is deterministic given the same `amazon_pairs.csv` and `intents.py`.
