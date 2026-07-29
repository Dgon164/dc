# THE STOREFRONT — copy, prices, and the week's plan
*(2026-07-28; every price and platform fact below comes from the
market scan — receipts in the session record. Paste-ready.)*

## The one sentence

> Buyers' #1 pain in 2026's AI-flooded data-gig market is that they
> cannot tell whose numbers to trust. **Receipts are the only offer
> in the category that makes trust checkable.** We sell that.

## The three offers (research-ranked by conversion)

### 1. Spreadsheet Cleaning + Audit with Receipts — the volume gig
"I will clean your Excel/CSV and hand back the cleaned file **plus a
changelog receipting every single change** — and an audit that flags
what needs your judgment instead of silently 'fixing' it."
- Basic $15: one file ≤5k rows, cleaned file + changelog, 24h
- Standard $50: + the audit (error/duplicate/outlier flags with
  receipts; we ask your tolerance rather than assume), 24h
- Premium $95: multi-file + joins + re-audit proving the mechanical
  passes are clean, 48h
- Run: `python standalone/clean_file.py file.csv [--resolution
  col=sigma]` → cleaned.csv + CHANGELOG.md. Mechanical fixes only
  (delimiters, headers, whitespace, missing markers, exact dupes —
  each receipted); anything needing judgment is FLAGGED, never
  guessed. A wild value that poisons the law gets the PASS B
  hold-out road and comes back as an OUTLIER-CANDIDATE with its
  receipt. See `cleaning_demo/` for the planted-error catch.
- The commodity cleaning market is $15–50 with 2–4 day turnaround
  and zero verifiability — we undercut on **turnaround and proof**,
  not price.

### 2. "Honest Numbers" Insight Report — the flagship
"Trend, cycle, anomalies, and forecast for every column of your
sales/finance/e-commerce CSV — every number traced to a receipt.
**If a column can't support a forecast, the report says NO RULE and
shows why. You will never get a made-up number.**"
- Core $45–95 (report + charts, 48h) · $125+ with dashboard, 72h
- Two tiers of claims in every report (owner ruling 2026-07-28):
  a **certified rule** passed the strictest checks; a **what's-there
  description** reports the plain trend graded SOLID/WEAK/UNSTABLE
  by whether it survives 200 random half-samples + a wandering-data
  check. Clients always get told what's in their data — with the
  honesty grade attached, never a guess dressed as certainty.
- Direct anti-workslop positioning: 40% of businesses hit AI slop
  (~2h/incident to fix), 39% distrust AI accuracy. Charts included
  — buyers expect them; the wrapper makes them.

### 3. Verify-the-Numbers (AI-output QA) — the growth niche
"I will check every number in your report, AI-generated analysis, or
dashboard against the raw data — **receipted pass/fail per claim**."
- $50–150, 24–48h. No charts needed; the receipts ARE the product.
- Rides a measured wave: QA hires up 9% on Upwork specifically to
  validate AI output. Also fits dissertation/survey buyers
  ($111–250/job) who pay for defensible methodology.

## Paste-ready posts

### r/forhire (needs an account ≥~5 days old with some karma —
one post per 7 days, rates REQUIRED in post)
> **[For Hire] Data analyst — every number in my report comes with a
> receipt. $40 flat: CSV/Excel audit or trend report in 24h.**
> I run your file through deterministic analysis software I build —
> not ChatGPT. Every claim traces to a receipt you can spot-check
> (the rows, the rule, the evidence), and where the data can't
> support a conclusion the report says NO RULE instead of inventing
> one. Deliverable: self-contained HTML report + charts + full
> receipt appendix. Sample report in comments. $40 single file /
> $95 multi-file or verification of an existing analysis. PayPal
> G&S / Venmo / Zelle.

### r/slavelabour (no account gate, low rates, same-day PayPal)
> $25: I'll audit your spreadsheet and hand you a receipted error
> log (cell / rule / evidence) + summary charts, today. $15 if under
> 1k rows. Every flag checkable — no AI guessing.

### Craigslist "computer gigs", nearest metro ($3–10 one-time fee)
> **Spreadsheet cleanup + honest summary report — $75–150, 48h
> turnaround.** Local data guy. I clean your Excel/QuickBooks
> export, flag every error with evidence you can check, and deliver
> charts + a plain-English writeup. No offshoring, no ChatGPT
> guesswork — deterministic software, every number receipted.
> Zelle/cash.

### Fiverr gig (the pipeline — first payout ~day 15+; post today,
paid in week 2–3)
- Title: "I will analyze your CSV and give you a report where every
  number has a receipt"
- Tiers: $15 basic (bank first 5 reviews) / $75 standard / $150 pro
- Gallery: the portfolio charts + a receipt-appendix screenshot +
  the NO RULE refusal captioned "the anti-hallucination guarantee."

### Cold DM / email (Discord freelance servers, local businesses)
> Quick offer: I'll take one CSV/Excel file off your hands and
> return a charted, plain-English report where every single number
> traces to a checkable receipt — and where the data's too thin to
> support a claim, the report says so instead of making one up.
> $45 flat, 48h, first one half-price if you let me use an
> anonymized chart as a portfolio sample. Pay by Zelle/Venmo when
> you're happy.

## Payment rails (set up TODAY, in this order)

1. **Zelle** via the existing bank — instant, free, no new-account
   hold. The only rail where Friday cash is GUARANTEED at payment.
2. **Venmo + Cash App** — 1.9–2.6% receive fee, ~1.75% instant
   transfer; same-day usable.
3. **PayPal invoices** — the rail strangers trust most, BUT new
   accounts hold first payments up to 21 days (mark order complete
   + buyer confirms receipt to release faster). Don't count it as
   Friday cash.
4. **SKIP this week:** Stripe (7–14 day unwaivable first payout),
   Gumroad (7-day hold, Friday-only payouts), Lemon Squeezy
   (13-day hold).

## Scam filters (non-negotiable)

Never accept a check. Never pay any fee to start. Urgency + "move to
WhatsApp" = walk away. Card-funded PayPal/Venmo can charge back —
for strangers, do a small paid pilot first. Zelle/Cash App received
funds effectively can't be reversed — prefer them for new clients.

## The per-job road (honesty laws apply to the business too)

1. Client file in → `python standalone/honest_report.py file.csv`
2. **Owner reads the report before it ships** (platform policies
   require human review; our laws require it anyway). Curate: drop
   trivial findings (e.g. the year column's trend), keep every
   refusal.
3. Deliver `report.html` (self-contained) + receipts.
4. Disclosure line, always: "Analysis produced by proprietary
   deterministic software and human-reviewed; every figure carries
   a receipt. Happy to walk through any number." (Research: clients
   prefer proactive transparency; discovered-not-disclosed AI does
   the most trust damage. Ours is not an LLM — say so plainly.)
5. Guarantee: "Any number you can't trace to a receipt = free
   revision or refund." (The QA-included pricing pattern that
   converts best.)
6. Client data: never enters Isaac's live mind, never leaves this
   machine, deleted on request. (dc privacy law.)

## The realistic math (honest)

Friday outcome from direct channels: **1–3 small gigs, $20–150
total** — grocery money, not rent. The marketplaces planted today
become the real pipeline in weeks 2–4 ($25→$75→$150 tiers as
reviews land). The differentiator is real and the research says it
answers the market's loudest complaint — but new-seller gravity
(reviews, response time) still applies. Volume of outreach this
week is the variable that moves the number.
