# Agnes Sys1 1.0 Decision Model Technical Report

> This is a web-readable presentation of the supplied five-page report. The [original PDF](PAPER.pdf) is the canonical source. Capability figures below are reproduced from the team-supplied evaluation table; the report did not independently re-evaluate them. The test-time training section specifies a method and evaluation plan, not a measured improvement.

## Abstract

Agnes Sys1 1.0 adapts Qwen3.5-9B to decision tasks: classification, tool selection, request routing, and ordinal scoring. The report describes block-bidirectional attention in full-attention layers, direct extraction of probabilities over request-specific candidate options, full fine-tuning on decision tasks, and a test-time training (TTT) approach. The team-supplied capability table reports an AA Composite Index of **71.2**.

## 1. Base model and decision interface

The language backbone is described as approximately 9 billion parameters, with 32 layers and a hidden dimension of 4,096. Its repeated pattern is eight groups of three Gated DeltaNet layers followed by one full-attention layer. The decision-oriented changes concern attention masks, output extraction, and training objectives.

| Item | Agnes Sys1 1.0 |
| --- | --- |
| Base model | Qwen3.5-9B |
| Backbone | Approximately 9B parameters; 32 layers; hidden dimension 4,096 |
| Layer pattern | 8 × (3 Gated DeltaNet layers + 1 full-attention layer) |
| Input | Shared state or source material, question specification, and candidate options |
| Output | Probability distribution over the current candidate options |

Let a task be `x = (s, q, O)`, where `s` is the state to evaluate, `q` specifies the question and rules, and `O` is the candidate set. The highest-probability option yields a classification; the affirmative option's probability yields a yes/no score; and probabilities over ordered levels can yield an expected score. Candidate options are supplied with each request rather than fixed in one business-specific head.

## 2. Architectural modifications

### Shared prefix and independent question suffixes

Each request has a shared prefix `P` containing the system prompt and state material, plus a question suffix `S` containing one question, letter-labeled options, and the answer extraction position. Multiple questions about the same material have separate suffixes but share prefix computation. A suffix cannot attend to another question's suffix, so their candidate sets do not influence one another.

### Block-bidirectional attention

The 24 Gated DeltaNet layers retain causal computation. In the eight full-attention layers, a prefix block has index 0 and the current suffix block has index 1; position `i` may attend to position `j` when `b(j) ≤ b(i)`.

| Query position | Accessible content | Consequence |
| --- | --- | --- |
| Prefix `P` | Entire prefix `P` | Bidirectional encoding of source material; cache does not depend on the question |
| Current suffix `S` | Entire prefix `P` and suffix `S` | Question and options participate jointly in representation computation |
| Other question suffixes | No cross-suffix attention | Each question is evaluated independently |

Unlike a standard causal mask, this lets earlier positions in a question incorporate later option definitions. Because the prefix cannot attend to the suffix, its cache can be reused across questions.

### Reading option probabilities from the LM head

The original language-model head is retained. At the answer position, the model takes hidden state `h`, selects logits `zᵢ = (Wᵁh)ℓᵢ` for candidate label tokens `ℓ₁ … ℓₖ`, and normalizes over only those candidates:

`pᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)`.

This produces a relative distribution over the current options without generating a JSON answer token by token or adding a fixed classification head for every business label set. The probabilities are **not automatically calibrated correctness probabilities**; that requires a separate calibration evaluation.

A standard LLM can also expose answer-position option logits. The report attributes the intended difference to question-block representations, decision-task supervision, a joint loss for probability quality, and training on uncertain samples for abstention behavior. Those modifications must be evaluated directly; exposing probabilities alone does not establish a gain. Each extra question still needs its own suffix forward pass, and actual latency depends on inputs, question count, and deployment.

## 3. Test-time training approach

The report proposes within-request adaptation: construct verifiable auxiliary tasks from the current material, perform a small number of gradient updates before prediction, rebuild caches, and then answer the original questions. This is a design specific to Agnes and is distinguished in the PDF from the implementation of the cited TTT paper.

### Parameters and supervision

Starting from checkpoint `θ₀`, updates are limited to scaling parameters `φ` of existing RMSNorm layers near the end of the language backbone. Other parameters and the visual encoder remain frozen. The number of updated layers is chosen on a development set. Each request gets an independent copy of `φ₀`; the updates are discarded after prediction.

Auxiliary multiple-choice tasks can ask about entity presence, explicit fields, event order, or directly stated attributes. Their labels must come from structured fields, source evidence, or deterministic programs. Unverifiable samples are excluded. Meaning-preserving views may reorder options or structured fields, with option outputs mapped back to semantic labels before comparison. If reliable auxiliary samples cannot be formed, the system falls back to the baseline model. Target business answers and evaluation-set labels are not used as adaptation supervision.

### Objective, safeguards, and cache lifecycle

The stated objective combines auxiliary supervision, consistency, baseline anchoring, and parameter regularization:

`L_TTT = L_aux + αL_cons + βL_anchor + γ‖φ − φ₀‖²₂`.

`L_aux` uses cross-entropy and Brier error; `L_cons` compares aligned view distributions with Jensen–Shannon divergence; `L_anchor` uses `KL(p₀ ∥ pφ)` on the same auxiliary tasks. The weights are selected on an independent development set.

The workflow initializes a request-local copy, builds auxiliary tasks, performs bounded updates with gradient clipping and a time limit, and falls back if loss is non-finite, drift exceeds a limit, or a timeout occurs. It then freezes adapted parameters, clears caches and recurrent states made under the old parameters, re-encodes the shared material, computes each suffix independently, and releases the adaptation state. Old KV caches and Gated DeltaNet recurrent states cannot be reused after parameters change. TTT adds auxiliary forward passes, backpropagation, and cache rebuilding, so ordinary single-pass latency assumptions do not apply.

The report proposes comparing 1, 3, and 5 update steps against the model without updates. The checkpoint, questions, candidate mapping, and scoring scripts would be fixed. Planned ablations isolate auxiliary supervision, consistency, and baseline constraints. Planned metrics include accuracy, non-hallucination rate, Brier/ECE calibration, abstention, P50/P95 latency, GPU memory, and fallback rate. **No TTT experiment results are reported in this document.**

## 4. Capability table supplied with the report

The following values are transcribed from the team-supplied evaluation image in the PDF. “Rank 1/2/3” are labels retained from that image, **not current real-time rankings**. The Agnes values follow prior internal estimates; the report conducted **no new independent re-evaluation**.

| Benchmark | d1-Ultra (Rank 1) | SubQ Snap 1.1 (Rank 2) | elephant-1 (Rank 3) | Agnes Sys1 1.0 |
| --- | ---: | ---: | ---: | ---: |
| HLE-MultiChoice | 15.0% | 17.5% | 18.3% | **38.8%** |
| When2Call | 92.6% | 94.6% | 94.8% | **95.2%** |
| WildGuardTest | 95.4% | 94.9% | 93.3% | **93.6%** |
| Omniscience accuracy (open-source subset) | 31.6% | 20.2% | 11.6% | **38.8%** |
| Omniscience non-hallucination rate (open-source subset) | 84.5% | 85.7% | 93.2% | **75.2%** |
| AA Composite Index | 65.2 | 65.0 | 64.7 | **71.2** |

For Agnes, the stated weighted calculation is `0.25 × 38.8 + 0.25 × 95.2 + 0.25 × 93.6 + 0.125 × 38.8 + 0.125 × 75.2 = 71.15`, rounded to 71.2. Against the table's 64.7-point reference, that is 6.5 points higher. The PDF also notes that Agnes's Omniscience non-hallucination rate is **18.0 percentage points lower** than that reference, so accuracy and non-hallucination must be considered together.

## References as described in the PDF

1. Official Qwen3.5-9B model card.
2. Publicly available technical report on the decision-model architecture and training, §§2.2–2.6.
3. Capability evaluation table supplied by the team.

The TTT section additionally names Sun et al., *Test-Time Training with Self-Supervision for Generalization under Distribution Shifts* (ICML 2020) as its methodological reference. See the [original PDF](PAPER.pdf) for the source wording.
