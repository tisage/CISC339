# Lab04 notebook integration plan (draft, not yet implemented)

Status: placeholder for the teaching-notebook work described in
conversation, to be written once flow/session/log data generation is
finalized and reviewed. This is a design sketch, not final wording -
expect to revise once real generated data is in hand.

## Bayesian Network chapter addition (after existing HW4-3)

New subsection: "Estimating a Bayesian Network from data" - kept
separate from HW4-3's hand-authored CPT version, not a replacement.

1. **Explanatory text**: how a CPT gets estimated from data (one/two
   sentences on `MaximumLikelihoodEstimator` - counting how often each
   combination of parent states co-occurs with each child state, no need
   to derive the math).
2. **Code cell**: load `flows.jsonl` into a DataFrame, build a
   `DiscreteBayesianNetwork` with the same structure as HW4-3
   (Traffic -> Alert <- Firewall -> Intrusion), fit it with
   `MaximumLikelihoodEstimator`, print the resulting CPDs.
3. **Display + guided question**: show the estimated CPD table(s) and
   ask students to interpret ONE specific cell out loud - e.g. "What does
   the number in the row Traffic=Abnormal, Firewall=Block mean? Write it
   as P(Alert=? | Traffic=?, Firewall=?)." Compare that cell's value to
   HW4-3's hand-authored equivalent.
4. **Prediction on a small sample set**: a curated ~20-record sample
   (not the full 40k dataset - see the flow/session ratio discussion),
   covering a deliberate mix of Traffic x Firewall x Alert combinations
   plus a couple of "stealthy attack" edge cases (traffic_level=Normal
   but label=attack). Run `VariableElimination` queries against the
   estimated model for each sample row, show the predicted
   P(Intrusion=Yes | evidence), and where ground truth (`label`) is
   available, show it alongside for comparison.
5. **Step-by-step guided questions** (not one big open question):
   - "For this record, what's the predicted probability of intrusion?"
   - "Is that above or below 50%? What would you conclude?"
   - "This record's real label was `attack` but predicted probability
     was low - why might that happen?" (leads into the traffic_level vs.
     label gap / stealthy-attack discussion)
   - "Does the Bayesian Network 'know' about attack_type? Why or why
     not?" (leads into scope/limitations of this model)

## HMM chapter addition/replacement

Mirrors the Bayesian Network treatment - hand-authored (existing
weather/umbrella example, kept) vs. data-estimated (new, from
`sessions.jsonl`) side by side.

1. Explanatory text: same idea as flow's MLE explanation, but for
   `hmmlearn`'s Baum-Welch/EM estimation of transition + emission
   matrices from sequences.
2. Code: load `sessions.jsonl`, encode to hmmlearn's integer format,
   `.fit()`, print estimated transition/emission matrices next to
   `session_scenarios.true_transition_matrix()` /
   `true_emission_matrix()` for comparison.
   - **Known issue to handle**: EM is sensitive to random
     initialization - some seeds converge to the true structure, others
     get stuck in a clearly worse local optimum (verified: `random_state`
     values 1/3/4 converged well in testing, 0/2 did not, at n=400
     sessions). Pin a verified-good `random_state` in the notebook code
     rather than exposing this instability to students as a live bug -
     it's real and worth a one-line mention, but not something to debug
     interactively in a probability/inference lab.
3. Run Viterbi + Forward-Backward on a few sample sessions (mix of
   all-benign and some-compromised), same comparison table style as the
   existing weather example, discuss where the two disagree.
4. Guided questions, same step-by-step style as the BN section above.

## LLM chapter (log data) - not yet designed

`logs.jsonl` generation is validated (schema, small-batch quality) but
the actual exercise this feeds is undefined - revisit once that chapter
exists. Do not build this section speculatively.

## Open items to confirm before writing real notebook cells

- Exact wording/phrasing of guided questions (draft above is a skeleton).
- Whether the ~20-record BN sample set is hand-picked from generated data
  or randomly sampled with a fixed seed + manual review pass.
- Whether HMM sample sessions for the Viterbi/Forward-Backward demo are
  hand-picked or randomly sampled.
