# CurrencyMorphism mathematical anchors

Use the pinned Lean 4.28.0 toolchain and locked mathlib dependencies:

```sh
lake build
lake env lean CheckAxioms.lean
```

`FiniteKL.lean` proves finite supported KL data processing for any deterministic
map, the ordinary-KL formula, and its specialization to actual path reversal
and pointwise coarse observation. The real `finiteKL` definition is interpreted
as KL only when the first law is supported on the reference law; it does not
represent infinity on singular pairs. The original `finiteKL_map_le` API remains.

`Duality.lean` constructs and normalizes finite Gibbs probability laws, proves
the entropy–cost variational identity, and derives constrained optimality and
the supporting shadow-price inequality under explicit budget feasibility and
complementary slackness. Resource budgets are hypotheses, not consequences of
packaging. The formal duality result covers a finite probability row; weighted
rows, price derivatives, limits, and packaging bounds are described in
`../review/mathematical_review.txt` and are not claimed as mechanized.

`CheckAxioms.lean` reports the transitive axioms of the principal exports. The
expected dependencies are only `propext`, `Classical.choice`, and `Quot.sound`.
The review and fresh numerical evidence are separate from the historical frozen
paper evidence under `docs/experiments/final/`.
