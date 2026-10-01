import CurrencyMorphism.FiniteKL

open scoped BigOperators

namespace CurrencyMorphism

variable {α : Type} [Fintype α] [DecidableEq α]

noncomputable def entropy (p : PMF α) : ℝ :=
  -∑ a, (p a).toReal * Real.log (p a).toReal

noncomputable def meanCost (p : PMF α) (u : α → ℝ) : ℝ :=
  ∑ a, (p a).toReal * u a

noncomputable def partitionFunction (u : α → ℝ) (lam : ℝ) : ℝ :=
  ∑ a, Real.exp (-lam * u a)

omit [DecidableEq α] in
lemma partitionFunction_pos [Nonempty α] (u : α → ℝ) (lam : ℝ) :
    0 < partitionFunction u lam := by
  classical
  let a : α := Classical.choice inferInstance
  unfold partitionFunction
  exact lt_of_lt_of_le (Real.exp_pos (-lam * u a))
    (Finset.single_le_sum (f := fun x => Real.exp (-lam * u x))
      (fun x _ => le_of_lt (Real.exp_pos (-lam * u x))) (Finset.mem_univ a))

/-- A concrete finite Gibbs law, with normalization proved rather than supplied. -/
noncomputable def gibbsPMF [Nonempty α] (u : α → ℝ) (lam : ℝ) : PMF α :=
  PMF.ofFintype (fun a => ENNReal.ofReal
    (Real.exp (-lam * u a) / partitionFunction u lam)) (by
      have hn : ∀ a, 0 ≤ Real.exp (-lam * u a) / partitionFunction u lam :=
        fun a => le_of_lt (div_pos (Real.exp_pos _) (partitionFunction_pos u lam))
      rw [← ENNReal.ofReal_sum_of_nonneg (fun a _ => hn a)]
      rw [← Finset.sum_div]
      change ENNReal.ofReal (partitionFunction u lam / partitionFunction u lam) = 1
      simp [ne_of_gt (partitionFunction_pos u lam)])

omit [DecidableEq α] in
lemma gibbsPMF_toReal [Nonempty α] (u : α → ℝ) (lam : ℝ) (a : α) :
    (gibbsPMF u lam a).toReal = Real.exp (-lam * u a) / partitionFunction u lam := by
  simp only [gibbsPMF, PMF.ofFintype_apply]
  exact ENNReal.toReal_ofReal (le_of_lt
    (div_pos (Real.exp_pos _) (partitionFunction_pos u lam)))

omit [DecidableEq α] in
lemma gibbsPMF_pos [Nonempty α] (u : α → ℝ) (lam : ℝ) (a : α) :
    0 < (gibbsPMF u lam a).toReal := by
  rw [gibbsPMF_toReal]
  exact div_pos (Real.exp_pos _) (partitionFunction_pos u lam)

omit [DecidableEq α] in
lemma gibbsPMF_log [Nonempty α] (u : α → ℝ) (lam : ℝ) (a : α) :
    Real.log (gibbsPMF u lam a).toReal =
      -lam * u a - Real.log (partitionFunction u lam) := by
  rw [gibbsPMF_toReal, Real.log_div (ne_of_gt (Real.exp_pos _))
    (ne_of_gt (partitionFunction_pos u lam)), Real.log_exp]

/-- Variational Gibbs identity. `hlog` is the explicit Gibbs recognition
hypothesis, NOT an assumed optimization conclusion. It is satisfied by
`q(a) = exp(-λ*u(a))/Z`, with `logZ = log Z`. -/
theorem gibbs_KL_identity (p q : PMF α) (u : α → ℝ) (lam logZ : ℝ)
    (hq : ∀ a, 0 < (q a).toReal)
    (hlog : ∀ a, Real.log (q a).toReal = -lam * u a - logZ) :
    finiteKL p q = -entropy p + lam * meanCost p u + logZ := by
  rw [finiteKL_eq_sum_log p q hq]
  have ht : ∀ a, (p a).toReal * Real.log ((p a).toReal / (q a).toReal) =
      (p a).toReal * Real.log (p a).toReal +
        lam * ((p a).toReal * u a) + (p a).toReal * logZ := by
    intro a
    by_cases hp : (p a).toReal = 0
    · simp [hp]
    · rw [Real.log_div hp (ne_of_gt (hq a)), hlog]
      ring
  simp_rw [ht]
  rw [Finset.sum_add_distrib, Finset.sum_add_distrib,
    ← Finset.mul_sum, ← Finset.sum_mul, sum_toReal]
  simp [entropy, meanCost]

/-- Gibbs laws maximize entropy minus priced cost; no existence of a
budget or resource law is inferred from packaging. -/
theorem gibbs_penalized_optimal (p q : PMF α) (u : α → ℝ) (lam logZ : ℝ)
    (hq : ∀ a, 0 < (q a).toReal)
    (hlog : ∀ a, Real.log (q a).toReal = -lam * u a - logZ) :
    entropy p - lam * meanCost p u ≤ entropy q - lam * meanCost q u := by
  have hp := finiteKL_nonneg p q
  rw [gibbs_KL_identity p q u lam logZ hq hlog] at hp
  have heq := gibbs_KL_identity q q u lam logZ hq hlog
  rw [finiteKL_self] at heq
  linarith

/-- A nonnegative Gibbs price, feasibility, and complementary slackness
certify the constrained entropy optimum. These resource assumptions are explicit. -/
theorem gibbs_budget_optimal (p q : PMF α) (u : α → ℝ) (lam logZ b : ℝ)
    (hq : ∀ a, 0 < (q a).toReal)
    (hlog : ∀ a, Real.log (q a).toReal = -lam * u a - logZ)
    (hlam : 0 ≤ lam) (hp : meanCost p u ≤ b)
    (_hq_feasible : meanCost q u ≤ b)
    (hslack : lam * (meanCost q u - b) = 0) :
    entropy p ≤ entropy q := by
  have hopt := gibbs_penalized_optimal p q u lam logZ hq hlog
  have hc := mul_le_mul_of_nonneg_left hp hlam
  nlinarith

/-- The price gives the supporting inequality for the entropy value as the
budget changes. At a differentiable value function this yields V'(b)=lam. -/
theorem gibbs_budget_shadow_bound (p q : PMF α) (u : α → ℝ)
    (lam logZ b b' : ℝ)
    (hq : ∀ a, 0 < (q a).toReal)
    (hlog : ∀ a, Real.log (q a).toReal = -lam * u a - logZ)
    (hlam : 0 ≤ lam) (hp : meanCost p u ≤ b')
    (hslack : lam * (meanCost q u - b) = 0) :
    entropy p ≤ entropy q + lam * (b' - b) := by
  have hopt := gibbs_penalized_optimal p q u lam logZ hq hlog
  have hc := mul_le_mul_of_nonneg_left hp hlam
  nlinarith
/-- The Gibbs optimizer certificate is inhabited for every finite nonempty
carrier and finite real price. Only feasibility and slackness remain hypotheses. -/
theorem maxent_budget_optimal [Nonempty α] (p : PMF α) (u : α → ℝ)
    (lam b : ℝ) (hlam : 0 ≤ lam) (hp : meanCost p u ≤ b)
    (hq : meanCost (gibbsPMF u lam) u ≤ b)
    (hslack : lam * (meanCost (gibbsPMF u lam) u - b) = 0) :
    entropy p ≤ entropy (gibbsPMF u lam) :=
  gibbs_budget_optimal p (gibbsPMF u lam) u lam
    (Real.log (partitionFunction u lam)) b
    (gibbsPMF_pos u lam) (gibbsPMF_log u lam) hlam hp hq hslack


end CurrencyMorphism
