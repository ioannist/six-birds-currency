import Mathlib.Probability.ProbabilityMassFunction.Constructions
import Mathlib.InformationTheory.KullbackLeibler.KLFun
import Mathlib.Analysis.Convex.Jensen
import Mathlib.Probability.Distributions.Uniform

open scoped BigOperators

namespace CurrencyMorphism

open InformationTheory

variable {α β : Type}

def fiber [Fintype α] [DecidableEq α] [DecidableEq β] (f : α → β) (b : β) : Finset α :=
  Finset.univ.filter (fun a => f a = b)

/-- Real finite KL form. It agrees with ordinary KL when `p` is supported
on `q`. Without that hypothesis, zero denominators use Lean's totalized
real division and this definition must NOT be interpreted as extended KL. -/
noncomputable def finiteKL [Fintype α] [DecidableEq α] (p q : PMF α) : ℝ :=
  ∑ a, (q a).toReal * klFun ((p a).toReal / (q a).toReal)

lemma map_apply_eq_sum_filter [Fintype α] [DecidableEq β]
    (f : α → β) (r : PMF α) (b : β) :
    (PMF.map f r) b = ∑ a with f a = b, r a := by
  rw [PMF.map_apply, tsum_fintype, Finset.sum_filter]
  refine Finset.sum_congr rfl ?_
  intro a ha
  by_cases h : b = f a
  · simp [h]
  · have h' : f a ≠ b := by simpa [eq_comm] using h
    simp [h, h']

lemma map_toReal_eq_sum_fiber [Fintype α] [DecidableEq α] [DecidableEq β]
    (f : α → β) (r : PMF α) (b : β) :
    ((PMF.map f r) b).toReal = ∑ a ∈ fiber f b, (r a).toReal := by
  rw [map_apply_eq_sum_filter]
  change (∑ a ∈ fiber f b, r a).toReal = ∑ a ∈ fiber f b, (r a).toReal
  exact ENNReal.toReal_sum fun a ha => PMF.apply_ne_top r a

lemma map_toReal_pos [Fintype α] [DecidableEq α] [DecidableEq β]
    (f : α → β) (hf : Function.Surjective f)
    (q : PMF α) (hq : ∀ a, 0 < (q a).toReal) (b : β) :
    0 < ((PMF.map f q) b).toReal := by
  rcases hf b with ⟨a0, ha0⟩
  rw [map_toReal_eq_sum_fiber]
  have hmem : a0 ∈ fiber f b := by simp [fiber, ha0]
  have hle : (q a0).toReal ≤ ∑ a ∈ fiber f b, (q a).toReal := by
    exact Finset.single_le_sum (fun a ha => ENNReal.toReal_nonneg) hmem
  exact lt_of_lt_of_le (hq a0) hle

lemma fiber_jensen_of_support [Fintype α] [DecidableEq α] [DecidableEq β]
    (f : α → β) (p q : PMF α)
    (hsupport : ∀ a, (q a).toReal = 0 → (p a).toReal = 0)
    (b : β)
    (hQ : 0 < ∑ a ∈ fiber f b, (q a).toReal) :
    (∑ a ∈ fiber f b, (q a).toReal) *
      klFun ((∑ a ∈ fiber f b, (p a).toReal) / (∑ a ∈ fiber f b, (q a).toReal))
      ≤
    ∑ a ∈ fiber f b, (q a).toReal * klFun ((p a).toReal / (q a).toReal) := by
  let t : Finset α := fiber f b
  let Qb : ℝ := ∑ a ∈ t, (q a).toReal
  let Pb : ℝ := ∑ a ∈ t, (p a).toReal
  let w : α → ℝ := fun a => (q a).toReal / Qb
  let x : α → ℝ := fun a => (p a).toReal / (q a).toReal

  have hQb : 0 < Qb := by simpa [Qb, t] using hQ
  have hQb_ne : Qb ≠ 0 := ne_of_gt hQb

  have h0 : ∀ a ∈ t, 0 ≤ w a := by
    intro a ha
    exact div_nonneg ENNReal.toReal_nonneg (le_of_lt hQb)

  have h1 : ∑ a ∈ t, w a = 1 := by
    calc
      ∑ a ∈ t, w a = (∑ a ∈ t, (q a).toReal) / Qb := by
        simpa [w] using (Finset.sum_div t (fun a => (q a).toReal) Qb).symm
      _ = Qb / Qb := by simp [Qb]
      _ = 1 := by field_simp [hQb_ne]

  have hmem : ∀ a ∈ t, x a ∈ Set.Ici (0 : ℝ) := by
    intro a ha
    have hqa_nonneg : 0 ≤ (q a).toReal := ENNReal.toReal_nonneg
    have hpa_nonneg : 0 ≤ (p a).toReal := ENNReal.toReal_nonneg
    exact div_nonneg hpa_nonneg hqa_nonneg

  have hj :
      klFun (∑ a ∈ t, w a * x a) ≤ ∑ a ∈ t, w a * klFun (x a) := by
    simpa [smul_eq_mul] using
      (InformationTheory.convexOn_klFun.map_sum_le (t := t) (w := w) (p := x) h0 h1 hmem)

  have hxsum : ∑ a ∈ t, w a * x a = Pb / Qb := by
    calc
      ∑ a ∈ t, w a * x a = ∑ a ∈ t, (p a).toReal / Qb := by
        refine Finset.sum_congr rfl ?_
        intro a ha
        by_cases hqa : (q a).toReal = 0
        · simp [w, x, hqa, hsupport a hqa]
        · dsimp [w, x]
          field_simp [hQb_ne, hqa]
      _ = (∑ a ∈ t, (p a).toReal) / Qb := by
        simpa using (Finset.sum_div t (fun a => (p a).toReal) Qb).symm
      _ = Pb / Qb := by simp [Pb]

  have hrhs : Qb * (∑ a ∈ t, w a * klFun (x a)) =
      ∑ a ∈ t, (q a).toReal * klFun ((p a).toReal / (q a).toReal) := by
    calc
      Qb * (∑ a ∈ t, w a * klFun (x a)) = ∑ a ∈ t, Qb * (w a * klFun (x a)) := by
        simpa using (Finset.mul_sum t (fun a => w a * klFun (x a)) Qb)
      _ = ∑ a ∈ t, (q a).toReal * klFun ((p a).toReal / (q a).toReal) := by
        refine Finset.sum_congr rfl ?_
        intro a ha
        have hw : Qb * w a = (q a).toReal := by
          dsimp [w]
          field_simp [hQb_ne]
        calc
          Qb * (w a * klFun (x a)) = (Qb * w a) * klFun (x a) := by ring
          _ = (q a).toReal * klFun (x a) := by simp [hw]
          _ = (q a).toReal * klFun ((p a).toReal / (q a).toReal) := by rfl

  have hmul := mul_le_mul_of_nonneg_left hj (le_of_lt hQb)
  calc
    (∑ a ∈ fiber f b, (q a).toReal) *
      klFun ((∑ a ∈ fiber f b, (p a).toReal) / (∑ a ∈ fiber f b, (q a).toReal))
        = Qb * klFun (Pb / Qb) := by simp [Qb, Pb, t]
    _ = Qb * klFun (∑ a ∈ t, w a * x a) := by rw [hxsum]
    _ ≤ Qb * (∑ a ∈ t, w a * klFun (x a)) := hmul
    _ = ∑ a ∈ t, (q a).toReal * klFun ((p a).toReal / (q a).toReal) := hrhs
    _ = ∑ a ∈ fiber f b, (q a).toReal * klFun ((p a).toReal / (q a).toReal) := by
      simp [t]


theorem finiteKL_map_le_of_support [Fintype α] [Fintype β] [DecidableEq α] [DecidableEq β]
    (f : α → β)
    (p q : PMF α)
    (hsupport : ∀ a, (q a).toReal = 0 → (p a).toReal = 0) :
    finiteKL (PMF.map f p) (PMF.map f q) ≤ finiteKL p q := by
  have hmain :
      ∀ b,
      (∑ a ∈ fiber f b, (q a).toReal) *
        klFun ((∑ a ∈ fiber f b, (p a).toReal) / (∑ a ∈ fiber f b, (q a).toReal))
        ≤
      ∑ a ∈ fiber f b, (q a).toReal * klFun ((p a).toReal / (q a).toReal) := by
    intro b
    by_cases hQ : 0 < ∑ a ∈ fiber f b, (q a).toReal
    · exact fiber_jensen_of_support f p q hsupport b hQ
    · have hzero : (∑ a ∈ fiber f b, (q a).toReal) = 0 :=
        le_antisymm (le_of_not_gt hQ)
          (Finset.sum_nonneg (fun _ _ => ENNReal.toReal_nonneg))
      rw [hzero, zero_mul]
      exact Finset.sum_nonneg fun a _ => mul_nonneg ENNReal.toReal_nonneg
        (klFun_nonneg (div_nonneg ENNReal.toReal_nonneg ENNReal.toReal_nonneg))

  have hL :
      finiteKL (PMF.map f p) (PMF.map f q)
      =
      ∑ b,
        (∑ a ∈ fiber f b, (q a).toReal) *
          klFun ((∑ a ∈ fiber f b, (p a).toReal) / (∑ a ∈ fiber f b, (q a).toReal)) := by
    unfold finiteKL
    refine Finset.sum_congr rfl ?_
    intro b hb
    rw [map_toReal_eq_sum_fiber (f := f) (r := q) (b := b)]
    rw [map_toReal_eq_sum_fiber (f := f) (r := p) (b := b)]

  have hR :
      finiteKL p q
      =
      ∑ b, ∑ a ∈ fiber f b, (q a).toReal * klFun ((p a).toReal / (q a).toReal) := by
    unfold finiteKL
    symm
    simpa [fiber] using (Finset.sum_fiberwise (s := Finset.univ) (g := f)
      (f := fun a => (q a).toReal * klFun ((p a).toReal / (q a).toReal)))

  rw [hL, hR]
  exact Finset.sum_le_sum (fun b hb => hmain b)


/-- Finite KL data processing with full-support reference and any deterministic map. -/
theorem finiteKL_map_le_of_pos [Fintype α] [Fintype β] [DecidableEq α] [DecidableEq β]
    (f : α → β) (p q : PMF α) (hq : ∀ a, 0 < (q a).toReal) :
    finiteKL (PMF.map f p) (PMF.map f q) ≤ finiteKL p q :=
  finiteKL_map_le_of_support f p q (fun a h => (ne_of_gt (hq a) h).elim)

/-- Original Jensen API. -/
lemma fiber_jensen [Fintype α] [DecidableEq α] [DecidableEq β]
    (f : α → β) (p q : PMF α) (hq : ∀ a, 0 < (q a).toReal)
    (b : β) (hQ : 0 < ∑ a ∈ fiber f b, (q a).toReal) :
    (∑ a ∈ fiber f b, (q a).toReal) *
      klFun ((∑ a ∈ fiber f b, (p a).toReal) / (∑ a ∈ fiber f b, (q a).toReal)) ≤
      ∑ a ∈ fiber f b, (q a).toReal * klFun ((p a).toReal / (q a).toReal) :=
  fiber_jensen_of_support f p q (fun a h => (ne_of_gt (hq a) h).elim) b hQ

/-- Original API, retained for downstream consumers. Surjectivity is unnecessary. -/
theorem finiteKL_map_le [Fintype α] [Fintype β] [DecidableEq α] [DecidableEq β]
    (f : α → β) (_hf : Function.Surjective f)
    (p q : PMF α) (hq : ∀ a, 0 < (q a).toReal) :
    finiteKL (PMF.map f p) (PMF.map f q) ≤ finiteKL p q :=
  finiteKL_map_le_of_pos f p q hq

lemma finiteKL_nonneg [Fintype α] [DecidableEq α] (p q : PMF α) :
    0 ≤ finiteKL p q := by
  exact Finset.sum_nonneg fun a _ =>
    mul_nonneg ENNReal.toReal_nonneg
      (klFun_nonneg (div_nonneg ENNReal.toReal_nonneg ENNReal.toReal_nonneg))

@[simp] lemma finiteKL_self [Fintype α] [DecidableEq α] (p : PMF α) :
    finiteKL p p = 0 := by
  unfold finiteKL
  apply Finset.sum_eq_zero
  intro a _
  by_cases h : (p a).toReal = 0
  · simp [h]
  · simp [div_self h, klFun_one]

lemma sum_toReal [Fintype α] (p : PMF α) : ∑ a, (p a).toReal = 1 := by
  rw [← ENNReal.toReal_sum (fun a _ => PMF.apply_ne_top p a)]
  have h : ∑ a, p a = 1 := by simpa [tsum_fintype] using p.tsum_coe
  simp [h]

/-- Semantic bridge to the conventional `sum p log(p/q)` formula, including
zero numerator terms under the convention `0 log 0 = 0`. -/
theorem finiteKL_eq_sum_log_of_support [Fintype α] [DecidableEq α]
    (p q : PMF α) (hsupport : ∀ a, (q a).toReal = 0 → (p a).toReal = 0) :
    finiteKL p q = ∑ a, (p a).toReal * Real.log ((p a).toReal / (q a).toReal) := by
  have hterm : ∀ a, (q a).toReal * klFun ((p a).toReal / (q a).toReal) =
      (p a).toReal * Real.log ((p a).toReal / (q a).toReal) +
        (q a).toReal - (p a).toReal := by
    intro a
    by_cases hqa : (q a).toReal = 0
    · simp [hqa, hsupport a hqa]
    · unfold klFun
      field_simp [hqa]
  unfold finiteKL
  simp_rw [hterm]
  rw [Finset.sum_sub_distrib, Finset.sum_add_distrib, sum_toReal, sum_toReal]
  ring

/-- Full-support specialization of the ordinary KL bridge. -/
theorem finiteKL_eq_sum_log [Fintype α] [DecidableEq α]
    (p q : PMF α) (hq : ∀ a, 0 < (q a).toReal) :
    finiteKL p q = ∑ a, (p a).toReal * Real.log ((p a).toReal / (q a).toReal) :=
  finiteKL_eq_sum_log_of_support p q (fun a h => (ne_of_gt (hq a) h).elim)

/-- Coarse observation and reversal commute. The reference law need only
be positive here; the pushforward need not be onto its declared codomain. -/
theorem finiteKL_reversal_map_le_of_support [Fintype α] [Fintype β]
    [DecidableEq α] [DecidableEq β]
    (f : α → β) (r : α → α) (s : β → β)
    (hcomm : ∀ a, f (r a) = s (f a))
    (p : PMF α)
    (hsupport : ∀ a, ((PMF.map r p) a).toReal = 0 → (p a).toReal = 0) :
    finiteKL (PMF.map f p) (PMF.map s (PMF.map f p)) ≤
      finiteKL p (PMF.map r p) := by
  have heq : PMF.map f (PMF.map r p) = PMF.map s (PMF.map f p) := by
    simp only [PMF.map_comp]
    congr 1
    funext a
    exact hcomm a
  rw [← heq]
  exact finiteKL_map_le_of_support f p (PMF.map r p) hsupport

/-- Full-support specialization of reversal/coarsening data processing. -/
theorem finiteKL_reversal_map_le [Fintype α] [Fintype β]
    [DecidableEq α] [DecidableEq β]
    (f : α → β) (r : α → α) (s : β → β)
    (hcomm : ∀ a, f (r a) = s (f a))
    (p : PMF α) (hq : ∀ a, 0 < ((PMF.map r p) a).toReal) :
    finiteKL (PMF.map f p) (PMF.map s (PMF.map f p)) ≤
      finiteKL p (PMF.map r p) :=
  finiteKL_reversal_map_le_of_support f r s hcomm p
    (fun a h => (ne_of_gt (hq a) h).elim)

def reversePath {n : ℕ} (x : Fin n → α) : Fin n → α := fun i => x i.rev

def observePath {n : ℕ} (f : α → β) (x : Fin n → α) : Fin n → β := fun i => f (x i)

lemma reversePath_involutive (n : ℕ) : Function.Involutive (@reversePath α n) := by
  intro x
  funext i
  simp [reversePath]

/-- Actual finite-path specialization: pointwise observation commutes with
path-order reversal, and sparse reversible support is allowed. -/
theorem finiteKL_path_observe_le [Fintype α] [Fintype β]
    [DecidableEq α] [DecidableEq β] (n : ℕ) (f : α → β)
    (p : PMF (Fin n → α))
    (hsupport : ∀ x, ((PMF.map reversePath p) x).toReal = 0 → (p x).toReal = 0) :
    finiteKL (PMF.map (observePath f) p)
      (PMF.map reversePath (PMF.map (observePath f) p)) ≤
      finiteKL p (PMF.map reversePath p) :=
  finiteKL_reversal_map_le_of_support (observePath f) reversePath reversePath
    (fun _ => rfl) p hsupport


def collapse4 : Fin 4 → Bool := fun i => i.1 < 2

example :
    finiteKL
      (PMF.map collapse4 (PMF.uniformOfFintype (Fin 4)))
      (PMF.map collapse4 (PMF.uniformOfFintype (Fin 4)))
    ≤
    finiteKL
      (PMF.uniformOfFintype (Fin 4))
      (PMF.uniformOfFintype (Fin 4)) := by
  refine finiteKL_map_le collapse4 ?_ _ _ ?_
  · intro b
    rcases b with _ | _
    · refine ⟨⟨2, by decide⟩, by decide⟩
    · refine ⟨⟨0, by decide⟩, by decide⟩
  · intro a
    simp [PMF.uniformOfFintype_apply]

end CurrencyMorphism
