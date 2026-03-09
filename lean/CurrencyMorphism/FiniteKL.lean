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

lemma fiber_jensen [Fintype α] [DecidableEq α] [DecidableEq β]
    (f : α → β) (p q : PMF α)
    (hq : ∀ a, 0 < (q a).toReal)
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
    have hqa_ne : ∀ a, (q a).toReal ≠ 0 := fun a => ne_of_gt (hq a)
    calc
      ∑ a ∈ t, w a * x a = ∑ a ∈ t, (p a).toReal / Qb := by
        refine Finset.sum_congr rfl ?_
        intro a ha
        have hqa_ne_a : (q a).toReal ≠ 0 := hqa_ne a
        dsimp [w, x]
        field_simp [hQb_ne, hqa_ne_a]
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


theorem finiteKL_map_le [Fintype α] [Fintype β] [DecidableEq α] [DecidableEq β]
    (f : α → β) (hf : Function.Surjective f)
    (p q : PMF α)
    (hq : ∀ a, 0 < (q a).toReal) :
    finiteKL (PMF.map f p) (PMF.map f q) ≤ finiteKL p q := by
  have hmain :
      ∀ b,
      (∑ a ∈ fiber f b, (q a).toReal) *
        klFun ((∑ a ∈ fiber f b, (p a).toReal) / (∑ a ∈ fiber f b, (q a).toReal))
        ≤
      ∑ a ∈ fiber f b, (q a).toReal * klFun ((p a).toReal / (q a).toReal) := by
    intro b
    have h := map_toReal_pos f hf q hq b
    have hQ : 0 < ∑ a ∈ fiber f b, (q a).toReal := by
      rw [map_toReal_eq_sum_fiber] at h
      exact h
    exact fiber_jensen f p q hq b hQ

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
