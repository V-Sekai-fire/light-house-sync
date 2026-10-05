/-
  QuestLHSync — identifiability (observability) of the lighthouse → headset alignment.

  Self-contained Lean 4 (no Mathlib).  Models the alignment the driver solves:

      p_headset  =  R • p_reference  +  t

  where `R` is a yaw rotation about gravity (1 DOF) and `t` a translation (3 DOF): 4 DOF total.
  Both spaces are gravity-aligned, so the rotation is a pure yaw; we model it only through the
  property the solver actually relies on — that it is a *linear* map of positions (`LinAct`).

  What the driver's observations give, and what we prove here:

    • A base station seen by the headset cameras contributes a correspondence
      (its pose in the reference frame ↔ its pose in headset space).

    • `ident_oriented`  —  ONE station whose *orientation* is observed pins (R,t) uniquely.
       This is exactly what a 1-station solver must capture: the base station's orientation,
       recoverable from its rotor sweep phase.  ⇒ k = 1 is doable *in principle*.

    • `baseline_rot` / `ident_baseline`  —  TWO (or more) stations pin R on their baseline
       (translation `t` cancels); with a non-degenerate baseline that determines R, then t.
       ⇒ k = 2, 3, 4 are doable with the *current* ray-only observations.  (QuestLHSync already
       stores stations in a map, so 3 and 4 are the same "fit of all" code path as 2.)

    • `one_point_underdetermined`  —  a single *position*-only observation leaves R completely
       free: for ANY rotation there is a translation reproducing the same observation.  This is
       why 1 station needs orientation, and why 2 coplanar stations seen edge-on keep the yaw
       "mirror" the driver's log reports until a 3rd station or a worn/held device breaks it.

  The concrete `Int` / `Unit` instances at the end witness that the axioms are consistent
  (the theorems are not vacuous).
-/

namespace QLHS
universe u

/-- Positions / translations: a commutative group, axioms stated explicitly (no Mathlib). -/
class CommPos (V : Type u) extends Add V, Neg V, Zero V where
  add_assoc : ∀ a b c : V, a + b + c = a + (b + c)
  add_comm  : ∀ a b : V, a + b = b + a
  zero_add  : ∀ a : V, (0 : V) + a = a
  add_neg   : ∀ a : V, a + -a = (0 : V)

namespace CommPos
variable {V : Type u} [CommPos V]

theorem add_zero (a : V) : a + 0 = a := by
  rw [add_comm]; exact zero_add a

theorem neg_add (a : V) : -a + a = (0 : V) := by
  rw [add_comm]; exact add_neg a

theorem add_left_cancel {x a b : V} (h : x + a = x + b) : a = b := by
  have h2 : -x + (x + a) = -x + (x + b) := by rw [h]
  rw [← add_assoc (-x) x a, ← add_assoc (-x) x b, neg_add, zero_add, zero_add] at h2
  exact h2

theorem add_right_cancel {a b x : V} (h : a + x = b + x) : a = b := by
  have h2 : a + x + -x = b + x + -x := by rw [h]
  rw [add_assoc, add_neg, add_zero, add_assoc, add_neg, add_zero] at h2
  exact h2

/-- `(x − y) + (y + c) = x + c`: the cancellation that eliminates the translation. -/
theorem collapse (x y c : V) : (x + -y) + (y + c) = x + c := by
  rw [add_assoc x (-y) (y + c), ← add_assoc (-y) y c, neg_add, zero_add]

end CommPos

/-- A linear action of the yaw group `G` on positions `V` (a rotation is a linear map). -/
class LinAct (G : Type u) (V : Type u) [CommPos V] where
  smul : G → V → V
  smul_add : ∀ g a b, smul g (a + b) = smul g a + smul g b
  smul_neg : ∀ g a, smul g (-a) = -(smul g a)

section Align
open CommPos LinAct
variable {G : Type u} {V : Type u} [CommPos V] [LinAct G V]

/-- The alignment the driver applies: `R • p + t`. -/
def align (R : G) (t p : V) : V := LinAct.smul R p + t

@[simp] theorem align_def (R : G) (t p : V) : align R t p = LinAct.smul R p + t := rfl

/--
  **Two stations (or more) pin the rotation on their baseline.**
  From two landmark correspondences the translation cancels, leaving the rotation's action on
  the baseline `a − b` determined.  (With a non-degenerate baseline this fixes `R`; this is the
  k = 2, 3, 4 case.)
-/
theorem baseline_rot {R R' : G} {t t' : V} {a b : V}
    (ha : align R t a = align R' t' a) (hb : align R t b = align R' t' b) :
    LinAct.smul R (a + -b) = LinAct.smul R' (a + -b) := by
  rw [align_def, align_def] at ha hb
  rw [LinAct.smul_add, LinAct.smul_neg, LinAct.smul_add, LinAct.smul_neg]
  apply add_right_cancel (x := LinAct.smul R b + t)
  rw [collapse, hb, collapse]
  exact ha

/--
  **One oriented station determines the whole alignment.**
  If the station's orientation is observed (so `R = R'`) and one position matches, then the
  translation is forced too.  ⇒ a single base station is sufficient *in principle* — provided
  the solver observes its orientation (rotor sweep phase).
-/
theorem ident_oriented {R R' : G} {t t' : V} (p : V)
    (hR : R = R') (hp : align R t p = align R' t' p) : R = R' ∧ t = t' := by
  refine ⟨hR, ?_⟩
  rw [align_def, align_def, hR] at hp
  exact add_left_cancel hp

/--
  **Two stations with a non-degenerate baseline determine the alignment.**
  `nondeg` packages "the baseline is not along the yaw-mirror axis": equal action on the
  baseline implies equal rotation.  Then the translation follows.
-/
theorem ident_baseline {R R' : G} {t t' : V} {a b : V}
    (ha : align R t a = align R' t' a) (hb : align R t b = align R' t' b)
    (nondeg : LinAct.smul R (a + -b) = LinAct.smul R' (a + -b) → R = R') :
    R = R' ∧ t = t' := by
  have hRR : R = R' := nondeg (baseline_rot ha hb)
  refine ⟨hRR, ?_⟩
  rw [align_def, align_def, hRR] at ha
  exact add_left_cancel ha

/--
  **One position alone is not enough.**
  For ANY rotation `R₂` there is a translation `t₂` reproducing the single observation of `p`:
  so a lone position cannot constrain the rotation.  This is the formal reason 1 station needs
  orientation, and 2 mirror-symmetric stations stay ambiguous until a 3rd (or a held device).
-/
theorem one_point_underdetermined (R R₂ : G) (t p : V) :
    ∃ t₂ : V, align R₂ t₂ p = align R t p := by
  refine ⟨(LinAct.smul R p + t) + -(LinAct.smul R₂ p), ?_⟩
  rw [align_def, align_def]
  rw [add_comm (LinAct.smul R p + t) (-(LinAct.smul R₂ p)), ← add_assoc, add_neg, zero_add]

end Align

/-! ### The axioms are consistent (witnessing instances, so nothing above is vacuous). -/

instance : CommPos Int where
  add := (· + ·)
  neg := (- ·)
  zero := 0
  add_assoc := by intro a b c; omega
  add_comm := by intro a b; omega
  zero_add := by intro a; omega
  add_neg := by intro a; omega

/-- The identity yaw action (trivial rotation group) on 1-D positions. -/
instance : LinAct Unit Int where
  smul := fun _ v => v
  smul_add := by intro _ a b; rfl
  smul_neg := by intro _ a; rfl

-- Sanity: the theorems instantiate at a concrete model.
example (p : Int) : ∃ t₂ : Int, align (G := Unit) () t₂ p = align () (0 : Int) p :=
  one_point_underdetermined (G := Unit) () () 0 p

#check @ident_oriented
#check @ident_baseline
#check @baseline_rot
#check @one_point_underdetermined

end QLHS
