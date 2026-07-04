# Confidently wrong — and caught

Cases where the base model was wrong with high raw confidence, but the uncertainty layer flagged them: conformal set size >= 2 (alpha = 0.1) or calibrated confidence below the abstention threshold (0.856, the cutoff for 70% coverage).

## Example 1

**Question:** Which of these is an example of water changing from water vapor to liquid?

- A. moisture forming on a mirror when you breathe on it  <- correct
- B. sweat forming on your body when you exercise
- C. ice cubes melting when you put them in a warm liquid  <- model's answer
- D. rivers drying up during a very hot summer

Raw confidence **0.95** — and wrong. Flagged because conformal set {A, C} (size 2) is not a singleton; and calibrated confidence 0.74 < abstention threshold 0.86 -> **abstain / escalate**.

## Example 2

**Question:** A pitcher throws a 0.15 kg baseball at 43 40 m/s towards the catcher. What is the momentum of the baseball while moving at 40 m/s?

- A. 0.025 kg x m/s
- B. 3.8 kg x m/s  <- model's answer
- C. 6.0 kg x m/s  <- correct
- D. 270 kg x m/s

Raw confidence **0.95** — and wrong. Flagged because conformal set {B, C} (size 2) is not a singleton; and calibrated confidence 0.70 < abstention threshold 0.86 -> **abstain / escalate**.

## Example 3

**Question:** Single-celled organisms that cause disease can be found in which domains?

- A. Archaea and Eukarya
- B. Bacteria and Eukarya  <- correct
- C. Archaea and Bacteria  <- model's answer
- D. Archaea, Bacteria, and Eukarya

Raw confidence **0.95** — and wrong. Flagged because conformal set {C, D} (size 2) is not a singleton; and calibrated confidence 0.73 < abstention threshold 0.86 -> **abstain / escalate**.
