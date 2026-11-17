# CLaM-H Global PairAcc evaluator

The evaluator contains the CLaM-H text, motion and movement encoders, the
Motion-X body/hand representation converter, CLIP tokenizer data, and the
published 623D normalization statistics. Upstream sources are identified in
the repository's `NOTICE.md`; the CLaM-H and CLIP MIT notices are retained here.

Call `scripts/evaluate_pairacc.py` from the repository root, or supply PairAcc
resources to `scripts/evaluate.py`. Both entry points use this implementation.
The checkpoint must contain `text_encoder`, `motion_encoder` and
`movement_encoder` state dictionaries. A MotionSpeech-adapted checkpoint is
needed for the paper's evaluator. SMPL-X assets and checkpoints are external.

The conversion restores root orientation and translation, selects 52 body/hand
joints at 197 common timestamps, and produces a 196-by-623 Motion-X sequence.
The scorer normalizes motion embeddings and pools normalized text chunks of at
most 32 words. For each level and seed it compares the matching caption score
against all other non-equivalent captions. Wins receive one point and exact
ties half a point. Caption equivalence ignores case and repeated whitespace.
The percentage is computed over all valid ordered comparisons; results are
then averaged across seeds separately for L1 and L2. A group with no valid
negative comparisons fails rather than receiving an invented score.

The additional `motionspeech_eval` modules provide dataset preparation,
evaluator adaptation, and evaluation of prepared features. Dataset preparation
and generated-motion scoring retain their explicit skeleton conventions; use
the generated-motion entry point for DuoFlow outputs.
