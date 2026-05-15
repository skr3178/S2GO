Select 2–3 diverse parts (e.g., Part‑01 + another from a different location/weather).
If you use 1/3 of the data, train for 3× the epochs (e.g., 36 instead of 12) to keep total gradient updates similar.
If you use 1/3 of the data, train for 3× the epochs (e.g., 36 instead of 12) to keep total gradient updates similar.

Diversity-filter your scenes: cherry-picking gives a "free" 2-3 mIoU at zero compute cost. Highest ROI.
2 parts is the sweet spot: 1 part is too narrow (memorization regime); 3+ parts has diminishing returns vs compute.
Always do the Stage 1 vs from-scratch A/B: the gap is the meaningful result. Absolute numbers without the gap are uninterpretable.
Use grad accumulation for effective batch=16: you're already doing this; keep it.
Don't go beyond 12 epochs: overfitting beyond that point on any subset.

train_datasets: ["01", "02", "03"]
scene_filter: diverse_curated
train_split: 80%
val_split: 20%
unseen_val_dataset: "04"

epochs: 12
model_size: medium
precision: amp/bf16
gradient_accumulation: true
save_every: 1 epoch