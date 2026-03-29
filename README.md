# Hebbian Cayley Workspace LM

A language model whose core dynamics learn without backpropagation.

The workspace state evolves through Cayley-orthogonal rotations — the brain parameters update via Hebbian learning (outer products modulated by prediction quality), while the embedding and classifier train with standard gradient descent. The two regimes run simultaneously from step 1.

## Architecture

A flat workspace vector `z ∈ R^1024` processes tokens sequentially. At each position:

1. **Inject**: gated update mixes the token embedding into the workspace
2. **Settle**: 5 ticks of dynamics — `z ← rms_norm(z + 0.1 * (scale * W @ gelu(z) + bias))`
3. **Classify**: cosine similarity between a linear projection of `z` and the embedding table

The dynamics matrix `W` is the mean of 16 Cayley bases. Each base is the Cayley transform `(I - A)(I + A)^{-1}` of a skew-symmetric generator `A`. The Cayley map guarantees all eigenvalues lie on the unit circle — the workspace rotates information without amplifying or destroying it.

`scale` and `bias` are token-conditioned (projected from the token embedding). Scale acts as per-dimension precision gating. Bias shifts the attractor location per token.

**Parameter split**:
- Gradient-trained (sensor/motor): embedding (49152×256), positional embedding, injection gate, classifier — ~13M params
- Hebbian-trained (brain): 16 skew-symmetric generators (16×1024×1024), scale/bias projections — ~17M params
- Total: ~25M

## Hebbian Learning Rule

After each token, the model computes a quality signal: cosine similarity between its predicted next-token embedding and the actual next-token embedding. This scalar, centered by subtracting the batch mean, modulates rank-1 outer product updates to the Cayley generators:

```
dA_i += quality * alpha_i * outer(z_post, gelu(z_pre))
dA_i ← (dA_i - dA_i^T) / 2     # project to skew-symmetric
```

The skew projection ensures the generators remain valid Cayley inputs. The quality centering makes the update contrastive — above-average predictions reinforce, below-average weaken.

`stop_gradient` on the brain parameters prevents gradient from flowing into the Cayley generators, scale, and bias projections. The gradient still flows through the workspace *state* (for embedding training), just not through the brain *parameters*.

## Training

Trained for 100K steps on WikiText-103 (289K sequences, 200 tokens each).

- Batch size: 16
- Sensor LR: 3e-4 (Adam)
- Hebbian LR: 5e-4 (applied every step)
- Settle: T=5 ticks, dt=0.1
- Tokenizer: StarCoder2 (49152 vocab)
- Hardware: RTX 5090, ~14 hours at 2.0 steps/s

## Results

| Step | PPL | Quality |
|------|-----|---------|
| 10K | ~75 | 0.18 |
| 30K | 46.8 | 0.30 |
| 60K | ~30 | 0.32 |
| 100K | **23.0** (best) | 0.32 |

For reference, the same architecture trained entirely with backpropagation (gradient through the brain) reaches PPL 37.7 at 30K steps. The Hebbian brain beats full BPTT given sufficient training — the 80,000:1 information compression (scalar quality signal vs full gradient) is overcome by volume of updates.

The quality signal (cosine similarity between readout and target) grows from 0.05 to 0.32 over training, confirming the Hebbian brain learns to produce increasingly accurate next-token predictions.

### Generation samples (temperature=0.8, top-k=40)

> **"The history of"** → The history of the game was criticised for the development of the game's development and the development of the game, creating a world map, such as the Mint tropical, and a Fordi inspired strategy.

> **"The city of London"** → The city of London in 1878, the Hudson, the site of the island of Baltimore, Lamuna Railway and the Royal Navy. The first single island is the largest opening (from 1843 to 1855).

> **"Scientists discovered"** → Scientists discovered in 2008, "The Pittsburgh" (1998), which became the most part of the Mexican state at a time when a 10,000 year old murder was destroyed.

Grammatical English with dates, proper nouns, and clause structure. Wikipedia-flavored (trained on wiki). Repetitive without further techniques like scheduled sampling.

## Dynamics visualization

After 100K steps of Hebbian learning, the 16 Cayley bases develop organized eigenvalue clusters on the unit circle (vs uniform distribution at initialization). The effective W (mean of bases) has spectral radius 0.923 — slight contraction from averaging orthogonal matrices, providing natural damping. Workspace trajectories during the 5-tick settle are distinct per prompt, confirming the dynamics differentiate between inputs.

## What didn't work

This architecture emerged from 15 experiments (h01-h15). Key failures that informed the design:

- No geometric constraint → workspace collapses (h01)
- Anti-symmetric-only Hebbian → 0.15% accuracy (h05 first attempt)
- Phase-switching gradient→Hebbian → catastrophic forgetting (h08)
- Block-partitioned workspace → fragments information for NLP (h09)
- Multirate (input/brain/memory loops) → added complexity without benefit at matched steps (h10)
- Identity gradient passthrough → embedding can't learn how brain transforms its output (h12)

## Files

- `model.py` — full model, training loop, generation
- `h14_100k.eqx` — trained checkpoint (100K steps, best PPL 23.0)

## Dependencies

```
jax jaxlib equinox optax wandb transformers
```

Data: WikiText-103, tokenized with StarCoder2, sequences truncated to 200 tokens. Precomputed as `precomputed_targets/wiki_long_512.npy`.

## wandb

Training run: https://wandb.ai/adew-me/iPC-Chain/runs/u3x52kjg
