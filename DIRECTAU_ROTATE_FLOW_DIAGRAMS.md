# DirectAU-RotatE Flow Diagrams

## 1. Model Architecture Flow

```
Knowledge Graph Triples: (h, r, t)
        ↓
    ┌───────────────────────────────────────┐
    │ Load Knowledge Graph Data             │
    │ • Entity Dictionary (IDs)             │
    │ • Relation Dictionary (IDs)           │
    │ • Train/Valid/Test Triple Lists       │
    └───────────────────────────────────────┘
        ↓
    ┌──────────────────────────┐  ┌─────────────────────────┐
    │ Entity Embeddings (E)    │  │ Relation Embeddings (R) │
    │ Complex-valued           │  │ Real-valued             │
    │ Shape: [n_e, 2*d]        │  │ Shape: [n_r, d]         │
    │ ├─ Real part             │  │ ├─ Phase values         │
    │ └─ Imaginary part        │  │ └─ Unit complex nums    │
    └──────────────────────────┘  └─────────────────────────┘
         ↓                                  ↓
         └──────────┬──────────────────────┘
                    ↓
         ┌──────────────────────────────┐
         │ Relation Mask Embeddings (M) │
         │ Real-valued                  │
         │ Shape: [n_r, d]              │
         │ → Sigmoid → [0, 1] Masks     │
         └──────────────────────────────┘
                    ↓
         ┌──────────────────────────────────────┐
         │  Query Composition & Scoring         │
         │  ─────────────────────────────────── │
         │  Mode 1: Tail Prediction (h,r,?)    │
         │  • Normalize h (complex)            │
         │  • Apply mask m(r)                  │
         │  • Rotate by r                      │
         │  • Score against all entities       │
         │                                      │
         │  Mode 2: Head Prediction (?,r,t)    │
         │  • Normalize t (complex)            │
         │  • Apply mask m(r)                  │
         │  • Rotate by inverse r              │
         │  • Score against all entities       │
         └──────────────────────────────────────┘
                    ↓
         ┌──────────────────────────┐
         │  Loss Computation        │
         │  ─────────────────────── │
         │  • Alignment Loss        │
         │  • Uniformity Loss       │
         │  • Total Loss            │
         └──────────────────────────┘
                    ↓
         ┌──────────────────────────┐
         │ Backpropagation &        │
         │ Gradient Descent         │
         └──────────────────────────┘
```

## 2. Training Loop Flow (Detailed)

```
START TRAINING
    ↓
FOR epoch FROM 1 TO max_epochs:
    │
    ├─→ [Shuffle training data]
    │
    ├─→ FOR each batch in training data:
    │   │
    │   ├─→ [Get positive triple: (h, r, t)]
    │   │
    │   ├─→ [Determine Mode: head-batch or tail-batch]
    │   │   ├─ head-batch: Sample negative heads (?, r, t)
    │   │   └─ tail-batch: Sample negative tails (h, r, ?)
    │   │
    │   ├─→ [Encode Positive Triple]
    │   │   ├─ Get embeddings: E[h], R[r], E[t]
    │   │   ├─ (Shape: [batch, 2*d])
    │   │   └─ (Shape: [batch, d])
    │   │
    │   ├─→ [Compose Query from Head (Tail Prediction)]
    │   │   ├─ Normalize head: head_norm = norm(E[h])
    │   │   ├─ Get mask: mask_r = sigmoid(M[r])
    │   │   ├─ Apply mask: masked_head = head_norm * mask_r
    │   │   ├─ Get relation unit: (cos(phase), sin(phase))
    │   │   ├─ Complex multiply: q = masked_head * relation_unit
    │   │   └─ Normalize: query = norm(q)
    │   │
    │   ├─→ [Compute Positive Score]
    │   │   ├─ score_pos = score_complex(query, normalize(E[t]))
    │   │
    │   ├─→ [Encode Negative Samples]
    │   │   ├─ Get negative entity embeddings
    │   │   └─ (Same process as positive, different entities)
    │   │
    │   ├─→ [Compute Negative Scores]
    │   │   ├─ scores_neg = [score_complex(query, E[neg_i]) for all neg_i]
    │   │
    │   ├─→ [Alignment Loss]
    │   │   ├─ loss_align = max(0, gamma + score_pos - scores_neg)
    │   │   └─ loss_align = mean(loss_align)
    │   │
    │   ├─→ [Extract Unique Embeddings for Uniformity]
    │   │   ├─ unique_queries = unique([h,r] pairs in batch)
    │   │   ├─ unique_tails = unique(tail IDs)
    │   │   ├─ q_unique_emb = compose_queries(unique_queries)
    │   │   └─ t_unique_emb = normalize(E[unique_tails])
    │   │
    │   ├─→ [Uniformity Loss (if gamma_uni > 0)]
    │   │   ├─ loss_uni_q = uniformity_loss(q_unique_emb)
    │   │   ├─ loss_uni_t = uniformity_loss(t_unique_emb)
    │   │   └─ loss_uni = 0.5 * (loss_uni_q + loss_uni_t)
    │   │
    │   ├─→ [Total Loss]
    │   │   ├─ loss = loss_align 
    │   │   ├─        + gamma_uni * loss_uni
    │   │   └─        + gamma_neg * loss_align
    │   │
    │   ├─→ [Backward Pass]
    │   │   └─ loss.backward()
    │   │
    │   ├─→ [Optimizer Step]
    │   │   ├─ optimizer.step()
    │   │   └─ optimizer.zero_grad()
    │   │
    │   └─→ [Log progress every log_steps]
    │
    ├─→ [VALIDATION on Valid Set]
    │   ├─ Pre-encode all entities
    │   ├─ FOR each test triple:
    │   │   ├─ Tail prediction: rank entities for (h, r, ?)
    │   │   ├─ Head prediction: rank entities for (?, r, t)
    │   │   └─ Compute MR, MRR, Hits@K
    │   ├─ Average metrics
    │   └─ valid_perf = mean_mrr
    │
    ├─→ [Check Best Performance]
    │   ├─ IF valid_perf > best_valid_perf:
    │   │   ├─ best_valid_perf = valid_perf
    │   │   ├─ Save checkpoint (model weights)
    │   │   └─ patience_counter = 0
    │   └─ ELSE:
    │       └─ patience_counter += 1
    │
    ├─→ [Early Stopping]
    │   ├─ IF patience_counter >= patience_threshold:
    │   │   └─ BREAK training
    │   └─ ELSE:
    │       └─ Continue to next epoch
    │
    └─→ [TEST EVALUATION (if enabled)]
        ├─ Use best checkpoint
        ├─ Evaluate on full test set
        ├─ WITH and WITHOUT filtering
        └─ Report final metrics

LOAD BEST CHECKPOINT
    ↓
RUN FINAL TEST
    ↓
REPORT RESULTS
    ↓
END TRAINING
```

## 3. Query Composition: Tail Prediction (h, r, ?)

```
Input:
  head_id ∈ [0, n_entity)
  relation_id ∈ [0, n_relation)
  
        ↓
    
    ┌─────────────────────────────────────┐
    │ Step 1: Retrieve Embeddings         │
    │                                     │
    │ head_emb = E[head_id]               │
    │ Shape: [2*d] = [Re, Im]             │
    │                                     │
    │ relation_emb = R[relation_id]       │
    │ Shape: [d]                          │
    │                                     │
    │ relation_mask_emb = M[relation_id]  │
    │ Shape: [d]                          │
    └─────────────────────────────────────┘
        ↓
    ┌─────────────────────────────────────┐
    │ Step 2: Normalize Head              │
    │                                     │
    │ head_re, head_im ← chunk(head_emb)  │
    │ norm = √(Re² + Im²)                 │
    │ head_re_norm = head_re / norm       │
    │ head_im_norm = head_im / norm       │
    │                                     │
    │ Result: Normalized on unit circle   │
    └─────────────────────────────────────┘
        ↓
    ┌─────────────────────────────────────┐
    │ Step 3: Compute & Apply Mask        │
    │                                     │
    │ mask = sigmoid(relation_mask_emb)   │
    │ Shape: [d] → values in [0, 1]       │
    │                                     │
    │ head_re_masked = head_re_norm * mask│
    │ head_im_masked = head_im_norm * mask│
    │                                     │
    │ Effect: Filter/emphasize dimensions │
    └─────────────────────────────────────┘
        ↓
    ┌─────────────────────────────────────┐
    │ Step 4: Compute Relation Unit       │
    │                                     │
    │ phase = sigmoid(relation_emb) * 2π  │
    │                                     │
    │ rel_re = cos(phase)                 │
    │ rel_im = sin(phase)                 │
    │                                     │
    │ Result: Unit complex number         │
    └─────────────────────────────────────┘
        ↓
    ┌─────────────────────────────────────┐
    │ Step 5: Complex Multiplication      │
    │         (Rotation)                  │
    │                                     │
    │ query_re = head_re_m * rel_re       │
    │            - head_im_m * rel_im     │
    │                                     │
    │ query_im = head_re_m * rel_im       │
    │            + head_im_m * rel_re     │
    │                                     │
    │ Implements: q = h̄ * r               │
    │ (ā denotes masked h)                │
    └─────────────────────────────────────┘
        ↓
    ┌─────────────────────────────────────┐
    │ Step 6: Normalize Query             │
    │                                     │
    │ norm = √(query_re² + query_im²)     │
    │ query_re_norm = query_re / norm     │
    │ query_im_norm = query_im / norm     │
    │                                     │
    │ Shape: [d] each (half of embedding) │
    └─────────────────────────────────────┘
        ↓
        
Output: query_re_norm, query_im_norm
        (Query embedding ready for scoring)
```

## 4. Scoring Process

```
Input:
  query_re [batch, d]
  query_im [batch, d]
  entity_re [n_entity, d]
  entity_im [n_entity, d]

        ↓
        
    ┌───────────────────────────────────────────┐
    │ Compute Complex Dot Product               │
    │                                           │
    │ FOR each batch_idx:                       │
    │   FOR each entity_idx:                    │
    │     score[b,e] = Σ_i (                    │
    │       query_re[b,i] * entity_re[e,i]  +   │
    │       query_im[b,i] * entity_im[e,i]  )   │
    │                                           │
    │ Result: [batch_size, n_entity]            │
    │ Higher score = better match               │
    └───────────────────────────────────────────┘
        ↓
        
    ┌───────────────────────────────────────────┐
    │ Rank Entities                             │
    │                                           │
    │ sorted_indices = argsort(scores, desc)    │
    │ rank = position of true entity in sorted  │
    │                                           │
    │ Metrics:                                  │
    │ • MR = rank (mean rank)                   │
    │ • MRR = 1/rank (reciprocal rank)          │
    │ • Hits@K = (rank <= K)                    │
    └───────────────────────────────────────────┘
```

## 5. Forward Pass with Batch Optimization

```
Input:
  head_ids [batch_size]
  relation_ids [batch_size]
  tail_ids [batch_size]

        ↓

    ┌────────────────────────────────────────────┐
    │ Step 1: Stack Query Pairs                  │
    │                                            │
    │ query_pairs = stack([head_ids, rel_ids])   │
    │ Shape: [batch_size, 2]                     │
    └────────────────────────────────────────────┘
        ↓

    ┌────────────────────────────────────────────┐
    │ Step 2: Find Unique Query Pairs            │
    │                                            │
    │ unique_q, q_inv = unique(query_pairs)     │
    │                                            │
    │ Example:                                   │
    │ query_pairs = [[0, 1], [0, 1], [2, 3]]     │
    │ unique_q = [[0, 1], [2, 3]]  # 2 unique   │
    │ q_inv = [0, 0, 1]  # mapping indices       │
    └────────────────────────────────────────────┘
        ↓

    ┌────────────────────────────────────────────┐
    │ Step 3: Compose Unique Queries             │
    │                                            │
    │ FOR each unique query pair:                │
    │   q_unique_emb[i] = compose_from_head(    │
    │     unique_q[i, 0],  # head_id            │
    │     unique_q[i, 1]   # relation_id         │
    │   )                                        │
    │                                            │
    │ Shape: [n_unique_queries, 2*d]            │
    └────────────────────────────────────────────┘
        ↓

    ┌────────────────────────────────────────────┐
    │ Step 4: Find Unique Tails                  │
    │                                            │
    │ unique_t, t_inv = unique(tail_ids)         │
    │                                            │
    │ Example:                                   │
    │ tail_ids = [5, 5, 7, 5]                    │
    │ unique_t = [5, 7]  # 2 unique              │
    │ t_inv = [0, 0, 1, 0]  # mapping            │
    └────────────────────────────────────────────┘
        ↓

    ┌────────────────────────────────────────────┐
    │ Step 5: Normalize Unique Tails             │
    │                                            │
    │ t_unique_emb = normalize(E[unique_t])      │
    │ Shape: [n_unique_tails, 2*d]              │
    └────────────────────────────────────────────┘
        ↓

    ┌────────────────────────────────────────────┐
    │ Step 6: Map Back to Batch Indices          │
    │                                            │
    │ q_batch = q_unique_emb[q_inv]             │
    │ t_batch = t_unique_emb[t_inv]             │
    │                                            │
    │ Shape: [batch_size, 2*d] each             │
    │                                            │
    │ Now: q_batch[i] corresponds to tail_ids[i]│
    │      t_batch[i] corresponds to tail_ids[i]│
    └────────────────────────────────────────────┘
        ↓

    ┌────────────────────────────────────────────┐
    │ Step 7: Compute Distances                  │
    │                                            │
    │ distances = ||q_batch - t_batch||₂        │
    │ Shape: [batch_size]                       │
    │                                            │
    │ Lower distance = better match              │
    └────────────────────────────────────────────┘
        ↓

Output: distances [batch_size]
        (Used for loss computation)
```

## 6. Uniformity Loss Computation

```
Input: Embeddings [n_samples, 2*d]

        ↓

    ┌──────────────────────────────────────────┐
    │ Step 1: Optional Subsampling              │
    │                                          │
    │ IF n_samples > max_samples:              │
    │   idx = random_perm(n_samples)           │
    │   embeddings = embeddings[idx[:max]]     │
    │   n = max_samples                        │
    │ ELSE:                                    │
    │   n = n_samples                          │
    │                                          │
    │ Purpose: Reduce O(n²) computation        │
    └──────────────────────────────────────────┘
        ↓

    ┌──────────────────────────────────────────┐
    │ Step 2: Chunked Pairwise Distance Comp   │
    │                                          │
    │ FOR chunk_i FROM 0 TO n STEP chunk_size: │
    │   FOR chunk_j FROM chunk_i TO n STEP...: │
    │                                          │
    │     xi = emb[chunk_i:i+chunk_size]      │
    │     xj = emb[chunk_j:j+chunk_size]      │
    │                                          │
    │     diff = xi[:, None, :] - xj[None, :] │
    │     dist_sq = (diff²).sum(dim=-1)        │
    │     weights = exp(-2 * dist_sq)          │
    │                                          │
    │     IF i == j:                           │
    │       Skip diagonal (self-pairs)         │
    │       pair_sum += weights_no_diag        │
    │     ELSE:                                │
    │       pair_sum += weights.sum()          │
    │       pair_count += weights.size         │
    └──────────────────────────────────────────┘
        ↓

    ┌──────────────────────────────────────────┐
    │ Step 3: Compute Uniformity               │
    │                                          │
    │ mean_weight = pair_sum / pair_count      │
    │ loss_uni = log(mean_weight)              │
    │                                          │
    │ Properties:                              │
    │ • Higher = more uniform                  │
    │ • Lower value worse (less uniform)       │
    │                                          │
    │ Goal: Maximize loss (reduce magnitude)   │
    └──────────────────────────────────────────┘
        ↓

Output: loss_uni (scalar)
        (Constraint to prevent collapse)
```

## 7. Complete Training → Validation → Testing Cycle

```
┌─────────────────────────────────────────────────────────────┐
│                     START EXPERIMENT                        │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 1: DATA LOADING & PREPARATION                        │
│  • Load entities, relations, triples                        │
│  • Create ID mappings (entity2id, relation2id)             │
│  • Build train/valid/test splits                           │
│  • Count triple frequencies (for subsampling)              │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 2: MODEL INITIALIZATION                              │
│  • Create embedding matrices:                              │
│    - Entity embeddings [n_entity, 2*d] (complex)          │
│    - Relation embeddings [n_relation, d] (real)           │
│    - Relation mask embeddings [n_relation, d] (real)      │
│  • Initialize optimizer (Adam)                            │
│  • Set hyperparameters (lr, gamma, etc.)                  │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 3: TRAINING LOOP (Multiple Epochs)                  │
│                                                             │
│  FOR each epoch:                                            │
│   ├─ Shuffle train data                                    │
│   ├─ FOR each batch (head-batch and tail-batch):           │
│   │   ├─ Sample negative entities                          │
│   │   ├─ Compose queries & score                           │
│   │   ├─ Compute losses                                    │
│   │   ├─ Backward & optimize                               │
│   │   └─ Log progress                                      │
│   │                                                         │
│   ├─ VALIDATION (every valid_steps):                       │
│   │   ├─ Pre-encode all entities                           │
│   │   ├─ FOR each valid triple:                            │
│   │   │   ├─ Tail prediction ranking                       │
│   │   │   └─ Head prediction ranking                       │
│   │   ├─ Compute MRR, MR, Hits@K                           │
│   │   └─ Check if best model                               │
│   │                                                         │
│   ├─ IF best performance:                                  │
│   │   └─ Save checkpoint                                   │
│   └─ IF patience exceeded:                                 │
│       └─ Break training                                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 4: LOAD BEST CHECKPOINT                              │
│  • Restore best model weights                              │
│  • Prepare for testing                                     │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│ PHASE 5: FINAL INFERENCE ON TEST SET                       │
│                                                             │
│  FOR each test triple (h, r, t):                           │
│   ├─ Tail Prediction (h, r, ?):                            │
│   │   ├─ Compose query from (h, r)                         │
│   │   ├─ Score against all entities                        │
│   │   ├─ Rank and find position of t                       │
│   │   └─ Compute MR, MRR, Hits@K                           │
│   │                                                         │
│   ├─ Head Prediction (?, r, t):                            │
│   │   ├─ Compose query from (t, inverse_r)                 │
│   │   ├─ Score against all entities                        │
│   │   ├─ Rank and find position of h                       │
│   │   └─ Compute MR, MRR, Hits@K                           │
│   │                                                         │
│   └─ Apply FILTERING (optional):                           │
│       ├─ Re-rank excluding invalid entities                │
│       └─ Recompute metrics                                 │
│                                                             │
│  Final Results:                                            │
│   • Mean MR (lower is better)                              │
│   • Mean MRR (higher is better)                            │
│   • Hits@1, Hits@3, Hits@10 (higher is better)            │
│   • Results with/without filtering                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│             REPORT AND SAVE RESULTS                        │
│  • Final metrics on test set                               │
│  • Model checkpoint saved                                  │
│  • Experiment configuration logged                         │
└─────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────┐
│                  END EXPERIMENT                            │
└─────────────────────────────────────────────────────────────┘
```

## 8. Bidirectional Prediction with Filtering

```
For test triple (h_test, r_test, t_test):

┌──────────────────────────────────────────────────┐
│ TAIL PREDICTION: (h_test, r_test, ?)             │
│                                                  │
│ Step 1: Query composition                       │
│   q = compose_from_head(h_test, r_test)         │
│                                                  │
│ Step 2: Score all entities                      │
│   scores = [score(q, E[e]) for e in all_ents]  │
│                                                  │
│ Step 3a: WITHOUT Filtering                      │
│   rank_no_filter = argsort(scores)              │
│   rank_t = index of t_test in rank_no_filter    │
│                                                  │
│ Step 3b: WITH Filtering                         │
│   valid_tails = {t' : (h_test, r_test, t') ∈ T}│
│   scores_filtered = scores.copy()               │
│   FOR e NOT in valid_tails:                     │
│     scores_filtered[e] = -∞                     │
│                                                  │
│   rank_filter = argsort(scores_filtered)        │
│   rank_t_filter = index of t_test               │
│                                                  │
│ Step 4: Compute metrics                         │
│   MR_t = rank_t                                 │
│   MRR_t = 1 / rank_t                            │
│   Hits@K_t = (rank_t ≤ K)                       │
│                                                  │
│   Same for filtered versions                    │
└──────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────┐
│ HEAD PREDICTION: (?, r_test, t_test)             │
│                                                  │
│ Step 1: Query composition (inverse relation)    │
│   q = compose_from_tail(t_test, r_test)        │
│                                                  │
│ Step 2: Score all entities                      │
│   scores = [score(q, E[e]) for e in all_ents]  │
│                                                  │
│ Step 3: Filtering                               │
│   valid_heads = {h' : (h', r_test, t_test) ∈ T}│
│   Apply same filtering as tail prediction      │
│                                                  │
│ Step 4: Compute metrics                         │
│   MR_h, MRR_h, Hits@K_h                        │
│                                                  │
└──────────────────────────────────────────────────┘
         ↓
┌──────────────────────────────────────────────────┐
│ AGGREGATE RESULTS                                │
│                                                  │
│ MR = (MR_t + MR_h) / 2                          │
│ MRR = (MRR_t + MRR_h) / 2                       │
│ Hits@K = (Hits@K_t + Hits@K_h) / 2             │
│                                                  │
└──────────────────────────────────────────────────┘
```

---

## Flow Summary Table

| Phase | Key Operations | Input | Output |
|-------|---------------|-------|--------|
| **Data Loading** | Index entities/relations, count frequencies | Raw triples | ID mappings, frequency counts |
| **Initialization** | Create embeddings, optimizer | Config | Model ready for training |
| **Training** | Sampling, composition, scoring, loss, backprop | Batches | Updated embeddings, logs |
| **Validation** | Pre-encode, score, rank | Valid triples | Metrics, best checkpoint |
| **Testing** | Pre-encode, compose, score, filter | Test triples | Final metrics with/without filter |

