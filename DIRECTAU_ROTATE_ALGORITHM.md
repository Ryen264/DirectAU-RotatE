# DirectAU-RotatE Algorithm: Relation-Aware Knowledge Graph Embedding

## Overview

**DirectAU-RotatE** is a Knowledge Graph Embedding model that extends RotatE with learnable **relation-aware directional masks** for improved link prediction. Unlike traditional RotatE which treats all relations uniformly, DirectAU-RotatE learns relation-specific masks that modulate entity embeddings during query composition, enabling the model to:

1. **Learn relation-specific transformations**: Each relation learns its own mask to emphasize/suppress relevant entity dimensions
2. **Maintain complex rotation semantics**: Preserves RotatE's rotation-based composition in complex space
3. **Enable bidirectional prediction**: Handles both head prediction (?, r, t) and tail prediction (h, r, ?)

---

## Architecture

### Components

```
Knowledge Graph Triples: (h, r, t)
        ↓
    ┌──────────────────────────────┐
    │ Load Knowledge Graph Data    │
    │ (entities and relations)     │
    └──────────────────────────────┘
        ↓         ↓         ↓
    [Head Entity][Relation][Tail Entity]
        ↓         ↓         ↓
    ┌────────────┴──────────┴────────────┐
    │    Entity Embeddings (Complex)     │
    │    [Re_part, Im_part]              │
    │    Shape: [n_entity, 2*hidden_dim]│
    └────────────┬──────────┬────────────┘
         ↓                   ↓
    Head Embedding      Tail Embedding
         ↓                   ↓
    ┌────────────────────────────┐
    │ Relation Embeddings        │
    │ (Real-valued)              │
    │ Shape: [n_relation, dim]   │
    └────────────────────────────┘
         ↓
    ┌────────────────────────────┐
    │ Relation Mask Embeddings   │
    │ (DirectAU Component)       │
    │ Shape: [n_relation, dim]   │
    └────────────────────────────┘
         ↓
    ┌──────────────────────────┐
    │ Query Composition &      │
    │ Scoring                  │
    └──────────────────────────┘
```

### Embedding Spaces

**Entity Embeddings**:
- Complex-valued vectors: $e_i = [Re_i, Im_i]$ where $Re_i, Im_i \in \mathbb{R}^{hidden\_dim}$
- Shape: `[n_entity, 2*hidden_dim]`
- Initialization: Uniform distribution over embedding range

**Relation Embeddings**:
- Real-valued vectors: $r_j \in \mathbb{R}^{hidden\_dim}$
- Shape: `[n_relation, hidden_dim]`
- Represents rotation phases in RotatE
- Initialization: Uniform distribution over embedding range

**Relation Mask Embeddings** (DirectAU-specific):
- Real-valued vectors: $m_j \in \mathbb{R}^{hidden\_dim}$
- Shape: `[n_relation, hidden_dim]`
- Passed through sigmoid to create masks in [0, 1]
- Modulates entity embeddings during composition

### Key Settings

```yaml
DirectAURotatE:
  hidden_dim: 500                         # Embedding dimension
  gamma: 12.0                             # Margin parameter
  embedding_range: (gamma + 2.0) / dim    # Initialization range
  
  # Training
  optimizer: Adam                         # Optimizer
  learning_rate: 0.0001                   # Learning rate
  max_steps: 100000                       # Total training steps
  batch_size: 1024                        # Mini-batch size
  test_batch_size: 4                      # Evaluation batch size
  
  # Negative sampling
  negative_sample_size: 128                # Negatives per positive
  negative_adversarial_sampling: true      # Use adversarial sampling
  adversarial_temperature: 1.0             # Temperature for sampling
  
  # Loss weights
  gamma_uni: 1.0                           # Uniformity loss weight
  gamma_neg: 1.0                           # Negative loss weight
  regularization: 0.0                      # L2 regularization
  
  # Validation & checkpointing
  valid_steps: 10000                       # Validation frequency
  save_checkpoint_steps: 10000             # Checkpoint frequency
```

---

## Core Operations

### 1. Complex Embedding Normalization

Normalize complex embeddings to unit norm on the complex plane.

**Formula**:
$$\text{normalize}(e) = \frac{[Re, Im]}{\sqrt{Re^2 + Im^2} + \epsilon}$$

**Code concept**:
```
FUNCTION normalize_complex_embedding(embedding):
    re_part, im_part ← split(embedding)              # Split [Re, Im]
    norm ← sqrt(re_part² + im_part²) + epsilon       # Compute complex norm
    re_norm ← re_part / norm
    im_norm ← im_part / norm
    RETURN [re_norm, im_norm]
END
```

### 2. Relation Phase Extraction

Convert relation embeddings to rotation angles in $[-\pi, \pi]$.

**Formula**:
$$\text{phase}(r_j) = \sigma(r_j) \times (2\pi)$$

where $\sigma$ is sigmoid function, converting relation embedding to [0, 1] then to angle [0, 2π].

**Code concept**:
```
FUNCTION relation_phase(relation_embedding):
    phase ← sigmoid(relation_embedding) * 2π    # Convert to [0, 2π]
    RETURN phase
END

FUNCTION relation_unit(relation_embedding):
    phase ← relation_phase(relation_embedding)
    re_unit ← cos(phase)
    im_unit ← sin(phase)
    RETURN re_unit, im_unit
END
```

### 3. Relation Mask Computation

Compute sigmoid-based masks from relation mask embeddings.

**Formula**:
$$\text{mask}(r_j) = \sigma(m_j)$$

where $m_j$ is the relation mask embedding and $\sigma$ is sigmoid.

**Result**: Mask values in $[0, 1]$ that modulate entity embeddings.

```
FUNCTION relation_mask(relation_ids):
    relation_mask_emb ← index_select(relation_mask_embedding, relation_ids)
    mask ← sigmoid(relation_mask_emb)                 # [batch_size, hidden_dim]
    RETURN unsqueeze(mask, 1)                         # [batch_size, 1, hidden_dim]
END
```

### 4. Query Composition (Tail Prediction)

Compose query embedding from head entity and relation for tail prediction task.

**Steps**:
1. Normalize head entity embedding (complex)
2. Apply relation mask to head embedding
3. Compose with relation rotation
4. Normalize result

**Formula**:
$$q_{re}, q_{im} = \text{normalize}\left(\text{compose}\left(\text{normalize}(h) \odot \text{mask}(r), \text{phase}(r)\right)\right)$$

where $\odot$ is element-wise multiplication, and composition uses complex multiplication.

**Code concept**:
```
FUNCTION compose_query_from_head(head, relation_emb, relation_ids):
    // 1. Normalize head entity
    head_re, head_im ← normalize_complex_embedding(head)
    
    // 2. Get and apply relation mask
    relation_mask ← relation_mask(relation_ids)       // [batch, 1, hidden_dim]
    head_re ← head_re * relation_mask
    head_im ← head_im * relation_mask
    
    // 3. Get relation rotation unit
    relation_re, relation_im ← relation_unit(relation_emb)
    
    // 4. Complex multiplication (rotation)
    query_re ← head_re * relation_re - head_im * relation_im
    query_im ← head_re * relation_im + head_im * relation_re
    
    // 5. Normalize result
    query_re_norm, query_im_norm ← normalize_complex_pair(query_re, query_im)
    
    RETURN query_re_norm, query_im_norm
END
```

### 5. Query Composition (Head Prediction)

Compose query from tail entity and inverse relation for head prediction.

**Key difference**: Uses conjugate (inverse) relation.

```
FUNCTION compose_query_from_tail(tail, relation_emb, relation_ids):
    // 1. Normalize tail entity
    tail_re, tail_im ← normalize_complex_embedding(tail)
    
    // 2. Apply relation mask
    relation_mask ← relation_mask(relation_ids)
    tail_re ← tail_re * relation_mask
    tail_im ← tail_im * relation_mask
    
    // 3. Get inverse relation rotation
    relation_re, relation_im ← relation_unit(relation_emb)
    
    // 4. Complex multiplication with conjugate (inverse)
    // For (?, r, t), compute: q = conj(t * r)
    query_re ← tail_re * relation_re + tail_im * relation_im       // Conjugate
    query_im ← -tail_re * relation_im + tail_im * relation_re
    
    // 5. Normalize
    query_re_norm, query_im_norm ← normalize_complex_pair(query_re, query_im)
    
    RETURN query_re_norm, query_im_norm
END
```

### 6. Scoring (Complex Dot Product)

Score candidates against query using complex dot product.

**Formula**:
$$\text{score}(q, c) = \sum_i (q_{re}^i \cdot c_{re}^i + q_{im}^i \cdot c_{im}^i)$$

**Code concept**:
```
FUNCTION score_complex(query_re, query_im, candidate_re, candidate_im):
    score ← (query_re * candidate_re + query_im * candidate_im).sum(dim=-1)
    RETURN score
END
```

---

## Loss Functions

### 1. Alignment Loss (Ranking Loss)

Use margin-ranking loss with negative sampling.

**Formula**:
$$L_{\text{align}} = \max(0, \gamma + d(\text{pos}) - d(\text{neg}))$$

where:
- $d(\text{pos})$ = distance to positive (golden truth) sample
- $d(\text{neg})$ = distance to negative sample
- $\gamma$ = margin parameter

**Code concept**:
```
FUNCTION alignment_loss(pos_score, neg_scores, gamma):
    // Higher score = better match, so negate for distance
    pos_dist ← -pos_score                                    // [batch]
    neg_dist ← -neg_scores                                   // [batch, neg_size]
    
    // Margin ranking loss
    loss ← max(0, gamma + pos_dist - neg_dist)              // [batch, neg_size]
    loss ← loss.mean()                                       // scalar
    
    RETURN loss
END
```

### 2. Uniformity Loss

Encourages uniform distribution of embeddings in complex space (optional).

**Purpose**: Prevent representation collapse by spreading embeddings evenly.

**Formula** (simplified):
$$L_{\text{uni}} = \log\left(\text{mean}_{i<j}\left[\exp(-2\|e_i - e_j\|^2)\right]\right)$$

**Implementation**:
- Computed on unique query embeddings in batch
- Computed on unique tail embeddings in batch
- Combined: $L_{\text{uni}} = 0.5 \times (L_{\text{uni}}^q + L_{\text{uni}}^t)$

### 3. Total Loss

**Formula**:
$$L_{\text{total}} = L_{\text{align}} + \gamma_{\text{uni}} \times L_{\text{uni}} + \gamma_{\text{neg}} \times L_{\text{neg}}$$

Where:
- $\gamma_{\text{uni}}$ = uniformity loss weight (default: 1.0)
- $\gamma_{\text{neg}}$ = negative loss weight (default: 1.0)

---

## Training Algorithm

### Data Preparation

1. **Index entities and relations** from train/valid/test files
2. **Count triple frequencies** for subsampling (like word2vec)
3. **Create positive/negative triple sets** for filtering during inference

### Training Loop

```
FOR epoch IN range(n_epoch):
    
    // Shuffle training data
    shuffled_indices ← random_permutation(n_train)
    
    optimizer.zero_grad()
    epoch_loss ← 0
    
    FOR batch_idx, batch IN enumerate_batches(train_data):
        
        // 1. Get positive sample
        head_batch, relation_batch, tail_batch ← batch
        
        // 2. Generate negative samples
        IF mode == 'head-batch':
            negative_heads ← sample_negatives(tail, relation)
        ELSE:  // tail-batch
            negative_tails ← sample_negatives(head, relation)
        
        // 3. Forward: compute scores
        pos_score ← model.forward(positive, mode='single')
        neg_score ← model.forward([positive, negatives], mode)
        
        // 4. Compute alignment loss
        loss_align ← alignment_loss(pos_score, neg_score, gamma)
        
        // 5. Extract unique embeddings for uniformity
        IF gamma_uni > 0:
            unique_queries ← unique([head, relation] pairs)
            unique_tails ← unique(tail_batch)
            
            loss_uni_q ← uniformity_loss(encode(unique_queries))
            loss_uni_t ← uniformity_loss(encode(unique_tails))
            loss_uni ← 0.5 * (loss_uni_q + loss_uni_t)
        ELSE:
            loss_uni ← 0
        
        // 6. Total loss
        loss ← loss_align + gamma_uni * loss_uni + gamma_neg * loss_align
        
        // 7. Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        epoch_loss ← epoch_loss + loss.item()
    
    // Validation
    valid_perf ← validate(model, valid_data)
    
    IF valid_perf > best_valid_perf:
        save_checkpoint(model)
        best_valid_perf ← valid_perf
    ELSE:
        patience_counter ← patience_counter + 1
    
    IF patience_counter >= early_stop_patience:
        BREAK
```

### Key Training Features

| Feature | Benefit |
|---------|---------|
| **Negative Sampling** | Efficient training with hard negative examples |
| **Adversarial Sampling** | Focus on hard negatives with probability ∝ loss |
| **Subsampling Weighting** | Down-weight frequent triples (like word2vec) |
| **Bidirectional Training** | Alternate head-batch and tail-batch modes |
| **Relation Masks** | Learn relation-specific entity dimension importance |
| **Complex Embeddings** | Leverage rotation semantics from RotatE |

---

## Inference / Testing Algorithm

### Link Prediction: (?, r, t) and (h, r, ?)

For each test triple (h, r, t):

```
1. Pre-encode all entities (done once per test set)
   entity_matrix ← [normalize(entity_0), ..., normalize(entity_n)]
   Shape: [n_entity, 2*hidden_dim]

2. For TAIL PREDICTION (h, r, ?):
   ├─ Compose query: q_tail = compose_query_from_head(h, r)
   ├─ Score all candidates: scores = score_complex(q_tail, entity_matrix)
   │  Higher score = better match
   ├─ Sort scores (descending)
   ├─ Find rank of true tail t
   └─ Compute metrics (MR, MRR, Hits@K)

3. For HEAD PREDICTION (?, r, t):
   ├─ Compose query: q_head = compose_query_from_tail(t, r)
   ├─ Score all candidates: scores = score_complex(q_head, entity_matrix)
   ├─ Sort scores (descending)
   ├─ Find rank of true head h
   └─ Compute metrics (MR, MRR, Hits@K)

4. Apply FILTERING (if enabled):
   ├─ Get valid entities S from training data
   ├─ For entities NOT in S:
   │   scores[invalid] ← -∞ (so they rank last)
   ├─ Recompute ranks

5. Average metrics:
   MR = (MR_tail + MR_head) / 2
   MRR = (MRR_tail + MRR_head) / 2
   Hits@K = (Hits@K_tail + Hits@K_head) / 2
```

### Code Snippet

```python
def test_link(model, test_data, nentity):
    entity_matrix = model.entity_embedding                    # [n_entity, 2*dim]
    
    for head, relation, tail in test_data:
        # Tail prediction
        q_tail = model.compose_query_from_head(head, relation)
        tail_scores = score_complex(q_tail, entity_matrix)
        tail_rank = ranking_metrics(tail_scores, tail)
        
        # Head prediction (inverse relation)
        q_head = model.compose_query_from_tail(tail, relation)
        head_scores = score_complex(q_head, entity_matrix)
        head_rank = ranking_metrics(head_scores, head)
        
        # Filter if needed
        if FILTERING_ENABLED:
            tail_scores[invalid_entities] = -INF
            head_scores[invalid_entities] = -INF
```

---

## Comparison: DirectAU-RotatE vs Standard RotatE

| Aspect | DirectAU-RotatE | Standard RotatE |
|--------|-----------------|-----------------|
| **Entity Embeddings** | Complex, normalized | Complex, normalized |
| **Relation Embeddings** | Real-valued | Real-valued |
| **Query Composition** | With relation mask | Direct rotation |
| **Relation Masks** | Learnable per relation | None |
| **Negative Sampling** | Yes | Yes |
| **Uniformity Loss** | Optional (γ_uni > 0) | N/A |
| **Directional Awareness** | Yes (via masks) | Limited |
| **Parameters** | More (mask embeddings) | Fewer |
| **Interpretability** | Masks show dimension importance | Limited |

---

## Complexity Analysis

| Operation | Time Complexity | Space Complexity |
|-----------|-----------------|------------------|
| **Forward pass** | $O(B \times d)$ | $O(B \times d)$ |
| **Negative sampling** | $O(B \times N)$ | $O(B \times N)$ |
| **Scoring all entities** | $O(n \times d)$ | $O(n \times d)$ |
| **Uniformity loss** | $O(m^2)$ with subsampling | $O(m \times d)$ |
| **Backward pass** | $O(B \times d)$ | $O(B \times d)$ |

Where: $B$ = batch size, $d$ = embedding dim, $n$ = num entities, $N$ = num negatives, $m$ = subsampled uniformity samples

---

## Key Innovations

1. **Relation-Aware Masking**: Each relation learns which dimensions of entities are important, enabling relation-specific embeddings without fully separate relation embeddings

2. **DirectAU Mechanism**: Directional masks allow asymmetric handling of (h, r) → t vs (?, r, t) patterns

3. **Negative Sampling**: Efficiently focuses training on challenging hard negatives using adversarial sampling

4. **Uniformity Constraint**: Optional uniformity loss prevents embedding collapse while maintaining rotation semantics

5. **Complex Space Semantics**: Leverages rotation in complex space, more expressive than TransE translation

---

## Configuration Example

```yaml
# WN18RR configuration for DirectAU-RotatE
model: DirectAU_RotatE
dataset: wn18rr

# Architecture
hidden_dim: 500
double_entity_embedding: true
double_relation_embedding: false

# Training hyperparameters
batch_size: 1024
negative_sample_size: 128
learning_rate: 0.0001
max_steps: 100000

# Loss weights
gamma: 12.0
gamma_uni: 1.0
gamma_neg: 1.0

# Validation
valid_steps: 10000
save_checkpoint_steps: 10000
```

---

## Summary

**DirectAU-RotatE** enhances the RotatE model with learnable relation-specific masks that:
- Enable directional awareness in knowledge graph reasoning
- Allow the model to learn which dimensions matter for each relation
- Maintain RotatE's rotation semantics while adding flexibility
- Improve link prediction performance through relation-aware transformations

The combination of negative sampling, uniformity constraints, and relation masks creates a powerful framework for knowledge graph embedding learning.
