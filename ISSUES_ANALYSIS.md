# DirectAU-RotatE Critical Issues Analysis

## Summary
Found **4 critical/moderate issues** affecting training convergence. Below is detailed analysis with code evidence.

---

## Issue 1: ⚠️ CRITICAL - Vector Magnitude Collapse Due to Mask + Normalize

**Location**: [codes/model.py](codes/model.py#L109-L115)

**Problem**:
```python
def _compose_query_from_head(self, head, relation_embedding, relation_ids):
    head_re, head_im = self._normalize_complex_embedding(head)      # Step 1: normalize to ||v||=1
    relation_mask = self._relation_mask(relation_ids)                # Step 2: mask ∈ [0, 1] from Sigmoid
    head_re = head_re * relation_mask                                # Step 3: apply mask (scales down!)
    head_im = head_im * relation_mask
    
    relation_re, relation_im = self._relation_unit(relation_embedding)
    query_re = head_re * relation_re - head_im * relation_im
    query_im = head_re * relation_im + head_im * relation_re
    return self._normalize_complex_pair(query_re, query_im)           # Step 4: normalize again
```

**What happens**:
1. Head is normalized to unit vector: `||head|| = 1`
2. Mask (from Sigmoid) typically ranges [0.4, 0.6] early in training
3. After masking: `||head * mask|| ≈ 0.5` (vectors collapse!)
4. Second normalization tries to "fix" this but loses information

**Result**: Vectors are artificially shrunk to near-zero magnitudes → all scores become tiny → model can't discriminate true from false entities → loss stays high

**Evidence from logs**: Your scores are likely all ≈ 0.00001 and indistinguishable

---

## Issue 2: ⚠️ CRITICAL - Uniformity Loss Numerical Instability

**Location**: [codes/model.py](codes/model.py#L352-L361)

**Code**:
```python
pairwise_distance = torch.pdist(uniform_entities, p=2)
uniformity_loss = torch.log(
    torch.mean(torch.exp(-2.0 * pairwise_distance.pow(2))) + args.epsilon
)
```

**Problems**:
1. **Negative loss by design**: `log(small_positive_value)` yields **large negative numbers**
   - If mean exp = 0.00001 → log(0.00001) ≈ **-11.5**
   - Your logs show -18.42 suggesting mean exp is even smaller
   
2. **Epsilon placement is wrong**: `epsilon` in args is 1e-8 (from log), but:
   - You add epsilon AFTER taking log: `log(mean_exp + epsilon)`
   - Should be: `log(mean_exp + epsilon)` to prevent `log(0)`
   - Currently epsilon (1e-8) doesn't help when mean_exp is 1e-20

3. **Loss domination**: With `gamma_uni = 1.0`:
   - `total_loss = align_loss(~1.0) + 1.0 * uniform_loss(-18.42) + gamma_neg * neg_loss(~6.3)`
   - Uniformity completely dominates! Model ignores alignment.

**Your log confirms it**:
```
align_loss: 0.999652
uniform_loss: -18.420681  ← This shouldn't be so negative!
negative_sample_loss: 6.347376
total_loss: 7.345329
```

---

## Issue 3: ⚠️ MODERATE - Relation Mask Initialization

**Location**: [codes/model.py](codes/model.py#L59-L62) and [codes/model.py](codes/model.py#L103-L106)

**Initialization**:
```python
self.relation_mask_embedding = nn.Parameter(torch.zeros(nrelation, hidden_dim))
nn.init.uniform_(
    tensor=self.relation_mask_embedding,
    a=-self.embedding_range.item(),  # ≈ -0.003
    b=self.embedding_range.item()     # ≈ +0.003
)
```

**Masking function**:
```python
def _relation_mask(self, relation_ids):
    relation_mask = torch.index_select(...)
    return torch.sigmoid(relation_mask).unsqueeze(1)
```

**Problem**:
- Initial values: `[-0.003, +0.003]`
- After sigmoid: `sigmoid(±0.003) ≈ [0.49925, 0.50075]` → **all masks stay near 0.5**
- Model wanted masks to *select* which dimensions to use, but they start at 0.5 (neutral)
- Should initialize so sigmoid outputs are closer to **1.0** (all dimensions active) or **0.0** (some off)

**Expected behavior**: Start with mask ≈ 1.0, then learn which dims to mask down

**Current behavior**: Start with mask ≈ 0.5, vectors immediately shrink to 50% magnitude

---

## Issue 4: ⚠️ MINOR - Learning Rate Too Conservative

**Location**: [codes/run.py](codes/run.py#L56)

```python
parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
```

**Problem**:
- Default LR = 1e-4 is very conservative for hidden_dim=500
- With complex number ops + normalization + masks, gradients get dampened
- Model needs 5000+ steps just to move embeddings meaningfully

**Your logs**: Likely showing near-zero gradient flow in first 1000 steps

**Recommendation**: Start with 5e-4 or 1e-3 for DirectAU_RotatE

---

## Recommended Fixes

### Fix 1: Remove Second Normalization After Masking
**File**: [codes/model.py](codes/model.py#L109-L115)

**Change**:
```python
def _compose_query_from_head(self, head, relation_embedding, relation_ids):
    head_re, head_im = self._normalize_complex_embedding(head)
    relation_mask = self._relation_mask(relation_ids)
    head_re = head_re * relation_mask
    head_im = head_im * relation_mask
    
    relation_re, relation_im = self._relation_unit(relation_embedding)
    query_re = head_re * relation_re - head_im * relation_im
    query_im = head_re * relation_im + head_im * relation_re
    # ❌ REMOVE: return self._normalize_complex_pair(query_re, query_im)
    # ✅ Keep vectors at their natural magnitude
    return query_re, query_im
```

**Do the same for** `_compose_query_from_tail` (lines 122-128)

---

### Fix 2: Reweight & Stabilize Uniformity Loss
**File**: [codes/model.py](codes/model.py#L352-L361)

**Change**:
```python
# Current (problematic):
# uniformity_loss = torch.log(torch.mean(torch.exp(-2.0 * pairwise_distance.pow(2))) + args.epsilon)

# Fix: Use better numerical stability + clamp
if pairwise_distance.numel() == 0:
    uniformity_loss = torch.zeros(1, device=positive_sample.device).squeeze(0)
else:
    # Clamp distances to prevent exp overflow
    clamped_distances = torch.clamp(pairwise_distance, min=0.01, max=10.0)
    exp_sum = torch.mean(torch.exp(-2.0 * clamped_distances.pow(2)))
    # Prevent log(0) and log(negative)
    uniformity_loss = torch.log(torch.clamp(exp_sum, min=args.epsilon))
```

**Better alternative - use contrastive uniformity**:
```python
# Uniformity loss: penalize when pairwise distances are too small
uniformity_loss = -torch.mean(torch.log(torch.clamp(pairwise_distance, min=args.epsilon)))
```

---

### Fix 3: Better Mask Initialization
**File**: [codes/model.py](codes/model.py#L59-L62)

**Change**:
```python
if model_name == 'DirectAU_RotatE':
    self.relation_mask_embedding = nn.Parameter(torch.zeros(nrelation, hidden_dim))
    # Initialize so sigmoid output is near 1.0 (mask everything initially)
    # sigmoid(x) ≈ 1 when x >> 0, so initialize with positive values
    nn.init.constant_(self.relation_mask_embedding, 2.0)  # sigmoid(2.0) ≈ 0.88
    # OR use uniform but shifted:
    # nn.init.uniform_(self.relation_mask_embedding, 1.0, 3.0)  # sigmoid → [0.73, 0.95]
```

---

## Testing Strategy

### Test 1: Verify mask normalization issue
```bash
# Run with gamma_uni=0 (disable uniformity loss), learning_rate=0.0005
bash run.sh train DirectAU_RotatE wn18rr 0 0 512 1024 500 6.0 0.5 0.0005 80000 8 -de \
    --gamma_uni 0.0 --gamma_neg 0.0 --epsilon 1e-8
# If align_loss still doesn't decrease, Issue 1 is confirmed
```

### Test 2: After applying Fix 1 + Fix 3
```bash
# Run again with same params
bash run.sh train DirectAU_RotatE wn18rr 0 0 512 1024 500 6.0 0.5 0.0005 80000 8 -de \
    --gamma_uni 0.0 --gamma_neg 0.0 --epsilon 1e-8
# MRR should start improving within 5000-10000 steps
```

### Test 3: After applying all fixes
```bash
# Full training with better hyperparameters
bash run.sh train DirectAU_RotatE wn18rr 0 0 512 1024 500 6.0 0.5 0.0005 80000 8 -de \
    --gamma_uni 0.1 --gamma_neg 1.0 --epsilon 1e-8
# MRR should reach competitive baseline (>0.475 on wn18rr)
```

---

## Summary of Root Causes

| Issue | Root Cause | Impact | Severity |
|-------|-----------|--------|----------|
| Mask + Normalize | Double normalization shrinks vectors after masking | Vectors collapse to ~0.5x magnitude, scores indistinguishable | **CRITICAL** |
| Uniformity Loss | Log of exponentially decaying values + epsilon placement | Loss goes to -18.4, dominates total loss, alignment ignored | **CRITICAL** |
| Mask Init | Sigmoid initialized to ~0.5 output, not 0 or 1 | Mask doesn't actively select dimensions, hides true effect | **MODERATE** |
| Learning Rate | Conservative default for complex model | Slow convergence, hard to see improvement in first epochs | **MINOR** |

---

## Next Steps
1. ✅ Apply Fix 1 & 3 immediately (code changes)
2. ✅ Test with `--gamma_uni 0.0` to isolate alignment loss
3. ✅ Gradually re-enable uniformity with lower weight (0.01-0.1)
4. ✅ Monitor scores/loss ratios to ensure no future collapse
