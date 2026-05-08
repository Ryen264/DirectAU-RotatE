# DirectAU-RotatE: Complete Pseudocode Reference

## Table of Contents
1. [Initialization](#initialization)
2. [Training](#training)
3. [Validation](#validation)
4. [Testing/Inference](#testing)
5. [Loss Functions](#loss-functions)
6. [Helper Functions](#helper-functions)

---

## Initialization

```python
# ============================================================================
# FUNCTION: Initialize DirectAURotatE Model
# ============================================================================

FUNCTION initialize_model(config):
    """
    Initialize all model components and training infrastructure
    """
    
    # Load knowledge graph data
    train_triples = load_triples(config.train_path)
    valid_triples = load_triples(config.valid_path)
    test_triples = load_triples(config.test_path)
    
    # Create entity and relation ID mappings
    entities = extract_unique_entities(train_triples)
    relations = extract_unique_relations(train_triples)
    
    entity2id = create_mapping(entities)
    relation2id = create_mapping(relations)
    
    n_entity = len(entity2id)
    n_relation = len(relation2id)
    
    # Convert triples to ID format
    train_triples_ids = convert_to_ids(train_triples, entity2id, relation2id)
    valid_triples_ids = convert_to_ids(valid_triples, entity2id, relation2id)
    test_triples_ids = convert_to_ids(test_triples, entity2id, relation2id)
    
    # Initialize embedding matrices
    hidden_dim = config.hidden_dim
    gamma = config.gamma
    epsilon = 2.0
    embedding_range = (gamma + epsilon) / hidden_dim
    
    // Entity embeddings (complex-valued)
    entity_embedding = torch.zeros(n_entity, 2 * hidden_dim)
    entity_embedding.uniform_(-embedding_range, embedding_range)
    entity_embedding = nn.Parameter(entity_embedding)
    
    // Relation embeddings (real-valued)
    relation_embedding = torch.zeros(n_relation, hidden_dim)
    relation_embedding.uniform_(-embedding_range, embedding_range)
    relation_embedding = nn.Parameter(relation_embedding)
    
    // Relation mask embeddings (DirectAU-specific)
    relation_mask_embedding = torch.zeros(n_relation, hidden_dim)
    relation_mask_embedding.uniform_(-embedding_range, embedding_range)
    relation_mask_embedding = nn.Parameter(relation_mask_embedding)
    
    // Gamma parameter (margin)
    gamma_param = nn.Parameter(torch.Tensor([gamma]), requires_grad=False)
    
    // Store in model dict
    model = {
        'entity_embedding': entity_embedding,
        'relation_embedding': relation_embedding,
        'relation_mask_embedding': relation_mask_embedding,
        'gamma': gamma_param,
        'embedding_range': embedding_range,
        'n_entity': n_entity,
        'n_relation': n_relation,
    }
    
    // Initialize optimizer
    trainable_params = [
        entity_embedding,
        relation_embedding,
        relation_mask_embedding
    ]
    
    optimizer = Adam(
        params=trainable_params,
        lr=config.learning_rate
    )
    
    RETURN model, optimizer, train_triples_ids, valid_triples_ids, test_triples_ids

END FUNCTION
```

---

## Training

```python
# ============================================================================
# FUNCTION: Normalize Complex Embedding
# ============================================================================

FUNCTION normalize_complex_embedding(embedding):
    """
    Normalize complex embedding to unit norm
    
    Input: embedding [batch, 2*d] = [Re, Im]
    Output: normalized [batch, 2*d] = [Re_norm, Im_norm]
    """
    
    epsilon = 1e-12
    re_part, im_part = torch.chunk(embedding, 2, dim=-1)
    
    norm = torch.sqrt(re_part.pow(2) + im_part.pow(2)).clamp_min(epsilon)
    
    re_norm = re_part / norm
    im_norm = im_part / norm
    
    RETURN torch.cat([re_norm, im_norm], dim=-1)

END FUNCTION


# ============================================================================
# FUNCTION: Compute Relation Phase
# ============================================================================

FUNCTION relation_phase(relation_embedding):
    """
    Convert relation embedding to rotation phase in [0, 2π]
    
    Input: relation_embedding [batch, d]
    Output: phase [batch, d]
    """
    
    phase = torch.sigmoid(relation_embedding) * (2.0 * π)
    
    RETURN phase

END FUNCTION


# ============================================================================
# FUNCTION: Compute Relation Unit (cos, sin of phase)
# ============================================================================

FUNCTION relation_unit(relation_embedding):
    """
    Convert relation embedding to unit complex representation
    
    Input: relation_embedding [batch, d]
    Output: (re_unit, im_unit) both [batch, d]
    """
    
    phase = relation_phase(relation_embedding)
    
    re_unit = torch.cos(phase)
    im_unit = torch.sin(phase)
    
    RETURN re_unit, im_unit

END FUNCTION


# ============================================================================
# FUNCTION: Compute Relation Mask
# ============================================================================

FUNCTION relation_mask(relation_ids, relation_mask_embedding):
    """
    Compute sigmoid-based mask for relations
    
    Input: relation_ids [batch], relation_mask_embedding [n_relation, d]
    Output: mask [batch, 1, d] with values in [0, 1]
    """
    
    relation_mask_emb = torch.index_select(
        relation_mask_embedding,
        dim=0,
        index=relation_ids
    )  // [batch, d]
    
    mask = torch.sigmoid(relation_mask_emb)  // [batch, d]
    
    RETURN mask.unsqueeze(1)  // [batch, 1, d]

END FUNCTION


# ============================================================================
# FUNCTION: Compose Query from Head (Tail Prediction)
# ============================================================================

FUNCTION compose_query_from_head(head, relation, relation_ids, 
                                 relation_mask_embedding, hidden_dim):
    """
    Compose query embedding: q = compose(h ⊙ mask(r), r)
    Used for tail prediction: (h, r, ?)
    
    Input:
        head [batch, 2*d] - head entity embedding
        relation [batch, d] - relation embedding
        relation_ids [batch] - relation IDs
        relation_mask_embedding [n_relation, d] - mask embeddings
    
    Output:
        query_re, query_im [batch, d] - normalized query components
    """
    
    // 1. Normalize head entity
    head_normalized = normalize_complex_embedding(head)
    head_re, head_im = torch.chunk(head_normalized, 2, dim=-1)
    head_re = head_re.squeeze(-1)  // [batch, d]
    head_im = head_im.squeeze(-1)  // [batch, d]
    
    // 2. Get and apply relation mask
    mask = relation_mask(relation_ids, relation_mask_embedding)  // [batch, 1, d]
    head_re = head_re.unsqueeze(1) * mask  // [batch, 1, d]
    head_im = head_im.unsqueeze(1) * mask  // [batch, 1, d]
    head_re = head_re.squeeze(1)  // [batch, d]
    head_im = head_im.squeeze(1)  // [batch, d]
    
    // 3. Get relation rotation unit (cos, sin)
    relation_re, relation_im = relation_unit(relation)  // [batch, d]
    
    // 4. Complex multiplication: (h ⊙ m) * r
    // (a + bi) * (c + di) = (ac - bd) + (ad + bc)i
    query_re = head_re * relation_re - head_im * relation_im  // [batch, d]
    query_im = head_re * relation_im + head_im * relation_re  // [batch, d]
    
    // 5. Normalize query result
    query_re_norm, query_im_norm = normalize_complex_pair(query_re, query_im)
    
    RETURN query_re_norm, query_im_norm

END FUNCTION


# ============================================================================
# FUNCTION: Normalize Complex Pair
# ============================================================================

FUNCTION normalize_complex_pair(re_part, im_part):
    """
    Normalize complex number pair to unit norm
    
    Input: re_part [batch, d], im_part [batch, d]
    Output: (re_norm, im_norm) both [batch, d]
    """
    
    epsilon = 1e-12
    norm = torch.sqrt(re_part.pow(2) + im_part.pow(2)).clamp_min(epsilon)
    
    re_norm = re_part / norm
    im_norm = im_part / norm
    
    RETURN re_norm, im_norm

END FUNCTION


# ============================================================================
# FUNCTION: Compose Query from Tail (Head Prediction)
# ============================================================================

FUNCTION compose_query_from_tail(tail, relation, relation_ids,
                                relation_mask_embedding, hidden_dim):
    """
    Compose query from tail for head prediction
    Used for: (?, r, t) - compute q = compose(t ⊙ mask(r), r_inverse)
    
    Input:
        tail [batch, 2*d] - tail entity embedding
        relation [batch, d] - relation embedding
        relation_ids [batch] - relation IDs
    
    Output:
        query_re, query_im [batch, d] - normalized query components
    """
    
    // 1. Normalize tail entity
    tail_normalized = normalize_complex_embedding(tail)
    tail_re, tail_im = torch.chunk(tail_normalized, 2, dim=-1)
    tail_re = tail_re.squeeze(-1)
    tail_im = tail_im.squeeze(-1)
    
    // 2. Apply relation mask
    mask = relation_mask(relation_ids, relation_mask_embedding)
    tail_re = tail_re.unsqueeze(1) * mask
    tail_im = tail_im.unsqueeze(1) * mask
    tail_re = tail_re.squeeze(1)
    tail_im = tail_im.squeeze(1)
    
    // 3. Get relation rotation unit
    relation_re, relation_im = relation_unit(relation)
    
    // 4. Complex multiplication with CONJUGATE (inverse)
    // For (?, r, t): q = conj(t * r)
    // conj(a + bi) = a - bi
    // So: (a + bi) * conj(c + di) = (a + bi) * (c - di)
    //                              = (ac + bd) + (bc - ad)i
    query_re = tail_re * relation_re + tail_im * relation_im
    query_im = -tail_re * relation_im + tail_im * relation_re
    
    // 5. Normalize
    query_re_norm, query_im_norm = normalize_complex_pair(query_re, query_im)
    
    RETURN query_re_norm, query_im_norm

END FUNCTION


# ============================================================================
# FUNCTION: Score Complex (Dot Product)
# ============================================================================

FUNCTION score_complex(query_re, query_im, candidate_re, candidate_im):
    """
    Compute complex dot product (scoring)
    
    Input:
        query_re [batch, d], query_im [batch, d]
        candidate_re [batch, d], candidate_im [batch, d]
    
    Output:
        score [batch] - dot product values (higher = better match)
    """
    
    // Complex dot product: (a + bi) · (c + di) = ac + bd
    score = (query_re * candidate_re + query_im * candidate_im).sum(dim=-1)
    
    RETURN score

END FUNCTION


# ============================================================================
# FUNCTION: Forward Pass (Single Mode)
# ============================================================================

FUNCTION forward_single_mode(sample, model):
    """
    Forward pass for single mode (positive triple only)
    
    Input:
        sample [batch, 3] - (head_id, relation_id, tail_id)
        model - model object with embeddings
    
    Output:
        score [batch] - scores for positive triples
    """
    
    head_ids = sample[:, 0]
    relation_ids = sample[:, 1]
    tail_ids = sample[:, 2]
    
    // Get embeddings
    head = torch.index_select(model.entity_embedding, 0, head_ids)  // [batch, 2*d]
    relation = torch.index_select(model.relation_embedding, 0, relation_ids)  // [batch, d]
    tail = torch.index_select(model.entity_embedding, 0, tail_ids)  // [batch, 2*d]
    
    // Compose query from head
    query_re, query_im = compose_query_from_head(
        head, relation, relation_ids,
        model.relation_mask_embedding, model.hidden_dim
    )  // [batch, d] each
    
    // Normalize tail
    tail_normalized = normalize_complex_embedding(tail)
    tail_re, tail_im = torch.chunk(tail_normalized, 2, dim=-1)
    tail_re = tail_re.squeeze(-1)
    tail_im = tail_im.squeeze(-1)
    
    // Score
    score = score_complex(query_re, query_im, tail_re, tail_im)  // [batch]
    
    RETURN score

END FUNCTION


# ============================================================================
# FUNCTION: Forward Pass (Head-Batch Mode)
# ============================================================================

FUNCTION forward_head_batch_mode(sample, model):
    """
    Forward pass for head-batch mode (negative head sampling)
    Ranking: (?, r, t) predictions
    
    Input:
        sample[0] [batch, 3] - positive tail_part (head_pos, r, t_pos)
        sample[1] [batch, n_neg] - negative head candidates
    
    Output:
        score [batch, n_neg] - scores for all (head_neg, r, t_pos) combinations
    """
    
    tail_part, head_part = sample
    
    batch_size = head_part.shape[0]
    n_neg = head_part.shape[1]
    
    // Extract IDs
    relation_ids = tail_part[:, 1]  // [batch]
    tail_ids = tail_part[:, 2]      // [batch]
    
    // Get embeddings for relation and tail (same for all negatives)
    relation = torch.index_select(model.relation_embedding, 0, relation_ids)
    relation = relation.unsqueeze(1)  // [batch, 1, d]
    
    tail = torch.index_select(model.entity_embedding, 0, tail_ids)
    tail = tail.unsqueeze(1)  // [batch, 1, 2*d]
    
    // Compose queries from all tail + relation
    query_re, query_im = compose_query_from_tail(
        tail, relation, relation_ids,
        model.relation_mask_embedding, model.hidden_dim
    )  // [batch, 1, d]
    
    // Get negative head embeddings
    head_neg = torch.index_select(
        model.entity_embedding,
        0,
        head_part.view(-1)
    ).view(batch_size, n_neg, -1)  // [batch, n_neg, 2*d]
    
    // Normalize heads
    head_normalized = normalize_complex_embedding(head_neg)
    head_re, head_im = torch.chunk(head_normalized, 2, dim=-1)
    head_re = head_re.squeeze(-1)  // [batch, n_neg, d]
    head_im = head_im.squeeze(-1)
    
    // Score: broadcast query [batch, 1, d] against heads [batch, n_neg, d]
    score = score_complex(
        query_re,  // [batch, 1, d]
        query_im,
        head_re,   // [batch, n_neg, d]
        head_im
    )  // [batch, n_neg]
    
    RETURN score

END FUNCTION


# ============================================================================
# FUNCTION: Alignment Loss
# ============================================================================

FUNCTION alignment_loss(pos_scores, neg_scores, gamma):
    """
    Margin-ranking loss
    
    Input:
        pos_scores [batch] - scores of positive triples
        neg_scores [batch, n_neg] - scores of negative triples
        gamma - margin parameter
    
    Output:
        loss [scalar] - average margin loss
    """
    
    // Margin ranking: max(0, gamma + neg_score - pos_score)
    // Higher score is better, so we want pos_score > neg_score + margin
    
    pos_scores = pos_scores.unsqueeze(1)  // [batch, 1]
    
    // loss = max(0, gamma + neg - pos)
    loss = torch.clamp(
        gamma + neg_scores - pos_scores,
        min=0.0
    )  // [batch, n_neg]
    
    loss = loss.mean()
    
    RETURN loss

END FUNCTION


# ============================================================================
# FUNCTION: Uniformity Loss (Chunked)
# ============================================================================

FUNCTION uniformity_loss(embeddings, max_samples=170, chunk_size=128):
    """
    Compute uniformity loss to prevent representation collapse
    
    Input:
        embeddings [n_samples, 2*d] - batch of embeddings
        max_samples - maximum samples for computation (for memory)
        chunk_size - chunk size for O(n²) computation
    
    Output:
        loss [scalar] - uniformity loss
    """
    
    n = embeddings.shape[0]
    
    IF n < 2:
        RETURN torch.tensor(0.0, device=embeddings.device)
    
    // Optional subsampling
    IF max_samples > 0 AND n > max_samples:
        indices = torch.randperm(n)[:max_samples]
        embeddings = embeddings[indices]
        n = max_samples
    
    pair_sum = 0.0
    pair_count = 0
    
    // Chunked computation for pairwise distances
    FOR i_start IN range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        xi = embeddings[i_start:i_end]  // [chunk_i, 2*d]
        
        FOR j_start IN range(i_start, n, chunk_size):
            j_end = min(j_start + chunk_size, n)
            xj = embeddings[j_start:j_end]  // [chunk_j, 2*d]
            
            // Pairwise squared Euclidean distance
            diff = xi.unsqueeze(1) - xj.unsqueeze(0)  // [chunk_i, chunk_j, 2*d]
            dist_sq = (diff.pow(2)).sum(dim=-1)       // [chunk_i, chunk_j]
            
            // Exponential kernel (Gaussian)
            weights = torch.exp(-2.0 * dist_sq)       // [chunk_i, chunk_j]
            
            IF i_start == j_start:
                // Skip diagonal (same embeddings)
                diagonal_mask = torch.eye(i_end - i_start)
                valid_weights = weights.masked_select(~diagonal_mask)
                pair_sum = pair_sum + valid_weights.sum() * 0.5
                pair_count = pair_count + valid_weights.numel() // 2
            ELSE:
                pair_sum = pair_sum + weights.sum()
                pair_count = pair_count + weights.numel()
    
    IF pair_count == 0:
        RETURN torch.tensor(0.0, device=embeddings.device)
    
    loss = torch.log(pair_sum / pair_count)
    
    RETURN loss

END FUNCTION


# ============================================================================
# FUNCTION: Extract Unique Embeddings for Uniformity (Optimization)
# ============================================================================

FUNCTION extract_unique_embeddings_for_uniformity(head_ids, relation_ids, tail_ids,
                                                  model):
    """
    Optimize by encoding only unique queries and tails for uniformity computation
    
    Input:
        head_ids [batch]
        relation_ids [batch]
        tail_ids [batch]
    
    Output:
        Dictionary with unique embeddings and inverse mappings
    """
    
    // Find unique query pairs (head, relation)
    query_pairs = torch.stack([head_ids, relation_ids], dim=1)  // [batch, 2]
    unique_queries, q_inverse = torch.unique(
        query_pairs,
        dim=0,
        return_inverse=True
    )
    
    // Encode unique queries
    unique_head_ids = unique_queries[:, 0]
    unique_relation_ids = unique_queries[:, 1]
    
    // Get embeddings
    unique_heads = torch.index_select(model.entity_embedding, 0, unique_head_ids)
    unique_relations = torch.index_select(model.relation_embedding, 0, unique_relation_ids)
    
    // Compose queries
    unique_query_re, unique_query_im = compose_query_from_head(
        unique_heads, unique_relations, unique_relation_ids,
        model.relation_mask_embedding, model.hidden_dim
    )
    unique_query_emb = torch.cat([unique_query_re, unique_query_im], dim=-1)
    
    // Find unique tails
    unique_tails, t_inverse = torch.unique(tail_ids, return_inverse=True)
    
    // Encode unique tails
    unique_tail_embs = torch.index_select(model.entity_embedding, 0, unique_tails)
    unique_tail_embs = normalize_complex_embedding(unique_tail_embs)
    
    RETURN {
        'unique_query_emb': unique_query_emb,
        'unique_tail_emb': unique_tail_embs,
        'q_inverse': q_inverse,
        't_inverse': t_inverse,
    }

END FUNCTION


# ============================================================================
# FUNCTION: Training Step
# ============================================================================

FUNCTION train_step(batch, model, optimizer, config):
    """
    Single training step with one batch
    
    Input:
        batch - (positive_sample, negative_sample, weight, mode)
        model - model object
        optimizer - optimizer
        config - configuration
    
    Output:
        loss [scalar] - total loss for batch
    """
    
    positive_sample, negative_sample, weight, mode = batch
    
    // FORWARD: Compute positive scores
    IF mode == 'head-batch':
        sample = (positive_sample, negative_sample)
        pos_score = forward_head_batch_mode(sample, model)
    ELSE:  // tail-batch
        sample = (positive_sample, negative_sample)
        // Similar for tail-batch mode
        pos_score = forward_tail_batch_mode(sample, model)
    
    // LOSS 1: Alignment (Margin Ranking)
    // For head-batch: first element is positive head
    pos_score_single = pos_score[:, 0]  // [batch]
    neg_scores = pos_score[:, 1:]       // [batch, n_neg-1]
    
    loss_align = alignment_loss(pos_score_single, neg_scores, model.gamma.item())
    
    // LOSS 2: Uniformity (Optional)
    loss_uni = 0.0
    
    IF config.gamma_uni > 0:
        head_ids = positive_sample[:, 0]
        relation_ids = positive_sample[:, 1]
        tail_ids = positive_sample[:, 2]
        
        unique_embs = extract_unique_embeddings_for_uniformity(
            head_ids, relation_ids, tail_ids, model
        )
        
        loss_uni_q = uniformity_loss(unique_embs['unique_query_emb'])
        loss_uni_t = uniformity_loss(unique_embs['unique_tail_emb'])
        loss_uni = 0.5 * (loss_uni_q + loss_uni_t)
    
    // TOTAL LOSS
    loss = loss_align + config.gamma_uni * loss_uni
    
    // BACKWARD & OPTIMIZE
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    RETURN loss.item()

END FUNCTION


# ============================================================================
# FUNCTION: Complete Training Loop
# ============================================================================

FUNCTION train_model(train_data, valid_data, test_data, model, optimizer, config):
    """
    Complete training with validation and early stopping
    
    Input:
        train_data - training dataset
        valid_data - validation dataset
        test_data - test dataset
        model - model object
        optimizer - optimizer
        config - configuration
    """
    
    best_mrr = 0.0
    patience_counter = 0
    current_step = 0
    
    FOR step IN range(config.max_steps):
        
        // Get batch from loader
        batch = get_next_batch(train_data)
        
        // Training step
        loss = train_step(batch, model, optimizer, config)
        current_step += 1
        
        // Log
        IF current_step % config.log_steps == 0:
            LOG "Step {current_step}: Loss = {loss:.6f}"
        
        // Validation
        IF current_step % config.valid_steps == 0:
            
            metrics = validate_model(model, valid_data, config)
            current_mrr = metrics['mrr']
            
            LOG "Step {current_step}: Valid MRR = {current_mrr:.4f}"
            
            // Save best checkpoint
            IF current_mrr > best_mrr:
                best_mrr = current_mrr
                save_checkpoint(model, current_step)
                patience_counter = 0
            ELSE:
                patience_counter += 1
            
            // Early stopping
            IF patience_counter >= config.patience:
                LOG "Early stopping at step {current_step}"
                BREAK
        
        // Checkpoint
        IF current_step % config.save_checkpoint_steps == 0:
            save_checkpoint(model, current_step)
    
    // Load best model
    best_model = load_checkpoint(best_model_path)
    
    // Final test
    test_results = test_model(best_model, test_data, config)
    
    RETURN test_results

END FUNCTION
```

---

## Validation

```python
# ============================================================================
# FUNCTION: Validate Model
# ============================================================================

FUNCTION validate_model(model, valid_data, config):
    """
    Evaluate model on validation set
    
    Input:
        model - trained model
        valid_data - validation triples
        config - configuration
    
    Output:
        metrics - validation metrics
    """
    
    model.eval()
    
    all_mrrs = []
    all_mrs = []
    all_hits1 = []
    all_hits3 = []
    all_hits10 = []
    
    // Pre-compute all entity embeddings
    all_entity_embs = normalize_complex_embedding(model.entity_embedding)
    
    FOR triple IN valid_data:
        head_id, relation_id, tail_id = triple
        
        // TAIL PREDICTION
        query_re, query_im = compose_query_from_head(
            all_entity_embs[head_id].unsqueeze(0),
            model.relation_embedding[relation_id].unsqueeze(0),
            torch.tensor([relation_id]),
            model.relation_mask_embedding,
            model.hidden_dim
        )
        
        tail_re, tail_im = torch.chunk(all_entity_embs, 2, dim=-1)
        scores_tail = score_complex(query_re, query_im, tail_re, tail_im)
        
        metrics_tail = ranking_metrics(scores_tail, tail_id)
        
        // HEAD PREDICTION
        query_re, query_im = compose_query_from_tail(
            all_entity_embs[tail_id].unsqueeze(0),
            model.relation_embedding[relation_id].unsqueeze(0),
            torch.tensor([relation_id]),
            model.relation_mask_embedding,
            model.hidden_dim
        )
        
        scores_head = score_complex(query_re, query_im, tail_re, tail_im)
        metrics_head = ranking_metrics(scores_head, head_id)
        
        // Average
        all_mrrs.append((metrics_tail['mrr'] + metrics_head['mrr']) / 2)
        all_mrs.append((metrics_tail['mr'] + metrics_head['mr']) / 2)
        all_hits1.append((metrics_tail['hits@1'] + metrics_head['hits@1']) / 2)
        all_hits3.append((metrics_tail['hits@3'] + metrics_head['hits@3']) / 2)
        all_hits10.append((metrics_tail['hits@10'] + metrics_head['hits@10']) / 2)
    
    model.train()
    
    RETURN {
        'mrr': mean(all_mrrs),
        'mr': mean(all_mrs),
        'hits@1': mean(all_hits1),
        'hits@3': mean(all_hits3),
        'hits@10': mean(all_hits10),
    }

END FUNCTION
```

---

## Testing/Inference

```python
# ============================================================================
# FUNCTION: Test Model
# ============================================================================

FUNCTION test_model(model, test_data, config):
    """
    Full inference on test set with optional filtering
    
    Input:
        model - trained model
        test_data - test triples
        config - configuration
    
    Output:
        results - test results with/without filtering
    """
    
    model.eval()
    
    // Build valid triple sets from training data
    valid_head = build_valid_head_dict(config.train_triples)
    valid_tail = build_valid_tail_dict(config.train_triples)
    
    all_metrics_no_filter = []
    all_metrics_filter = []
    
    // Pre-compute all entity embeddings
    all_entity_embs = normalize_complex_embedding(model.entity_embedding)
    
    FOR triple IN test_data:
        head_id, relation_id, tail_id = triple
        
        // TAIL PREDICTION (h, r, ?)
        
        // Query composition
        query_re, query_im = compose_query_from_head(
            all_entity_embs[head_id].unsqueeze(0),
            model.relation_embedding[relation_id].unsqueeze(0),
            torch.tensor([relation_id]),
            model.relation_mask_embedding,
            model.hidden_dim
        )
        
        tail_re, tail_im = torch.chunk(all_entity_embs, 2, dim=-1)
        scores_tail = score_complex(query_re, query_im, tail_re, tail_im)
        
        // Without filtering
        metrics_tail_no_filter = ranking_metrics(scores_tail, tail_id)
        
        // With filtering
        IF config.use_filter:
            valid_entities = valid_tail[(head_id, relation_id)]
            scores_tail_filter = scores_tail.clone()
            invalid_mask = ~torch.isin(torch.arange(model.n_entity), valid_entities)
            scores_tail_filter[invalid_mask] = -infinity
            
            metrics_tail_filter = ranking_metrics(scores_tail_filter, tail_id)
        
        // HEAD PREDICTION (?, r, t)
        
        query_re, query_im = compose_query_from_tail(
            all_entity_embs[tail_id].unsqueeze(0),
            model.relation_embedding[relation_id].unsqueeze(0),
            torch.tensor([relation_id]),
            model.relation_mask_embedding,
            model.hidden_dim
        )
        
        scores_head = score_complex(query_re, query_im, tail_re, tail_im)
        
        // Without filtering
        metrics_head_no_filter = ranking_metrics(scores_head, head_id)
        
        // With filtering
        IF config.use_filter:
            valid_entities = valid_head[(relation_id, tail_id)]
            scores_head_filter = scores_head.clone()
            invalid_mask = ~torch.isin(torch.arange(model.n_entity), valid_entities)
            scores_head_filter[invalid_mask] = -infinity
            
            metrics_head_filter = ranking_metrics(scores_head_filter, head_id)
        
        // Aggregate
        metrics_no_filter = average_metrics([metrics_tail_no_filter, metrics_head_no_filter])
        all_metrics_no_filter.append(metrics_no_filter)
        
        IF config.use_filter:
            metrics_filter = average_metrics([metrics_tail_filter, metrics_head_filter])
            all_metrics_filter.append(metrics_filter)
    
    // Final results
    results = {
        'no_filter': aggregate_metrics(all_metrics_no_filter),
        'filter': aggregate_metrics(all_metrics_filter) IF config.use_filter ELSE None
    }
    
    RETURN results

END FUNCTION
```

---

## Loss Functions

```python
# ============================================================================
# FUNCTION: Ranking Metrics
# ============================================================================

FUNCTION ranking_metrics(scores, target_id, k_list=[1, 3, 10]):
    """
    Compute ranking metrics: MR, MRR, Hits@K
    
    Input:
        scores [n_entity] - entity scores
        target_id - true entity ID
        k_list - list of K values
    
    Output:
        metrics dictionary
    """
    
    _, sorted_indices = torch.sort(scores, descending=True)
    
    target_rank = torch.nonzero(sorted_indices == target_id)[0, 0] + 1
    
    RETURN {
        'mr': target_rank.item(),
        'mrr': 1.0 / target_rank.item(),
        'hits@1': 1 IF target_rank <= 1 ELSE 0,
        'hits@3': 1 IF target_rank <= 3 ELSE 0,
        'hits@10': 1 IF target_rank <= 10 ELSE 0,
    }

END FUNCTION
```

---

## Helper Functions

```python
# ============================================================================
# FUNCTION: Build Valid Tail Dictionary
# ============================================================================

FUNCTION build_valid_tail_dict(triples):
    """
    Build dictionary mapping (head, relation) -> set of valid tails
    for filtering during evaluation
    """
    
    valid_tail = {}
    
    FOR head, relation, tail IN triples:
        key = (head, relation)
        
        IF key NOT IN valid_tail:
            valid_tail[key] = set()
        
        valid_tail[key].add(tail)
    
    RETURN valid_tail

END FUNCTION


# ============================================================================
# FUNCTION: Aggregate Metrics
# ============================================================================

FUNCTION aggregate_metrics(metrics_list):
    """
    Average metrics across all test triples
    """
    
    IF len(metrics_list) == 0:
        RETURN None
    
    avg_mr = mean([m['mr'] for m in metrics_list])
    avg_mrr = mean([m['mrr'] for m in metrics_list])
    avg_hits1 = mean([m['hits@1'] for m in metrics_list])
    avg_hits3 = mean([m['hits@3'] for m in metrics_list])
    avg_hits10 = mean([m['hits@10'] for m in metrics_list])
    
    RETURN {
        'MR': avg_mr,
        'MRR': avg_mrr,
        'Hits@1': avg_hits1,
        'Hits@3': avg_hits3,
        'Hits@10': avg_hits10,
    }

END FUNCTION
```

---

## Complete Training Pseudocode Summary

```
BEGIN TRAINING

    Initialize model embeddings
    Initialize optimizer
    Load training/validation/test data
    
    FOR each training epoch:
        
        FOR each batch:
            // 1. Forward pass
            pos_scores ← forward_pass(positive_triples)
            neg_scores ← forward_pass(negative_triples)
            
            // 2. Compute losses
            loss_align ← margin_ranking_loss(pos_scores, neg_scores)
            loss_uni ← uniformity_loss(unique_embeddings)
            loss_total ← loss_align + γ_uni * loss_uni
            
            // 3. Backward & optimize
            loss_total.backward()
            optimizer.step()
        
        // 4. Validation
        valid_metrics ← validate(validation_set)
        
        // 5. Save best checkpoint
        IF valid_metrics['mrr'] > best_mrr:
            save_checkpoint()
        
        // 6. Early stopping
        IF patience_exceeded:
            break
    
    // 7. Load best model
    Load best checkpoint
    
    // 8. Final test
    test_metrics ← test(test_set)
    
    RETURN test_metrics

END TRAINING
```

---

## Configuration Example

```yaml
# DirectAU-RotatE Configuration

# Model
model: DirectAU_RotatE
hidden_dim: 500
gamma: 12.0
double_entity_embedding: true
double_relation_embedding: false

# Data
dataset: wn18rr
data_path: ./data/wn18rr/
batch_size: 1024
test_batch_size: 4
negative_sample_size: 128

# Training
learning_rate: 0.0001
max_steps: 100000
warm_up_steps: null

# Loss weights
gamma_uni: 1.0
gamma_neg: 1.0
regularization: 0.0

# Validation & Checkpointing
valid_steps: 10000
save_checkpoint_steps: 10000
log_steps: 100

# Optimization
negative_adversarial_sampling: true
adversarial_temperature: 1.0
uni_weight: false

# Inference
use_filter: true
```

