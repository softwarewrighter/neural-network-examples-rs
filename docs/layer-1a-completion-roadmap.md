# Layer 1a Completion Roadmap

**Repository**: neural-network-examples-rs (THIS REPO - Layer 1a)
**Purpose**: Core building blocks used by Layer 2 specialized models
**Status**: ~50% complete
**Goal**: Complete all components that will be imported by TRM, Text-Diffusion, RAG, etc.

**Note**: CNN and RNN deferred to Layer 1b (neural-network-concepts-rs) - educational only

---

## Current State ✅

**Completed Examples**:
- ✅ example-1-forward-propagation (XOR with random weights)
- ✅ example-2-backward-propagation-and
- ✅ example-2-backward-propagation-or
- ✅ example-2-backward-propagation-xor
- ✅ example-3-complex-boolean-parity-3bit
- ✅ example-3-complex-boolean-majority-3
- ✅ example-3-multi-output-half-adder
- ✅ example-3-multi-output-full-adder
- ✅ example-4-modern-optimizers ⭐ **DONE**

**Completed Crates**:
- ✅ neural_net_types (Layer, Network, error types, serialization)
- ✅ neural_net_core (forward, backward, optimizers, activation)
- ✅ neural_net_viz (SVG visualization)

**Completed Documentation**:
- ✅ README with project overview
- ✅ docs/learnings.md (lessons learned)
- ✅ docs/history.md (C++ to Rust migration)
- ✅ docs/multi-repo-strategy.md (repository separation strategy)
- ✅ Each example has comprehensive inline documentation
- ✅ All examples have tests (negative + positive)

---

## Remaining Building Blocks

### Priority 1: Essential (Used by ALL Layer 2 Models) ⭐

#### 1. ✅ Modern Optimizers (Example-4) **DONE**
**What**: Adam, RMSprop, AdamW, SGD+Momentum
**Why**: Industry standard, required by TRM, Diffusion, RAG, everything
**Status**: ✅ Complete
**Timeline**: Week 1-2

#### 2. Modern Activations (Example-5) ⭐ **NEXT**
**What**: ReLU, Leaky ReLU, GELU, Swish
**Why**: Prevent vanishing gradients, required for deep networks, TRM uses GELU
**Timeline**: 1 week (Week 3)
**Complexity**: Low
**Used by**: TRM, Text-Diffusion, all Layer 2 models

#### 3. Deep Networks + Normalization (Example-6)
**What**: 4-6 layers, residual connections, layer normalization
**Why**: TRM uses deep supervision, transformers use layer norm
**Timeline**: 1-2 weeks (Week 4-5)
**Complexity**: Medium
**Used by**: TRM (deep supervision), Text-Diffusion (transformers)

#### 4. Regularization (Example-7)
**What**: Dropout, L1/L2 regularization
**Why**: Prevent overfitting in larger models
**Timeline**: 1 week (Week 6)
**Complexity**: Low
**Used by**: Text-Diffusion, possibly TRM

#### 5. Attention Mechanism (Example-8) ⭐⭐⭐
**What**: Scaled dot-product attention, multi-head attention
**Why**: Foundation for transformers, critical for Text-Diffusion
**Timeline**: 2-3 weeks (Week 7-9)
**Complexity**: High
**Used by**: Text-Diffusion (core component), possibly RAG

#### 6. Embeddings (Example-9) ⭐⭐⭐
**What**: Learned embeddings, positional encoding
**Why**: Map discrete tokens to continuous, needed for Text-Diffusion, TRM
**Timeline**: 1-2 weeks (Week 10-11)
**Complexity**: Medium
**Used by**: Text-Diffusion, TRM, RAG

#### 7. Training Utilities (Example-10)
**What**: Learning rate schedules, early stopping, gradient clipping
**Why**: Training stability for complex models
**Timeline**: 1 week (Week 12)
**Complexity**: Low
**Used by**: All Layer 2 models

---

## Deferred to Layer 1b (Educational Concepts)

These are valuable for learning but **not required** for our Layer 2 models:

- ❌ CNNs (Conv2D, MaxPool) → neural-network-concepts-rs
- ❌ RNNs (LSTM, GRU) → neural-network-concepts-rs
- ❌ Advanced CNN architectures → neural-network-concepts-rs
- ❌ Sequence-to-sequence models → neural-network-concepts-rs

**Why deferred?**
1. Not used by TRM, Text-Diffusion, RAG, or SNNs
2. Still valuable for education (will be in separate repo)
3. Keeps THIS repo focused on what Layer 2 actually needs
4. Students can skip to Layer 2 after completing THIS repo

---

## Detailed Plan for Remaining Examples

### Example-5: Modern Activations (Week 3) ⭐ NEXT

**Goal**: Show ReLU/GELU > Sigmoid for deep networks

**What to Implement**:
```rust
// In neural_net_core/src/activation.rs

pub trait Activation {
    fn forward(&self, x: f32) -> f32;
    fn backward(&self, x: f32) -> f32;
}

pub struct ReLU;
impl Activation for ReLU {
    fn forward(&self, x: f32) -> f32 {
        x.max(0.0)
    }
    fn backward(&self, x: f32) -> f32 {
        if x > 0.0 { 1.0 } else { 0.0 }
    }
}

pub struct LeakyReLU { pub alpha: f32 }  // alpha = 0.01
pub struct GELU;  // Used in transformers, TRM
pub struct Swish;  // x * sigmoid(x)
pub struct Tanh;
```

**Example Task**: Deep network (4-6 layers) showing gradient flow

**Expected Output**:
```
=== Activation Comparison (4-layer network) ===

Sigmoid:
  Layer 1 gradient: 0.12
  Layer 2 gradient: 0.06
  Layer 3 gradient: 0.02 (vanishing!)
  Layer 4 gradient: 0.005 (almost dead)
  Failed to train

ReLU:
  All layers have healthy gradients
  Accuracy: 94% ⭐

GELU (used in transformers/TRM):
  Smooth approximation to ReLU
  Accuracy: 95% ⭐
```

**Deliverables**:
- [ ] Refactor activation trait (currently exists, needs extension)
- [ ] ReLU, Leaky ReLU, GELU, Swish, Tanh implementations
- [ ] Example with 4-layer network
- [ ] Gradient flow visualization
- [ ] **Tests for each activation** (negative + positive required - see docs/learnings.md)
- [ ] Documentation with mathematical formulas

**Testing Requirement** (ALL EXAMPLES):
- ⚠️ **MANDATORY**: Each example MUST include negative and positive tests
- Negative test: `test_<task>_untrained_has_high_error()` - Verifies problem non-trivial
- Positive test: `test_<task>_network_trains()` - Verifies training works
- See `docs/learnings.md` "Testing Strategy" section for detailed pattern

**Timeline**: 1 week

---

### Example-6: Deep Networks + Normalization (Week 4-5)

**Goal**: Build deeper networks with residual connections and normalization

**What to Implement**:
```rust
// In neural_net_core/src/layer.rs

pub struct LayerNorm {
    epsilon: f32,  // 1e-5
    gamma: Array1<f32>,  // Scale
    beta: Array1<f32>,   // Shift
}

impl LayerNorm {
    fn forward(&self, x: &Array1<f32>) -> Array1<f32> {
        let mean = x.mean().unwrap();
        let var = x.var(0.0);
        let normalized = (x - mean) / (var + self.epsilon).sqrt();
        &self.gamma * &normalized + &self.beta
    }
}

// Residual connections (conceptual)
fn residual_block(x: &Array, layer: &Layer) -> Array {
    x + layer.forward(x)  // Skip connection
}
```

**Example Task**: 6-layer network on complex task

**Expected Output**:
```
=== Deep Network Comparison ===

6-layer without residuals/normalization:
  Gradient vanishes
  Fails to converge

6-layer with layer normalization:
  Trains successfully
  Accuracy: 92% ⭐

6-layer with residuals + layer norm:
  Faster convergence
  Accuracy: 96% ⭐
```

**Deliverables**:
- [ ] Layer normalization
- [ ] Batch normalization (optional)
- [ ] Residual connection pattern
- [ ] Example with 4-6 layer networks
- [ ] Gradient flow analysis
- [ ] Tests
- [ ] Documentation

**Timeline**: 1-2 weeks

---

### Example-7: Regularization (Week 6)

**Goal**: Prevent overfitting

**What to Implement**:
```rust
pub struct Dropout {
    pub p: f32,  // Dropout probability (e.g., 0.5)
    pub training: bool,
}

impl Dropout {
    pub fn forward(&self, x: &Array, rng: &mut impl Rng) -> Array {
        if self.training {
            let mask = Array::random_using(x.dim(), Uniform::new(0.0, 1.0), rng);
            let scale = 1.0 / (1.0 - self.p);
            x * mask.mapv(|v| if v > self.p { scale } else { 0.0 })
        } else {
            x.clone()
        }
    }
}

pub fn l1_regularization(weights: &Array, lambda: f32) -> f32 {
    lambda * weights.mapv(|x| x.abs()).sum()
}

pub fn l2_regularization(weights: &Array, lambda: f32) -> f32 {
    lambda * weights.mapv(|x| x * x).sum()
}
```

**Example Task**: Deliberately overfit, then show regularization helps

**Expected Output**:
```
=== Overfitting Demo ===

Large network (100 hidden units), small data (50 examples):

No regularization:
  Train accuracy: 100%
  Test accuracy: 62% (overfitting!)

With Dropout (p=0.5):
  Train accuracy: 94%
  Test accuracy: 86% ⭐

With L2 (lambda=0.01):
  Train accuracy: 96%
  Test accuracy: 88% ⭐
```

**Deliverables**:
- [ ] Dropout layer
- [ ] L1/L2 regularization functions
- [ ] Example demonstrating overfitting prevention
- [ ] Tests
- [ ] Documentation

**Timeline**: 1 week

---

### Example-8: Attention Mechanism (Week 7-9) ⭐⭐⭐

**Goal**: Foundation for transformers (Text-Diffusion)

**What to Implement**:
```rust
// In neural_net_core/src/attention.rs

pub struct ScaledDotProductAttention {
    scale: f32,  // 1/sqrt(d_k)
}

impl ScaledDotProductAttention {
    pub fn forward(&self, Q: &Array2<f32>, K: &Array2<f32>, V: &Array2<f32>) -> Array2<f32> {
        // Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
        let scores = Q.dot(&K.t()) / self.scale;
        let weights = softmax(&scores, Axis(1));
        weights.dot(V)
    }
}

pub struct MultiHeadAttention {
    num_heads: usize,
    d_model: usize,
    W_Q: Array2<f32>,
    W_K: Array2<f32>,
    W_V: Array2<f32>,
    W_O: Array2<f32>,
}
```

**Example Tasks**:
1. Simple attention on sequences
2. Multi-head attention
3. Attention weight visualization

**Expected Output**:
```
=== Attention Mechanism ===

Input sequence: [x, =, y, +, z]
Query: "What variables are used?"

Attention weights:
  x: 0.45 ⭐
  =: 0.05
  y: 0.42 ⭐
  +: 0.02
  z: 0.43 ⭐

Attention learned to focus on variables!
```

**Deliverables**:
- [ ] Scaled dot-product attention
- [ ] Multi-head attention
- [ ] Self-attention
- [ ] Example demonstrating attention
- [ ] Attention weight visualization
- [ ] Tests
- [ ] Documentation with mathematical formulas

**Timeline**: 2-3 weeks

---

### Example-9: Embeddings (Week 10-11) ⭐⭐⭐

**Goal**: Map discrete tokens to continuous vectors

**What to Implement**:
```rust
// In neural_net_core/src/embedding.rs

pub struct Embedding {
    table: Array2<f32>,  // [vocab_size, embedding_dim]
}

impl Embedding {
    pub fn forward(&self, tokens: &[usize]) -> Array2<f32> {
        // Lookup embeddings for each token
        Array2::from_shape_fn((tokens.len(), self.table.ncols()), |(i, j)| {
            self.table[[tokens[i], j]]
        })
    }

    pub fn backward(&mut self, tokens: &[usize], grad_output: &Array2<f32>) {
        // Accumulate gradients for each token's embedding
        for (i, &token) in tokens.iter().enumerate() {
            for j in 0..self.table.ncols() {
                self.table[[token, j]] += grad_output[[i, j]];
            }
        }
    }
}

pub struct PositionalEncoding {
    max_len: usize,
    d_model: usize,
    encoding: Array2<f32>,
}

impl PositionalEncoding {
    pub fn new(max_len: usize, d_model: usize) -> Self {
        let mut encoding = Array2::zeros((max_len, d_model));
        for pos in 0..max_len {
            for i in 0..(d_model / 2) {
                let angle = (pos as f32) / 10000_f32.powf(2.0 * i as f32 / d_model as f32);
                encoding[[pos, 2 * i]] = angle.sin();
                encoding[[pos, 2 * i + 1]] = angle.cos();
            }
        }
        Self { max_len, d_model, encoding }
    }
}
```

**Example Task**: Learn word similarities

**Expected Output**:
```
=== Learned Embeddings ===

Vocabulary: [function, def, class, struct, int, str, ...]

Similar words (cosine similarity):
  function ↔ def: 0.89
  class ↔ struct: 0.92
  int ↔ str: 0.71

Embeddings captured semantics!
```

**Deliverables**:
- [ ] Embedding layer with backprop
- [ ] Positional encoding
- [ ] Example: learning embeddings on small vocab
- [ ] Similarity visualization
- [ ] Tests
- [ ] Documentation

**Timeline**: 1-2 weeks

---

### Example-10: Training Utilities (Week 12)

**Goal**: Improve training stability and efficiency

**What to Implement**:
```rust
pub struct LearningRateSchedule {
    initial_lr: f32,
    warmup_steps: usize,
    current_step: usize,
}

impl LearningRateSchedule {
    pub fn get_lr(&mut self) -> f32 {
        self.current_step += 1;
        if self.current_step < self.warmup_steps {
            // Linear warmup
            self.initial_lr * (self.current_step as f32 / self.warmup_steps as f32)
        } else {
            // Cosine decay or constant
            self.initial_lr
        }
    }
}

pub struct EarlyStopping {
    patience: usize,
    best_loss: f32,
    counter: usize,
}

impl EarlyStopping {
    pub fn should_stop(&mut self, current_loss: f32) -> bool {
        if current_loss < self.best_loss {
            self.best_loss = current_loss;
            self.counter = 0;
            false
        } else {
            self.counter += 1;
            self.counter >= self.patience
        }
    }
}

pub fn gradient_clipping(grads: &mut Array2<f32>, max_norm: f32) {
    let norm = grads.mapv(|x| x * x).sum().sqrt();
    if norm > max_norm {
        *grads *= max_norm / norm;
    }
}
```

**Example Task**: Show training improvements

**Expected Output**:
```
=== Training Improvements ===

Without learning rate warmup:
  Diverges in first 100 iterations

With warmup (500 steps):
  Stable training, converges ⭐

Without gradient clipping:
  Exploding gradients at iteration 234

With gradient clipping (max_norm=1.0):
  Stable throughout ⭐
```

**Deliverables**:
- [ ] Learning rate schedules (warmup, cosine decay)
- [ ] Early stopping
- [ ] Gradient clipping
- [ ] Example showing benefits
- [ ] Tests
- [ ] Documentation

**Timeline**: 1 week

---

## Completion Timeline

| Week | Example | Components | Status |
|------|---------|------------|--------|
| 1-2 | Example-4 | Optimizers (Adam, RMSprop, AdamW) | ✅ **DONE** |
| 3 | Example-5 | Activations (ReLU, GELU, Swish) | 🔲 **NEXT** |
| 4-5 | Example-6 | Deep networks, residuals, layer norm | 🔲 |
| 6 | Example-7 | Regularization (Dropout, L1/L2) | 🔲 |
| 7-9 | Example-8 | Attention mechanisms | 🔲 |
| 10-11 | Example-9 | Embeddings | 🔲 |
| 12 | Example-10 | Training utilities | 🔲 |

**Total**: ~12 weeks (~3 months)

**Note**: CNN (Example-8) and RNN (Example-7) from original plan moved to Layer 1b

---

## After Completion

Once all examples are complete, THIS repo (Layer 1a) will have:

✅ **Core building blocks** imported by Layer 2 models
✅ **Reusable crates** (neural_net_core, neural_net_types, neural_net_viz)
✅ **Comprehensive examples** (10 examples total)
✅ **Full test coverage** (95+ tests)
✅ **Excellent documentation** (theory + practice + roadmaps)
✅ **Visualizations** for understanding

Then we can:
1. **Start Layer 2 repos**: TRM, Text-Diffusion, RAG
2. **Optionally create Layer 1b**: neural-network-concepts-rs (CNN, RNN for education)
3. **Build on solid foundation** without revisiting basics

---

## Recommended Order

**Follow sequential order** (don't jump ahead):

1. ✅ Example-4: Optimizers **DONE**
2. **Example-5: Activations** ⭐ **NEXT** (1 week)
3. Example-6: Deep Networks (1-2 weeks)
4. Example-7: Regularization (1 week)
5. Example-8: Attention (2-3 weeks)
6. Example-9: Embeddings (1-2 weeks)
7. Example-10: Training Utilities (1 week)

**Why sequential?**
- Each builds on previous concepts
- Avoids complexity jumps
- Maintains clear learning path
- No dependencies skipped

---

## Next Steps

**Immediate**: Start Example-5 (Modern Activations)

1. Create `examples/example-5-modern-activations/` directory
2. Extend activation trait in `neural_net_core/src/activation.rs`
3. Implement ReLU, LeakyReLU, GELU, Swish
4. Create comparison example on 4-layer network
5. Add gradient flow visualization
6. Write tests and documentation

**Timeline**: 1 week

---

**Last Updated**: 2025-10-14
**Completion Target**: ~12 weeks from now (mid-January 2026)
**After Layer 1a**: Start Layer 2 (TRM) or Layer 1b (CNN/RNN educational)
