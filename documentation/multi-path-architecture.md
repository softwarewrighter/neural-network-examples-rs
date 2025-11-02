# Multi-Path Architecture: Toward Efficient Cooperative AI Agents

**Vision**: Build a cooperative team of specialist AI coding agents that:
- Run on old/affordable hardware (multi-core CPUs, old GPUs with high VRAM)
- Achieve higher accuracy per kWh than GPT/LLMs
- Learn continuously without full retraining
- Generate high-quality software on predictable cadence
- Operate sustainably (solar-powered, low carbon footprint)

---

## Current State

✅ **What We Have:**
- Basic feedforward networks (3-layer)
- Backpropagation (SGD)
- Boolean logic + simple arithmetic
- Checkpoint save/load
- SVG visualization

✅ **Foundation Established:**
- Clean Rust codebase (no ML framework dependencies)
- Educational examples with comprehensive tests
- Modular architecture (neural_net_core, neural_net_viz)

---

## The Multi-Path Strategy

Instead of one path to one technology, we build **parallel learning tracks** that:
1. Share a common efficient foundation
2. Reach different specialized capabilities
3. Can be combined for hybrid approaches
4. Optimize for different constraints (memory, compute, energy)

```
                    Shared Foundation (Phase 0-2)
                              |
        ┌─────────────────────┼─────────────────────┐
        |                     |                     |
   Path A: HRM/TRM       Path B: Diffusion     Path C: BDH/SNN
   (Reasoning)           (Generation)          (Biological)
        |                     |                     |
        └──────────┬──────────┴──────────┬──────────┘
                   |                     |
            Path D: RAG              Path E: Hybrid
         (Knowledge+Learning)      (Combinations)
```

---

## Shared Foundation (Benefits ALL Paths)

### Phase 0: Core Efficiency (Example-4) - 1-2 weeks

**What**: Optimize what we have for efficiency

**Implement:**
- [ ] Memory profiling (track allocations)
- [ ] SIMD operations (ndarray with BLAS)
- [ ] Quantization (f32 → f16 or int8)
- [ ] Benchmark suite (ops/sec, memory/sample)
- [ ] CPU cache optimization

**Why This Matters:**
- Establishes efficiency baseline
- Measures "accuracy per kWh"
- Proves we can beat GPUs on old CPUs for small models
- Foundation for all specialist agents

**Example Output:**
```
=== XOR Benchmark ===
Operations: 1M forward passes
Time: 2.3s (CPU), 1.8s (GPU)
Energy: 5.2 Wh (CPU), 12.1 Wh (GPU)
Accuracy/kWh: CPU wins 2.3× ⭐

CPU: Intel Xeon E5-2670 (2012, $20 used)
GPU: Comparable performance to RTX 3060
```

**Deliverable**: Benchmark comparing old CPU vs modern GPU on simple tasks

---

### Phase 1: Better Training (Example-5) - 2-3 weeks

**What**: Modern optimizers and activations

**Implement:**
- [ ] ReLU, Leaky ReLU, GELU activations
- [ ] Adam optimizer (industry standard)
- [ ] Learning rate schedules (warmup, decay)
- [ ] Gradient clipping (stability)
- [ ] Mixed precision training (f16 where safe)

**Why This Matters:**
- Required by ALL advanced models
- Faster convergence = less energy
- Better generalization = smaller models
- Foundation for HRM, TRM, Diffusion, everything

**Example Output:**
```
=== Training Efficiency Comparison ===
Task: 3-bit parity (1000 examples)

SGD + Sigmoid:
  - 5000 iterations, 45 seconds
  - Final accuracy: 92%
  - Energy: 18 Wh

Adam + ReLU:
  - 1200 iterations, 12 seconds ⭐
  - Final accuracy: 97%
  - Energy: 4.3 Wh ⭐
  - 4.2× more efficient!
```

**Deliverable**: "example-5-efficient-training" comparing optimizers

---

### Phase 2: Deeper & Modular (Example-6) - 2-3 weeks

**What**: Build flexible, composable architectures

**Implement:**
- [ ] Layer abstraction (Dense, Conv, Norm, etc.)
- [ ] ModuleList / Sequential composition
- [ ] 4-6 layer networks
- [ ] Residual connections (optional)
- [ ] Layer normalization
- [ ] Dropout (regularization)

**Why This Matters:**
- Flexible foundation for all architectures
- Easy to experiment with different designs
- Required for transformers, SNNs, everything
- Modular = reusable across paths

**Example Output:**
```rust
// Easy composition
let model = Sequential::new(vec![
    Dense::new(10, 64).with_activation(ReLU),
    LayerNorm::new(64),
    Dense::new(64, 32).with_activation(ReLU),
    Dense::new(32, 1).with_activation(Sigmoid),
]);

// Or custom:
struct MyNetwork {
    encoder: Sequential,
    decoder: Sequential,
}
```

**Deliverable**: "example-6-modular-architecture" with flexible layer system

---

## Path A: Recursive Reasoning (HRM → TRM)

**Goal**: Small models that iteratively refine answers (like human reasoning)

**Use Case**: Code debugging, logical reasoning, constraint satisfaction

**Efficiency**: Tiny models (5-30M params) beat LLMs (100B+) on specific tasks

---

### Phase A1: Simple Recursion (Example-7a) - 1 week

**Concept**: Reuse same network multiple times

**Implement:**
```rust
// Simple recursive network
fn recursive_forward(net: &Network, x: &Array, depth: usize) -> Array {
    let mut output = x.clone();
    for _ in 0..depth {
        output = net.forward(&output);
    }
    output
}
```

**Task**: XOR chain - XOR(XOR(XOR(a, b), c), d)

**Expected Result:**
- Single-pass: ~60% accuracy
- 3-pass recursive: ~90% accuracy
- Proves recursion > depth

**Deliverable**: "example-7a-simple-recursion"

---

### Phase A2: HRM (Hierarchical Reasoning) (Example-7b) - 2 weeks

**Concept**: Two networks at different frequencies (Controller + Worker)

**Implement:**
- Fast network f_L (low-level, runs often)
- Slow network f_H (high-level, runs rarely)
- Dual latent features (z_L, z_H)
- Fixed recursion pattern (n=2 f_L, 1 f_H)

**Task**: Mini-Sudoku (4x4 grid)

**Expected Result:**
- Single network: ~30% accuracy
- HRM (27M params): ~85% accuracy
- Emergent dimensionality hierarchy

**Deliverable**: "example-7b-hrm-sudoku"

---

### Phase A3: TRM (Tiny Recursion) (Example-7c) - 2-3 weeks

**Concept**: Simplify HRM - one tiny network, latent + answer

**Implement:**
- Single 2-layer network
- Two features: z (reasoning), y (answer)
- z = net(x, y, z)  # Update reasoning
- y = net(y, z)     # Update answer
- Deep supervision (train on each step)
- Detached recursion (T-1 free, 1 backprop)

**Task**: Pattern completion, simple logic puzzles

**Expected Result:**
- HRM (27M): ~55% on puzzles
- TRM (7M): ~87% on same puzzles ⭐
- 4× fewer parameters, better accuracy

**Deliverable**: "example-7c-trm-patterns"

---

### Path A Summary

```
Example-7a: Simple Recursion (XOR chain)
    ↓
Example-7b: HRM (Mini-Sudoku)
    ↓
Example-7c: TRM (Patterns)
    ↓
Specialist Agent: LogicReasoner
  - Task: Code logic bugs, constraint solving
  - Size: 5-10M parameters
  - Hardware: Runs on old CPU (8-core Xeon)
  - Efficiency: 100× cheaper than GPT-4 for logic tasks
```

---

## Path B: Text Diffusion (Generation)

**Goal**: Generate text via iterative denoising (like DALL-E for text)

**Use Case**: Code generation, documentation, test case creation

**Efficiency**: Parallel generation (vs autoregressive), controllable, smaller models

---

### Phase B1: Embeddings & Tokenization (Example-8a) - 1-2 weeks

**Concept**: Map discrete tokens to continuous vectors

**Implement:**
- Byte-pair encoding (BPE) tokenizer
- Learned embedding table
- Positional encodings
- Reverse embedding (detokenization)

**Task**: Simple language (variable names, keywords)

**Expected Result:**
- Vocabulary: ~500 tokens (code-specific)
- Embedding dim: 128-256
- Can embed/decode code snippets

**Deliverable**: "example-8a-embeddings"

---

### Phase B2: Attention Mechanism (Example-8b) - 2-3 weeks

**Concept**: Attend to relevant parts of sequence

**Implement:**
- Scaled dot-product attention
- Multi-head attention (4-8 heads)
- Self-attention (attend to own sequence)
- Masked attention (autoregressive)

**Task**: Sequence-to-sequence (variable renaming)

**Expected Result:**
- Input: "x = y + z"
- Attention: Focuses on "x" and "=" when predicting assignment
- Output: Correct renamed variables

**Deliverable**: "example-8b-attention"

---

### Phase B3: Transformer Encoder-Decoder (Example-8c) - 2-3 weeks

**Concept**: Full transformer for seq2seq

**Implement:**
- Encoder (self-attention + FFN)
- Decoder (masked self-attention + cross-attention + FFN)
- Positional encoding
- Layer norm

**Task**: Code translation (Python-like → Rust-like pseudocode)

**Expected Result:**
- Small transformer (4 layers, 512 hidden, 8 heads)
- ~20M parameters
- Learns simple code patterns

**Deliverable**: "example-8c-transformer"

---

### Phase B4: Masked Diffusion (MDLM) (Example-8d) - 3-4 weeks

**Concept**: Diffusion in discrete token space

**Implement:**
- Forward process: Gradually mask tokens
- Reverse process: Predict masked tokens
- Absorbing state diffusion
- Training: Mixture of MLM losses

**Task**: Code completion with diffusion

**Expected Result:**
- Input: "fn add(a: i32, [MASK]) -> [MASK]"
- Diffusion steps: 16
- Output: "fn add(a: i32, b: i32) -> i32"
- Parallel generation (all tokens simultaneously)

**Deliverable**: "example-8d-text-diffusion"

---

### Path B Summary

```
Example-8a: Embeddings
    ↓
Example-8b: Attention
    ↓
Example-8c: Transformer
    ↓
Example-8d: Text Diffusion
    ↓
Specialist Agent: CodeGenerator
  - Task: Generate boilerplate, tests, docs
  - Size: 20-50M parameters
  - Hardware: Old GPU with high VRAM (GTX 1080 Ti, 11GB)
  - Efficiency: Parallel generation = 5× faster than autoregressive
```

---

## Path C: Biological / Spiking (BDH → SNNs)

**Goal**: Biologically plausible, energy-efficient learning

**Use Case**: Always-on monitoring, real-time event processing

**Efficiency**: Event-driven (only compute on spikes), 1000× less energy

---

### Phase C1: Spiking Neurons (Example-9a) - 2 weeks

**Concept**: Neurons fire discrete spikes instead of continuous values

**Implement:**
- LIF (Leaky Integrate-and-Fire) neuron
- Spike timing
- Refractory period
- Encoding: Rate coding, temporal coding

**Task**: Spike-based XOR

**Expected Result:**
- Input: Spike trains for [1, 0]
- Processing: Spikes propagate through network
- Output: Spike rate encodes XOR result

**Deliverable**: "example-9a-spiking-neurons"

---

### Phase C2: STDP Learning (Example-9b) - 2 weeks

**Concept**: Hebbian learning - "neurons that fire together wire together"

**Implement:**
- Spike-Timing-Dependent Plasticity (STDP)
- Weight update based on spike timing
- Unsupervised learning
- Synaptic traces

**Task**: Pattern recognition with STDP

**Expected Result:**
- Network learns to recognize spike patterns
- No backprop needed (local learning rule)
- Biologically plausible

**Deliverable**: "example-9b-stdp-learning"

---

### Phase C3: Surrogate Gradients (Example-9c) - 2-3 weeks

**Concept**: Backprop through spikes using smooth approximation

**Implement:**
- Surrogate gradient function (differentiable spike)
- Backpropagation Through Time (BPTT) for spikes
- Hybrid: STDP + surrogate gradients

**Task**: Spike-based classification

**Expected Result:**
- Train SNN with backprop
- Combine benefits: energy efficiency + supervised learning
- Match accuracy of regular networks

**Deliverable**: "example-9c-snn-backprop"

---

### Phase C4: BDH (Baby Dragon Hatchling) (Example-9d) - 3-4 weeks

**Concept**: Scale-free spiking network with Hebbian plasticity

**Implement:**
- Scale-free network topology (power-law degree distribution)
- Hebbian synaptic plasticity as memory
- Local neuron interactions
- Attention-like mechanism for spikes

**Task**: Simple language understanding with spikes

**Expected Result:**
- 10M spiking neurons
- Rivals GPT-2 on simple tasks
- Event-driven computation
- Extremely low power

**Deliverable**: "example-9d-bdh-network"

---

### Path C Summary

```
Example-9a: Spiking Neurons
    ↓
Example-9b: STDP Learning
    ↓
Example-9c: Surrogate Gradients
    ↓
Example-9d: BDH Network
    ↓
Specialist Agent: EventMonitor
  - Task: Real-time log analysis, anomaly detection
  - Size: 10M spiking neurons
  - Hardware: Neuromorphic chip (Intel Loihi) OR CPU with event queue
  - Efficiency: 1000× less energy than GPU for real-time tasks
```

---

## Path D: RAG + Continual Learning

**Goal**: Knowledge retrieval + learning without full retraining

**Use Case**: Code search, documentation, learning from feedback

**Efficiency**: Small model + big memory >> big model

---

### Phase D1: Vector Database (Example-10a) - 2 weeks

**Concept**: Store and retrieve embeddings efficiently

**Implement:**
- Vector storage (in-memory or disk)
- Cosine similarity search
- Approximate nearest neighbors (simple grid-based)
- Indexing for fast retrieval

**Task**: Code snippet search

**Expected Result:**
```
Query: "parse JSON"
Retrieved: [
  "serde_json::from_str(...)",
  "serde::Deserialize trait",
  "JSON parsing examples"
]
```

**Deliverable**: "example-10a-vector-db"

---

### Phase D2: Dense Retrieval (Example-10b) - 2 weeks

**Concept**: Encode queries and documents into same embedding space

**Implement:**
- Query encoder (small network)
- Document encoder (same network)
- Similarity scoring
- Top-K retrieval

**Task**: Find relevant code given natural language query

**Expected Result:**
- Query: "function to read file"
- Encoded: [0.2, -0.5, 0.8, ...]
- Top match: fs::read_to_string() documentation

**Deliverable**: "example-10b-dense-retrieval"

---

### Phase D3: RAG (Retrieval-Augmented Generation) (Example-10c) - 2-3 weeks

**Concept**: Small model + retrieved knowledge > big model alone

**Implement:**
- Query encoding
- Document retrieval (top-K)
- Context injection (prepend retrieved docs)
- Answer generation (small transformer/TRM)

**Task**: Code Q&A

**Expected Result:**
```
Question: "How do I handle errors in Rust?"
Retrieved: [Result<T,E> docs, ? operator examples]
Generated: "Use Result<T,E> and the ? operator for propagation..."
```

**Deliverable**: "example-10c-rag-system"

---

### Phase D4: Continual Learning (Example-10d) - 3-4 weeks

**Concept**: Learn from feedback without catastrophic forgetting

**Implement:**
- Experience replay buffer
- Positive/negative reinforcement signals
- Incremental weight updates (EMA-based)
- Episodic memory (store important examples)

**Task**: Learn from code review feedback

**Expected Result:**
```
Generated Code: "let x = y.clone(); let z = x.clone();"
Feedback: "Unnecessary clone, use borrow"
Learning: Update weights to prefer borrowing
Next Generation: "let x = &y; let z = &x;"
```

**Deliverable**: "example-10d-continual-learning"

---

### Path D Summary

```
Example-10a: Vector Database
    ↓
Example-10b: Dense Retrieval
    ↓
Example-10c: RAG System
    ↓
Example-10d: Continual Learning
    ↓
Specialist Agent: KnowledgeRetriever
  - Task: Code search, documentation lookup
  - Size: 5M model + 100GB vector DB
  - Hardware: CPU with large RAM (128GB+)
  - Efficiency: Tiny model + big memory = cheap inference
```

---

## Path E: Hybrid Combinations

**Goal**: Combine the best of each path for maximum capability

---

### Hybrid 1: TRM + RAG (Reasoning with Knowledge)

**Combination**:
- TRM for recursive reasoning
- RAG for knowledge retrieval
- Each recursion step can retrieve additional context

**Architecture:**
```rust
fn trm_with_rag(question, y, z, db, n=6) {
    for i in 0..n {
        // Retrieve relevant knowledge
        context = db.retrieve(&question, &y, &z);

        // Update reasoning with context
        z = net(&question, &context, &y, &z);
    }
    y = net(&y, &z);
    return (y, z);
}
```

**Use Case**: Complex code reasoning with documentation lookup

**Deliverable**: "example-11a-trm-rag"

---

### Hybrid 2: Diffusion + RAG (Controlled Generation)

**Combination**:
- Text diffusion for parallel generation
- RAG for grounding in knowledge base

**Architecture:**
- Retrieve relevant code examples
- Use as conditioning for diffusion
- Generate code that matches patterns in examples

**Use Case**: Generate code following project conventions

**Deliverable**: "example-11b-diffusion-rag"

---

### Hybrid 3: SNN + TRM (Efficient Reasoning)

**Combination**:
- Spiking neurons for energy efficiency
- TRM-style recursive refinement

**Architecture:**
- Spike-based encoding
- Recursive spiking network
- Event-driven computation

**Use Case**: Always-on code monitoring with reasoning

**Deliverable**: "example-11c-snn-trm"

---

### Hybrid 4: Multi-Agent Cooperation

**Combination**: ALL paths working together!

**Architecture:**
```
User Query
    ↓
LogicReasoner (TRM) → "What needs to be done?"
    ↓
KnowledgeRetriever (RAG) → "What patterns exist?"
    ↓
CodeGenerator (Diffusion) → "Generate implementation"
    ↓
EventMonitor (SNN) → "Watch for issues"
    ↓
ContinualLearner (RAG+TRM) → "Learn from feedback"
```

**Use Case**: Full software development pipeline

**Deliverable**: "example-11d-multi-agent"

---

## Efficiency Optimization Strategies

### Strategy 1: Hardware-Specific Specialization

**Old Multi-Core CPUs** (Intel Xeon E5-2670, 16 cores, $20 used):
- Best for: TRM, RAG, vector search
- Why: High core count, large RAM, cheap
- Optimization: SIMD, parallel batch processing

**Old GPUs with High VRAM** (GTX 1080 Ti, 11GB, $150 used):
- Best for: Diffusion, Transformers, embeddings
- Why: Parallel computation, large memory
- Optimization: Batch processing, mixed precision

**Neuromorphic Chips** (Intel Loihi, if available):
- Best for: SNNs, BDH, event monitoring
- Why: Event-driven, ultra-low power
- Optimization: Spike timing, sparse computation

---

### Strategy 2: Model Quantization

**Techniques:**
- f32 → f16 (2× smaller, minimal accuracy loss)
- f32 → int8 (4× smaller, need careful calibration)
- Mixed precision (f16 activations, f32 gradients)

**Expected Savings:**
- Memory: 2-4× reduction
- Inference speed: 1.5-3× faster on CPU
- Energy: 2-3× less

**Implementation**: Start in Phase 0 (core efficiency)

---

### Strategy 3: Knowledge Distillation

**Technique**: Train small specialist from large generalist

**Process:**
1. Train large model on broad tasks
2. Generate labeled data with large model
3. Train tiny specialist on specific task
4. Deploy only tiny specialist

**Result:**
- 100M general model → 5M specialist
- 95% of accuracy on specific task
- 20× faster, 20× less energy

---

### Strategy 4: Sparse Activation

**Technique**: Only activate subset of neurons per input

**Methods:**
- Top-K activation (only keep K largest activations)
- Mixture of Experts (route to relevant expert)
- Conditional computation

**Expected Savings:**
- Computation: 5-10× reduction
- Energy: 3-5× less
- Minimal accuracy loss

---

## Parallel Development Strategy

### Team of 1 Developer (You)

**Month 1-2: Foundation for All**
- Week 1-2: Phase 0 (Core Efficiency) ⭐
- Week 3-4: Phase 1 (Better Training) ⭐
- Week 5-6: Phase 2 (Modular Architecture) ⭐
- Week 7-8: Documentation, benchmarks

**Month 3-4: First Specialist (Pick One)**

**Option A: TRM Path** (Recommended - fastest results)
- Week 9-10: Simple Recursion
- Week 11-12: HRM
- Week 13-14: TRM
- Week 15-16: Polish + deploy

**Option B: Diffusion Path** (Most versatile)
- Week 9-10: Embeddings
- Week 11-12: Attention
- Week 13-14: Transformer
- Week 15-16: Masked Diffusion

**Option C: SNN Path** (Most efficient)
- Week 9-10: Spiking Neurons
- Week 11-12: STDP
- Week 13-14: Surrogate Gradients
- Week 15-16: BDH basics

**Month 5-6: Second Specialist**
- Pick different path from Month 3-4
- Reuse foundation from Month 1-2

**Month 7-8: RAG + Continual Learning**
- Week 25-26: Vector DB
- Week 27-28: Retrieval
- Week 29-30: RAG
- Week 31-32: Continual Learning

**Month 9-10: Hybrids**
- Combine specialists
- Build multi-agent system

---

### Parallel Work (If Team >1)

**Developer 1**: TRM Path (reasoning)
**Developer 2**: Diffusion Path (generation)
**Developer 3**: SNN Path (efficiency)
**Developer 4**: RAG Path (knowledge)

All share: Foundation (Phase 0-2)

Meet monthly to integrate

---

## Measuring Success: Accuracy per kWh

### Benchmark Suite

Create unified benchmark that ALL paths must pass:

**Tasks:**
1. Logic puzzle (3-bit parity, Sudoku 4x4)
2. Code completion (fill in [MASK])
3. Pattern recognition (find bug in code)
4. Knowledge retrieval (find relevant function)

**Metrics:**
- Accuracy (% correct)
- Latency (ms per inference)
- Energy (Wh per 1000 inferences)
- **Accuracy per kWh** (primary metric)

**Target:**
- Beat GPT-3.5 on specialized tasks
- Use 1/1000th the energy
- Run on $20 used CPU

**Example Benchmark Output:**
```
=== Logic Puzzle Benchmark ===

GPT-3.5 (175B):
  Accuracy: 82%
  Energy: 120 Wh / 1000 inferences
  Accuracy/kWh: 683

TRM (7M):
  Accuracy: 87%
  Energy: 0.8 Wh / 1000 inferences ⭐
  Accuracy/kWh: 108,750 ⭐

159× more efficient!
```

---

## Recommended Next Steps

### Immediate (This Week):

1. **Commit current work**:
   ```bash
   git add docs/multi-path-architecture.md
   git commit -m "Add multi-path roadmap for specialist agents"
   git push
   ```

2. **Choose starting path**:
   - **Option A**: Start Phase 0 (Core Efficiency) - benefits everything
   - **Option B**: Start Path A (TRM) - fastest to working prototype
   - **Option C**: Start Path D (RAG) - most immediately useful

3. **Set up benchmark framework**:
   - Create `benchmarks/` directory
   - Implement energy measurement (CPU power draw)
   - Define standard tasks
   - Establish baseline with current code

### This Month:

**Week 1-2: Phase 0 (Core Efficiency)**
- Memory profiling
- SIMD optimization
- Quantization (f16)
- Energy benchmarking

**Week 3-4: Choose First Path**
- My recommendation: **Path A (TRM)** because:
  1. Simplest to implement (builds on what we have)
  2. Most impressive results (tiny model beats huge LLMs)
  3. Teaches core concepts (recursion, reasoning)
  4. Good foundation for hybrids

### This Quarter (3 months):

- Month 1: Foundation (Phase 0-2)
- Month 2: First specialist (TRM or Diffusion)
- Month 3: RAG basics + integration

---

## My Recommendation: Start with Phase 0 + TRM

**Why This Order:**

1. **Phase 0 (Core Efficiency)** first:
   - Establishes measurement framework
   - Optimizes foundation for all paths
   - Proves efficiency thesis (old hardware can compete)
   - ~2 weeks

2. **Path A (TRM)** next:
   - Builds on current backprop implementation
   - Shows immediate impressive results (7M beats 175B on puzzles)
   - Teaches recursive reasoning (useful for all agents)
   - ~6 weeks

3. **Path D (RAG)** third:
   - Adds memory/knowledge capability
   - Enables continual learning
   - Combines with TRM for powerful hybrid
   - ~6 weeks

4. **Other paths** as needed:
   - Diffusion for generation tasks
   - SNNs for always-on monitoring
   - Hybrids for complex workflows

**Total to first useful multi-agent system: ~4 months**

---

## Questions to Consider

1. **Which specialist agent do you need first?**
   - LogicReasoner (TRM) - debugging, constraint solving
   - CodeGenerator (Diffusion) - boilerplate, tests
   - KnowledgeRetriever (RAG) - documentation, search
   - EventMonitor (SNN) - real-time analysis

2. **What hardware do you have access to?**
   - Old multi-core CPUs? (Great for TRM, RAG)
   - Old GPUs with VRAM? (Good for Diffusion)
   - Lots of RAM? (Perfect for RAG)

3. **What's the first task for the agent team?**
   - Automated testing?
   - Code review?
   - Documentation generation?
   - Bug detection?

**Should we start with Phase 0 (Core Efficiency) to establish the measurement framework?**

Or jump straight to a specialist path?
