# Multi-Repository Strategy

**Philosophy**: Each repository has a clear, focused purpose. Foundational concepts in the base repo, specialized implementations in separate repos, combinations in integration repos, and applications in the final layer.

---

## Repository Hierarchy

```
Layer 1: Foundation (THIS REPO)
    ↓ (depends on)
Layer 2: Specialized Models (TRM, SNN, Diffusion, etc.)
    ↓ (depends on)
Layer 3: Hybrid Combinations
    ↓ (depends on)
Layer 4: Software Development Agents
    ↓ (depends on)
Layer 5: Real-World Applications
```

---

## Layer 1: neural-network-examples-rs (THIS REPO) ✅

**Purpose**: Educational building blocks and foundational concepts

**Scope**: ONLY fundamental, general-purpose neural network components

**What Belongs Here:**
- ✅ Basic feedforward networks (3-layer)
- ✅ Backpropagation (SGD)
- ✅ Sigmoid activation
- ✅ Boolean logic examples (AND, OR, XOR)
- ✅ Simple arithmetic (adders)
- ✅ Visualization (SVG)
- ✅ Checkpoint save/load
- 🔲 Better optimizers (Adam, RMSprop, AdamW)
- 🔲 Modern activations (ReLU, Leaky ReLU, GELU, Swish)
- 🔲 Deeper networks (4-6 layers)
- 🔲 Residual connections
- 🔲 Layer normalization, Batch normalization
- 🔲 Dropout
- 🔲 RNNs (basic LSTM/GRU cells)
- 🔲 CNNs (Conv2D, MaxPool, etc.)
- 🔲 Attention mechanism (scaled dot-product, multi-head)
- 🔲 Embeddings (learned embeddings, positional encoding)

**What Does NOT Belong Here:**
- ❌ TRM/HRM (too specialized → separate repo)
- ❌ SNNs/BDH (too specialized → separate repo)
- ❌ Diffusion models (too specialized → separate repo)
- ❌ RAG systems (too specialized → separate repo)
- ❌ Multi-agent systems (too complex → separate repo)
- ❌ Production applications (→ Layer 5)

**Deliverables:**
1. **Crates** (importable by other repos):
   - `neural_net_core` - Network, layers, optimizers, activations
   - `neural_net_viz` - Visualization tools
   - `neural_net_utils` - Common utilities

2. **Examples** (educational demonstrations):
   - `example-1-forward-propagation` ✅
   - `example-2-backward-propagation-{and,or,xor}` ✅
   - `example-3-complex-boolean-{parity,majority}` ✅
   - `example-3-multi-output-{half-adder,full-adder}` ✅
   - `example-4-optimizers` (Adam, RMSprop)
   - `example-5-activations` (ReLU, GELU)
   - `example-6-deep-networks` (4-6 layers, residuals)
   - `example-7-rnns` (LSTM, GRU)
   - `example-8-cnns` (Conv2D for image classification)
   - `example-9-attention` (Self-attention, multi-head)
   - `example-10-embeddings` (Word embeddings)

3. **Documentation**:
   - Tutorial for each building block
   - API documentation
   - Theory explanations
   - Roadmap for future repos

**Status**: ~40% complete
- ✅ Feedforward + backprop
- ✅ Basic examples
- 🔲 Missing: optimizers, activations, RNN, CNN, attention

**Timeline to Complete**: 2-3 months (see detailed roadmap below)

---

## Layer 2: Specialized Model Repositories

Each specialized model gets its own focused repository that imports crates from Layer 1.

---

### Repo 2a: trm-reasoning-rs

**Purpose**: Tiny Recursion Model (TRM) for recursive reasoning

**Depends On**: neural-network-examples-rs (uses neural_net_core, neural_net_viz)

**Scope**:
- TRM architecture (2-layer recursive network)
- Latent + answer features (z, y)
- Deep supervision training
- Detached recursion (T-1 free, 1 backprop)
- Halting mechanism
- EMA for stability

**Examples**:
- Pattern completion
- Logic puzzles (Sudoku 4x4, 9x9)
- ARC-AGI-like tasks
- Code logic debugging

**Deliverables**:
- `trm_core` crate
- Educational examples
- Benchmarks vs baselines
- Pre-trained models

**Timeline**: 2-3 months after Layer 1 complete

---

### Repo 2b: hrm-reasoning-rs

**Purpose**: Hierarchical Reasoning Model (HRM) - predecessor to TRM

**Depends On**: neural-network-examples-rs

**Scope**:
- Dual-network architecture (Controller + Worker)
- Different frequency recursion
- Hierarchical features (z_L, z_H)
- Deep supervision
- ACT (adaptive computational time)

**Examples**:
- Mini-Sudoku
- Maze solving
- Multi-step reasoning

**Deliverables**:
- `hrm_core` crate
- Comparison to TRM
- Ablation studies

**Timeline**: 2-3 months (can be parallel with TRM)

---

### Repo 2c: spiking-networks-rs

**Purpose**: Spiking Neural Networks (SNNs) and Baby Dragon Hatchling (BDH)

**Depends On**: neural-network-examples-rs

**Scope**:
- LIF (Leaky Integrate-and-Fire) neurons
- STDP (Spike-Timing-Dependent Plasticity)
- Surrogate gradients for backprop
- Event-driven computation
- BDH architecture (scale-free, Hebbian)

**Examples**:
- Spike-based XOR
- Event classification
- Real-time monitoring
- Energy efficiency demos

**Deliverables**:
- `snn_core` crate
- `bdh` crate
- Neuromorphic hardware examples (if available)
- Energy benchmarks

**Timeline**: 3-4 months (more complex)

---

### Repo 2d: text-diffusion-rs

**Purpose**: Text generation via discrete diffusion (MDLM, masked diffusion)

**Depends On**: neural-network-examples-rs

**Scope**:
- Transformer architecture (from Layer 1 attention)
- Masked diffusion process
- Absorbing state diffusion
- Forward/reverse diffusion
- Sampling algorithms

**Examples**:
- Code completion
- Text generation
- Controlled generation

**Deliverables**:
- `text_diffusion` crate
- Pre-trained small models
- Comparison to autoregressive

**Timeline**: 3-4 months

---

### Repo 2e: rag-learning-rs

**Purpose**: Retrieval-Augmented Generation (RAG) + Continual Learning

**Depends On**: neural-network-examples-rs

**Scope**:
- Vector database (from scratch)
- Dense retrieval
- RAG system (small model + big memory)
- Continual learning without catastrophic forgetting
- Experience replay

**Examples**:
- Code search
- Documentation retrieval
- Learning from feedback

**Deliverables**:
- `vector_db` crate
- `rag_core` crate
- `continual_learning` crate

**Timeline**: 2-3 months

---

## Layer 3: Hybrid Combination Repositories

Combine specialized models for enhanced capabilities.

---

### Repo 3a: trm-rag-hybrid-rs

**Purpose**: TRM reasoning + RAG knowledge retrieval

**Depends On**:
- trm-reasoning-rs
- rag-learning-rs

**Scope**:
- Each recursion step can retrieve knowledge
- Reasoning grounded in documentation
- Continual learning from feedback

**Examples**:
- Code debugging with docs lookup
- Complex reasoning with knowledge base

**Timeline**: 1-2 months

---

### Repo 3b: diffusion-rag-hybrid-rs

**Purpose**: Text diffusion + RAG for controlled generation

**Depends On**:
- text-diffusion-rs
- rag-learning-rs

**Scope**:
- Retrieve code examples
- Generate following project conventions
- Knowledge-grounded generation

**Examples**:
- Generate code matching style guide
- Test generation from examples

**Timeline**: 1-2 months

---

### Repo 3c: snn-trm-hybrid-rs

**Purpose**: Spiking networks + TRM reasoning

**Depends On**:
- spiking-networks-rs
- trm-reasoning-rs

**Scope**:
- Event-driven recursive reasoning
- Energy-efficient logic

**Examples**:
- Always-on monitoring with reasoning
- Low-power inference

**Timeline**: 2 months

---

## Layer 4: Software Development Agent Repository

**Repo**: ai-dev-agents-rs

**Purpose**: Cooperative team of specialist AI coding agents

**Depends On**:
- All Layer 2 specialized models
- Relevant Layer 3 hybrids

**Scope**:
- Agent orchestration framework
- Task decomposition
- Agent communication
- Feedback loops
- Continual learning

**Agents**:
1. **LogicReasoner** (TRM) - Debug, constraints
2. **CodeGenerator** (Diffusion) - Boilerplate, tests
3. **KnowledgeRetriever** (RAG) - Docs, search
4. **EventMonitor** (SNN) - Real-time analysis
5. **LearnCoordinator** (RAG + all) - Feedback, improvement

**Examples**:
- Automated test generation
- Code review pipeline
- Documentation generation
- Bug detection and fixing

**Deliverables**:
- `dev_agents` crate
- Agent orchestration framework
- Example multi-agent workflows

**Timeline**: 3-4 months

---

## Layer 5: Real-World Application Repositories

**Purpose**: Actual software projects built by the agent team

**Examples**:
- **Repo**: rust-web-app-demo (built by agents)
- **Repo**: cli-tool-generator (built by agents)
- **Repo**: code-analyzer (built by agents)

**Depends On**: ai-dev-agents-rs

**Timeline**: Ongoing, as agents become capable

---

## Development Sequence

### Phase 1: Complete Layer 1 (THIS REPO) ⭐ CURRENT FOCUS

**Timeline**: 2-3 months
**Status**: ~40% complete

**Remaining Work**:
1. Optimizers (Adam, RMSprop) - 1-2 weeks
2. Activations (ReLU, GELU, etc.) - 1 week
3. Deeper networks + residuals - 1-2 weeks
4. RNNs (LSTM, GRU) - 2-3 weeks
5. CNNs (Conv2D, MaxPool) - 2-3 weeks
6. Attention mechanism - 2-3 weeks
7. Embeddings - 1-2 weeks
8. Documentation polish - 1 week

**Total**: ~12-16 weeks

---

### Phase 2: Pick First Specialized Model (Layer 2)

**Options** (after Layer 1 complete):

**Option A: TRM** (trm-reasoning-rs)
- Fastest to implement (simplest architecture)
- Most impressive results (tiny beats huge)
- Good learning experience

**Option B: RAG** (rag-learning-rs)
- Most immediately useful
- Can enhance all other models
- Foundation for continual learning

**Option C: SNN** (spiking-networks-rs)
- Most energy-efficient
- Novel approach
- Harder to implement

**Recommendation**: Start with **TRM** (fastest results) then **RAG** (enhances everything)

**Timeline**: 2-3 months per specialized model

---

### Phase 3: Hybrids (Layer 3)

After 2-3 specialized models are complete, start combining.

**First Hybrid**: TRM + RAG (reasoning + knowledge)

**Timeline**: 1-2 months

---

### Phase 4: Dev Agents (Layer 4)

Once you have TRM, RAG, and possibly Diffusion:
- Build agent orchestration
- Create multi-agent system

**Timeline**: 3-4 months

---

### Phase 5: Applications (Layer 5)

Use dev agents to build real projects.

**Timeline**: Ongoing

---

## Total Timeline Estimate

| Phase | Duration | Cumulative |
|-------|----------|------------|
| Phase 1: Layer 1 (THIS REPO) | 3 months | 3 months |
| Phase 2a: TRM | 2-3 months | 5-6 months |
| Phase 2b: RAG | 2-3 months | 7-9 months |
| Phase 2c: Diffusion | 3 months | 10-12 months |
| Phase 3: First Hybrid (TRM+RAG) | 1-2 months | 11-14 months |
| Phase 4: Dev Agents | 3-4 months | 14-18 months |
| Phase 5: Applications | Ongoing | - |

**Realistic timeline to working dev agent system: 12-18 months**

---

## Repository Dependencies

```
neural-network-examples-rs (Layer 1)
├── trm-reasoning-rs (Layer 2a)
├── hrm-reasoning-rs (Layer 2b)
├── spiking-networks-rs (Layer 2c)
├── text-diffusion-rs (Layer 2d)
└── rag-learning-rs (Layer 2e)
    ├── trm-rag-hybrid-rs (Layer 3a)
    │   └── ai-dev-agents-rs (Layer 4)
    ├── diffusion-rag-hybrid-rs (Layer 3b)
    │   └── ai-dev-agents-rs (Layer 4)
    └── snn-trm-hybrid-rs (Layer 3c)
        └── ai-dev-agents-rs (Layer 4)
            └── [Real applications] (Layer 5)
```

---

## Benefits of Multi-Repo Approach

### 1. **Clear Scope**
Each repo has a focused purpose, easier to maintain and understand.

### 2. **Modularity**
Specialized models are separate, can develop in parallel.

### 3. **Reusability**
Layer 1 crates are imported by all Layer 2 repos.

### 4. **Versioning**
Each specialized model can have its own version, release cycle.

### 5. **Learning Path**
Clear progression: foundations → specialization → combination → application.

### 6. **Collaboration**
Different people can work on different Layer 2 repos independently.

### 7. **Experimentation**
Can try different approaches in separate repos without affecting others.

---

## What Goes in THIS Repo (Final Decision)

**Include** (general-purpose building blocks):
- ✅ Feedforward networks
- ✅ Backpropagation
- ✅ Basic activations (Sigmoid)
- 🔲 Modern activations (ReLU, GELU, Swish)
- 🔲 Optimizers (SGD, Adam, RMSprop, AdamW)
- 🔲 Regularization (Dropout, L1/L2)
- 🔲 Normalization (Batch, Layer)
- 🔲 Deeper networks (4-6 layers, residuals)
- 🔲 RNNs (vanilla, LSTM, GRU)
- 🔲 CNNs (Conv2D, MaxPool, etc.)
- 🔲 Attention (scaled dot-product, multi-head)
- 🔲 Embeddings (learned, positional)
- ✅ Visualization tools
- ✅ Checkpoint save/load
- 🔲 Training utilities (learning rate schedules, early stopping)

**Exclude** (too specialized):
- ❌ TRM/HRM → trm-reasoning-rs, hrm-reasoning-rs
- ❌ SNNs/BDH → spiking-networks-rs
- ❌ Diffusion → text-diffusion-rs
- ❌ RAG → rag-learning-rs
- ❌ Hybrids → Layer 3 repos
- ❌ Agents → ai-dev-agents-rs

---

## Next Steps for THIS Repo

### Immediate (This Week):
1. ✅ Create multi-repo strategy document (this file)
2. 🔲 Create detailed roadmap for completing Layer 1
3. 🔲 Identify which building block to add first

### This Month:
1. Pick first missing building block (recommendation: Adam optimizer)
2. Implement with educational example
3. Add tests, documentation, visualization
4. Repeat for next building block

### This Quarter:
Complete all missing building blocks in Layer 1.

---

## Long-Term Vision Documents (Not for Implementation Now)

These documents capture future plans but should NOT be implemented in THIS repo:

- ✅ `docs/ROADMAP.md` - Long-term vision for all technologies
- ✅ `docs/research-summary-2024-2025.md` - Research on TRM, HRM, BDH, SNNs
- ✅ `docs/multi-path-architecture.md` - Detailed plans for all paths
- ✅ `docs/multi-repo-strategy.md` - This document

These serve as:
- Reference for future repos
- Motivation for why we're building foundations
- Roadmap for where we're going
- Don't clutter the current implementation focus

---

## Summary

**THIS REPO (neural-network-examples-rs)**:
- **Purpose**: Educational building blocks
- **Scope**: General-purpose NN components only
- **Status**: ~40% complete
- **Focus**: Complete foundational examples (RNN, CNN, Attention, etc.)
- **Timeline**: 2-3 months to complete

**FUTURE REPOS**:
- Each specialized model (TRM, SNN, Diffusion, RAG) gets its own repo
- They import crates from THIS repo
- Can develop in parallel once Layer 1 is complete

**MAKES SENSE?** ✅

Let's focus on completing the missing building blocks in THIS repo before branching out.
