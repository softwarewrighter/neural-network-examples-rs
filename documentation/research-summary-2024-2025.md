# Cutting-Edge ML Research Summary (2024-2025)

Research findings for the specific models you want to implement.

---

## 1. Text-Diffusion Models

### Google Gemini Diffusion (2025)
**Status**: Commercial deployment, announced at Google I/O 2025

**Key Features**:
- First commercial-grade diffusion model matching autoregressive performance
- **Speed**: 1,479 tokens/second (5× faster than comparable models)
- Outperforms Gemini 2.0 Flash-Lite on coding (30.9% vs 28.5% LiveCodeBench)
- Strong mathematical reasoning capabilities

### MDLM (NeurIPS 2024 - Best Paper Award Track)
**Paper**: "Simple and Effective Masked Diffusion Language Model"
**GitHub**: github.com/kuleshov-group/mdlm

**Architecture**:
- Masked discrete diffusion (absorbing state diffusion)
- SUBS (Substitution-based) parameterization
- Simplifies to mixture of masked language modeling losses
- 6-8× better generative perplexity than GPT-2
- 32× fewer network evaluations

**Training Process**:
1. Forward: Gradually mask tokens (absorbing diffusion)
2. Reverse: Predict masked tokens (denoising)
3. Loss: Mixture of MLM losses at different noise levels

### Discrete Diffusion for Text Summarization (2024)
**Paper**: "Discrete Diffusion Language Model for Efficient Text Summarization" (ACL 2025)

**Innovation**:
- Semantic-aware noising process for long sequences
- CrossMamba architecture (Mamba adapted to encoder-decoder)
- Outperforms existing discrete diffusion on Gigaword, CNN/DailyMail, Arxiv
- Much faster inference than autoregressive

**Key Insight**: Diffusion enables parallel generation, bidirectional context, better controllability

---

## 2. Hierarchical Reasoning Model (HRM)

**Institution**: Sapient Intelligence (Singapore AI lab, founded 2024)
**Papers**:
- "Hierarchical Reasoning Model" (arXiv 2506.21734)
- "Critical Supplementary Material" (arXiv 2510.00355)

### Architecture: Two-Network System

**Controller Module** (Slow, High-Level):
- Abstract deliberate reasoning
- System 2 thinking (Kahneman)
- ~3× higher-dimensional representational space
- Plans and guides

**Worker Module** (Fast, Low-Level):
- Detailed computations
- System 1 thinking
- Executes plans

### Performance
- **27M parameters** (tiny!)
- **1000 training examples** (minimal data!)
- Nearly **perfect on complex Sudoku**
- Nearly **perfect on maze path-finding**
- **Zero failures** where o3-mini-high, Claude 3.7, DeepSeek-R1 all failed (0% accuracy)

### Emergent Property
**Dimensionality Hierarchy**: Spontaneously develops during training
- High-level module: Higher-dimensional space (abstract concepts)
- Low-level module: Lower-dimensional space (concrete operations)
- Not explicitly programmed - emerges from architecture

### Biological Inspiration
Based on dual-process theory of cognition:
- System 1: Fast, intuitive, automatic
- System 2: Slow, deliberate, effortful

---

## 3. Tiny Recursion Model (TRM)

**Institution**: Samsung SAIL Montreal
**Paper**: "Less is More: Recursive Reasoning with Tiny Networks" (arXiv 2510.04871)
**GitHub**: github.com/SamsungSAILMontreal/TinyRecursiveModels
**License**: MIT (Open Source)

### Architecture: Single-Network Self-Refinement

**Size**: 7M parameters (2-layer network)

**Recursive Mechanism**:
```
for k in 1..K improvement steps:
    z_k = update_latent(x, y_{k-1}, z_{k-1})  # Refine representation
    y_k = update_answer(y_{k-1}, z_k)          # Refine answer
```

Each iteration corrects errors from previous step → **self-improving reasoning**

### Performance
- **45% on ARC-AGI-1** (abstract reasoning)
- **8% on ARC-AGI-2** (harder abstract reasoning)
- **87.4% on Sudoku-Extreme**
- Outperforms models 10,000× larger on these tasks
- <0.01% of parameters compared to LLMs

### Key Innovation
**Recursive refinement** instead of one-shot prediction:
- Start with initial answer
- Iteratively improve using self-feedback
- Converge to stable, correct solution
- Minimizes overfitting through progressive refinement

---

## 4. Baby Dragon Hatchling (BDH)

**Institution**: Pathway
**Paper**: "The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain" (arXiv 2509.26507)
**GitHub**: github.com/pathwaycom/bdh

### Vision
Bridge artificial intelligence and neuroscience - **biologically plausible AI**

### Architecture: Scale-Free Neuron Network

**Core Principles**:
- Scale-free network of locally-interacting neuron particles
- Spiking neurons with Hebbian learning
- Synaptic plasticity as working memory
- Attention-based state space sequence learning

**Memory Mechanism**:
- **Working memory = Synaptic plasticity**
- Individual synapses strengthen when hearing/reasoning about concepts
- Persistent connections form during language processing
- Biologically plausible memory without external memory module

### Performance
- **10M parameters**
- Rivals GPT-2 on language and translation tasks
- Maintains biological plausibility while matching transformer performance

### Biological Features
1. **Spiking Neurons**: Discrete events instead of continuous activations
2. **Hebbian Learning**: "Neurons that fire together wire together"
3. **Local Interactions**: No global coordination required
4. **Scale-Free Network**: Power-law degree distribution (like brain)

### Theoretical Foundation
Provides framework for understanding how **reasoning and generalization emerge** from biologically-inspired principles

---

## 5. Spiking Neural Networks (Recent Advances 2024-2025)

### Training Methods Revolution

#### Spike-Aware Data Pruning (SADP) - October 2025
**Innovation**: Reduce training time by 35% on ImageNet

**Method**:
- Determine example selection probability ∝ gradient norm
- Reduces gradient variance
- Maintains accuracy with less data
- Continuous spike timing adjustment

#### Surrogate Gradient Method
**Current Standard**: Direct training with sufficient flexibility

**Enables**:
- Novel SNN architectures
- Spatial-temporal dynamics exploration
- End-to-end backpropagation through spikes

### Transformer-Based SNNs (2024)

**Dual Spike Self-Attention (DSSA)**:
- 79.40% top-1 accuracy on ImageNet-1K
- Reasonable scaling for large datasets

**Spiking Convolutional Stem (SCS)**:
- 80.38% accuracy on ImageNet-1K
- Supplementary layers for better feature extraction

**SGLFormer** (State-of-the-Art):
- **83.73% top-1 accuracy on ImageNet-1K**
- 64M parameters
- Groundbreaking for SNNs on large-scale vision

### Large-Scale Training: Densely Additive Networks (DANet)
**Problem**: Gradient vanishing in deep SNNs
**Solution**: Densely additive (DA) residual connections
**Result**: Enables training of very deep SNNs

### Energy Efficiency
- Continuous spike timing adjustment enables fine-tuning
- Event-driven computation (only compute on spikes)
- Neuromorphic hardware compatibility (Loihi, TrueNorth)
- 1000× energy savings vs conventional DNNs

---

## Implementation Strategy: From Here to There

### Immediate Dependencies for ALL Goals

1. **Better Optimizers** (Adam, RMSprop)
   - All 5 models use modern optimizers
   - Critical for training stability

2. **Multiple Activations** (ReLU, GELU, etc.)
   - Text-diffusion: GELU in transformers
   - HRM/TRM: Various activations
   - BDH: Spiking activations
   - SNNs: Spike response functions

3. **Deeper Networks** (4+ layers)
   - All models use multiple layers
   - Gradient flow management essential

### Path 1: TRM (Fastest to Implement) ⭐ RECOMMENDED START

**Why First**:
- Simplest architecture (single 2-layer network)
- Only 7M parameters
- Self-contained recursive mechanism
- No complex dependencies (no embeddings, no attention)
- Open-source MIT license with code

**Implementation Steps**:
1. Build 2-layer network (already can do this!)
2. Add recursive update loop
3. Implement latent + answer update
4. Train on ARC-AGI-like tasks

**Timeline**: 2-3 weeks

**Skills Gained**:
- Recursive reasoning
- Latent space refinement
- Self-improving systems
- Small-data training

### Path 2: HRM (Next Easiest)

**Why Second**:
- Two small networks (27M total)
- Simple recurrence at different frequencies
- Clear separation of concerns (Controller/Worker)
- Minimal training data (1000 examples)

**Implementation Steps**:
1. Build two RNN modules
2. Implement different update frequencies
3. Train on Sudoku/maze tasks
4. Analyze emergent dimensionality hierarchy

**Timeline**: 3-4 weeks

**Skills Gained**:
- Multi-module coordination
- Emergent properties
- System 1/System 2 thinking
- Hierarchical reasoning

### Path 3: BDH (Biologically Inspired)

**Why Third**:
- Requires spiking neuron implementation
- Hebbian learning rules
- Scale-free network topology
- Complex but self-contained

**Implementation Steps**:
1. Implement LIF (Leaky Integrate-and-Fire) neurons
2. Add STDP (Spike-Timing-Dependent Plasticity)
3. Build scale-free network generator
4. Train on language tasks

**Timeline**: 5-6 weeks

**Skills Gained**:
- Spiking neural networks
- Biological learning rules
- Complex network topologies
- Event-driven computation

### Path 4: SNNs (General Framework)

**Why Fourth**:
- Foundation for BDH and other bio-inspired models
- Recent advances make training feasible
- Energy efficiency benefits

**Implementation Steps**:
1. LIF neuron model
2. Surrogate gradient method
3. DSSA (Dual Spike Self-Attention)
4. Train on vision tasks (MNIST → ImageNet)

**Timeline**: 6-8 weeks

**Skills Gained**:
- Spike encoding/decoding
- Temporal dynamics
- Neuromorphic computing
- Energy-efficient AI

### Path 5: Text-Diffusion (Most Complex)

**Why Last**:
- Requires transformers (attention, embeddings, etc.)
- Complex training (diffusion process)
- Longest dependency chain
- But most powerful for text generation

**Implementation Steps**:
1. Build transformer encoder-decoder
2. Implement masked diffusion (MDLM approach)
3. Add semantic-aware noising
4. Train on summarization/generation tasks

**Timeline**: 10-12 weeks

**Skills Gained**:
- Transformer architecture
- Diffusion processes
- Parallel generation
- Bidirectional context

---

## Recommended Learning Sequence

### Phase 1: Foundation (Weeks 1-2)
- [ ] ReLU, GELU, Leaky ReLU activations
- [ ] Adam, RMSprop optimizers
- [ ] 4-5 layer networks
- [ ] Residual connections

### Phase 2: TRM Implementation (Weeks 3-4) ⭐
- [ ] 2-layer recursive network
- [ ] Latent + answer update loops
- [ ] ARC-AGI-style puzzles
- [ ] Self-refinement visualization

### Phase 3: HRM Implementation (Weeks 5-7)
- [ ] Controller + Worker RNN modules
- [ ] Dual-frequency updates
- [ ] Sudoku puzzle environment
- [ ] Dimensionality hierarchy analysis

### Phase 4: Spiking Foundations (Weeks 8-10)
- [ ] LIF neuron implementation
- [ ] STDP learning rule
- [ ] Surrogate gradient training
- [ ] MNIST with SNNs

### Phase 5: BDH Implementation (Weeks 11-14)
- [ ] Scale-free network generation
- [ ] Hebbian plasticity
- [ ] Spiking attention mechanism
- [ ] Language task training

### Phase 6: Text-Diffusion (Weeks 15-20)
- [ ] Transformer implementation
- [ ] Masked diffusion process
- [ ] MDLM training
- [ ] Text generation demos

---

## Research Papers to Study

### Must-Read (In Order)

1. **TRM**: "Less is More: Recursive Reasoning with Tiny Networks" (2510.04871)
2. **HRM**: "Hierarchical Reasoning Model" (2506.21734)
3. **BDH**: "The Dragon Hatchling" (2509.26507)
4. **MDLM**: "Simple and Effective Masked Diffusion Language Model" (NeurIPS 2024)
5. **SNNs**: "Direct Training High-Performance Deep Spiking Neural Networks" (PMC11322636)

### Supporting Papers

- "Discrete Diffusion Language Model for Efficient Text Summarization" (ACL 2025)
- "Efficient Training of SNNs by Spike-aware Data Pruning" (2510.04098)
- "Rethinking Residual Connection in Training Large-Scale SNNs" (2024)

---

## Code Repositories to Study

1. **TRM**: github.com/SamsungSAILMontreal/TinyRecursiveModels (MIT License)
2. **MDLM**: github.com/kuleshov-group/mdlm (NeurIPS 2024)
3. **BDH**: github.com/pathwaycom/bdh (Pathway)
4. **Awesome-DLMs**: github.com/VILA-Lab/Awesome-DLMs (Survey of diffusion LMs)

---

## Key Insights for Implementation

### TRM Insight
**Recursion > Depth**: A tiny 2-layer network recursing beats deep one-shot networks

### HRM Insight
**Emergent Hierarchy**: Architecture induces dimensionality hierarchy without explicit programming

### BDH Insight
**Memory = Plasticity**: Synaptic plasticity IS working memory (no separate memory module)

### MDLM Insight
**Masked = Simplified**: Absorbing state diffusion reduces to mixture of MLM losses

### SNN Insight
**Surrogate Gradients**: Enable backprop through discrete spikes for end-to-end training

---

## Next Immediate Steps

Based on this research, I recommend:

### Option A: Fast Path to Results (TRM First) ⭐ RECOMMENDED
1. **Week 1**: Activations + optimizers foundation
2. **Week 2**: 2-layer recursive network architecture
3. **Week 3-4**: TRM implementation + ARC-AGI puzzles
4. **Result**: Working recursive reasoning model in 1 month

### Option B: Bio-Inspired Path (SNNs → BDH)
1. **Week 1-2**: Foundation + LIF neurons
2. **Week 3-4**: STDP + surrogate gradients
3. **Week 5-8**: SNN training on MNIST
4. **Week 9-12**: BDH with Hebbian learning
5. **Result**: Biologically plausible AI in 3 months

### Option C: Text Generation Path (Transformer → MDLM)
1. **Week 1-2**: Foundation + embeddings
2. **Week 3-6**: Attention mechanisms
3. **Week 7-10**: Transformer architecture
4. **Week 11-16**: Masked diffusion training
5. **Result**: Text generation model in 4 months

---

## My Recommendation

**Start with TRM** because:
1. ✅ Shortest path to a working advanced system (1 month)
2. ✅ Minimal dependencies (no embeddings, no attention)
3. ✅ Open-source code available to reference
4. ✅ Teaches recursive reasoning (applicable to all other models)
5. ✅ 7M parameters = fast training on CPU
6. ✅ Impressive demos (45% ARC-AGI, Sudoku-Extreme)

Then you can:
- Add HRM (multi-module) concepts
- Add BDH (biological) principles
- Add SNN (spiking) mechanics
- Eventually: Combine everything with text-diffusion + RAG

This gives you a working cutting-edge model quickly, then builds up complexity.

**What do you think? Should we start with TRM?**
