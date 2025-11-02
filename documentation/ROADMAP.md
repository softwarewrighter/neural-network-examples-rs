# Neural Network Examples - Learning Roadmap

**Vision**: Build advanced ML concepts from scratch in Rust to demonstrate:
- Text-diffusion models
- Hierarchical reasoning systems
- Evolutionary learning ("baby dragon hatchlings")
- Tiny recursion models (self-modifying networks)
- Spiking neural networks (SNNs)
- RAG (Retrieval-Augmented Generation) for continuous learning

**Philosophy**: Educational implementation from scratch, minimal third-party ML libraries, maximum understanding.

---

## Current State (Phase 0-1: COMPLETE ✅)

**What We Have:**
- Basic feedforward networks (2-3 layers)
- Backpropagation training
- Sigmoid activation function
- Boolean logic & simple arithmetic examples
- Visualization (SVG network diagrams)
- Checkpoint save/load

**Capabilities:**
- XOR, parity, majority functions
- Half-adder, full-adder circuits
- Single hidden layer learning

**Limitations:**
- Only sigmoid activation
- No sequence handling
- No attention mechanisms
- No probabilistic modeling
- No temporal dynamics
- No hierarchical structures

---

## Phase 2: Foundation Expansion (NEXT 2-3 weeks)

Build the essential primitives needed for advanced architectures.

### 2.1: Activation Function Library ⭐ CRITICAL
**Why**: Different activations enable different learning dynamics

- [x] Sigmoid (already have)
- [ ] **ReLU** (Rectified Linear Unit) - Standard for deep networks
- [ ] **Leaky ReLU** - Prevents dying neurons
- [ ] **Tanh** - Centered activation
- [ ] **Softmax** - Multi-class classification
- [ ] **GELU** (Gaussian Error Linear Unit) - Used in transformers
- [ ] **Swish/SiLU** - Modern smooth activation

**Examples**: Demonstrate same XOR problem with different activations, compare convergence.

### 2.2: Advanced Optimizers ⭐ CRITICAL
**Why**: SGD alone won't scale to complex models

- [x] SGD (what we have now)
- [ ] **SGD with Momentum** - Accelerated convergence
- [ ] **RMSprop** - Adaptive learning rates per parameter
- [ ] **Adam** - Combines momentum + RMSprop (industry standard)
- [ ] **AdamW** - Adam with proper weight decay

**Examples**: Training comparison on same task, loss curve visualization.

### 2.3: Deeper Networks (3-5 layers)
**Why**: Hierarchical features needed for complex tasks

- [ ] Multi-layer architecture (4-5 hidden layers)
- [ ] Gradient flow analysis
- [ ] Vanishing/exploding gradient demonstration
- [ ] **Batch Normalization** - Stabilize training
- [ ] **Layer Normalization** - Better for sequences
- [ ] **Residual Connections** - Enable very deep networks

**Examples**: Deep XOR, visual hierarchy demonstration.

### 2.4: Regularization Techniques
**Why**: Prevent overfitting on real data

- [ ] **L1/L2 Regularization** - Weight penalties
- [ ] **Dropout** - Random neuron disabling
- [ ] **Early Stopping** - Validation-based stopping
- [ ] **Data Augmentation** - Expand training set

**Examples**: Overfitting demonstration, regularization comparison.

---

## Phase 3: Sequence & Embeddings (3-4 weeks)

Enable working with text and sequences - prerequisite for all NLP-based goals.

### 3.1: Word Embeddings ⭐ CRITICAL for Text
**Why**: Map discrete tokens to continuous vectors

- [ ] **One-hot encoding** - Baseline representation
- [ ] **Learned embeddings** - Trainable lookup table
- [ ] **Positional encodings** - Add position information (for transformers)
- [ ] **Byte-pair encoding (BPE)** - Subword tokenization

**Examples**: Word similarity, embedding visualization (t-SNE/PCA).

### 3.2: Recurrent Neural Networks (RNNs)
**Why**: Handle sequential data, maintain state

- [ ] **Vanilla RNN** - Basic recurrence
- [ ] **LSTM** (Long Short-Term Memory) - Solves vanishing gradients
- [ ] **GRU** (Gated Recurrent Unit) - Simpler LSTM
- [ ] **Bidirectional RNNs** - Context from both directions

**Examples**: Character-level text generation, sequence prediction.

### 3.3: Attention Mechanisms ⭐ CRITICAL for Transformers
**Why**: Foundation for modern NLP, reasoning, diffusion models

- [ ] **Scaled Dot-Product Attention** - Core attention formula
- [ ] **Multi-Head Attention** - Parallel attention patterns
- [ ] **Self-Attention** - Attend to own sequence
- [ ] **Cross-Attention** - Attend to different sequence
- [ ] **Masked Attention** - Autoregressive generation

**Examples**: Attention visualization, sequence-to-sequence translation.

---

## Phase 4: Transformer Architecture (4-5 weeks)

The backbone of modern NLP - needed for text-diffusion, reasoning, RAG.

### 4.1: Basic Transformer ⭐ CRITICAL PATH
**Why**: Enables all text-based advanced goals

- [ ] **Encoder**: Self-attention + FFN layers
- [ ] **Decoder**: Masked self-attention + cross-attention + FFN
- [ ] **Encoder-Decoder**: Full transformer
- [ ] **Decoder-Only**: GPT-style (for generation)
- [ ] **Encoder-Only**: BERT-style (for understanding)

**Examples**: Simple translation, text classification.

### 4.2: Training Techniques for Transformers
- [ ] **Gradient Clipping** - Prevent exploding gradients
- [ ] **Warmup Learning Rate** - Stabilize early training
- [ ] **Label Smoothing** - Better calibration
- [ ] **Mixed Precision Training** - Faster computation

### 4.3: Text Generation
- [ ] **Greedy Decoding** - Take most likely token
- [ ] **Beam Search** - Track top K sequences
- [ ] **Sampling** - Temperature-based randomness
- [ ] **Top-K / Top-P (Nucleus) Sampling** - Controlled diversity

**Examples**: Simple language model, story completion.

---

## Phase 5: Probabilistic & Generative Models (5-6 weeks)

Enable sampling, generation, and uncertainty - needed for diffusion and evolutionary systems.

### 5.1: Variational Autoencoders (VAEs)
**Why**: Learn latent representations, sampling

- [ ] **Encoder-Decoder Architecture**
- [ ] **Reparameterization Trick** - Backprop through sampling
- [ ] **KL Divergence Loss** - Regularize latent space
- [ ] **Conditional VAEs** - Controlled generation

**Examples**: Image/text reconstruction, latent space exploration.

### 5.2: Generative Adversarial Networks (GANs)
**Why**: Adversarial training concepts

- [ ] **Generator Network**
- [ ] **Discriminator Network**
- [ ] **Minimax Training** - Adversarial loss
- [ ] **Wasserstein GAN** - Stable training

**Examples**: Simple image generation, mode collapse demonstration.

### 5.3: Energy-Based Models
**Why**: Foundation for diffusion models

- [ ] **Energy Functions** - Score-based modeling
- [ ] **Contrastive Divergence** - Training EBMs
- [ ] **Langevin Dynamics** - Sampling from energy

---

## Phase 6: Diffusion Models (6-8 weeks)

### 6.1: Denoising Diffusion Probabilistic Models (DDPMs) ⭐ TEXT-DIFFUSION GOAL
**Why**: One of your primary goals

- [ ] **Forward Diffusion Process** - Add noise gradually
- [ ] **Reverse Diffusion Process** - Denoise step-by-step
- [ ] **Noise Prediction Network** - U-Net style
- [ ] **Training Objective** - Predict noise at each step
- [ ] **Sampling Algorithm** - Generate from noise

**Examples**: Image generation (MNIST), visual diffusion process.

### 6.2: Text Diffusion ⭐ PRIMARY GOAL
**Why**: Your explicit goal

- [ ] **Discrete Diffusion** - For tokens instead of continuous pixels
- [ ] **Embedding Space Diffusion** - Diffuse in embedding space
- [ ] **Absorbing State Diffusion** - Mask-based diffusion
- [ ] **Autoregressive Diffusion** - Combine with transformers

**Examples**: Text generation via diffusion, compare to autoregressive.

### 6.3: Guidance & Conditioning
- [ ] **Classifier Guidance** - Steer generation
- [ ] **Classifier-Free Guidance** - Better steering
- [ ] **Conditioning Mechanisms** - Controlled generation

---

## Phase 7: Hierarchical & Graph Structures (7-9 weeks)

### 7.1: Graph Neural Networks (GNNs) ⭐ HIERARCHICAL REASONING GOAL
**Why**: Enable reasoning over structured knowledge

- [ ] **Message Passing** - Node communication
- [ ] **Graph Attention Networks (GATs)** - Attention on graphs
- [ ] **Graph Convolutions (GCNs)** - Neighborhood aggregation
- [ ] **Hierarchical Graphs** - Multi-scale structures

**Examples**: Knowledge graph reasoning, hierarchical classification.

### 7.2: Tree-Structured Networks
**Why**: Explicit hierarchical reasoning

- [ ] **Tree LSTMs** - Recurrence over trees
- [ ] **Tree Attention** - Attend to tree structures
- [ ] **Recursive Networks** - Bottom-up/top-down processing

### 7.3: Hierarchical Reasoning Model ⭐ PRIMARY GOAL
**Why**: Your explicit goal

- [ ] **Multi-Level Abstraction** - Coarse-to-fine reasoning
- [ ] **Planning Networks** - Goal-directed reasoning
- [ ] **Working Memory** - Explicit memory modules
- [ ] **Compositional Reasoning** - Combine sub-solutions

**Examples**: Multi-step problem solving, planning tasks.

---

## Phase 8: Meta-Learning & Recursion (8-10 weeks)

### 8.1: Meta-Learning (Learning to Learn)
**Why**: Foundation for self-modifying systems

- [ ] **MAML** (Model-Agnostic Meta-Learning) - Fast adaptation
- [ ] **Reptile** - First-order meta-learning
- [ ] **Meta-Networks** - Networks that output networks

### 8.2: Neural Architecture Search (NAS)
- [ ] **Evolutionary NAS** - Genetic algorithms for architectures
- [ ] **Gradient-Based NAS** - Differentiable architecture search
- [ ] **Weight Sharing** - Efficient search

### 8.3: Tiny Recursion Model ⭐ PRIMARY GOAL
**Why**: Your explicit goal - self-modifying networks

- [ ] **Self-Referential Networks** - Networks operating on themselves
- [ ] **Weight Prediction Networks** - Networks generating weights
- [ ] **Hyper-Networks** - Meta-parameter generation
- [ ] **Neural Turing Machines** - External memory + attention

**Examples**: Self-improving optimization, adaptive architectures.

---

## Phase 9: Evolutionary & Neuroevolution (9-11 weeks)

### 9.1: Genetic Algorithms
**Why**: Foundation for evolutionary approaches

- [ ] **Population Management** - Maintain diverse agents
- [ ] **Fitness Evaluation** - Score agents
- [ ] **Selection** - Tournament, roulette wheel
- [ ] **Crossover** - Combine parent solutions
- [ ] **Mutation** - Random variation

### 9.2: Neuroevolution ⭐ BABY DRAGON HATCHLINGS GOAL
**Why**: Your explicit goal - evolving neural agents

- [ ] **NEAT** (NeuroEvolution of Augmenting Topologies) - Evolve topology + weights
- [ ] **ES** (Evolution Strategies) - Gradient-free optimization
- [ ] **Fitness Landscapes** - Visualize evolution
- [ ] **Speciation** - Protect innovation
- [ ] **Agent Lifecycles** - Birth, growth, death, reproduction

**Examples**: Evolving XOR solver, game-playing agents, "dragon hatchlings" ecosystem.

### 9.3: Competitive Evolution
- [ ] **Co-Evolution** - Agents compete/cooperate
- [ ] **Predator-Prey Dynamics** - Ecosystem simulation
- [ ] **Resource Competition** - Survival of fittest

---

## Phase 10: Spiking Neural Networks (10-12 weeks)

### 10.1: Neuron Models ⭐ SPIKING NETWORKS GOAL
**Why**: Your explicit goal - biologically plausible learning

- [ ] **Leaky Integrate-and-Fire (LIF)** - Basic spiking neuron
- [ ] **Izhikevich Model** - Rich dynamics, computationally efficient
- [ ] **Hodgkin-Huxley** - Biologically detailed (optional)
- [ ] **Adaptive Exponential (AdEx)** - Modern balance

### 10.2: Spike-Based Learning
- [ ] **STDP** (Spike-Timing-Dependent Plasticity) - Hebbian learning
- [ ] **R-STDP** - Reward-modulated STDP
- [ ] **Surrogate Gradients** - Backprop through spikes
- [ ] **Temporal Coding** - Encode information in spike timing

### 10.3: SNN Architectures
- [ ] **Feedforward SNNs** - Spike propagation
- [ ] **Recurrent SNNs** - Temporal dynamics
- [ ] **Convolutional SNNs** - Spatial processing
- [ ] **Neuromorphic Encoding** - Convert continuous to spikes

**Examples**: Event-based vision, temporal pattern recognition, low-power inference.

---

## Phase 11: Knowledge & Retrieval (11-13 weeks)

### 11.1: Vector Databases (From Scratch)
**Why**: Foundation for RAG

- [ ] **Vector Storage** - Efficient embedding storage
- [ ] **Similarity Search** - Cosine similarity, Euclidean distance
- [ ] **Approximate Nearest Neighbors (ANN)** - HNSW, IVF
- [ ] **Indexing Structures** - Fast retrieval

### 11.2: Retrieval Mechanisms ⭐ RAG GOAL
**Why**: Your explicit goal - continuous learning

- [ ] **Dense Retrieval** - Embedding-based search
- [ ] **Sparse Retrieval** - BM25, TF-IDF
- [ ] **Hybrid Retrieval** - Combine dense + sparse
- [ ] **Re-ranking** - Refine retrieval results

### 11.3: Retrieval-Augmented Generation (RAG) ⭐ PRIMARY GOAL
**Why**: Your explicit goal

- [ ] **Query Encoding** - Convert question to embedding
- [ ] **Document Retrieval** - Find relevant context
- [ ] **Context Injection** - Add retrieved docs to prompt
- [ ] **Answer Generation** - Generate with context
- [ ] **Continuous Learning** - Update knowledge base
- [ ] **Memory Management** - Long-term/short-term memory

**Examples**: Question answering, knowledge-grounded generation.

---

## Phase 12: Integration & Advanced Systems (14+ weeks)

### 12.1: Combined Systems ⭐ ULTIMATE GOALS

**Text-Diffusion + RAG**:
- Diffusion models that retrieve and incorporate knowledge
- Continuous learning through retrieval

**Hierarchical Reasoning + RAG**:
- Multi-level reasoning with knowledge retrieval
- Planning with external memory

**Spiking Networks + Evolution**:
- Evolve spiking network architectures
- Energy-efficient learning agents

**Recursive Models + Meta-Learning**:
- Self-modifying systems that learn to learn
- Adaptive reasoning strategies

**Dragon Hatchlings Ecosystem**:
- Evolutionary population of learning agents
- Spiking or standard networks as "brains"
- Fitness based on task performance
- RAG-based knowledge sharing between generations
- Hierarchical reasoning for complex behaviors
- Recursive self-improvement

### 12.2: Visualization & Analysis Tools
- [ ] **Training Dashboards** - Real-time monitoring
- [ ] **Attention Visualizations** - What networks focus on
- [ ] **Embedding Explorers** - Interactive latent spaces
- [ ] **Evolution Simulators** - Population dynamics
- [ ] **Spike Raster Plots** - Temporal activity
- [ ] **Knowledge Graph Viewers** - RAG memory visualization

---

## Dependency Graph (What Enables What)

```
Phase 1 (Current) → Phase 2 (Foundation)
                                ↓
                         Phase 3 (Sequences)
                                ↓
                         Phase 4 (Transformers) ──→ TEXT-DIFFUSION (Phase 6.2)
                                ↓                            ↓
                         Phase 5 (Probabilistic)      RAG (Phase 11.3)
                                ↓                            ↑
                         Phase 6 (Diffusion) ←───────────────┘

Phase 2 → Phase 7 (Graphs/Hierarchy) → HIERARCHICAL REASONING (7.3)

Phase 2 → Phase 8 (Meta-Learning) → TINY RECURSION (8.3)

Phase 2 → Phase 9 (Evolution) → BABY DRAGONS (9.2)

Phase 2 → Phase 10 (SNNs) → SPIKING NETWORKS (10.1-10.3)
                        ↓
                Phase 9 (Evolution) → Evolved SNNs

All converge → Phase 12 (Integration)
```

---

## Timeline Estimate

**Conservative Estimate** (Learning + Implementation + Documentation):
- Phase 2: 2-3 weeks
- Phase 3: 3-4 weeks
- Phase 4: 4-5 weeks
- Phase 5: 5-6 weeks
- Phase 6: 6-8 weeks
- Phase 7: 7-9 weeks
- Phase 8: 8-10 weeks
- Phase 9: 9-11 weeks
- Phase 10: 10-12 weeks
- Phase 11: 11-13 weeks
- Phase 12: 14+ weeks

**Total**: ~6-8 months of focused development

**Parallel Paths**: Some phases can be done in parallel:
- Phases 7, 8, 9, 10 are relatively independent after Phase 2
- Could work on SNNs while implementing transformers
- Evolution can progress alongside hierarchical reasoning

---

## Recommended Immediate Next Steps

Given your ultimate goals, here's the optimal path:

### Critical Path 1: Text-Diffusion + RAG
1. **Phase 2.1**: Activation functions (ReLU, GELU) - 3 days
2. **Phase 2.2**: Adam optimizer - 3 days
3. **Phase 3.1**: Embeddings + tokenization - 1 week
4. **Phase 3.3**: Attention mechanisms - 2 weeks
5. **Phase 4.1**: Basic transformer - 3 weeks
6. **Phase 5.1**: VAEs (optional but helpful) - 1 week
7. **Phase 6.1**: Image diffusion (practice) - 2 weeks
8. **Phase 6.2**: Text diffusion ⭐ GOAL - 3 weeks
9. **Phase 11**: RAG ⭐ GOAL - 2 weeks

**Time to first major goal**: ~3-4 months

### Critical Path 2: Hierarchical Reasoning
1. **Phase 2.1-2.2**: Activations + optimizers - 1 week
2. **Phase 2.3**: Deep networks - 1 week
3. **Phase 3.3**: Attention - 2 weeks
4. **Phase 7.1**: GNNs - 2 weeks
5. **Phase 7.2**: Tree networks - 2 weeks
6. **Phase 7.3**: Hierarchical reasoning ⭐ GOAL - 3 weeks

**Time to goal**: ~2.5-3 months

### Critical Path 3: Baby Dragons + SNNs
1. **Phase 2.1-2.2**: Activations + optimizers - 1 week
2. **Phase 9.1**: Genetic algorithms - 1 week
3. **Phase 9.2**: Neuroevolution ⭐ GOAL - 2 weeks
4. **Phase 10.1-10.3**: Spiking networks ⭐ GOAL - 4 weeks
5. **Phase 9.3**: Competitive evolution - 2 weeks

**Time to goal**: ~2.5 months

### Critical Path 4: Tiny Recursion
1. **Phase 2**: Full foundation - 2 weeks
2. **Phase 8.1**: Meta-learning - 3 weeks
3. **Phase 8.3**: Self-modifying networks ⭐ GOAL - 3 weeks

**Time to goal**: ~2 months

---

## My Recommendation

**Start with the foundation that benefits ALL paths:**

### Week 1-2: Phase 2 Foundation
- [ ] ReLU, Leaky ReLU, GELU activations
- [ ] Adam optimizer
- [ ] Deeper networks (4-5 layers)
- [ ] Better visualization (loss curves)

**Why**: Every advanced system needs these. Build once, use everywhere.

### Then pick ONE primary path based on interest:

**Option A**: Text-focused (Diffusion + RAG)
- Most technically challenging
- Covers transformers (widely applicable)
- Longest path but most complete skillset

**Option B**: Evolution-focused (Dragons + SNNs)
- Fastest to working demos
- Most fun/visual results
- Biological inspiration angle

**Option C**: Reasoning-focused (Hierarchical + Recursion)
- Deep AI concepts
- Meta-learning is cutting-edge research
- Leads to AGI-related topics

---

## Question for You

Before we start Phase 2, which ultimate goal excites you most?

1. **Text-Diffusion + RAG** - Controllable text generation with knowledge
2. **Baby Dragon Hatchlings** - Evolving agents in ecosystem
3. **Hierarchical Reasoning** - Multi-level planning/problem-solving
4. **Tiny Recursion** - Self-modifying networks
5. **Spiking Networks** - Biologically plausible learning

Or do you want to build the foundation first and decide later?

This will help me prioritize what examples/infrastructure to build next.
