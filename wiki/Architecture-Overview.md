# Architecture Overview

This page provides a comprehensive overview of the neural network platform architecture, including system design, component relationships, and key architectural decisions.

## System Architecture

The project follows a **layered architecture** with clear separation of concerns:

```mermaid
graph TB
    subgraph "Application Layer"
        WASM[Yew WASM Frontend]
        CLI[Command-Line Examples]
    end

    subgraph "Library Layer"
        ANIM[neural-net-animator]
        VIZ[neural-net-viz]
        CORE[neural-net-core]
        TYPES[neural-net-types]
    end

    subgraph "Foundation Layer"
        NDARRAY[ndarray]
        SERDE[serde]
        RAND[rand]
    end

    WASM --> ANIM
    WASM --> VIZ
    CLI --> CORE
    CLI --> VIZ
    ANIM --> VIZ
    ANIM --> TYPES
    VIZ --> TYPES
    CORE --> TYPES
    TYPES --> NDARRAY
    TYPES --> SERDE
    CORE --> RAND
```

### Layer Descriptions

#### Application Layer
- **Yew WASM Frontend**: Interactive web application for visualizing neural networks
- **Command-Line Examples**: Educational examples demonstrating specific concepts

#### Library Layer
- **neural-net-types**: Foundation data structures (Network, Layer, errors)
- **neural-net-core**: Algorithms (forward/backward propagation, optimization)
- **neural-net-viz**: SVG visualization and rendering
- **neural-net-animator**: Animation framework for training visualization

#### Foundation Layer
- **ndarray**: Multi-dimensional array operations
- **serde**: Serialization/deserialization
- **rand**: Random number generation

## Crate Dependency Graph

```mermaid
graph LR
    TYPES[neural-net-types]
    CORE[neural-net-core]
    VIZ[neural-net-viz]
    ANIM[neural-net-animator]

    CORE --> TYPES
    VIZ --> TYPES
    ANIM --> TYPES
    ANIM --> VIZ

    style TYPES fill:#e1f5ff
    style CORE fill:#fff4e1
    style VIZ fill:#e8f5e9
    style ANIM fill:#f3e5f5
```

**Key Principle**: No circular dependencies. All crates depend on `neural-net-types` as the foundation.

## Component Architecture

### Network Structure

```mermaid
classDiagram
    class FeedForwardNetwork {
        -layers: Vec~Layer~
        -targets: Option~Vec~f32~~
        +new_with_config(input, hidden, output) Result
        +forward(inputs) Result~Vec~f32~~
        +train_single(inputs, targets, lr) Result
        +get_layer(index) Option~Layer~
    }

    class Layer {
        -index: usize
        -num_neurons: usize
        -weights: Option~Array2~f32~~
        -inputs: Vec~f32~
        -outputs: Vec~f32~
        -deltas: Vec~f32~
        +new(index, num_neurons, prev_size) Self
        +forward_propagate(...) Result
        +backward_propagate(...) Result
        +update_weights(...)
    }

    class NeuralNetError {
        <<enumeration>>
        InvalidConfig
        DimensionMismatch
        IoError
        TrainingError
    }

    FeedForwardNetwork "1" *-- "3..*" Layer : owns
    FeedForwardNetwork ..> NeuralNetError : returns
    Layer ..> NeuralNetError : returns
```

### Key Design Decisions

#### 1. Unidirectional Ownership

**C++ Pattern (Problematic):**
```
Layer* ←→ FFN*  (bidirectional pointers)
```

**Rust Pattern (Clean):**
```
FeedForwardNetwork → Vec<Layer>  (unidirectional ownership)
```

The network owns all layers, eliminating circular references and potential memory leaks.

#### 2. Trait-Based Extensibility

```mermaid
classDiagram
    class Activation {
        <<trait>>
        +activate(x: f32) f32
        +derivative(output: f32) f32
    }

    class Sigmoid {
        +activate(x: f32) f32
        +derivative(output: f32) f32
    }

    class Linear {
        +activate(x: f32) f32
        +derivative(output: f32) f32
    }

    class ReLU {
        +activate(x: f32) f32
        +derivative(output: f32) f32
    }

    Activation <|.. Sigmoid : implements
    Activation <|.. Linear : implements
    Activation <|.. ReLU : implements
```

All extensible components (activations, optimizers, layers) use traits for polymorphism.

## Data Structure Design

### Layer Memory Layout

```mermaid
graph LR
    subgraph "Input Layer (index=0)"
        IW[weights: None]
        II[inputs: Vec~f32~]
        IO[outputs: Vec~f32~]
        ID[deltas: Vec~f32~]
    end

    subgraph "Hidden Layer (index=1)"
        HW[weights: Array2~f32~]
        HI[inputs: Vec~f32~]
        HO[outputs: Vec~f32~]
        HD[deltas: Vec~f32~]
    end

    subgraph "Output Layer (index=2)"
        OW[weights: None]
        OI[inputs: Vec~f32~]
        OO[outputs: Vec~f32~]
        OD[deltas: Vec~f32~]
    end
```

**Key Points:**
- Input layer has no weights (passthrough)
- Hidden layer stores weight matrix (prev_size × num_neurons)
- Output layer has no weights (uses hidden layer outputs directly)
- All layers maintain inputs, outputs, and deltas for backpropagation

### Weight Matrix Layout

For a hidden layer connecting previous layer (size M) to current layer (size N):

```
Weights[M][N] where:
  - M = number of neurons in previous layer
  - N = number of neurons in current layer
  - weights[i][j] = connection strength from neuron i to neuron j
```

## Algorithm Flow Architecture

### Forward Propagation Flow

```mermaid
sequenceDiagram
    participant Client
    participant Network
    participant InputLayer
    participant HiddenLayer
    participant OutputLayer

    Client->>Network: forward(inputs)
    Network->>InputLayer: forward_propagate(inputs, Linear)
    InputLayer-->>Network: outputs (passthrough)

    Network->>HiddenLayer: forward_propagate(prev_outputs, Sigmoid)
    HiddenLayer->>HiddenLayer: weighted_sum = weights × prev_outputs
    HiddenLayer->>HiddenLayer: outputs = sigmoid(weighted_sum)
    HiddenLayer-->>Network: outputs

    Network->>OutputLayer: forward_propagate(prev_outputs, Linear)
    OutputLayer-->>Network: outputs (passthrough)

    Network-->>Client: final_outputs
```

### Backpropagation Flow

```mermaid
sequenceDiagram
    participant Client
    participant Network
    participant OutputLayer
    participant HiddenLayer
    participant InputLayer

    Client->>Network: train_single(inputs, targets, lr)
    Network->>Network: forward(inputs)

    Network->>OutputLayer: backward_propagate(targets, None)
    OutputLayer->>OutputLayer: deltas = targets - outputs
    OutputLayer-->>Network: deltas

    Network->>HiddenLayer: backward_propagate(prev_outputs, next_layer)
    HiddenLayer->>HiddenLayer: compute error from next layer
    HiddenLayer->>HiddenLayer: deltas = error × sigmoid_derivative
    HiddenLayer->>HiddenLayer: update_weights(lr)
    HiddenLayer-->>Network: deltas

    Network->>InputLayer: backward_propagate(inputs, next_layer)
    InputLayer-->>Network: done (no weights to update)

    Network-->>Client: Ok(())
```

## Error Handling Architecture

```mermaid
graph TD
    OP[Operation] --> CHECK{Valid?}
    CHECK -->|Yes| SUCCESS[Ok~T~]
    CHECK -->|No| ERROR[Err~NeuralNetError~]

    ERROR --> IC[InvalidConfig]
    ERROR --> DM[DimensionMismatch]
    ERROR --> IO[IoError]
    ERROR --> TE[TrainingError]

    IC --> CTX[Add context]
    DM --> CTX
    IO --> CTX
    TE --> CTX

    CTX --> PROP[Propagate with ?]
    PROP --> CALLER[Caller handles]
```

**Strategy:**
- All fallible operations return `Result<T, NeuralNetError>`
- Never panic in library code
- Provide context in error messages
- Use `?` operator for clean propagation

## Visualization Architecture

### SVG Generation Pipeline

```mermaid
graph LR
    NET[Network State] --> META[Extract Metadata]
    META --> VIZ[NetworkVisualization]
    VIZ --> SVG[SVG Generator]
    SVG --> FILE[Output File]

    CONFIG[VisualizationConfig] --> VIZ

    subgraph "Configuration"
        CONFIG
    end

    subgraph "Rendering"
        VIZ
        SVG
    end
```

### Animation Framework

```mermaid
graph TB
    SCRIPT[Animation Script] --> TIMELINE[Timeline]
    TIMELINE --> FRAME1[Frame 1]
    TIMELINE --> FRAME2[Frame 2]
    TIMELINE --> FRAMEN[Frame N]

    FRAME1 --> VIZ1[Visualize State 1]
    FRAME2 --> VIZ2[Visualize State 2]
    FRAMEN --> VIZN[Visualize State N]

    VIZ1 --> SVG1[SVG 1]
    VIZ2 --> SVG2[SVG 2]
    VIZN --> SVGN[SVG N]
```

## Security & Safety

### Memory Safety

```mermaid
graph LR
    RUST[Rust Ownership] --> NOSEG[No Segfaults]
    RUST --> NODATA[No Data Races]
    RUST --> NOLEAK[No Memory Leaks]

    BORROW[Borrow Checker] --> VALID[References Always Valid]
    BORROW --> NOALIAS[No Aliasing Violations]

    NOSEG --> SAFE[Memory Safe]
    NODATA --> SAFE
    NOLEAK --> SAFE
    VALID --> SAFE
    NOALIAS --> SAFE
```

**Guarantees:**
- No unsafe code in library (Phases 1-4)
- All references validated at compile time
- No manual memory management
- Thread-safe by default

## Performance Considerations

### Optimization Strategy

```mermaid
graph TB
    CODE[Code] --> PROFILE[Profile]
    PROFILE --> HOTSPOT{Hot Path?}

    HOTSPOT -->|Yes| OPT[Optimize]
    HOTSPOT -->|No| DONE[Done]

    OPT --> BLAS[Use BLAS/LAPACK]
    OPT --> SIMD[SIMD Operations]
    OPT --> CACHE[Cache Locality]
    OPT --> INLINE[Inline Functions]

    BLAS --> VERIFY[Verify Performance]
    SIMD --> VERIFY
    CACHE --> VERIFY
    INLINE --> VERIFY

    VERIFY --> PROFILE
```

**Optimizations Applied:**
1. **BLAS Integration**: `ndarray` with optimized linear algebra
2. **Memory Layout**: Contiguous memory for cache efficiency
3. **Borrowing**: Use references to avoid copies
4. **Iterator Chains**: Lazy evaluation

## Testing Architecture

### Test Pyramid

```mermaid
graph TB
    E2E[End-to-End Tests]
    INT[Integration Tests]
    UNIT[Unit Tests]

    E2E --> EX1[XOR Example]
    E2E --> EX2[Boolean Logic]

    INT --> NET[Network Training]
    INT --> IO[File I/O]

    UNIT --> ACT[Activations]
    UNIT --> FWD[Forward Prop]
    UNIT --> BWD[Backward Prop]
    UNIT --> LAYER[Layer Operations]

    style E2E fill:#ffebee
    style INT fill:#fff3e0
    style UNIT fill:#e8f5e9
```

**Coverage:**
- **Unit Tests**: Individual functions and methods
- **Integration Tests**: Multiple components working together
- **End-to-End Tests**: Complete examples (negative + positive)

## Related Pages

- [[Core-Components]] - Detailed crate descriptions
- [[Training-Algorithms]] - Algorithm implementations
- [[Data-Flow]] - Sequence diagrams for data flow
- [[Error-Handling]] - Error handling patterns
- [[Testing-Strategy]] - Comprehensive testing approach

## External Documentation

- [Architecture Document](../../blob/main/documentation/architecture.md) - Full architecture specification
- [PRD](../../blob/main/documentation/PRD.md) - Product requirements
- [Process Guide](../../blob/main/documentation/process.md) - Development workflow
