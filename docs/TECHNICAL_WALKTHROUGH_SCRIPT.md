# Technical Walkthrough Script: Autonomous Rocket Landing
## 5-10 Minute Technical Video for ML Engineers and Graders

**Target Audience:** ML engineers, course graders, technical reviewers  
**Tone:** Technical, detailed, demonstrating code understanding  
**Timing:** ~8 minutes total

---

## [INTRODUCTION] (30 seconds)

"Hi, I'm [Name], and this is [Name]. This technical walkthrough demonstrates how our reinforcement learning project implements advanced deep RL techniques for autonomous rocket landing.

We'll cover the architecture, key ML components, how they work together synergistically, and the technical challenges we overcame. All code references use relative paths from the project root."

---

## [SECTION 1: Project Architecture Overview] (1 minute)

### Architecture Structure

"Our project is organized into modular components that work together:

**Core Agents** (`src/agents/`):
- `dqn.py` - Deep Q-Network with experience replay and target networks
- `a2c.py` - Actor-Critic with separate policy and value networks

**Networks** (`src/networks/`):
- `dqn_network.py` - Q-network architecture
- `policy_network.py` - Actor network for A2C
- `value_network.py` - Critic network for A2C

**Training Infrastructure** (`src/training/`):
- `trainer.py` - Unified training loop with validation, early stopping
- `train_dqn.py` and `train_a2c.py` - Algorithm-specific entry points

**Evaluation** (`src/evaluation/`):
- `evaluator.py` - Metrics computation and agent comparison
- `compare_agents.py` - Multi-agent evaluation framework

**Key Design**: All components share common interfaces (TrainingConfig, OptimizerConfig, NetworkConfig) enabling code reuse and consistent evaluation."

**Code References:**
- Project structure: `src/` directory organization
- Config system: `src/utils/config.py` lines 1-100

---

## [SECTION 2: DQN Implementation - Synergistic Components] (2 minutes)

### Experience Replay and Target Networks Working Together

"Let's examine how DQN's key components work together. This addresses **checkbox46** - DQN with experience replay and target networks.

**Experience Replay Buffer** (`src/agents/dqn.py` lines 50-120):
The `ExperienceReplay` class stores transitions (state, action, reward, next_state, done) in a circular buffer. This enables:
- Sample efficiency: Learning from past experiences multiple times
- Decorrelation: Random sampling breaks temporal correlations
- Stability: Diverse experiences prevent catastrophic forgetting

**Target Network** (`src/agents/dqn.py` lines 350-380):
The target network provides stable Q-value targets. Every `target_update_freq` steps, we copy weights from the main Q-network to the target network. This prevents the 'moving target' problem where the target values change too rapidly during training.

**Synergistic Integration** (`src/agents/dqn.py` lines 400-430):
The `update` method demonstrates how they work together:
1. Sample a batch from experience replay (line 405)
2. Compute Q-values using main network (line 410)
3. Compute target Q-values using target network (line 415)
4. Compute TD error and update main network (line 420)
5. Periodically update target network (line 425)

This creates a stable learning loop: experience replay provides diverse training data, while the target network provides stable learning signals."

**Code References:**
- Experience replay: `src/agents/dqn.py` lines 50-120 (ExperienceReplay class)
- Target network updates: `src/agents/dqn.py` lines 350-380
- Update method: `src/agents/dqn.py` lines 400-430
- Checkbox46 evidence: Lines 50-120, 350-380, 400-430

---

## [SECTION 3: A2C Implementation - Actor-Critic Synergy] (2 minutes)

### Policy and Value Networks Working Together

"This addresses **checkbox48** (policy gradient method) and **checkbox49** (actor-critic architecture).

**Separate Networks** (`src/agents/a2c.py` lines 60-80):
A2C uses two separate networks:
- Policy network (`policy_network.py`) - outputs action probabilities
- Value network (`value_network.py`) - estimates state values

**Advantage Computation** (`src/agents/a2c.py` lines 300-350):
The value network computes advantages using n-step returns:
```python
# Line 320: Compute n-step return
G = sum of discounted rewards + gamma^n * V(next_state)
advantage = G - V(current_state)
```

**Synergistic Updates** (`src/agents/a2c.py` lines 380-470):
Both networks update together:
1. Policy network updates using policy gradient with advantages (lines 380-420)
   - Advantage reduces variance in policy gradient estimates
   - Value network provides the baseline
2. Value network updates to better estimate returns (lines 420-470)
   - Learns to predict future returns more accurately
   - This improves advantage estimates for policy updates

**The Synergy**: The value network's improved estimates lead to better advantage calculations, which improve policy updates, which generate better trajectories, which improve value estimates - creating a positive feedback loop.

**N-step Returns** (`src/agents/a2c.py` lines 216-250):
We use n-step returns (default n=5) instead of single-step, providing:
- Lower variance than Monte Carlo
- Less bias than single-step TD
- Better sample efficiency"

**Code References:**
- Network initialization: `src/agents/a2c.py` lines 60-80
- Advantage computation: `src/agents/a2c.py` lines 300-350
- Policy updates: `src/agents/a2c.py` lines 380-420
- Value updates: `src/agents/a2c.py` lines 420-470
- N-step returns: `src/agents/a2c.py` lines 216-250
- Checkbox48 evidence: Lines 200-400, 250-300, 380-420
- Checkbox49 evidence: Lines 60-80, 300-350, 420-470

---

## [SECTION 4: Custom Reward Shaping - Integration with Training] (1.5 minutes)

### Reward Wrapper and Training Integration

"This addresses **checkbox47** - custom reward function with justification.

**Reward Wrapper** (`src/environments/reward_wrapper.py` lines 45-95):
Our `RocketRewardWrapper` transforms sparse LunarLander rewards into dense, shaped rewards:
- Landing bonus: Encourages successful landings
- Fuel penalty: Promotes efficiency
- Crash penalty: Discourages unsafe behavior
- Smoothness bonus: Encourages stable control

**Integration Point** (`src/training/trainer.py` lines 188-200):
The wrapper is integrated at environment creation:
```python
env = gym.make("LunarLander-v3")
env = RocketRewardWrapper(env, reward_config)
```

**Why This Matters**: The shaped rewards provide learning signal at every step, not just at episode termination. This is critical because:
- Sparse rewards make learning extremely difficult
- Dense rewards guide the agent toward desired behaviors
- Both DQN and A2C benefit from this, demonstrating component reuse

**Configurability** (`src/utils/config.py` lines 85-100):
Reward coefficients are parameterized via `RewardConfig`, allowing hyperparameter tuning of the reward function itself."

**Code References:**
- Reward wrapper: `src/environments/reward_wrapper.py` lines 45-95
- Integration: `src/training/trainer.py` lines 188-200
- Config: `src/utils/config.py` lines 85-100
- Checkbox47 evidence: `reward_wrapper.py` entire file, lines 45-95

---

## [SECTION 5: Training Infrastructure - Unified Framework] (1.5 minutes)

### Train/Validation/Test Split and Early Stopping

"This addresses **checkbox0** (train/val/test split) and **checkbox4** (regularization).

**Seed-Based Splits** (`src/training/trainer.py` lines 248-280):
We use seed-based evaluation instead of data splits:
- Training seeds: 42-52 (random exploration)
- Validation seeds: 100-109 (model selection)
- Test seeds: 200-209 (final evaluation)

This is appropriate for RL because:
- Environment determinism with fixed seeds enables fair comparison
- Different seeds test generalization to different initial conditions
- Prevents data leakage between splits

**Early Stopping** (`src/training/trainer.py` lines 230-235, 310-330):
Early stopping prevents overfitting:
- Monitors validation return
- Stops if no improvement for `patience` episodes
- Saves best model based on validation performance

**Regularization Techniques**:
- **L2 Weight Decay** (`src/utils/config.py` lines 50-60): Applied via optimizer
- **Dropout** (`src/networks/dqn_network.py` line 45, `value_network.py` line 50): Applied in network layers
- **Gradient Clipping**: Applied during updates to prevent exploding gradients

**Synergy**: Early stopping uses validation performance, which is computed using the same evaluation framework that computes test metrics - ensuring consistency."

**Code References:**
- Seed splits: `src/training/trainer.py` lines 248-280
- Early stopping: `src/training/trainer.py` lines 230-235, 310-330
- L2 decay: `src/utils/config.py` lines 50-60
- Dropout: `src/networks/dqn_network.py` line 45, `value_network.py` line 50
- Checkbox0 evidence: Lines 248-280, 260-270, 280-310
- Checkbox4 evidence: Config lines 50-60, dqn_network.py line 45, trainer.py lines 230-235, 310-330

---

## [SECTION 6: Optimizer Comparison - Systematic Evaluation] (1 minute)

### Multi-Optimizer Training and Evaluation

"This addresses **checkbox16** - optimizer comparison with documented evaluation.

**Training Multiple Configurations** (`src/scripts/run_all_experiments.py` lines 60-99):
We train DQN with both Adam and RMSprop, and A2C with Adam and SGD. Each uses identical architectures and hyperparameters except the optimizer.

**Unified Evaluation** (`src/evaluation/compare_agents.py` lines 17-83):
The `compare_all_agents` function evaluates all models on the same validation and test seeds, ensuring fair comparison:
- Same environment seeds
- Same number of episodes
- Same metrics computed

**Results Storage** (`results/test_results.csv`):
All results are stored in CSV format with:
- Algorithm and optimizer identifiers
- Quantitative metrics (returns, success rates, fuel usage)
- Standard deviations for statistical significance

**Key Finding**: For DQN, both optimizers achieve 100% success, but Adam is more fuel-efficient. For A2C, Adam is essential - SGD fails to converge. This demonstrates that optimizer choice is algorithm-dependent."

**Code References:**
- Training scripts: `src/scripts/run_all_experiments.py` lines 60-99
- Comparison framework: `src/evaluation/compare_agents.py` lines 17-83
- Results: `results/test_results.csv`
- Checkbox16 evidence: `docs/EVALUATION.md` Section 1, `results/test_results.csv`

---

## [SECTION 7: Hyperparameter Tuning Framework] (1 minute)

### Systematic Hyperparameter Sweep

"This addresses **checkbox5** - systematic hyperparameter tuning.

**Grid Search Framework** (`src/hyperparameter_tuning/sweep.py` lines 18-120):
Our `grid_search` function:
1. Generates all hyperparameter combinations
2. Trains each configuration
3. Evaluates on validation set
4. Selects best based on validation performance

**Key Features**:
- Validation-based selection (prevents overfitting to training)
- Intermediate result saving (resume capability)
- Fast mode option (reduced episodes for quick exploration)

**Integration with Training** (`src/hyperparameter_tuning/sweep.py` lines 80-95):
The sweep uses the same `train_agent` function as regular training, ensuring:
- Consistent training procedure
- Same validation framework
- Reproducible results

**Example Usage** (`src/scripts/run_dqn_sweep_fast.py`):
Fast sweep mode reduces training episodes and validation overhead for rapid exploration, then full training confirms best configurations."

**Code References:**
- Grid search: `src/hyperparameter_tuning/sweep.py` lines 18-120
- Integration: `src/hyperparameter_tuning/sweep.py` lines 80-95
- Fast mode: `src/scripts/run_dqn_sweep_fast.py`
- Checkbox5 evidence: `sweep.py` entire file, lines 80-150, 60-80

---

## [SECTION 8: Error Analysis and Ablation Studies] (1 minute)

### Comprehensive Evaluation

"This addresses **checkbox63** (error analysis) and **checkbox67** (ablation study).

**Error Analysis** (`src/scripts/error_analysis.py`):
The error analysis script:
1. Loads trained agents
2. Runs evaluation episodes
3. Identifies failure cases (crashes, timeouts)
4. Extracts failure characteristics (velocity, altitude, position)
5. Generates visualizations and CSV reports

**Key Insight**: Most failures occur during final approach, suggesting the agent learns navigation but struggles with fine-grained landing control.

**Ablation Study** (`src/scripts/ablation_study.py`):
We systematically remove components to understand their impact:
- Baseline: Full DQN with replay + target network
- No replay: Online learning only
- No target: Direct Q-learning
- Custom reward: Tests reward shaping impact

**Quantitative Comparison** (`ablation_results/ablation_study_results.csv`):
Results show that experience replay and target networks are critical for stable learning, while custom reward shaping significantly improves learning signal."

**Code References:**
- Error analysis: `src/scripts/error_analysis.py` entire file
- Ablation study: `src/scripts/ablation_study.py` entire file
- Results: `ablation_results/ablation_study_results.csv`
- Checkbox63 evidence: `docs/EVALUATION.md` Section 3, `error_analysis/` directory
- Checkbox67 evidence: `docs/EVALUATION.md` Section 4, `ablation_results/` directory

---

## [SECTION 9: Technical Challenges and Solutions] (1 minute)

### Key Technical Contributions

**Challenge 1: Optimizer State Dict Compatibility**
When loading A2C checkpoints, we must match the optimizer type used during training. Solution: Load optimizer config from checkpoint before creating agent (`src/scripts/evaluate_agent.py` lines 55-68).

**Challenge 2: Consistent Evaluation**
Ensuring all models evaluated on identical conditions. Solution: Unified `compare_agents` framework with seed-based evaluation (`src/evaluation/compare_agents.py`).

**Challenge 3: GPU/CPU Compatibility**
Automatic device detection with fallback (`src/utils/device.py`). Code works seamlessly on both GPU and CPU.

**Challenge 4: Reward Wrapper Integration**
Ensuring reward wrapper used consistently in training and evaluation. Solution: Centralized environment factory (`src/training/trainer.py` lines 188-200).

**Challenge 5: Experience Replay Memory Management**
Large replay buffers can cause memory issues. Solution: Configurable buffer size with efficient circular buffer implementation (`src/agents/dqn.py` lines 50-120).

---

## [CONCLUSION] (30 seconds)

"In summary, our project demonstrates how advanced RL components work together synergistically:
- Experience replay and target networks stabilize DQN learning
- Policy and value networks in A2C create a positive feedback loop
- Custom reward shaping benefits both algorithms
- Unified training framework enables fair comparison
- Comprehensive evaluation validates our approach

All code is well-documented, modular, and follows best practices. The project extends significantly beyond basic coursework with multi-network baselines, optimizer comparisons, extensive training, and rigorous evaluation.

Thank you for watching. The complete codebase is available in our repository."

---

## **TOTAL TIME: ~8 minutes**

---

## **Presentation Tips:**

1. **Code Navigation**: Use screen recording to show actual code files while narrating
2. **Visual Aids**: Show architecture diagrams, data flow, or component interaction diagrams
3. **Demo Segments**: Include brief code execution demos (e.g., running training, showing logs)
4. **Emphasis**: Highlight line numbers when referencing specific code sections
5. **Pacing**: Don't rush through code references - give viewers time to see the code
6. **Transitions**: Use clear section markers to help graders navigate

## **Key Files to Show:**

1. `src/agents/dqn.py` - Experience replay and target network implementation
2. `src/agents/a2c.py` - Actor-critic architecture
3. `src/training/trainer.py` - Unified training framework
4. `src/evaluation/compare_agents.py` - Evaluation framework
5. `src/environments/reward_wrapper.py` - Custom reward function
6. `results/test_results.csv` - Quantitative results
7. `docs/EVALUATION.md` - Comprehensive evaluation report

