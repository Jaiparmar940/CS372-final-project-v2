# Technical Walkthrough Script: Autonomous Rocket Landing
## 5-10 Minute Technical Video for ML Engineers and Graders

**Target Audience:** ML engineers, course graders, technical reviewers  
**Tone:** Technical, detailed, demonstrating code understanding  
**Timing:** ~6 minutes total

---

## [INTRODUCTION] (20 seconds)

"Hi, I'm [Name], and this is [Name]. This walkthrough shows how our RL project implements advanced deep RL techniques. We'll cover key ML components, how they work together, and code locations for each feature."

**Code Reference:** Project structure in `src/` directory

---

## [SECTION 1: DQN - Experience Replay & Target Networks] (1.5 minutes)

"This addresses **checkbox46** - DQN with experience replay and target networks.

**Experience Replay** (`src/agents/dqn.py` lines 24-67):
The `ExperienceReplay` class stores transitions in a circular buffer. `__init__` at line 30, `sample` at line 53, enabling sample-efficient learning from past experiences.

**Target Network** (`src/agents/dqn.py` lines 132-136, 299-305):
Target network initialized at line 132, copied from main network at line 135. Periodic weight copying every `target_update_freq` steps (lines 299-305) provides stable Q-value targets, preventing the moving target problem.

**Synergistic Integration** (`src/agents/dqn.py` lines 251-305):
The `update` method shows how they work together: sample batch from replay buffer (line 262), compute Q-values with main network (line 272), compute targets with target network (line 276), update main network (lines 283-290), periodically update target (lines 299-305)."

**Code References:**
- Experience replay class: `src/agents/dqn.py` lines 24-67
- Target network init: `src/agents/dqn.py` lines 132-136
- Update method: `src/agents/dqn.py` lines 251-305
- Target updates: `src/agents/dqn.py` lines 299-305

---

## [SECTION 2: A2C - Actor-Critic Architecture] (1.5 minutes)

"This addresses **checkbox48** (policy gradient) and **checkbox49** (actor-critic).

**Separate Networks** (`src/agents/a2c.py` lines 69-70):
Two networks: policy network (`src/networks/policy_network.py`) outputs action probabilities, value network (`src/networks/value_network.py`) estimates state values.

**Advantage Computation** (`src/agents/a2c.py` lines 214-247):
`compute_advantages` method computes n-step returns (lines 229-241) and advantages: `advantage = G - V(state)` where G is n-step return (line 244).

**Synergistic Updates** (`src/agents/a2c.py` lines 249-335):
`update` method: compute advantages (line 265), policy network updates using policy gradient with advantages (lines 324-328), value network updates to better estimate returns (lines 331-335). Improved value estimates lead to better advantages, which improve policy updates - creating a positive feedback loop."

**Code References:**
- Network init: `src/agents/a2c.py` lines 69-70
- Advantage computation: `src/agents/a2c.py` lines 214-247
- Update method: `src/agents/a2c.py` lines 249-335
- Policy updates: `src/agents/a2c.py` lines 324-328
- Value updates: `src/agents/a2c.py` lines 331-335

---

## [SECTION 3: Custom Reward Shaping] (1 minute)

"This addresses **checkbox47** - custom reward function.

**Reward Wrapper** (`src/environments/reward_wrapper.py` lines 75-144):
`RocketRewardWrapper.step` method transforms sparse LunarLander rewards into dense shaped rewards: fuel penalty (lines 101-104), smoothness penalty (lines 107-112), landing bonus (lines 123-125), crash penalty (lines 130-132).

**Integration** (`src/training/trainer.py` lines 478-501):
`create_env_factory` function wraps environment with `RocketRewardWrapper` at line 501. Shaped rewards provide learning signal at every step, not just episode termination.

**Configurability** (`src/utils/config.py` lines 38-45):
Reward coefficients parameterized via `RewardConfig` class (landing_bonus, fuel_penalty, crash_penalty, smoothness_penalty) for hyperparameter tuning."

**Code References:**
- Reward wrapper step: `src/environments/reward_wrapper.py` lines 75-144
- Integration: `src/training/trainer.py` lines 478-501 (create_env_factory)
- Config: `src/utils/config.py` lines 38-45 (RewardConfig class)

---

## [SECTION 4: Training Infrastructure] (1 minute)

"This addresses **checkbox0** (train/val/test split) and **checkbox4** (regularization).

**Seed-Based Splits** (`src/training/trainer.py` lines 248-282):
Training on random training seeds (line 250), validation on separate validation seeds (lines 279-281). Training seeds 42-52, validation seeds 100-109, test seeds 200-209. Environment determinism enables fair comparison.

**Early Stopping** (`src/training/trainer.py` lines 232-235, 310-324):
EarlyStopping class initialization (lines 232-235). Monitors validation return (lines 310-324), stops if no improvement for `patience` episodes, saves best model.

**Regularization**:
- L2 Weight Decay: `src/utils/config.py` lines 100-104 (weight_decay parameter in OptimizerConfig)
- Dropout: `src/networks/dqn_network.py` lines 111-112, `value_network.py` lines 108-109"

**Code References:**
- Seed splits: `src/training/trainer.py` lines 248-282
- Early stopping: `src/training/trainer.py` lines 232-235, 310-324
- L2 decay: `src/utils/config.py` lines 100-104
- Dropout: `src/networks/dqn_network.py` lines 111-112

---

## [SECTION 5: Optimizer Comparison & Hyperparameter Tuning] (1 minute)

"This addresses **checkbox16** (optimizer comparison) and **checkbox5** (hyperparameter tuning).

**Multi-Optimizer Training** (`src/scripts/run_all_experiments.py` lines 60-99):
DQN with Adam/RMSprop, A2C with Adam/SGD. Identical architectures, only optimizer differs.

**Unified Evaluation** (`src/evaluation/compare_agents.py` lines 17-83):
`compare_all_agents` evaluates all models on same seeds, ensuring fair comparison. Results in `results/test_results.csv`.

**Grid Search** (`src/hyperparameter_tuning/sweep.py` lines 18-120):
Systematic hyperparameter sweep with validation-based selection. Uses same `train_agent` function for consistency."

**Code References:**
- Training: `src/scripts/run_all_experiments.py` lines 60-99
- Comparison: `src/evaluation/compare_agents.py` lines 17-83
- Grid search: `src/hyperparameter_tuning/sweep.py` lines 18-120

---

## [SECTION 6: Error Analysis & Ablation Studies] (30 seconds)

"This addresses **checkbox63** (error analysis) and **checkbox67** (ablation study).

**Error Analysis** (`src/scripts/error_analysis.py`):
Identifies failure cases, extracts characteristics (velocity, altitude, position), generates visualizations. Results in `error_analysis/` directory.

**Ablation Study** (`src/scripts/ablation_study.py`):
Systematically removes components (replay buffer, target network, custom reward) to understand impact. Results in `ablation_results/ablation_study_results.csv`."

**Code References:**
- Error analysis: `src/scripts/error_analysis.py`
- Ablation: `src/scripts/ablation_study.py`
- Results: `ablation_results/ablation_study_results.csv`

---

## [CONCLUSION] (20 seconds)

"In summary, our project demonstrates synergistic RL components: experience replay and target networks stabilize DQN, policy and value networks create feedback loops in A2C, custom rewards benefit both, and unified evaluation enables fair comparison. All code is modular and well-documented."

**Code Reference:** Complete codebase in `src/` directory

---

---

## **TOTAL TIME: ~6 minutes**

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

