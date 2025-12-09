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

**Experience Replay** (`src/agents/dqn.py` lines 50-120):
The `ExperienceReplay` class stores transitions in a circular buffer, enabling sample-efficient learning from past experiences.

**Target Network** (`src/agents/dqn.py` lines 350-380):
Periodic weight copying every `target_update_freq` steps provides stable Q-value targets, preventing the moving target problem.

**Synergistic Integration** (`src/agents/dqn.py` lines 400-430):
The `update` method shows how they work together: sample batch from replay buffer (line 405), compute Q-values with main network (line 410), compute targets with target network (line 415), update main network (line 420), periodically update target (line 425)."

**Code References:**
- Experience replay: `src/agents/dqn.py` lines 50-120
- Target updates: `src/agents/dqn.py` lines 350-380  
- Update method: `src/agents/dqn.py` lines 400-430

---

## [SECTION 2: A2C - Actor-Critic Architecture] (1.5 minutes)

"This addresses **checkbox48** (policy gradient) and **checkbox49** (actor-critic).

**Separate Networks** (`src/agents/a2c.py` lines 60-80):
Two networks: policy network (`src/networks/policy_network.py`) outputs action probabilities, value network (`src/networks/value_network.py`) estimates state values.

**Advantage Computation** (`src/agents/a2c.py` lines 300-350):
Value network computes advantages using n-step returns: `advantage = G - V(state)` where G is n-step return.

**Synergistic Updates** (`src/agents/a2c.py` lines 380-470):
Policy network updates using policy gradient with advantages (lines 380-420), value network updates to better estimate returns (lines 420-470). Improved value estimates lead to better advantages, which improve policy updates - creating a positive feedback loop."

**Code References:**
- Network init: `src/agents/a2c.py` lines 60-80
- Advantage: `src/agents/a2c.py` lines 300-350
- Policy updates: `src/agents/a2c.py` lines 380-420
- Value updates: `src/agents/a2c.py` lines 420-470

---

## [SECTION 3: Custom Reward Shaping] (1 minute)

"This addresses **checkbox47** - custom reward function.

**Reward Wrapper** (`src/environments/reward_wrapper.py` lines 45-95):
`RocketRewardWrapper` transforms sparse LunarLander rewards into dense shaped rewards with landing bonus, fuel penalty, crash penalty, and smoothness bonus.

**Integration** (`src/training/trainer.py` lines 188-200):
Wrapper integrated at environment creation. Shaped rewards provide learning signal at every step, not just episode termination.

**Configurability** (`src/utils/config.py` lines 85-100):
Reward coefficients parameterized via `RewardConfig` for hyperparameter tuning."

**Code References:**
- Reward wrapper: `src/environments/reward_wrapper.py` lines 45-95
- Integration: `src/training/trainer.py` lines 188-200
- Config: `src/utils/config.py` lines 85-100

---

## [SECTION 4: Training Infrastructure] (1 minute)

"This addresses **checkbox0** (train/val/test split) and **checkbox4** (regularization).

**Seed-Based Splits** (`src/training/trainer.py` lines 248-280):
Training seeds 42-52, validation seeds 100-109, test seeds 200-209. Environment determinism enables fair comparison.

**Early Stopping** (`src/training/trainer.py` lines 230-235, 310-330):
Monitors validation return, stops if no improvement for `patience` episodes, saves best model.

**Regularization**:
- L2 Weight Decay: `src/utils/config.py` lines 50-60
- Dropout: `src/networks/dqn_network.py` line 45, `value_network.py` line 50"

**Code References:**
- Seed splits: `src/training/trainer.py` lines 248-280
- Early stopping: `src/training/trainer.py` lines 230-235, 310-330
- L2 decay: `src/utils/config.py` lines 50-60
- Dropout: `src/networks/dqn_network.py` line 45

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

