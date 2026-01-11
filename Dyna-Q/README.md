# RL Racetrack-v0 🏎️

Reinforcement Learning agents for the `racetrack-v0` environment from [highway-env](https://github.com/Farama-Foundation/HighwayEnv).

## 🎯 Project Overview

This project implements and compares multiple model-based RL algorithms:

| Algorithm | File | Description |
|-----------|------|-------------|
| **Basic Dyna-Q** | `dyna_q_basic_trainer.py` | Standard Dyna-Q with tabular Q-learning |
| **Dyna-Q+** | `dyna_q_plus_trainer.py` | Exploration bonus for unvisited states |
| **Prioritized Sweeping** | `prioritized_sweeping_trainer.py` | TD-error priority queue planning |
| **Traffic Aware** | `dyna_q_traffic_aware_trainer.py` | Reward shaping for traffic avoidance |

## 🚀 Quick Start

### Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install gymnasium highway-env numpy matplotlib
```

### Training a Model

```bash
# Train any of the agents
python3 dyna_q_basic_trainer.py
python3 dyna_q_plus_trainer.py
python3 prioritized_sweeping_trainer.py
python3 dyna_q_traffic_aware_trainer.py
```

### Running a Demo

```bash
# Watch a trained agent in action
python3 run_loop_demo.py
```

### Evaluate & Compare Models

```bash
# Generate comparison graphs
python3 evaluate_models.py
```

## 📊 Results

After training, run `evaluate_models.py` to generate performance comparisons:
- Episode reward curves (smoothed)
- Box plot distributions
- Summary statistics (max, mean, std)

## 🧠 Algorithm Details

### Dyna-Q
Standard model-based RL: learns from real experience + simulated planning steps.

### Dyna-Q+
Adds exploration bonus: `κ × √(time_since_visit)` to encourage visiting "forgotten" states.

### Prioritized Sweeping
Uses TD-error priority queue - updates states with largest errors first and propagates changes backward.

### Traffic Aware
Adds distance-based penalties for approaching other vehicles on the track.

## 📁 File Structure

```
├── dyna_q_basic_trainer.py      # Basic Dyna-Q training
├── dyna_q_plus_trainer.py       # Dyna-Q+ with exploration bonus
├── prioritized_sweeping_trainer.py  # Priority-based planning
├── dyna_q_traffic_aware_trainer.py  # Traffic avoidance
├── run_loop_demo.py             # Visual demo runner
├── evaluate_models.py           # Model comparison tool
└── README.md
```

## 📝 License

MIT License
