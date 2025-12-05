# Addressing Function Approximation Error in Actor-Critic Methods

PyTorch implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3). Forked from the [paper](https://arxiv.org/abs/1802.09477) repository [here](https://github.com/sfujim/TD3).

Method is tested on [MuJoCo](http://www.mujoco.org/) continuous control tasks in [OpenAI gym](https://github.com/openai/gym).
Networks are trained using [PyTorch 2.9](https://github.com/pytorch/pytorch) and Python 3.9+.

## Reimplementation Results
We rerun their MuJoCo experiments with the exact same hyperparameters and get the following results.

### Performance (Time and Final Reward)

Based on [results/final_rewards_table.tex](results/final_rewards_table.tex) and [results/final_rewards.csv](results/final_rewards.csv):

| Environment               | TD3 Reward (±)  | TD3 Time (min ±) | DDPG Reward (±) | DDPG Time (min ±) |
| ------------------------- | --------------- | ---------------- | --------------- | ----------------- |
| Ant-v5                    | 3944.0 ± 785.0  | 69.0 ± 0.5       | 1308.5 ± 196.3  | 76.8 ± 0.7        |
| HalfCheetah-v5            | 11057.6 ± 91.8  | 55.4 ± 0.2       | 9273.9 ± 517.9  | 59.2 ± 0.2        |
| Hopper-v5                 | 3304.1 ± 88.6   | 56.2 ± 0.6       | 2102.8 ± 442.5  | 59.8 ± 0.2        |
| InvertedDoublePendulum-v5 | 7695.4 ± 1623.5 | 56.2 ± 0.2       | 9275.6 ± 23.6   | 59.8 ± 0.2        |
| InvertedPendulum-v5       | 1000.0 ± 0.0    | 53.4 ± 0.2       | 1000.0 ± 0.0    | 57.4 ± 0.2        |
| Reacher-v5                | -3.1 ± 0.7      | 48.6 ± 0.2       | -5.9 ± 1.6      | 53.2 ± 0.2        |
| Walker2d-v5               | 4253.9 ± 130.7  | 57.4 ± 0.4       | 1988.5 ± 408.2  | 58.8 ± 0.2        |


### Visualizations
The [notebook](./algorithm_comparison.ipynb) provides an interactive way to run and compare all algorithms. Here we provide comparison plots for all environments:

Here we provide comparison plots for all environments:

<p align="center">
  <img src="results/Ant-v5_comparison.png" width="30%" />
  <img src="results/Hopper-v5_comparison.png" width="30%" />
  <img src="results/InvertedPendulum-v5_comparison.png" width="30%" />
</p>

<p align="center">
  <img src="results/InvertedDoublePendulum-v5_comparison.png" width="30%" />
  <img src="results/Reacher-v5_comparison.png" width="30%" />
  <img src="results/Walker2d-v5_comparison.png" width="30%" />
</p>

<!-- ![Ant-v5 Comparison](results/Ant-v5_comparison.png)

![HalfCheetah-v5 Comparison](results/HalfCheetah-v5_comparison.png)

![Hopper-v5 Comparison](results/Hopper-v5_comparison.png)

![InvertedPendulum-v5 Comparison](results/InvertedPendulum-v5_comparison.png)

![InvertedDoublePendulum-v5 Comparison](results/InvertedDoublePendulum-v5_comparison.png)

![Reacher-v5 Comparison](results/Reacher-v5_comparison.png)

![Walker2d-v5 Comparison](results/Walker2d-v5_comparison.png) -->


## Project Structure

- `DDPG.py` - Original DDPG implementation (400-300 architecture)
- `OurDDPG.py` - Re-tuned DDPG implementation (256-256 architecture)
- `TD3.py` - TD3 implementation with twin critics
- `QR_TD3.py` - Quantile Regression TD3 implementation
- `Gaussian_TD3.py` - Gaussian TD3 implementation
- `utils.py` - Replay buffer implementation
- `main.py` - Training script for running experiments
- `plot_results.py`, `plot_results_alpha.py`, `plot_results_k.py` - Script for generating comparison plots
- `compare_td3_ddpg.py` - Multi-seed statistical comparison script
- `algorithm_comparison.ipynb` - **Jupyter notebook for interactive experimentation** (imports from Python files)
- `log_files` folder - Folder with log files from different experiments

## Installation

### Prerequisites
- Python 3.9 or higher
- CUDA-capable GPU (recommended)

### Setup

1. **Clone the repository** (if applicable):
```bash
cd DRL_Group2_DDPG_TD3
```

2. **Create a virtual environment** (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Dependencies

The project requires the following packages (see `requirements.txt`):
- `gymnasium` - OpenAI Gym environments (with MuJoCo support via `gymnasium[mujoco]`)
- `matplotlib` - Plotting and visualization
- `numpy` - Numerical computations
- `torch` - PyTorch (CUDA support recommended for GPU training)
- `tqdm` - Progress bars
- `tensorboard` - TensorBoard logging and visualization
- `jupyter` - Jupyter notebook interface
- `ipykernel` - Jupyter kernel for Python

**Note**: For macOS, install PyTorch CPU version. For Linux/Windows with CUDA, install CUDA-enabled PyTorch.

## Usage

### Running the Jupyter Notebook

The **algorithm_comparison.ipynb** notebook provides an interactive way to run and compare all algorithms:

1. **Open the notebook**:
   - Open `algorithm_comparison.ipynb` in Jupyter/VS Code
   - Select a Python kernel with the required packages installed

2. **Configure and run**:
   - Follow the setup instructions in the first cell of the notebook
   - Set `ENV_NAME`, `SEED`, and `POLICY` in the configuration cell
   - Run all cells sequentially
   - Change `POLICY` to compare different algorithms

The notebook includes setup instructions in the first cell.

### Running Experiments via Command Line

#### Minimal Example

Run a single experiment with default hyperparameters:

```bash
python main.py --policy TD3 --env Reacher-v5 --seed 0
```

#### QRTD3 and GaussianTD3 Additional Parameters

For QRTD3 and GaussianTD3, you can also specify:
- `--alpha` (default: 0.5) - Risk parameter for QRTD3/GaussianTD3
- `--K` (default: 5) - Number of quantiles for QRTD3

```bash
python main.py \
  --policy QRTD3 \
  --env Reacher-v5 \
  --seed 0 \
  --alpha 0.5 \
  --K 5
```

### Running on PACE (To recreate all Experiments)

To run all experiments on PACE:

```bash
sbatch scripts/queue_all.sh
```

This will submit jobs for:
- `queue_relevant.sh` - Ant, Walker, InvertedPendulum (TD3 and DDPG)
- `queue_other.sh` - Other environments (TD3 and DDPG)
- `run_gaussiantd3.sh` - Gaussain TD3 experiments
- `run_qrtd3.sh` - Quantile TD3 experiments

To run specific algorithm variants:

```bash
sbatch scripts/run_qrtd3.sh      # QRTD3 experiments
sbatch scripts/run_gaussiantd3.sh  # GaussianTD3 experiments
```

#### Supported Policies

The `--policy` argument accepts one of the following:
- `TD3` - Twin Delayed Deep Deterministic Policy Gradient (default)
- `DDPG` - Deep Deterministic Policy Gradient (original 400-300 architecture)
- `OurDDPG` - Re-tuned DDPG (256-256 architecture)
- `QRTD3` - Quantile Regression TD3
- `GaussianTD3` - Gaussian TD3

**Note**: If `--policy` is not specified, `TD3` is used by default.

## Algorithms Comparison

### Algorithm Variants

| Feature | DDPG (DDPG.py) | OurDDPG (OurDDPG.py) | TD3 (TD3.py) | QRTD3 (QR_TD3.py) | GaussianTD3 (Gaussian_TD3.py) |
|---------|----------------|---------------------|--------------|-------------------|------------------------------|
| **Architecture** | 400-300 | 256-256 | 256-256 (twin) | 256-256 (twin) | 256-256 (twin) |
| **Learning Rate** | 1e-4 (actor) | 3e-4 | 3e-4 | 3e-4 | 3e-4 |
| **Tau** | 0.001 | 0.005 | 0.005 | 0.005 | 0.005 |
| **Batch Size** | 64 | 256 | 256 | 256 | 256 |
| **Critic Networks** | Single | Single | Twin (Q1, Q2) | Twin (Quantile) | Twin (Gaussian) |
| **Policy Updates** | Every step | Every step | Delayed (every 2 steps) | Delayed (every 2 steps) | Delayed (every 2 steps) |
| **Target Smoothing** | No | No | Yes | Yes | Yes |
| **Special Features** | - | - | Clipped Double Q-Learning | Quantile Regression | Gaussian Distribution |

**Note**: All implementations use the exact code from their respective files for accurate replication of results.


## Project Modifications

### Changes from Original TD3 Paper Repository

1. **Environment Updates**: Updated from `gym` to `gymnasium` (v1.2.1)
2. **Environment Names**: Changed from `-v2` to `-v5` (e.g., `HalfCheetah-v2` → `HalfCheetah-v5`)
3. **PyTorch Version**: Updated to PyTorch 2.9 with CUDA 12.6 support
4. **Algorithm Extensions**: Added QRTD3 (Quantile Regression TD3) and GaussianTD3 variants
5. **Notebook Added**: New Jupyter notebook (`algorithm_comparison.ipynb`) that imports from Python files for easy experimentation
6. **Visualization**: Uses Matplotlib for direct comparisons and plotting
7. **Comparison Scripts**: Added `compare_td3_ddpg.py` for multi-seed statistical analysis and `plot_results.py` for generating comparison plots


## Citation

```bibtex
@inproceedings{fujimoto2018addressing,
  title={Addressing Function Approximation Error in Actor-Critic Methods},
  author={Fujimoto, Scott and Hoof, Herke and Meger, David},
  booktitle={International Conference on Machine Learning},
  pages={1582--1591},
  year={2018}
}
```

## References

- [TD3 Paper](https://arxiv.org/abs/1802.09477)
- [DDPG Paper](https://arxiv.org/abs/1509.02971)
- [TD3 Official Implementation](https://github.com/sfujim/TD3)
- [DQN Paper](https://arxiv.org/pdf/1710.10044)
- [OpenAI Baselines](https://github.com/openai/baselines) - For DDPG, PPO, TRPO, ACKTR comparisons
- [Learned Agent Video](https://youtu.be/x33Vw-6vzso)

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in hyperparameters:
```python
batch_size = 128  # Instead of 256
```

### Jupyter Kernel Dies
Reduce max_timesteps or eval_freq to lower memory usage:
```python
max_timesteps = int(2e5)  # Instead of 5e5
```

## License

This project is based on the original TD3 implementation. Please cite the paper if you use this code.
