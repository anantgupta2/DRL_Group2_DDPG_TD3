# Addressing Function Approximation Error in Actor-Critic Methods

PyTorch implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3). Forked from the [paper](https://arxiv.org/abs/1802.09477) repository [here](https://github.com/sfujim/TD3).

Method is tested on [MuJoCo](http://www.mujoco.org/) continuous control tasks in [OpenAI gym](https://github.com/openai/gym).
Networks are trained using [PyTorch 2.9](https://github.com/pytorch/pytorch) and Python 3.9+.

## Reimplementation Results
We rerun their MuJoCo experiments with the exact same hyperparameters and get the following results.

### Visualizations
The [notebook](./algorithm_comparison.ipynb) provides an interactive way to run and compare all algorithms. Here we provide comparison plots for all environments:

![Ant-v5 Comparison](results/Ant-v5_comparison.png)

![HalfCheetah-v5 Comparison](results/HalfCheetah-v5_comparison.png)

![Hopper-v5 Comparison](results/Hopper-v5_comparison.png)

![InvertedPendulum-v5 Comparison](results/InvertedPendulum-v5_comparison.png)

![InvertedDoublePendulum-v5 Comparison](results/InvertedDoublePendulum-v5_comparison.png)

![Reacher-v5 Comparison](results/Reacher-v5_comparison.png)

![Walker2d-v5 Comparison](results/Walker2d-v5_comparison.png)


## Project Structure

- `DDPG.py` - Original DDPG implementation (400-300 architecture)
- `OurDDPG.py` - Re-tuned DDPG implementation (256-256 architecture)
- `TD3.py` - TD3 implementation with twin critics
- `QR_TD3.py` - Quantile Regression TD3 implementation
- `Gaussian_TD3.py` - Gaussian TD3 implementation
- `utils.py` - Replay buffer implementation
- `main.py` - Training script for running experiments
- `plot_results.py` - Script for generating comparison plots
- `compare_td3_ddpg.py` - Multi-seed statistical comparison script
- `algorithm_comparison.ipynb` - **Jupyter notebook for interactive experimentation** (imports from Python files)

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

#### Full Example with All Hyperparameters

```bash
python main.py \
  --policy TD3 \
  --env Reacher-v5 \
  --seed 0 \
  --max_timesteps 1000000 \
  --start_timesteps 25000 \
  --eval_freq 5000 \
  --batch_size 256 \
  --discount 0.99 \
  --tau 0.005 \
  --policy_noise 0.2 \
  --noise_clip 0.5 \
  --policy_freq 2 \
  --expl_noise 0.1
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

## Final Results

**Table 3: Final Average Return (Mean ± Standard Error over Five Seeds)**

| Environment               | DDPG              | TD3               | GaussianTD3       | QRTD3             |
| ------------------------- | ----------------- | ----------------- | ----------------- | ----------------- |
| Ant-v5                    | 1308.5 ± 175.6    | 3944.0 ± 702.1    | **4144.8 ± 743.8** | 3458.9 ± 821.7    |
| HalfCheetah-v5            | 9273.9 ± 463.2    | **11057.6 ± 82.1** | 10438.3 ± 637.2   | 9114.3 ± 331.7    |
| Hopper-v5                 | 2102.8 ± 395.8    | 3304.1 ± 79.3     | **3374.4 ± 11.3**  | 3260.6 ± 83.4     |
| InvertedDoublePendulum-v5 | 9275.6 ± 21.1     | 7695.4 ± 1452.1   | **9323.4 ± 1.5**   | 9316.6 ± 0.7      |
| InvertedPendulum-v5       | 1000.0 ± 0.0      | 1000.0 ± 0.0      | 813.0 ± 167.3     | **1000.0 ± 0.0**  |
| Reacher-v5                | -5.9 ± 1.5        | **-3.1 ± 0.6**     | -3.3 ± 0.6        | -3.3 ± 0.7        |
| Walker2d-v5               | 1988.5 ± 365.1    | 4253.9 ± 116.9    | **5204.1 ± 273.8** | 2390.7 ± 446.0    |

*Note: Bold values indicate the best performance for each environment.*

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
- [OpenAI Baselines](https://github.com/openai/baselines) - For DDPG, PPO, TRPO, ACKTR comparisons
- [Learned Agent Video](https://youtu.be/x33Vw-6vzso)

## License

This project is based on the original TD3 implementation. Please cite the paper if you use this code.
