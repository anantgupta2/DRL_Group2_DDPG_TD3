# Addressing Function Approximation Error in Actor-Critic Methods

PyTorch implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3). If you use our code or data please cite the [paper](https://arxiv.org/abs/1802.09477).

Method is tested on [MuJoCo](http://www.mujoco.org/) continuous control tasks in [OpenAI gym](https://github.com/openai/gym).
Networks are trained using [PyTorch 2.9](https://github.com/pytorch/pytorch) and Python 3.9+.

## Project Structure

- `DDPG.py` - Original DDPG implementation (400-300 architecture)
- `OurDDPG.py` - Re-tuned DDPG implementation (256-256 architecture)
- `TD3.py` - TD3 implementation with twin critics
- `utils.py` - Replay buffer implementation
- `main.py` - Training script for running experiments
- `DDPG+TD3_notebook.ipynb` - **Self-contained Jupyter notebook for experimentation**

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
- `gymnasium==1.2.1` - OpenAI Gym environments
- `matplotlib==3.10.7` - Plotting and visualization
- `numpy==2.3.4` - Numerical computations
- `torch==2.9.0+cu126` - PyTorch with CUDA support
- `tqdm==4.67.1` - Progress bars
- `tensorboard` - TensorBoard logging and visualization
- `jupyter` - Jupyter notebook interface
- `ipykernel` - Jupyter kernel for Python

## Usage

### Running the Jupyter Notebook (Recommended for Quick Start)

The **DDPG+TD3_notebook.ipynb** provides a self-contained, interactive way to run and compare both algorithms:

1. **Launch Jupyter**:
```bash
jupyter notebook
```

2. **Open the notebook**:
   - Navigate to `DDPG+TD3_notebook.ipynb` in your browser

3. **Run all cells**:
   - Click "Cell" → "Run All" or run cells individually

4. **View TensorBoard** (embedded in notebook):
   - TensorBoard will appear inline showing real-time training progress
   - Compare DDPG vs TD3 metrics side-by-side

**Notebook Features:**
- ✅ Completely self-contained (no external file imports needed)
- ✅ Mathematical theory and explanations
- ✅ Full implementations of DDPG and TD3 from source files
- ✅ Real-time TensorBoard visualization
- ✅ Automated training and evaluation
- ✅ Comparative analysis and plots
- ✅ Optimized for Reacher-v5 (fastest environment)

### Running Python Scripts

For full-scale experiments with multiple environments:

1. **Run a single experiment**:
```bash
python main.py --env HalfCheetah-v2
```

2. **Run all experiments** (reproduces paper results):
```bash
./run_experiments.sh
```

### Hyperparameters

Modify hyperparameters via command-line arguments to `main.py`:

```bash
python main.py \
  --env Reacher-v5 \
  --seed 0 \
  --max_timesteps 200000 \
  --start_timesteps 10000 \
  --eval_freq 5000 \
  --batch_size 256 \
  --discount 0.99 \
  --tau 0.005 \
  --policy_noise 0.2 \
  --noise_clip 0.5 \
  --policy_freq 2 \
  --expl_noise 0.1
```

## Algorithms Comparison

### DDPG vs TD3

| Feature | DDPG (DDPG.py) | OurDDPG (OurDDPG.py) | TD3 (TD3.py) |
|---------|----------------|---------------------|--------------|
| **Architecture** | 400-300 | 256-256 | 256-256 (twin) |
| **Learning Rate** | 1e-4 (actor) | 3e-4 | 3e-4 |
| **Tau** | 0.001 | 0.005 | 0.005 |
| **Batch Size** | 64 | 256 | 256 |
| **Critic Networks** | Single | Single | Twin (Q1, Q2) |
| **Policy Updates** | Every step | Every step | Delayed (every 2 steps) |
| **Target Smoothing** | No | No | Yes |

**Note**: The notebook uses the exact implementations from DDPG.py and TD3.py for accurate replication of results.

## Results

### Environment Performance (1M timesteps)

Based on [results/training_times.csv](results/training_times.csv):

| Environment | Avg Training Time |
|------------|-------------------|
| **Reacher-v5** | **~51 min** (fastest) |
| InvertedPendulum-v5 | ~55 min |
| HalfCheetah-v5 | ~57 min |
| InvertedDoublePendulum-v5 | ~58 min |
| Hopper-v5 | ~58 min |
| Walker2d-v5 | ~58 min |
| Ant-v5 | ~73 min |
| Humanoid-v5 | ~104 min |

Learning curves from the paper can be found under `/learning_curves`. Each file contains 201 evaluations (every 5000 timesteps over 1M steps).

Results are saved as NumPy arrays where each evaluation is the average reward over 10 episodes with no exploration.

## TensorBoard Visualization

### Notebook (Inline)
TensorBoard is embedded directly in the notebook:
- Automatically launches when you run the notebook
- Updates in real-time during training
- No separate terminal needed

### Command Line
For standalone TensorBoard viewing:

```bash
tensorboard --logdir=runs
```

Then open `http://localhost:6006` in your browser.

**Available Metrics:**
- Training/Episode_Reward - Reward per episode during training
- Training/Episode_Length - Episode lengths
- Evaluation/Average_Reward - Evaluation performance over time
- Loss/Critic - Critic network loss
- Loss/Actor - Actor network loss

## Project Modifications

### Changes from Original TD3 Paper Repository

1. **Environment Updates**: Updated from `gym` to `gymnasium` (v1.2.1)
2. **Environment Names**: Changed from `-v2` to `-v5` (e.g., `HalfCheetah-v2` → `HalfCheetah-v5`)
3. **PyTorch Version**: Updated to PyTorch 2.9 with CUDA 12.6 support
4. **Notebook Added**: New self-contained Jupyter notebook for easy experimentation
5. **TensorBoard**: Added comprehensive logging and visualization
6. **Timestep Reduction**: Notebook uses 200K steps for Reacher-v5 (converges faster)

### Code is Representative But Not Identical to Paper

Minor adjustments have been made to improve performance and compatibility. Learning curves in the paper reflect the original implementation.

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

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in hyperparameters:
```python
batch_size = 128  # Instead of 256
```

### Jupyter Kernel Dies
Reduce max_timesteps or eval_freq to lower memory usage:
```python
max_timesteps = int(1e5)  # Instead of 2e5
```

### TensorBoard Not Updating
Refresh the page or restart the TensorBoard cell.

## License

This project is based on the original TD3 implementation. Please cite the paper if you use this code.
