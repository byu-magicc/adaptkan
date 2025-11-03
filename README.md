# AdaptKAN
Code respository for "Automatic Grid Updates for Komolgorov Arnold Networks using Layer Histograms."

AdaptKAN provides implementations of Kolmogorov-Arnold Networks (KANs) with support for adaptable continual learning, available in JAX.

![Demo](assets/digits.gif)

## Features

- **Customizable Layers**: Control noise injection, grid refinements, and pruning strategies.
- **Continual Learning**: Adapt model weights online with minimal overhead.
- **Data Utilities**: Parsers and dataset builders for the benchmarks in the paper.
- **Training Scripts**: Convenient scripts for Feynman and Lyapunov tasks.
- **Examples**: Jupyter notebooks in `examples/` to get started quickly.

## Installation 🚀

The following steps will guide you through setting up the project using **uv**.

### Prerequisites
* Git
* [uv](https://github.com/astral-sh/uv) (which we will install into the environment)

---
### Setup Steps

1.  **Clone the Repository**
    ```bash
    git clone git@github.com:your-username/adaptkan.git
    cd adaptkan
    ```

2.  **Install `uv`**
    Install uv to easily setup the environment.
    ```bash
    wget -qO- https://astral.sh/uv/install.sh | sh
    ```

3.  **Sync Dependencies from Lock File**
    Use `uv` to install the exact versions of all external packages from the lock file. This guarantees a reproducible base environment.
    ```bash
    cd adaptkan
    uv sync --extra cuda # or cpu
    ```
4.  **Activate the Environment**
    This command activates the virtual environment for the project.
    ```bash
    source .venv/bin/activate
    ```

Your environment is now complete and ready to use!

## Quick Start

### JAX

The project uses Equinox and Jax. We keep track of two different things during training, ```model``` which is the AdaptKAN model and ```state``` which stores histogram information. See [stateful API reference](https://docs.kidger.site/equinox/api/nn/stateful/) for more details on how to work with ```state```.

```python
import equinox as eqx
from adaptkan.jax.model import AdaptKANJax

model, state = eqx.nn.make_with_state(AdaptKANJax)(width=[2, 5, 1])
```

## Project Structure

```
.
├── adaptkan/           # Source code
│   ├── common/         # Data processing utilities
│   ├── data/           # Dataset CSVs/raw files
│   └── jax/            # JAX implementation code
├── assets/             # Assets used for the github repo
├── scripts/            # Standalone training scripts
├── examples/           # Jupyter notebooks
├── results/            # Generated outputs
├── helloadaptkan.ipynb # Jupyter notebook with minimal working example
├── pyproject.toml      # Project setup config
├── README.md           # This file
└── LICENSE             # GPL-3.0 License
```

## Contributing

Contributions are welcome! Please:
1. Fork the repo.
2. Create a branch (`git checkout -b feature/my-feature`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push (`git push origin feature/my-feature`).
5. Open a pull request.

## License

This project is under the GPL-3.0 License. See [LICENSE](LICENSE) for details.

## Citation

Please use of this project in your code with: TODO