# Contributing to Physics-Informed Quantum Reservoir Transformer

Thank you for your interest in contributing! This is an active research project and we welcome contributions of all kinds.

---

## 🤝 How to Contribute

### Reporting Bugs
1. Search existing [Issues](https://github.com/Sangeeth0301/Physics-Informed-Quantum-Reservoir-Transformer-for-Bearing-SHM/issues) first
2. If not found, open a new issue with:
   - A clear, descriptive title
   - Steps to reproduce
   - Expected vs. actual behavior
   - Your environment (OS, Python version, GPU/CPU)
   - Relevant error traceback

### Suggesting Enhancements
- Open a GitHub Issue with the `enhancement` label
- Describe the motivation and expected benefit clearly

### Submitting Code Changes
1. **Fork** the repository
2. **Create a branch**: `git checkout -b feature/your-feature-name`
3. **Make changes** following the code style guidelines below
4. **Test** your changes: `python scripts/run_all_reproduction.py`
5. **Commit** with a clear message: `git commit -m "feat: add IMS dataset loader"`
6. **Push**: `git push origin feature/your-feature-name`
7. **Open a Pull Request** against `main`

---

## 🎯 Areas Where Help Is Needed

| Area | Description | Difficulty |
|---|---|:---:|
| **IMS Dataset** | Implement `scripts/09_load_ims_and_run_pipeline.py` fully | Medium |
| **XJTU-SY Dataset** | Complete `scripts/14_xjtu_generalization.py` | Medium |
| **Hardware QPU** | Swap PennyLane simulator for real QPU backend | Hard |
| **Benchmarking** | Add more classical baselines (LSTM, TCN, etc.) | Easy |
| **Documentation** | Improve docstrings in `src/` modules | Easy |
| **Tests** | Add pytest unit tests for `src/quantum/` | Medium |

---

## 🧹 Code Style Guidelines

- **Python**: Follow [PEP 8](https://pep8.org/)
- **Docstrings**: Use Google-style docstrings
- **Type hints**: Use type hints for all public functions
- **Commits**: Use [Conventional Commits](https://www.conventionalcommits.org/) format
  - `feat:` for new features
  - `fix:` for bug fixes
  - `docs:` for documentation changes
  - `refactor:` for code refactoring
  - `test:` for test additions

---

## 📋 Development Setup

```bash
git clone https://github.com/Sangeeth0301/Physics-Informed-Quantum-Reservoir-Transformer-for-Bearing-SHM.git
cd Physics-Informed-Quantum-Reservoir-Transformer-for-Bearing-SHM
python -m venv .venv
.\.venv\Scripts\activate  # Windows
pip install -r requirements.txt
pip install pylint pytest  # Dev tools
```

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).

---

*Thank you for helping advance quantum-classical machine learning for predictive maintenance!* ⚛️
