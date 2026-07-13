# Contributing to OncoPredict AI

First of all, thank you for taking the time to contribute! This project is designed as an enterprise-grade MLOps system. Please follow these guidelines to ensure code quality and consistent architecture.

## Development Setup

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/breast-cancer-prediction.git
   cd breast-cancer-prediction
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies in Editable Mode**
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```

4. **Verify Pipeline Execution**
   ```bash
   python verify_pipeline_phase3.py
   python verify_backend_phase4.py
   ```

---

## Coding Standards

* **PEP 8 Compliance**: Follow the standard Python style guidelines. Use a formatter like `black` or `ruff` if possible.
* **Type Hints**: All functions and classes must include descriptive type hints for arguments and return types.
* **Docstrings**: Document classes and public functions using Google or Sphinx docstring formats.
* **Logging & Exceptions**: Use the centralized logging system (`src/logger`) and wrap computations in `CustomException` (`src/exception`) blocks to capture tracing diagnostics.

---

## Git Commit Guidelines

* Create semantic commit messages:
  * `feat:` for new features or ML pipeline components.
  * `fix:` for bug fixes.
  * `docs:` for documentation updates.
  * `test:` for adding or modifying tests.
  * `refactor:` for code restructurings that don't add features.
* Always develop on a separate branch (e.g. `feat/new-component`) and open a Pull Request against the `main` branch.

---

## Testing Contributions

* All modifications must be backed by tests under the `tests/` directory.
* Run the test suite before submitting:
  ```bash
  pytest
  ```
