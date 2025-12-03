# Contributing to OpenNeuralEngine

Thank you for your interest in contributing! This guide will help you get started.

## 📋 Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Documentation](#documentation)
- [Submitting Changes](#submitting-changes)

---

## Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what's best for the community
- Show empathy towards other contributors

---

## Getting Started

### 1. Fork and Clone
```bash
git clone https://github.com/YOUR_USERNAME/OpenNeuralEngine.git
cd OpenNeuralEngine
```

### 2. Set Up Development Environment
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Install in development mode
pip install -e ".[dev]"
```

### 3. Run Tests
```bash
# Run full test suite
pytest tests/ -v

# Run specific test file
pytest tests/test_layers.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

---

## Development Workflow

### Branch Naming
- `feature/your-feature-name` - New features
- `fix/bug-description` - Bug fixes
- `docs/update-description` - Documentation updates
- `refactor/component-name` - Code refactoring

### Commit Messages
Follow conventional commits format:
```
type(scope): brief description

Longer description if needed

- Bullet points for details
- Breaking changes should be noted
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style/formatting
- `refactor`: Code refactoring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(attention): add Flash Attention 2.0 support

- Implemented FA2 kernel for faster attention
- Added backward compatibility with FA1
- 2x speedup on A100 GPUs

fix(quantization): resolve INT8 calibration overflow

The calibration range calculation was incorrect for
negative activations, causing overflow. Fixed by using
symmetric quantization range.
```

---

## Code Standards

### Python Style
We use **Ruff** for linting and formatting:

```bash
# Format code
ruff format .

# Lint code
ruff check .

# Fix auto-fixable issues
ruff check --fix .
```

### Type Hints
Always use type hints for function signatures:

```python
from typing import Optional, List, Dict, Any
import torch

def process_batch(
    inputs: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    config: Dict[str, Any] = None,
) -> List[torch.Tensor]:
    """Process a batch of inputs.
    
    Args:
        inputs: Input tensor of shape (batch, seq_len).
        labels: Optional labels for supervised training.
        config: Configuration dictionary.
    
    Returns:
        List of output tensors.
    """
    pass
```

### Docstrings
Use Google-style docstrings:

```python
def my_function(param1: int, param2: str) -> bool:
    """Brief one-line description.
    
    Longer description with more details about what the function
    does and how it works.
    
    Args:
        param1: Description of param1.
        param2: Description of param2.
    
    Returns:
        Description of return value.
    
    Raises:
        ValueError: When param1 is negative.
    
    Example:
        >>> my_function(42, "hello")
        True
    """
```

---

## Testing

### Writing Tests
Place tests in `tests/` directory:

```python
import pytest
import torch
from src.Core_Models.layers import SwiGLU

class TestSwiGLU:
    """Test suite for SwiGLU activation."""
    
    def test_forward_shape(self):
        """Test output shape matches input."""
        layer = SwiGLU(hidden_dim=256)
        x = torch.randn(2, 10, 256)
        output = layer(x)
        
        assert output.shape == x.shape
    
    def test_gradient_flow(self):
        """Test gradients flow properly."""
        layer = SwiGLU(hidden_dim=256)
        x = torch.randn(2, 10, 256, requires_grad=True)
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
```

### Test Coverage
Aim for >90% test coverage. Check with:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Documentation

### Code Documentation
- **Every module** should have a module-level docstring
- **Every class** should have a class docstring
- **Every public function** should have a docstring
- **Complex logic** should have inline comments

### User Documentation
When adding features, update:
- `README.md` - If it affects quick start or main features
- `docs/CAPABILITIES.md` - For new capabilities
- `docs/FAQ.md` - For common questions
- Relevant guides in `docs/`

### Script Documentation
For new scripts in `scripts/`:
- Add comprehensive `--help` text
- Include examples in docstring
- Update `scripts/README.md`

---

## Submitting Changes

### Before Submitting
1. **Run tests**: `pytest tests/ -v`
2. **Check linting**: `ruff check .`
3. **Format code**: `ruff format .`
4. **Update docs**: If you changed user-facing features
5. **Test manually**: Run your changes end-to-end

### Pull Request Process

1. **Update CHANGELOG** (if applicable)
2. **Write clear PR description**:
   ```markdown
   ## What
   Brief description of changes
   
   ## Why
   Motivation and context
   
   ## How
   Technical implementation details
   
   ## Testing
   How you tested the changes
   
   ## Screenshots
   (if applicable)
   ```

3. **Link related issues**: Use `Fixes #123` or `Relates to #456`
4. **Request review**: Tag relevant maintainers
5. **Address feedback**: Respond to review comments
6. **Keep PR focused**: One feature/fix per PR

### Review Criteria
- Code follows style guidelines
- Tests pass and coverage is maintained
- Documentation is updated
- No merge conflicts
- Commits are clean and well-described

---

## Project Structure Guidelines

### Where to Add New Code

**Core Models** (`src/Core_Models/`):
- New layer types → `layers.py`
- Attention mechanisms → `efficient_attention.py`
- Training utilities → `advanced_training.py`
- New model architectures → Create new file, update `builders.py`

**Utilities** (`utils/`):
- Data processing → `tokenization.py`
- Model I/O → `model_io.py`, `model_loading.py`
- Device management → `device_manager.py`
- Architecture detection → `architecture_inspector.py`

**Scripts** (`scripts/`):
- CLI tools → New file in `scripts/`
- Update `scripts/README.md`

**Tests** (`tests/`):
- Mirror structure of `src/`
- Name files `test_*.py`

**Documentation** (`docs/`):
- User guides → `docs/`
- API reference → Inline docstrings
- Update `docs/INDEX.md`

---

## Questions?

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Discussions**: Use GitHub Discussions for questions
- **Code Review**: Tag maintainers for review

Thank you for contributing! 🎉
