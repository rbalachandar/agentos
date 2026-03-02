# Claude Agent Guidelines for AgentOS

This file contains guidelines and checklists for Claude (and other AI assistants) working on the AgentOS codebase.

## Pre-Commit Checklist

ALWAYS run these checks before committing changes:

```bash
# 1. Check Python syntax
python -m py_compile src/agentos/**/*.py

# 2. Type checking (mypy)
mypy src/agentos/

# 3. Linting (ruff)
ruff check src/agentos/
ruff format --check src/agentos/

# 4. Run tests
pytest tests/

# 5. Import check
PYTHONPATH=/Users/bramakri/dev/repos/agentos/src python -c "import agentos; print('OK')"
```

## Post-Change Review Checklist

After making ANY changes, review for:

### 1. Correctness
- [ ] Logic is correct and matches the paper/intent
- [ ] Edge cases are handled (empty inputs, None values, single items)
- [ ] Error messages are clear and actionable
- [ ] No hardcoded values that should be configurable

### 2. Completeness
- [ ] All imports are added to `__init__.py` if needed
- [ ] New classes/functions are exported from top-level `__init__.py`
- [ ] Documentation (docstrings) is complete
- [ ] Type hints are correct and complete
- [ ] Configuration parameters have validation

### 3. Errors
- [ ] No `ModuleNotFoundError` - check imports
- [ ] No `AttributeError` - check class attributes
- [ ] No `TypeError` - check type hints and unions (use `from __future__ import annotations`)
- [ ] No circular imports
- [ ] Exception handling is appropriate (not too broad, not missing)

### 4. Unused Code
- [ ] No unused imports
- [ ] No unused variables (check `_` prefix for intentionally unused)
- [ ] No commented-out code blocks
- [ ] No dead/unreachable code
- [ ] **CRITICAL: All new public methods are actually called/used somewhere**
  - For each new public method added, verify it has at least one caller
  - Use grep: `grep -r "method_name(" src/` to find callers
  - If no caller exists, either add one or make it private (`_method_name`)

### 5. Performance
- [ ] No O(n²) where O(n) is possible
- [ ] No unnecessary file I/O or network calls
- [ ] Appropriate use of caching/memoization
- [ ] Large data structures use appropriate types (numpy arrays vs lists)

### 6. Formatting
- [ ] Line length ≤ 100 characters (per ruff config)
- [ ] Proper indentation (4 spaces)
- [ ] Blank lines between classes/functions (2 blank lines before class, 1 before function)
- [ ] No trailing whitespace
- [ ] Consistent quote style (double quotes for docstrings, single for code)

### 7. Type Hints
- [ ] All public functions have type hints
- [ ] Use `X | None` for optional types (requires `from __future__ import annotations`)
- [ ] Use `list[X]` not `List[X]` (requires Python 3.9+)
- [ ] Use `dict[K, V]` not `Dict[K, V]`
- [ ] Forward references use `TYPE_CHECKING` pattern

### 8. Documentation
- [ ] All modules have docstrings
- [ ] All public classes have docstrings
- [ ] All public functions/methods have docstrings with Args/Returns
- [ ] Complex logic has inline comments
- [ ] README is updated if user-facing changes

### 9. Git/Repository
- [ ] `.gitignore` is updated for new files
- [ ] New source files are tracked (not in `data/`)
- [ ] Commit message follows convention: `git log --oneline -5`
- [ ] Branch is correct (not on `master` for features)
- [ ] No sensitive data (tokens, passwords) in code

### 10. Python Version
- [ ] Code works with Python 3.10+
- [ ] No features requiring Python 3.11+ unless documented
- [ ] `from __future__ import annotations` for union types

### 11. Complexity Analysis (REQUIRED)
After implementing any feature, assess complexity:

**Complexity Levels:**
- **Low**: Simple data transfer, config changes, <50 LOC, 1-2 files
- **Medium**: Algorithm changes, new class, 50-200 LOC, 2-4 files
- **High**: New subsystem, multiple interacting components, >200 LOC, >4 files

**For Medium+ Complexity:**
- [ ] Is there a simpler approach that achieves the same goal?
- [ ] Can this be broken into smaller, independent changes?
- [ ] Are we adding abstraction without clear benefit?
- [ ] Would a future maintainer understand this without reading a paper?

**Anti-Patterns to Avoid:**
- Implementing a pattern just because it's "standard"
- Adding infrastructure before it's needed (YAGNI)
- Creating complex hierarchies for simple problems

**Self-Review Question:**
> "If I were reviewing this PR, would I ask 'why is this so complex?'"

If yes, simplify before committing.

## Self-Correction Protocol

When you detect an issue:

1. **Acknowledge it** - "I noticed X is wrong..."
2. **Explain the impact** - "This causes Y to fail because..."
3. **Fix it immediately** - Don't wait, fix it now
4. **Verify the fix** - Run the relevant check
5. **Check for similar issues** - "Are there other places with this problem?"

## Common Mistakes to Avoid

### Import Issues
```python
# WRONG - causes circular import
from agentos.agentos import AgentOS  # in agent.py

# CORRECT - use TYPE_CHECKING
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from agentos.agentos import AgentOS
```

### Type Hints in Python 3.9+
```python
# WRONG - causes TypeError
def foo(x: str | None) -> str: ...

# CORRECT - add future import
from __future__ import annotations
def foo(x: str | None) -> str: ...
```

### Mutable Default Arguments
```python
# WRONG
def foo(items: list = []) -> None: ...

# CORRECT
def foo(items: list | None = None) -> None:
    if items is None:
        items = []
```

### Exception Handling
```python
# WRONG - too broad
try:
    ...
except Exception:
    pass  # hides all errors

# CORRECT - specific exceptions
try:
    ...
except (ValueError, KeyError) as e:
    logger.error(f"Specific error: {e}")
    raise
```

## Code Review Patterns

### Before Closing a Task
1. Read the entire file you modified
2. Search for TODO/FIXME/HACK comments you may have left
3. Run the full checklist above
4. **CRITICAL: Integration Check**
   - For each new public method: `grep -r "method_name(" src/` to verify it's called
   - For each new class: verify it's instantiated somewhere
   - For each new module: verify it's imported in `__init__.py` if public
5. Ask: "Would this pass a code review from a senior developer?"

### Before Creating Issues/PRs
1. Ensure all tests pass
2. Update documentation
3. Check formatting with ruff
4. Write a clear commit message
5. Verify the branch is correct

## Project-Specific Guidelines

### AgentOS Architecture
- Respect the separation of concerns: Kernel, Memory, Scheduler, Sync, I/O
- Don't bypass the S-MMU for memory operations
- Use SemanticPageTable for tracking slice locations
- Follow the paper's algorithms where specified

### Memory Hierarchy
- L1 is for active attention (small, fast)
- L2 is for deep context (medium)
- L3 is for knowledge base (large, slow)
- Always use importance scores for paging decisions

### Multi-Agent System
- Agents communicate through DSM (DistributedSharedMemory)
- Sync pulses are triggered by CSP orchestrator
- Use the scheduler for thread management
- Don't directly access other agents' memory

### Testing
- Unit tests go in `tests/unit/`
- Integration tests go in `tests/integration/`
- Use pytest fixtures for common setup
- Mock external dependencies (models, databases)

## Commands to Use

### Quick Check After Changes
```bash
# One-liner to check everything
PYTHONPATH=./src python -c "import agentos; print('✓ Imports OK')" && \
ruff check src/ && echo "✓ Ruff OK"
```

### Format Code
```bash
ruff format src/
```

### Fix Imports
```bash
ruff check --fix src/ --select I
```

## When in Doubt

1. **Read existing code** - Follow the patterns you see
2. **Ask for clarification** - Better to ask than guess
3. **Start simple** - Implement the minimal viable version first
4. **Test incrementally** - Test each part as you build it
5. **Document assumptions** - Comment if something is unusual

## Remember

- You are an assistant helping build research software
- Correctness > Speed
- Clarity > Cleverness
- Test > Don't Test
- Document > Don't Document
