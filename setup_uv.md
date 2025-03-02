# Setting up '**uv**' for Python

```python
# Install the uv package
pip install uv

# Install different python version
uv python install 3.10

# Creating venv with uv
uv venv --python=python3.10

# Installing packages
uv pip install pandas

# Creating project
uv init --lib lib_pkg

# Adding packages to project
uv add pandas
```
