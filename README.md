# Coder

An AI-powered coding agent

## Requirements

### Language Server Protocol (LSP)

**LSP servers are required for this project to function.** The agent uses LSP servers to analyze and understand code across different languages.

Before running the project, you must install the required LSP servers for the languages in your workspace.

### Supported Languages

The following languages are supported:

- **Python**
- **TypeScript**
- **JavaScript**
- **Go**
- **Rust**

## LSP Installation

Choose the installation method that works best for your system. The commands below use common package managers available on different platforms.

### Python (PyRight)

**Using npm:**
```bash
npm install -g pyright
```

**Using pip/pipx:**
```bash
pip install pyright
# or
pipx install pyright
```

**Using conda:**
```bash
conda install -c conda-forge pyright
```

### TypeScript & JavaScript (TypeScript Language Server)

**Using npm:**
```bash
npm install -g typescript typescript-language-server
```

**Using yarn:**
```bash
yarn global add typescript typescript-language-server
```

**Using pnpm:**
```bash
pnpm install -g typescript typescript-language-server
```

### Go (gopls)

**Using Go:**
```bash
go install golang.org/x/tools/gopls@latest
```

**Using Homebrew (macOS/Linux):**
```bash
brew install gopls
```

**Using apt (Debian/Ubuntu):**
```bash
sudo apt install -y golang-gopls
```

### Rust (rust-analyzer)

**Using rustup:**
```bash
rustup component add rust-analyzer
```

**Using Homebrew (macOS/Linux):**
```bash
brew install rust-analyzer
```

**Using apt (Debian/Ubuntu):**
```bash
sudo apt install -y rust-analyzer
```

**Using dnf (Fedora/CentOS):**
```bash
sudo dnf install -y rust-analyzer
```

**Using Chocolatey (Windows):**
```bash
choco install rust-analyzer
```

## Quick Start

1. **Install LSP servers** for the languages in your workspace (see [LSP Installation](#lsp-installation) above)

2. **Install Python dependencies:**
   ```bash
   pip install -e .
   # or using uv
   uv sync
   ```

3. **Configure environment variables** (if needed):
   ```bash
   cp .env.example .env  # if available
   # Edit .env with your settings
   ```

4. **Run the agent:**
   ```bash
   python main.py
   # or using uv
   uv run main.py
   ```

## Troubleshooting

### LSP Server Not Found

If you get an error indicating that an LSP server is not found:

1. Verify the server is installed by running its executable directly
2. Ensure the installation directory is in your `PATH`
3. Reinstall the LSP server using the appropriate command from [LSP Installation](#lsp-installation)

### Verification

To verify LSP servers are installed correctly, test each one:

```bash
# Python
pyright --version

# TypeScript/JavaScript
typescript-language-server --version

# Go
gopls version

# Rust
rust-analyzer --version
```

## Development

### Running Tests

```bash
pytest tests/
```

### Installing LSP for Development

The project includes a helper script for installing LSP servers:

```bash
python scripts/install_lsps.py
```

This script will:
- Detect your operating system
- Identify available package managers
- Suggest and run appropriate installation commands

## Project Structure

```
core/
├── agents/          # AI agent implementations
├── cli/             # Command-line interface
├── client/          # LSP client implementation
├── common/          # Shared utilities
├── model/           # Data models
└── tools/           # Tool implementations (including LSP tools)
```

## Contributing

When adding support for new languages:

1. Add the language to `config.py` (LSP_SERVER_COMMANDS)
2. Update this README with installation instructions
3. Add test cases for the new language