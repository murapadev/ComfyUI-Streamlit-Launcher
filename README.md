# ComfyUI Workflow Launcher - Streamlit Edition

A streamlined interface for managing ComfyUI projects built with Streamlit.

## Features

- Create new ComfyUI workflows from templates
- Import existing ComfyUI workflows from JSON files
- Manage project lifecycle (start, stop, delete)
- View system information and model details
- Track project status and ports
- Responsive UI with a clean, modern design

## Installation

1. Clone the repository:
```bash
git clone https://github.com/murapadev/comfyui-launcher.git
cd comfyui-launcher/streamlit
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Start the Streamlit application:
```bash
streamlit run app.py
```

The application will be available at http://localhost:8501 by default.

### Creating a New Workflow

1. Click on "Create New Workflow" in the sidebar or from the home page
2. Enter a name for your workflow
3. Select a template from the available options
4. Optionally specify a fixed port
5. Click "Create Workflow"

### Importing a Workflow

1. Click on "Import Workflow" in the sidebar or from the home page
2. Enter a name for your workflow
3. Choose to either upload a JSON file or paste JSON content
4. Optionally specify a fixed port or skip model validation
5. Click "Import Workflow"

### Managing Workflows

From the home page, you can:
- Start ready workflows
- Open or stop running workflows
- Delete any workflow

### Settings

The settings page provides information about:
- System information and directory paths
- Port configuration and availability
- Model storage and details

## Directory Structure

- `projects/`: Contains all created ComfyUI projects
- `models/`: Shared models directory used by all projects
- `templates/`: Template workflows for creating new projects

## Requirements

- Python 3.8+
- Streamlit 1.24.0+
- psutil
- requests

## Copyright

## Copyright and Attribution

This project is based on code from:

- [ComfyUI-Launcher](https://github.com/ComfyWorkflows/ComfyUI-Launcher) - Licensed under [AGPL-3.0](https://github.com/ComfyWorkflows/ComfyUI-Launcher/blob/bb6690462780abecaa733814d02f8ccee1b0a829/server/utils.py)

All derivative work maintains compliance with the original license terms.