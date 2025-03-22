  # ComfyUI Workflow Launcher - Streamlit Edition

![GitHub last commit](https://img.shields.io/github/last-commit/murapadev/ComfyUI-Streamlit-Launcher)
![GitHub issues](https://img.shields.io/github/issues/murapadev/ComfyUI-Streamlit-Launcher)
![GitHub stars](https://img.shields.io/github/stars/murapadev/ComfyUI-Streamlit-Launcher?style=social)
![GitHub forks](https://img.shields.io/github/forks/murapadev/ComfyUI-Streamlit-Launcher?style=social)
![License](https://img.shields.io/github/license/murapadev/ComfyUI-Streamlit-Launcher)

🚀 A user-friendly interface for managing **ComfyUI** projects with **Streamlit**.

⚠️ **Work in Progress**: This project is actively being developed. Some features may change or be incomplete.

---

## ✨ Features

- **Create workflows** from predefined templates
- **Import workflows** from JSON files
- **Manage projects** (start, stop, delete)
- **Monitor system info** and model details
- **Track workflow status** and port usage
- **Modern, responsive UI** for seamless interaction

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/murapadev/ComfyUI-Streamlit-Launcher.git
cd comfyui-launcher/streamlit
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Start the application

```bash
streamlit run app.py
```

The app will be available at [http://localhost:8501](http://localhost:8501) by default.

### Creating a New Workflow

1. Click **"Create New Workflow"** (sidebar or home page)
2. Enter a **workflow name**
3. Select a **template**
4. (Optional) Specify a **fixed port**
5. Click **"Create Workflow"**

### Importing a Workflow

1. Click **"Import Workflow"**
2. Enter a **workflow name**
3. Upload a **JSON file** or paste JSON content
4. (Optional) Set a **fixed port** or skip model validation
5. Click **"Import Workflow"**

### Managing Workflows

From the home page, you can:

- **Start** ready workflows
- **Open/Stop** running workflows
- **Delete** workflows

### ⚙️ Settings

Access system information and configurations, including:

- **System & directory info**
- **Port configuration**
- **Model storage details**

---

## 📂 Directory Structure

- `projects/` - Created ComfyUI projects
- `models/` - Shared models directory
- `templates/` - Workflow templates

---

## 📌 Requirements

- Python **3.8+**
- `streamlit>=1.24.0`
- `requests>=2.28.1`
- `psutil>=5.9.0`
- `tqdm>=4.64.1`
- `pillow>=9.3.0`
- `watchdog>=3.0.0`
- `pandas>=1.5.0`
- `plotly>=5.13.0`
- `streamlit-option-menu>=0.3.2`
- `streamlit-extras>=0.3.0`
- `python-dotenv>=1.0.0`
- `celery`
- *(Optional)* `torch>=2.0.0` (for ML functionality)

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. **Fork** the repository
2. **Create a new branch** (`git checkout -b feature-name`)
3. **Commit your changes** (`git commit -m "Add feature"`)
4. **Push to your branch** (`git push origin feature-name`)
5. **Submit a Pull Request** 🚀

---

## 📜 License & Attribution

This project is based on [ComfyUI-Launcher](https://github.com/ComfyWorkflows/ComfyUI-Launcher) ([AGPL-3.0 License](https://github.com/ComfyWorkflows/ComfyUI-Launcher/blob/bb6690462780abecaa733814d02f8ccee1b0a829/server/utils.py)). All derivative work complies with the original license terms.

📌 **Star this project** ⭐ if you find it useful!

