import streamlit as st

# Page config must be the first Streamlit command
st.set_page_config(
    page_title="ComfyUI Launcher",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Regular imports after st.set_page_config
import os
import json
import subprocess
import time
import sys
import shutil
import threading
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import psutil
import pandas as pd
from streamlit_option_menu import option_menu
from streamlit_extras.app_logo import add_logo
from streamlit_extras.colored_header import colored_header
from streamlit_extras.metric_cards import style_metric_cards
import plotly.express as px
from utils import (
    slugify,
    setup_files_from_launcher_json,
    get_launcher_state,
    set_launcher_state_data,
    get_project_port,
    is_port_in_use,
    run_command,
    run_command_in_project_comfyui_venv,
    is_launcher_json_format,
    get_launcher_json_for_workflow_json,
    check_url_structure,
    get_config,
    update_config,
    CONFIG_FILEPATH,
    DEFAULT_CONFIG,
)
from tasks import (

        create_comfyui_project,
)
from settings import (
    PROJECTS_DIR,
    MODELS_DIR,
    TEMPLATES_DIR,
    PROJECT_MIN_PORT,
    PROJECT_MAX_PORT,
    ALLOW_OVERRIDABLE_PORTS_PER_PROJECT,
)

# GPU detection function
def detect_gpu() -> Tuple[bool, str]:
    """
    Detect if a GPU is available and return its information without requiring torch.
    
    Returns:
        Tuple of (has_gpu, gpu_info)
    """
    # Check for NVIDIA GPUs using nvidia-smi
    try:
        nvidia_output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], 
            stderr=subprocess.DEVNULL,
            text=True
        )
        # Successfully ran nvidia-smi, so we have NVIDIA GPU(s)
        first_gpu = nvidia_output.strip().split('\n')[0]
        return True, f"NVIDIA {first_gpu}"
    except (subprocess.SubprocessError, FileNotFoundError):
        # No NVIDIA GPU or nvidia-smi not available
        pass
    
    # Check for Apple Silicon (M1/M2) GPU
    if sys.platform == "darwin":  # macOS
        try:
            model = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
            if "Apple" in model:
                return True, "Apple Silicon GPU"
        except (subprocess.SubprocessError, FileNotFoundError):
            pass
    
    # Check for AMD GPUs on Linux
    if sys.platform == "linux":
        if os.path.exists("/sys/class/drm/"):
            try:
                # Look for AMD GPU directories
                for device in os.listdir("/sys/class/drm/"):
                    if device.startswith("card") and os.path.exists(f"/sys/class/drm/{device}/device/vendor"):
                        with open(f"/sys/class/drm/{device}/device/vendor", "r") as f:
                            vendor_id = f.read().strip()
                            # AMD vendor ID is 0x1002
                            if vendor_id == "0x1002":
                                return True, "AMD GPU"
            except (FileNotFoundError, PermissionError):
                pass
    
    # No GPU detected
    return False, "CPU Only"


# Initialize session state
if 'dark_theme' not in st.session_state:
    st.session_state.dark_theme = True
if 'show_advanced' not in st.session_state:
    st.session_state.show_advanced = False
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

# Initialize session state for project operations
if 'operation_status' not in st.session_state:
    st.session_state.operation_status = None
if 'active_page' not in st.session_state:
    st.session_state.active_page = "home"
if 'selected_project' not in st.session_state:
    st.session_state.selected_project = None
if 'last_refresh' not in st.session_state:
    st.session_state.last_refresh = time.time()
if 'template_filter' not in st.session_state:
    st.session_state.template_filter = ""
if 'port_range' not in st.session_state:
    st.session_state.port_range = (PROJECT_MIN_PORT, PROJECT_MAX_PORT)
if 'missing_models' not in st.session_state:
    st.session_state.missing_models = []
if 'import_json' not in st.session_state:
    st.session_state.import_json = None
if 'config' not in st.session_state:
    st.session_state.config = get_config()

# Create the required directories if they don't exist
os.makedirs(PROJECTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Improved sidebar with option menu
with st.sidebar:
    add_logo("https://raw.githubusercontent.com/comfyanonymous/ComfyUI/master/web/favicon.png")
    
    # Get the current active page for default selection
    default_idx = 0
    menu_options = ["Home", "New Workflow", "Import", "Models", "Settings"]
    if st.session_state.active_page != "home":
        # Convert active_page back to menu option format for correct default selection
        current_page = st.session_state.active_page.replace("_", " ").title()
        if current_page in menu_options:
            default_idx = menu_options.index(current_page)
    
    selected = option_menu(
        "ComfyUI Launcher",
        menu_options,
        icons=['house', 'plus-circle', 'cloud-upload', 'archive', 'gear'],
        menu_icon="rocket",
        default_index=default_idx,
        styles={
            "container": {"padding": "5px"},
            "icon": {"color": "orange", "font-size": "25px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px"}
        }
    )
    
    # Update the active page in session state and force a rerun if it changed
    new_active_page = selected.lower().replace(" ", "_")
    if st.session_state.active_page != new_active_page:
        st.session_state.active_page = new_active_page
        st.rerun()
    
    # System metrics in sidebar
    st.divider()
    colored_header(label="System Metrics", description="Current system status", color_name="blue-70")
    
    # Memory usage
    try:
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Memory Usage", f"{memory_usage}%")
        with col2:
            st.metric("Memory Used", f"{memory_used_gb:.1f}GB")
    except:
        st.warning("Could not get memory information")
    
    # GPU info without torch dependency
    has_gpu, gpu_info = detect_gpu()
    if has_gpu:
        st.success(f"🎮 GPU: {gpu_info}")
    else:
        st.warning("💻 Using CPU mode")
    
    # Auto-refresh toggle with rate limiting
    st.divider()
    st.checkbox("Auto-refresh", value=st.session_state.auto_refresh, key="auto_refresh")
    if st.session_state.auto_refresh:
        now = time.time()
        # Refresh only if 5 seconds have passed since last refresh
        if now - st.session_state.last_refresh >= 5:
            st.session_state.last_refresh = now
            st.rerun()

# Project type with proper typing
class Project:
    def __init__(self, data: Dict[str, Any]):
        self.id: str = data.get('id', '')
        self.state: Dict[str, Any] = data.get('state', {})
        self.project_folder_name: str = data.get('project_folder_name', '')
        self.project_folder_path: str = data.get('project_folder_path', '')
        self.last_modified: float = data.get('last_modified', 0)
        self.port: Optional[int] = data.get('port')
        
    @property
    def name(self) -> str:
        return self.state.get('name', self.id)
    
    @property
    def status_message(self) -> str:
        return self.state.get('status_message', 'Unknown')
    
    @property
    def project_state(self) -> str:
        return self.state.get('state', 'unknown')

# Get all projects
def get_projects() -> List[Project]:
    projects = []
    for proj_folder in os.listdir(PROJECTS_DIR):
        full_proj_path = os.path.join(PROJECTS_DIR, proj_folder)
        if not os.path.isdir(full_proj_path):
            continue
        
        launcher_state, _ = get_launcher_state(full_proj_path)
        if not launcher_state:
            continue
        
        project_port = get_project_port(proj_folder)
        
        project_data = {
            "id": proj_folder,
            "state": launcher_state,
            "project_folder_name": proj_folder,
            "project_folder_path": full_proj_path,
            "last_modified": os.stat(full_proj_path).st_mtime,
            "port": project_port
        }
        
        projects.append(Project(project_data))
    
    # Sort projects by last modified (descending)
    projects.sort(key=lambda x: x.last_modified, reverse=True)
    return projects

# Enhanced project cards in home page
def render_project_card(project: Project):
    with st.container():
        st.markdown(f"""
        <div class="project-card">
            <h3>{project.name}</h3>
            <span class="badge badge-{project.project_state}">{project.project_state.capitalize()}</span>
            <p>{project.status_message}</p>
            <p>Port: {project.port if project.port else 'Not assigned'}</p>
            <p>Last modified: {datetime.fromtimestamp(project.last_modified).strftime('%Y-%m-%d %H:%M:%S')}</p>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if project.project_state == 'ready':
                if st.button("▶️ Start", key=f"start_{project.id}", use_container_width=True):
                    with st.spinner("Starting project..."):
                        result = start_project(project.id)
                        if result and result.get("success"):
                            st.success(f"Project started on port {result.get('port')}")
                            st.rerun()
            elif project.project_state == 'running':
                if st.button("⏹️ Stop", key=f"stop_{project.id}", use_container_width=True):
                    with st.spinner("Stopping project..."):
                        result = stop_project(project.id)
                        if result and result.get("success"):
                            st.success("Project stopped")
                            st.rerun()
                
                # Add web UI link
                if project.port:
                    st.markdown(f"""
                    <a href="http://localhost:{project.port}" target="_blank" 
                      class="web-ui-link">
                        🔗 Open Web UI
                    </a>
                    """, unsafe_allow_html=True)
        
        with col2:
            if st.button("🗑️ Delete", key=f"delete_{project.id}", use_container_width=True):
                with st.spinner("Deleting project..."):
                    result = delete_project(project.id)
                    if result and result.get("success"):
                        st.success("Project deleted")
                        st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

# Enhanced home page
def home_page():
    st.markdown("<h1 class='main-header'>ComfyUI Projects</h1>", unsafe_allow_html=True)
    
    # Quick stats
    projects = get_projects()
    running = len([p for p in projects if p.project_state == "running"])
    total = len(projects)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Projects", total)
    with col2:
        st.metric("Running Projects", running)
    with col3:
        st.metric("Available Ports", PROJECT_MAX_PORT - PROJECT_MIN_PORT - running)
    
    style_metric_cards()
    
    # Project filtering
    st.text_input("🔍 Filter projects", key="project_filter", placeholder="Type to filter...")
    filter_text = st.session_state.get("project_filter", "").lower()
    
    # Filter projects
    if filter_text:
        projects = [p for p in projects if filter_text in p.name.lower()]
    
    if not projects:
        st.info("No projects found. Create a new workflow or import one to get started.")
        if st.button("➕ Create New Workflow", use_container_width=True):
            st.session_state.active_page = "new"
            st.rerun()
    else:
        # Display projects in a grid
        cols = st.columns(3)
        for i, project in enumerate(projects):
            with cols[i % 3]:
                render_project_card(project)

# Create new workflow page
def new_workflow_page():
    st.markdown("<h1 class='main-header'>Create New Workflow</h1>", unsafe_allow_html=True)
    
    # Get templates from templates directory
    templates = []
    for template_folder in os.listdir(TEMPLATES_DIR):
        template_path = os.path.join(TEMPLATES_DIR, template_folder)
        if not os.path.isdir(template_path):
            continue
        
        # Check if launcher.json or workflow.json exists
        launcher_json_path = os.path.join(template_path, "launcher.json")
        workflow_json_path = os.path.join(template_path, "workflow.json")
        
        if os.path.exists(launcher_json_path) or os.path.exists(workflow_json_path):
            templates.append({
                "id": template_folder,
                "name": template_folder.replace("_", " ").title(),
                "path": template_path
            })
    
    # Filter input
    st.text_input("Filter templates", key="template_filter")
    
    if st.session_state.template_filter:
        templates = [t for t in templates if st.session_state.template_filter.lower() in t["name"].lower()]
    
    # Form for creating a new workflow
    with st.form("create_workflow_form", clear_on_submit=True):
        st.markdown('<div class="form-container">', unsafe_allow_html=True)
        
        name = st.text_input("Workflow Name", placeholder="My Awesome Workflow")
        
        # Display templates as cards
        st.subheader("Select Template")
        
        template_id = None
        template_cols = st.columns(2)
        for i, template in enumerate(templates):
            with template_cols[i % 2]:
                template_selected = st.checkbox(
                    template["name"], 
                    key=f"template_{template['id']}",
                    help=f"Select {template['name']} template"
                )
                if template_selected:
                    template_id = template["id"]
        
        use_fixed_port = st.checkbox("Use fixed port")
        port = None
        if use_fixed_port:
            port = st.number_input("Port", min_value=PROJECT_MIN_PORT, max_value=PROJECT_MAX_PORT)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        submit = st.form_submit_button("Create Workflow")
        
        if submit and name and template_id:
            with st.spinner(f"Creating workflow '{name}'. This may take a few minutes..."):
                result = create_new_project(name, template_id, port if use_fixed_port else None)
                if result and result.get("success"):
                    st.success(f"Creating workflow '{name}'. This may take a few minutes.")
                    st.session_state.active_page = "home"
                    st.rerun()

def import_workflow_page():
    st.markdown("<h1 class='main-header'>Import Workflow</h1>", unsafe_allow_html=True)
    
    # Check if we have missing models to handle
    if st.session_state.missing_models:
        st.warning(f"Missing models detected: {len(st.session_state.missing_models)} models need to be resolved.")
        
        # Handle missing models
        with st.form("resolve_missing_models_form"):
            st.subheader("Missing Models")
            
            resolved_models = []
            for i, model in enumerate(st.session_state.missing_models):
                st.markdown(f"### Model {i+1}: {model.get('filename', 'Unknown')}")
                st.text(f"Node Type: {model.get('node_type', 'Unknown')}")
                st.text(f"Path: {model.get('dest_relative_path', 'Unknown')}")
                
                # Select source
                source_type = st.radio(f"Source for {model.get('filename', 'Unknown')}", 
                                      ["Skip", "URL"], key=f"source_{i}")
                
                if source_type == "URL":
                    url = st.text_input("Model URL", key=f"url_{i}")
                    if url and check_url_structure(url):
                        resolved_models.append({
                            "filename": model.get('filename'),
                            "node_type": model.get('node_type'),
                            "dest_relative_path": model.get('dest_relative_path'),
                            "source": {
                                "url": url,
                                "file_id": None
                            }
                        })
            
            skip_validation = st.checkbox("Skip model validation")
            
            submit = st.form_submit_button("Import with Resolved Models")
            
            if submit:
                if st.session_state.import_json:
                    name = st.session_state.import_name
                    with st.spinner(f"Importing workflow '{name}'. This may take a few minutes..."):
                        result = import_workflow(name, st.session_state.import_json, 
                                              skip_validation, resolved_models,
                                              st.session_state.import_port)
                        if result and result.get("success"):
                            st.success(f"Importing workflow '{name}'. This may take a few minutes.")
                            # Clear session state
                            st.session_state.missing_models = []
                            st.session_state.import_json = None
                            st.session_state.import_name = None
                            st.session_state.import_port = None
                            st.session_state.active_page = "home"
                            st.rerun()
                        elif result.get("missing_models"):
                            st.session_state.missing_models = result.get("missing_models")
                            st.rerun()
                        else:
                            st.error(result.get("error", "Failed to import workflow"))
        
        if st.button("Cancel Import"):
            st.session_state.missing_models = []
            st.session_state.import_json = None
            st.session_state.import_name = None
            st.session_state.import_port = None
            st.rerun()
            
    else:
        # Main import form 
        tabs = st.tabs(["Upload JSON", "Paste JSON"])
        
        with tabs[0]:
            with st.form("import_workflow_upload_form"):
                name = st.text_input("Workflow Name", placeholder="My Imported Workflow")
                uploaded_file = st.file_uploader("Upload workflow JSON file", type=["json"])
                
                use_fixed_port = st.checkbox("Use fixed port", key="upload_fixed_port")
                port = None
                if use_fixed_port:
                    port = st.number_input("Port", min_value=PROJECT_MIN_PORT, max_value=PROJECT_MAX_PORT, key="upload_port")
                
                skip_model_validation = st.checkbox("Skip model validation", key="upload_skip_validation")
                
                submit = st.form_submit_button("Import Workflow")
                
                if submit and name and uploaded_file:
                    try:
                        import_json = json.loads(uploaded_file.getvalue().decode("utf-8"))
                        
                        # Store in session state in case we need to handle missing models
                        st.session_state.import_json = import_json
                        st.session_state.import_name = name
                        st.session_state.import_port = port if use_fixed_port else None
                        
                        with st.spinner(f"Processing workflow '{name}'..."):
                            result = import_workflow(name, import_json, skip_model_validation, [], port if use_fixed_port else None)
                            if isinstance(result, dict):
                                if not result.get("success"):
                                    if result.get("missing_models"):
                                        st.session_state.missing_models = result.get("missing_models")
                                        st.rerun()
                                    else:
                                        st.error(result.get("error", "Failed to import workflow"))
                                else:
                                    st.success(f"Importing workflow '{name}'. This may take a few minutes.")
                                    st.session_state.active_page = "home"
                                    st.rerun()
                    except json.JSONDecodeError:
                        st.error("Invalid JSON format")
        
        with tabs[1]:
            with st.form("import_workflow_paste_form"):
                name = st.text_input("Workflow Name", placeholder="My Imported Workflow", key="paste_name")
                json_text = st.text_area("Paste JSON content", height=300)
                
                use_fixed_port = st.checkbox("Use fixed port", key="paste_fixed_port")
                port = None
                if use_fixed_port:
                    port = st.number_input("Port", min_value=PROJECT_MIN_PORT, max_value=PROJECT_MAX_PORT, key="paste_port")
                
                skip_model_validation = st.checkbox("Skip model validation", key="paste_skip_validation")
                
                submit = st.form_submit_button("Import Workflow")
                
                if submit and name and json_text:
                    try:
                        import_json = json.loads(json_text)
                        
                        # Store in session state in case we need to handle missing models
                        st.session_state.import_json = import_json
                        st.session_state.import_name = name
                        st.session_state.import_port = port if use_fixed_port else None
                        
                        with st.spinner(f"Processing workflow '{name}'..."):
                            result = import_workflow(name, import_json, skip_model_validation, [], port if use_fixed_port else None)
                            if isinstance(result, dict):
                                if not result.get("success"):
                                    if result.get("missing_models"):
                                        st.session_state.missing_models = result.get("missing_models")
                                        st.rerun()
                                    else:
                                        st.error(result.get("error", "Failed to import workflow"))
                                else:
                                    st.success(f"Importing workflow '{name}'. This may take a few minutes.")
                                    st.session_state.active_page = "home"
                                    st.rerun()
                    except json.JSONDecodeError:
                        st.error("Invalid JSON format")

# Add the missing Models page function
def models_page():
    """
    Display and manage models for ComfyUI projects.
    """
    st.markdown("<h1 class='main-header'>Models Library</h1>", unsafe_allow_html=True)
    
    # Models dashboard with statistics
    if os.path.exists(MODELS_DIR):
        # Calculate total size of models
        total_size = 0
        file_count = 0
        for dirpath, dirnames, filenames in os.walk(MODELS_DIR):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    file_size = os.path.getsize(fp)
                    total_size += file_size
                    file_count += 1
                except (FileNotFoundError, PermissionError):
                    pass
        
        # Display stats at the top
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Models", file_count)
        with col2:
            st.metric("Total Size", f"{total_size / (1024**3):.2f} GB")
        with col3:
            categories = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
            st.metric("Categories", len(categories))
        
        style_metric_cards()
        
        # Add model management options
        st.markdown("<h2 class='sub-header'>Model Management</h2>", unsafe_allow_html=True)
        
        tabs = st.tabs(["Browse Models", "Add Model", "Import/Export"])
        
        with tabs[0]:
            # Categories filter
            categories = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
            
            # Filter dropdown
            selected_category = st.selectbox("Select Model Category", ["All Categories"] + categories)
            
            # Model search
            model_search = st.text_input("🔍 Search Models", placeholder="Enter model name...")
            
            if selected_category == "All Categories":
                # Display all categories in expanders
                for category in categories:
                    with st.expander(category.capitalize()):
                        category_path = os.path.join(MODELS_DIR, category)
                        models = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
                        
                        # Apply search filter
                        if model_search:
                            models = [m for m in models if model_search.lower() in m.lower()]
                        
                        if models:
                            # Convert to DataFrame for nicer display
                            model_data = []
                            for model in sorted(models):
                                model_path = os.path.join(category_path, model)
                                model_size = os.path.getsize(model_path) / (1024**2)  # MB
                                model_data.append({"Model": model, "Size (MB)": f"{model_size:.1f}"})
                            
                            model_df = pd.DataFrame(model_data)
                            st.dataframe(model_df, use_container_width=True)
                        else:
                            st.info("No models found matching your search criteria")
            else:
                # Display only the selected category
                category_path = os.path.join(MODELS_DIR, selected_category)
                if os.path.exists(category_path) and os.path.isdir(category_path):
                    models = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
                    
                    # Apply search filter
                    if model_search:
                        models = [m for m in models if model_search.lower() in m.lower()]
                    
                    if models:
                        # Convert to DataFrame for nicer display
                        model_data = []
                        for model in sorted(models):
                            model_path = os.path.join(category_path, model)
                            model_size = os.path.getsize(model_path) / (1024**2)  # MB
                            last_modified = datetime.fromtimestamp(os.path.getmtime(model_path))
                            model_data.append({
                                "Model": model, 
                                "Size (MB)": f"{model_size:.1f}",
                                "Last Modified": last_modified.strftime('%Y-%m-%d %H:%M')
                            })
                        
                        model_df = pd.DataFrame(model_data)
                        st.dataframe(model_df, use_container_width=True)
                    else:
                        st.info("No models found matching your search criteria")
        
        with tabs[1]:
            st.subheader("Add a New Model")
            
            with st.form("add_model_form"):
                # Model category
                model_category = st.selectbox("Model Category", categories)
                
                # File uploader
                uploaded_file = st.file_uploader("Upload Model File", type=["ckpt", "safetensors", "pt", "pth", "bin", "onnx"])
                
                # URL input alternative
                model_url = st.text_input("Or Provide Model URL", placeholder="https://example.com/model.safetensors")
                
                # Model name
                model_name = st.text_input("Model Name (optional)", 
                                          placeholder="Leave blank to use filename")
                
                submit = st.form_submit_button("Add Model")
                
                if submit and (uploaded_file or (model_url and check_url_structure(model_url))):
                    st.info("This feature is not fully implemented yet. Coming soon!")
        
        with tabs[2]:
            st.subheader("Import/Export Models")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### Export Models List")
                if st.button("Export Models Inventory"):
                    # Create a simple CSV inventory
                    model_inventory = []
                    for category in categories:
                        category_path = os.path.join(MODELS_DIR, category)
                        for model in os.listdir(category_path):
                            if os.path.isfile(os.path.join(category_path, model)):
                                model_path = os.path.join(category_path, model)
                                size_mb = os.path.getsize(model_path) / (1024**2)
                                model_inventory.append({
                                    "Category": category,
                                    "Model": model,
                                    "Size_MB": round(size_mb, 2)
                                })
                    
                    if model_inventory:
                        model_df = pd.DataFrame(model_inventory)
                        # Create a download link for the CSV
                        csv = model_df.to_csv(index=False)
                        st.download_button(
                            label="Download Inventory CSV",
                            data=csv,
                            file_name="comfyui_models_inventory.csv",
                            mime="text/csv"
                        )
            
            with col2:
                st.markdown("### Import Models")
                st.info("Batch import feature coming soon!")
    else:
        st.warning(f"Models directory {MODELS_DIR} not found!")
        if st.button("Create Models Directory"):
            os.makedirs(MODELS_DIR, exist_ok=True)
            st.success(f"Created models directory at {MODELS_DIR}")
            st.rerun()

# Settings page
def settings_page():
    st.markdown("<h1 class='main-header'>Settings</h1>", unsafe_allow_html=True)
    
    tabs = st.tabs(["API Keys", "System", "Port Configuration", "Models"])
    
    with tabs[0]:
        st.markdown("<h2 class='sub-header'>API Keys & Credentials</h2>", unsafe_allow_html=True)
        
        # CivitAI API Key setting
        with st.form("civitai_form"):
            st.markdown('<div class="settings-section">', unsafe_allow_html=True)
            st.markdown('<div class="settings-header">CivitAI API Key</div>', unsafe_allow_html=True)
            
            # Get current CivitAI API key from config
            # Reload the config to ensure we have the latest version
            current_config = get_config()
            civitai_api_key = current_config.get('credentials', {}).get('civitai', {}).get('apikey', '')
            
            new_api_key = st.text_input(
                "CivitAI API Key", 
                value=civitai_api_key,
                type="password",
                help="Enter your CivitAI API key to download models directly",
                key="civitai_api_key"
            )
            
            st.markdown("""
            <div class="settings-description">
                You can get your CivitAI API key from your 
                <a href="https://civitai.com/user/account" target="_blank" class="settings-link">CivitAI account settings page</a>.
                <br>This key is saved locally and ONLY used to download missing models directly from CivitAI.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.form_submit_button("Save CivitAI Settings"):
                try:
                    # Update the configuration
                    new_config = {
                        "credentials": {
                            "civitai": {
                                "apikey": new_api_key
                            }
                        }
                    }
                    update_config(new_config)
                    # Update the session state with the new config
                    st.session_state.config = get_config()
                    st.success("CivitAI API key saved successfully!")
                except Exception as e:
                    st.error(f"Failed to save API key: {str(e)}")

    with tabs[1]:
        st.markdown("<h2 class='sub-header'>System Information</h2>", unsafe_allow_html=True)
        
        with st.form("system_settings_form"):
            st.markdown('<div class="settings-section">', unsafe_allow_html=True)
            st.markdown('<div class="settings-header">Directories</div>', unsafe_allow_html=True)
            
            st.info(f"Projects Directory: {PROJECTS_DIR}")
            st.info(f"Models Directory: {MODELS_DIR}")
            st.info(f"Templates Directory: {TEMPLATES_DIR}")
            
            custom_projects_dir = st.text_input("Custom Projects Directory (requires restart)", value=PROJECTS_DIR)
            custom_models_dir = st.text_input("Custom Models Directory (requires restart)", value=MODELS_DIR)
            custom_templates_dir = st.text_input("Custom Templates Directory (requires restart)", value=TEMPLATES_DIR)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.form_submit_button("Save Directory Settings"):
                # Note: In a real implementation, you'd update environment variables or a config file
                # For this demo, we'll just show a success message with instructions
                if custom_projects_dir != PROJECTS_DIR or custom_models_dir != MODELS_DIR or custom_templates_dir != TEMPLATES_DIR:
                    st.warning("Directory changes require restarting the application.")
                    st.info("To apply these changes, set the environment variables before starting the app:\n"
                           f"PROJECTS_DIR={custom_projects_dir}\n"
                           f"MODELS_DIR={custom_models_dir}\n"
                           f"TEMPLATES_DIR={custom_templates_dir}")
                else:
                    st.success("No changes were made to directory settings.")
            
        # System Status section
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown('<div class="settings-header">System Status</div>', unsafe_allow_html=True)
        
        # Get system memory info
        try:
            memory = psutil.virtual_memory()
            st.info(f"Memory: {memory.percent}% used ({memory.used / (1024**3):.2f} GB / {memory.total / (1024**3):.2f} GB)")
        except:
            st.warning("Could not get memory information")
        
        # Get running projects count
        running_projects = len([p for p in get_projects() if p.project_state == "running"])
        st.info(f"Running Projects: {running_projects}")
        
        # Get Python and Streamlit versions
        st.info(f"Python Version: {sys.version.split()[0]}")
        st.info(f"Streamlit Version: {st.__version__}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[2]:
        st.markdown("<h2 class='sub-header'>Port Configuration</h2>", unsafe_allow_html=True)
        
        with st.form("port_settings_form"):
            st.markdown('<div class="settings-section">', unsafe_allow_html=True)
            
            st.info(f"Default Port Range: {PROJECT_MIN_PORT} - {PROJECT_MAX_PORT}")
            st.info(f"Allow Overridable Ports: {ALLOW_OVERRIDABLE_PORTS_PER_PROJECT}")
            
            custom_min_port = st.number_input("Minimum Port", min_value=1024, max_value=65000, value=PROJECT_MIN_PORT)
            custom_max_port = st.number_input("Maximum Port", min_value=1024, max_value=65000, value=PROJECT_MAX_PORT)
            allow_overridable_ports = st.checkbox("Allow Overridable Ports Per Project", value=ALLOW_OVERRIDABLE_PORTS_PER_PROJECT)
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            if st.form_submit_button("Save Port Settings"):
                # Note: In a real implementation, you'd update environment variables or a config file
                # For this demo, we'll just show a success message with instructions
                if custom_min_port != PROJECT_MIN_PORT or custom_max_port != PROJECT_MAX_PORT or allow_overridable_ports != ALLOW_OVERRIDABLE_PORTS_PER_PROJECT:
                    st.warning("Port configuration changes require restarting the application.")
                    st.info("To apply these changes, set the environment variables before starting the app:\n"
                           f"PROJECT_MIN_PORT={custom_min_port}\n"
                           f"PROJECT_MAX_PORT={custom_max_port}\n"
                           f"ALLOW_OVERRIDABLE_PORTS_PER_PROJECT={'true' if allow_overridable_ports else 'false'}")
                else:
                    st.success("No changes were made to port settings.")
        
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        st.markdown('<div class="settings-header">Port Usage</div>', unsafe_allow_html=True)
        
        # Display a table of port usage
        port_data = []
        for port in range(PROJECT_MIN_PORT, min(PROJECT_MAX_PORT+1, PROJECT_MIN_PORT+10)):
            in_use = is_port_in_use(port)
            status = "In use" if in_use else "Available"
            status_color = "🔴" if in_use else "🟢"
            port_data.append({"Port": port, "Status": f"{status_color} {status}"})
        
        # Convert to DataFrame for nicer display
        port_df = pd.DataFrame(port_data)
        st.table(port_df)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tabs[3]:
        st.markdown("<h2 class='sub-header'>Models</h2>", unsafe_allow_html=True)
        
        st.markdown('<div class="settings-section">', unsafe_allow_html=True)
        
        if os.path.exists(MODELS_DIR):
            # Calculate total size of models
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(MODELS_DIR):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    total_size += os.path.getsize(fp)
            
            st.info(f"Total Models Size: {total_size / (1024**3):.2f} GB")
            
            # List model categories
            categories = [d for d in os.listdir(MODELS_DIR) if os.path.isdir(os.path.join(MODELS_DIR, d))]
            
            for category in categories:
                with st.expander(category.capitalize()):
                    category_path = os.path.join(MODELS_DIR, category)
                    models = [f for f in os.listdir(category_path) if os.path.isfile(os.path.join(category_path, f))]
                    
                    if models:
                        # Convert to DataFrame for nicer display
                        model_data = []
                        for model in sorted(models):
                            model_path = os.path.join(category_path, model)
                            model_size = os.path.getsize(model_path) / (1024**2)  # MB
                            model_data.append({"Model": model, "Size (MB)": f"{model_size:.1f}"})
                        
                        model_df = pd.DataFrame(model_data)
                        st.dataframe(model_df)
                    else:
                        st.text("No models found in this category")
        
        if st.button("Open Models Folder"):
            # This will only work if running locally
            try:
                if sys.platform == "win32":
                    subprocess.run(["explorer", MODELS_DIR], check=True)
                elif sys.platform == "darwin":
                    subprocess.run(["open", MODELS_DIR], check=True)
                else:
                    subprocess.run(["xdg-open", MODELS_DIR], check=True)
            except:
                st.error("Could not open models folder. Please open it manually.")
        
        st.markdown('</div>', unsafe_allow_html=True)

# Project operation functions
def create_new_project(name: str, template_id: str, port: Optional[int] = None) -> Dict[str, Any]:
    """
    Create a new ComfyUI project.
    
    Args:
        name: The name of the project
        template_id: The ID of the template to use
        port: Optional port number to use
        
    Returns:
        Dict with success status and project ID
    """
    id = slugify(name)
    project_path = os.path.join(PROJECTS_DIR, id)
    
    if os.path.exists(project_path):
        st.error(f"Project with id {id} already exists")
        return {"success": False, "error": f"Project with id {id} already exists"}
    
    models_path = MODELS_DIR
    
    # Load launcher.json or workflow.json from template
    launcher_json = None
    template_folder = os.path.join(TEMPLATES_DIR, template_id)
    template_launcher_json_path = os.path.join(template_folder, "launcher.json")
    
    if os.path.exists(template_launcher_json_path):
        with open(template_launcher_json_path, "r") as f:
            launcher_json = json.load(f)
    else:
        template_workflow_json_path = os.path.join(template_folder, "workflow.json")
        if os.path.exists(template_workflow_json_path):
            with open(template_workflow_json_path, "r") as f:
                template_workflow_json = json.load(f)
            res = get_launcher_json_for_workflow_json(template_workflow_json, resolved_missing_models=[], skip_model_validation=True)
            if res["success"] and res["launcher_json"]:
                launcher_json = res["launcher_json"]
            else:
                st.error(res["error"])
                return {"success": False, "error": res["error"]}
    
    os.makedirs(project_path)
    set_launcher_state_data(
        project_path,
        {"id": id, "name": name, "status_message": "Downloading ComfyUI...", "state": "download_comfyui"},
    )
    
    # Start the project creation process in a separate thread
    thread = threading.Thread(
        target=create_comfyui_project,
        args=(project_path, models_path),
        kwargs={"id": id, "name": name, "launcher_json": launcher_json, "port": port, "create_project_folder": False}
    )
    thread.start()
    
    return {"success": True, "id": id}

def import_workflow(
    name: str, 
    import_json: Dict[str, Any], 
    skipping_model_validation: bool = False, 
    resolved_missing_models: List[Dict[str, Any]] = None,
    port: Optional[int] = None
) -> Dict[str, Any]:
    """
    Import a workflow.
    
    Args:
        name: The name of the project
        import_json: The workflow JSON to import
        skipping_model_validation: Whether to skip model validation
        resolved_missing_models: List of resolved missing models
        port: Optional port number to use
        
    Returns:
        Dict with success status and project ID or missing models
    """
    if not name or not import_json:
        return {"success": False, "error": "Name and JSON content are required"}
        
    if resolved_missing_models is None:
        resolved_missing_models = []
        
    id = slugify(name)
    project_path = os.path.join(PROJECTS_DIR, id)
    
    if os.path.exists(project_path):
        return {"success": False, "error": f"Project with id {id} already exists"}
    
    models_path = MODELS_DIR
    
    # Process the imported JSON
    if is_launcher_json_format(import_json):
        launcher_json = import_json
    else:
        # Convert workflow JSON to launcher JSON
        skip_model_validation = True if skipping_model_validation else False
        res = get_launcher_json_for_workflow_json(import_json, resolved_missing_models, skip_model_validation)
        
        if res.get("success") and res.get("launcher_json"):
            launcher_json = res["launcher_json"]
        elif not res.get("success") and res.get("error") == "MISSING_MODELS" and len(res.get("missing_models", [])) > 0:
            return {"success": False, "missing_models": res["missing_models"], "error": res["error"]}
        else:
            return {"success": False, "error": res.get("error", "Failed to process workflow JSON")}
    
    os.makedirs(project_path)
    set_launcher_state_data(
        project_path,
        {"id": id, "name": name, "status_message": "Downloading ComfyUI...", "state": "download_comfyui"},
    )
    
    # Start the project creation process in a separate thread
    thread = threading.Thread(
        target=create_comfyui_project,
        args=(project_path, models_path),
        kwargs={"id": id, "name": name, "launcher_json": launcher_json, "port": port, "create_project_folder": False}
    )
    thread.start()
    
    return {"success": True, "id": id}

# Start project function updated to detect GPU without torch
def start_project(id: str) -> Dict[str, Any]:
    """
    Start a ComfyUI project.
    
    Args:
        id: The ID of the project to start
        
    Returns:
        Dict with success status and port
    """
    project_path = os.path.join(PROJECTS_DIR, id)
    
    if not os.path.exists(project_path):
        st.error(f"Project with id {id} does not exist")
        return {"success": False, "error": f"Project with id {id} does not exist"}
    
    launcher_state, _ = get_launcher_state(project_path)
    if not launcher_state or launcher_state["state"] != "ready":
        st.error(f"Project with id {id} is not ready")
        return {"success": False, "error": f"Project with id {id} is not ready"}
    
    port = get_project_port(id)
    if not port:
        st.error("No free port found")
        return {"success": False, "error": "No free port found"}
    
    if is_port_in_use(port):
        st.error(f"Port {port} is already in use")
        return {"success": False, "error": f"Port {port} is already in use"}
    
    # Start the project
    command = f"python main.py --port {port} --listen 0.0.0.0"
    
    # Check for GPU availability without torch
    has_gpu, _ = detect_gpu()
    if not has_gpu:
        st.warning("No GPU detected. Using CPU mode.")
        command += " --cpu"
    
    pid = run_command_in_project_comfyui_venv(project_path, command, in_bg=True)
    
    if not pid:
        st.error("Failed to start the project")
        return {"success": False, "error": "Failed to start the project"}
    
    # Wait for port to be bound
    max_wait_secs = 60
    while max_wait_secs > 0:
        max_wait_secs -= 1
        if is_port_in_use(port):
            break
        time.sleep(1)
    
    set_launcher_state_data(
        project_path, {"state": "running", "status_message": "Running...", "port": port, "pid": pid}
    )
    
    return {"success": True, "port": port}

def stop_project(id: str) -> Dict[str, Any]:
    """
    Stop a running ComfyUI project.
    
    Args:
        id: The ID of the project to stop
        
    Returns:
        Dict with success status
    """
    project_path = os.path.join(PROJECTS_DIR, id)
    
    if not os.path.exists(project_path):
        st.error(f"Project with id {id} does not exist")
        return {"success": False, "error": f"Project with id {id} does not exist"}
    
    launcher_state, _ = get_launcher_state(project_path)
    if not launcher_state or launcher_state["state"] != "running":
        st.error(f"Project with id {id} is not running")
        return {"success": False, "error": f"Project with id {id} is not running"}
    
    # Kill the process
    try:
        pid = launcher_state["pid"]
        parent = psutil.Process(pid)
        for child in parent.children(recursive=True):
            child.terminate()
        parent.terminate()
    except:
        pass
    
    set_launcher_state_data(project_path, {"state": "ready", "status_message": "Ready", "port": None, "pid": None})
    return {"success": True}

def delete_project(id: str) -> Dict[str, Any]:
    """
    Delete a ComfyUI project.
    
    Args:
        id: The ID of the project to delete
        
    Returns:
        Dict with success status
    """
    project_path = os.path.join(PROJECTS_DIR, id)
    
    if not os.path.exists(project_path):
        st.error(f"Project with id {id} does not exist")
        return {"success": False, "error": f"Project with id {id} does not exist"}
    
    # Stop the project if it's running
    launcher_state, _ = get_launcher_state(project_path)
    if launcher_state and launcher_state["state"] == "running":
        stop_project(id)
    
    # Delete the project folder
    shutil.rmtree(project_path, ignore_errors=True)
    return {"success": True}

# Render the appropriate page based on the active page
if st.session_state.active_page == "home":
    home_page()
elif st.session_state.active_page == "new_workflow":
    new_workflow_page()
elif st.session_state.active_page == "import":
    import_workflow_page()
elif st.session_state.active_page == "models":
    models_page()
elif st.session_state.active_page == "settings":
    settings_page()
else:
    # Fallback to home page if unknown page
    st.session_state.active_page = "home"
    st.rerun()

st.markdown("""
<style>
    /* ...existing code... */
    
    /* Fix st.metric styling for dark mode */
    .stMetric {
        background-color: #1e2227 !important;
        color: #ffffff !important;
        border: 1px solid #2d3139;
        border-radius: 8px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)