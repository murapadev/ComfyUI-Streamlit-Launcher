import os
import json
import os.path

# Load configuration from config.json
CONFIG_FILEPATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

def get_config():
    """
    Load the configuration from config.json file.
    If the file doesn't exist, create it with default values.
    
    Returns:
        dict: Configuration dictionary
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(CONFIG_FILEPATH), exist_ok=True)
        
        if os.path.exists(CONFIG_FILEPATH):
            with open(CONFIG_FILEPATH, "r") as f:
                config = json.load(f)
                return config
        else:
            # Define default configuration
            default_config = {
                "credentials": {
                    "civitai": {
                        "apikey": ""
                    }
                },
                "directories": {
                    "projects": "./projects",
                    "models": "./models",
                    "templates": "./templates"
                },
                "port_configuration": {
                    "allow_overridable_ports": True,
                    "project_min_port": 4001,
                    "project_max_port": 4100,
                    "server_port": 8501
                }
            }
            
            # Create the config file with default values
            with open(CONFIG_FILEPATH, "w") as f:
                json.dump(default_config, f, indent=2)
            
            return default_config
    except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
        print(f"Warning: Could not load config file: {e}")
        # Return default configuration as fallback
        return {
            "credentials": {"civitai": {"apikey": ""}},
            "directories": {
                "projects": "./projects",
                "models": "./models",
                "templates": "./templates"
            },
            "port_configuration": {
                "allow_overridable_ports": True,
                "project_min_port": 4001,
                "project_max_port": 4100,
                "server_port": 8501
            }
        }

# Load configuration
config = get_config()

# Get directory configurations with environment variable overrides
# Environment variables take precedence over config file values
PROJECTS_DIR = os.environ.get("PROJECTS_DIR", config["directories"]["projects"])
MODELS_DIR = os.environ.get("MODELS_DIR", config["directories"]["models"])
TEMPLATES_DIR = os.environ.get("TEMPLATES_DIR", config["directories"]["templates"])

# Create directories if they don't exist
os.makedirs(PROJECTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(TEMPLATES_DIR, exist_ok=True)

# Get port configuration with environment variable overrides
ALLOW_OVERRIDABLE_PORTS_PER_PROJECT = os.environ.get(
    "ALLOW_OVERRIDABLE_PORTS_PER_PROJECT", 
    str(config["port_configuration"]["allow_overridable_ports"])
).lower() == "true"

PROJECT_MIN_PORT = int(os.environ.get(
    "PROJECT_MIN_PORT", 
    str(config["port_configuration"]["project_min_port"])
))

PROJECT_MAX_PORT = int(os.environ.get(
    "PROJECT_MAX_PORT", 
    str(config["port_configuration"]["project_max_port"])
))

SERVER_PORT = int(os.environ.get(
    "SERVER_PORT", 
    str(config["port_configuration"]["server_port"])
))