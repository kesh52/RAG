import os
import re
import yaml
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

def _resolve_env_vars(val):
    """Recursively walks a nested dict/list structure and resolves any env variables in strings."""
    if isinstance(val, dict):
        return {k: _resolve_env_vars(v) for k, v in val.items()}
    elif isinstance(val, list):
        return [_resolve_env_vars(v) for v in val]
    elif isinstance(val, str):
        # Matches patterns like ${VAR_NAME} or ${VAR_NAME:default_value}
        pattern = re.compile(r'\$\{(\w+)(?::([^}]*))?\}')
        
        def replace(match):
            var_name = match.group(1)
            default_val = match.group(2)  # None if there was no colon
            
            env_val = os.getenv(var_name)
            if env_val is None:
                return default_val if default_val is not None else ""
            return env_val

        return pattern.sub(replace, val)
    return val


class Config:
    """Config manager that loads YAML configuration with environment variable resolution."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self._config = {}
        self.load()

    def load(self):
        """Loads and parses the configuration file."""
        if not os.path.exists(self.config_path):
            # Check relative to root directory (src/utils/../../ parent directory)
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            resolved_path = os.path.join(root_dir, self.config_path)
            if os.path.exists(resolved_path):
                self.config_path = resolved_path
            else:
                raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
                
        with open(self.config_path, "r") as f:
            raw_config = yaml.safe_load(f) or {}
            
        # Resolve all environment variables inside the configuration dict
        self._config = _resolve_env_vars(raw_config)

    def get(self, key_path: str, default=None):
        """Gets a configuration property using dot-notation (e.g. 'database.host')."""
        keys = key_path.split(".")
        val = self._config
        for key in keys:
            if isinstance(val, dict):
                val = val.get(key)
            else:
                return default
        return val if val is not None else default


# Global configuration singleton instance
config = Config()

