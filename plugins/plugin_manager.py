import importlib
import os
import inspect
import numpy as np
from typing import Dict, Type
from plugins.base_plugin import BasePlugin

class PluginManager:
    """Manages loading and accessing plugins"""
    
    def __init__(self):
        self.plugins: Dict[str, BasePlugin] = {}
        
    def load_plugins(self, plugin_dir: str = "plugins"):
        """Load all plugins from the specified directory
        
        Args:
            plugin_dir: Directory containing plugin files
        """
        # Get the absolute path to the plugins directory
        abs_plugin_dir = os.path.abspath(plugin_dir)
        
        # Ensure plugins directory exists
        if not os.path.exists(abs_plugin_dir):
            os.makedirs(abs_plugin_dir)
            
        # Add plugins directory to Python path if not already there
        if abs_plugin_dir not in os.sys.path:
            os.sys.path.insert(0, abs_plugin_dir)
            
        # Load each .py file in the plugins directory
        for filename in os.listdir(abs_plugin_dir):
            if filename.endswith('.py') and not filename.startswith('__'):
                module_name = filename[:-3]  # Remove .py extension
                try:
                    module = importlib.import_module(module_name)
                    
                    # Find all classes in the module that inherit from BasePlugin
                    for name, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, BasePlugin) and 
                            obj != BasePlugin):
                            # Initialize plugin
                            plugin_instance = obj()
                            self.plugins[obj.SLUG] = plugin_instance
                            
                except Exception as e:
                    print(f"Error loading plugin {module_name}: {e}")
                    
    def get_plugin(self, slug: str) -> BasePlugin:
        """Get a plugin instance by its slug"""
        return self.plugins[slug]
    
    def get_plugin_choices(self) -> Dict[str, str]:
        """Get a dictionary of plugin choices for dropdown menus"""
        return {slug: plugin.NAME for slug, plugin in self.plugins.items()}
    
    def run_plugin(self, slug: str, image: np.ndarray, **kwargs) -> np.ndarray:
        """Run a specific plugin with the given image and arguments"""
        plugin = self.get_plugin(slug)
        return plugin.run(image, **kwargs)