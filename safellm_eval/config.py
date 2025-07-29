"""
Basic configuration management system.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration for a single model."""
    name: str
    provider: str
    api_key: Optional[str] = None
    enabled: bool = True


class ConfigManager:
    """Basic configuration manager."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager."""
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        self.config = self._load_default_config()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "models": [
                ModelConfig(name="mock-gpt-4", provider="mock", enabled=True),
                ModelConfig(name="mock-claude", provider="mock", enabled=True)
            ],
            "evaluation": {
                "batch_size": 10,
                "max_workers": 2,
                "output_dir": "./results",
                "log_level": "INFO"
            },
            "languages": [
                "English", "Chinese", "Spanish", "French", "German", 
                "Arabic", "Hindi", "Swahili", "Russian", "Portuguese"
            ],
            "domains": [
                "Healthcare", "Legal", "Education", "Finance", "Safety", "General"
            ]
        }
    
    def get_enabled_models(self) -> List[ModelConfig]:
        """Get list of enabled models."""
        models = self.config.get("models", [])
        return [model for model in models if model.enabled]
    
    def validate_config(self) -> Dict[str, List[str]]:
        """Validate configuration and return validation results."""
        issues = {
            "errors": [],
            "warnings": [],
            "info": []
        }
        
        # Check models
        models = self.config.get("models", [])
        if not models:
            issues["errors"].append("No models configured")
        
        enabled_models = self.get_enabled_models()
        if not enabled_models:
            issues["warnings"].append("No enabled models found")
        
        # Check evaluation settings
        eval_config = self.config.get("evaluation", {})
        batch_size = eval_config.get("batch_size", 0)
        if batch_size <= 0:
            issues["warnings"].append("Invalid batch size")
        
        # Info messages
        issues["info"].append(f"Found {len(enabled_models)} enabled models")
        issues["info"].append(f"Configured for {len(self.config.get('languages', []))} languages")
        issues["info"].append(f"Configured for {len(self.config.get('domains', []))} domains")
        
        return issues
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration."""
        enabled_models = self.get_enabled_models()
        
        return {
            "models": {
                "total": len(self.config.get("models", [])),
                "enabled": len(enabled_models),
                "providers": list(set(m.provider for m in enabled_models))
            },
            "evaluation": self.config.get("evaluation", {}),
            "languages": {
                "count": len(self.config.get("languages", [])),
                "languages": self.config.get("languages", [])
            },
            "domains": {
                "count": len(self.config.get("domains", [])),
                "domains": self.config.get("domains", [])
            }
        }