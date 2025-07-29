"""
Configuration management for PromptInjector security testing tool.
Handles settings, API keys, logging configuration, and test parameters.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class APIConfig:
    """API configuration settings"""
    target_api_key: str
    analyzer_api_key: str
    target_base_url: Optional[str] = None
    analyzer_base_url: Optional[str] = None
    timeout: int = 30
    max_retries: int = 3


@dataclass
class ModelConfig:
    """Model configuration settings"""
    target_model: str = "gpt-3.5-turbo"
    analyzer_model: str = "gpt-4"
    max_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0


@dataclass
class TestConfig:
    """Test execution configuration"""
    concurrent_tests: int = 3
    rate_limit_delay: float = 1.0
    max_test_time: int = 120  # seconds
    default_static_tests: int = 100
    default_adaptive_tests: int = 50
    adaptive_generation_rounds: int = 3
    min_confidence_threshold: float = 0.5


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    log_file: str = "prompt_injector.log"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    console_output: bool = True


@dataclass
class SecurityConfig:
    """Security and safety configuration"""
    enable_content_filtering: bool = True
    log_sensitive_data: bool = False
    anonymize_results: bool = True
    export_restrictions: bool = True
    audit_trail: bool = True


@dataclass
class PromptInjectorConfig:
    """Main configuration class"""
    api: APIConfig
    models: ModelConfig
    testing: TestConfig
    logging: LoggingConfig
    security: SecurityConfig
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'PromptInjectorConfig':
        """Create config from dictionary"""
        return cls(
            api=APIConfig(**config_dict.get('api', {})),
            models=ModelConfig(**config_dict.get('models', {})),
            testing=TestConfig(**config_dict.get('testing', {})),
            logging=LoggingConfig(**config_dict.get('logging', {})),
            security=SecurityConfig(**config_dict.get('security', {}))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'api': asdict(self.api),
            'models': asdict(self.models),
            'testing': asdict(self.testing),
            'logging': asdict(self.logging),
            'security': asdict(self.security)
        }


class ConfigManager:
    """Manages configuration loading, saving, and environment variable handling"""
    
    DEFAULT_CONFIG_FILE = "prompt_injector_config.json"
    ENV_PREFIX = "PI_"  # PromptInjector environment variable prefix
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or self.DEFAULT_CONFIG_FILE
        self.config: Optional[PromptInjectorConfig] = None
        
    def load_config(self) -> PromptInjectorConfig:
        """Load configuration from file and environment variables"""
        
        # Start with default config
        config_dict = self._get_default_config()
        
        # Load from file if it exists
        if Path(self.config_file).exists():
            try:
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                    config_dict = self._merge_configs(config_dict, file_config)
            except Exception as e:
                logging.warning(f"Failed to load config file {self.config_file}: {e}")
        
        # Override with environment variables
        env_config = self._load_from_environment()
        config_dict = self._merge_configs(config_dict, env_config)
        
        # Validate required fields
        self._validate_config(config_dict)
        
        self.config = PromptInjectorConfig.from_dict(config_dict)
        return self.config
    
    def save_config(self, config: PromptInjectorConfig):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config.to_dict(), f, indent=2)
            logging.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logging.error(f"Failed to save config to {self.config_file}: {e}")
            raise
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration values"""
        return {
            'api': {
                'target_api_key': '',
                'analyzer_api_key': '',
                'target_base_url': None,
                'analyzer_base_url': None,
                'timeout': 30,
                'max_retries': 3
            },
            'models': {
                'target_model': 'gpt-3.5-turbo',
                'analyzer_model': 'gpt-4',
                'max_tokens': 500,
                'temperature': 0.7,
                'top_p': 1.0,
                'frequency_penalty': 0.0,
                'presence_penalty': 0.0
            },
            'testing': {
                'concurrent_tests': 3,
                'rate_limit_delay': 1.0,
                'max_test_time': 120,
                'default_static_tests': 100,
                'default_adaptive_tests': 50,
                'adaptive_generation_rounds': 3,
                'min_confidence_threshold': 0.5
            },
            'logging': {
                'level': 'INFO',
                'log_file': 'prompt_injector.log',
                'max_file_size': 10 * 1024 * 1024,
                'backup_count': 5,
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'console_output': True
            },
            'security': {
                'enable_content_filtering': True,
                'log_sensitive_data': False,
                'anonymize_results': True,
                'export_restrictions': True,
                'audit_trail': True
            }
        }
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_config = {}
        
        # API configuration
        if os.getenv(f'{self.ENV_PREFIX}TARGET_API_KEY'):
            env_config.setdefault('api', {})['target_api_key'] = os.getenv(f'{self.ENV_PREFIX}TARGET_API_KEY')
        
        if os.getenv(f'{self.ENV_PREFIX}ANALYZER_API_KEY'):
            env_config.setdefault('api', {})['analyzer_api_key'] = os.getenv(f'{self.ENV_PREFIX}ANALYZER_API_KEY')
        
        if os.getenv(f'{self.ENV_PREFIX}TARGET_BASE_URL'):
            env_config.setdefault('api', {})['target_base_url'] = os.getenv(f'{self.ENV_PREFIX}TARGET_BASE_URL')
        
        if os.getenv(f'{self.ENV_PREFIX}ANALYZER_BASE_URL'):
            env_config.setdefault('api', {})['analyzer_base_url'] = os.getenv(f'{self.ENV_PREFIX}ANALYZER_BASE_URL')
        
        # Model configuration
        if os.getenv(f'{self.ENV_PREFIX}TARGET_MODEL'):
            env_config.setdefault('models', {})['target_model'] = os.getenv(f'{self.ENV_PREFIX}TARGET_MODEL')
        
        if os.getenv(f'{self.ENV_PREFIX}ANALYZER_MODEL'):
            env_config.setdefault('models', {})['analyzer_model'] = os.getenv(f'{self.ENV_PREFIX}ANALYZER_MODEL')
        
        # Test configuration
        if os.getenv(f'{self.ENV_PREFIX}CONCURRENT_TESTS'):
            env_config.setdefault('testing', {})['concurrent_tests'] = int(os.getenv(f'{self.ENV_PREFIX}CONCURRENT_TESTS'))
        
        if os.getenv(f'{self.ENV_PREFIX}RATE_LIMIT_DELAY'):
            env_config.setdefault('testing', {})['rate_limit_delay'] = float(os.getenv(f'{self.ENV_PREFIX}RATE_LIMIT_DELAY'))
        
        # Logging configuration
        if os.getenv(f'{self.ENV_PREFIX}LOG_LEVEL'):
            env_config.setdefault('logging', {})['level'] = os.getenv(f'{self.ENV_PREFIX}LOG_LEVEL')
        
        if os.getenv(f'{self.ENV_PREFIX}LOG_FILE'):
            env_config.setdefault('logging', {})['log_file'] = os.getenv(f'{self.ENV_PREFIX}LOG_FILE')
        
        # Security configuration
        if os.getenv(f'{self.ENV_PREFIX}ENABLE_CONTENT_FILTERING'):
            env_config.setdefault('security', {})['enable_content_filtering'] = os.getenv(f'{self.ENV_PREFIX}ENABLE_CONTENT_FILTERING').lower() == 'true'
        
        if os.getenv(f'{self.ENV_PREFIX}LOG_SENSITIVE_DATA'):
            env_config.setdefault('security', {})['log_sensitive_data'] = os.getenv(f'{self.ENV_PREFIX}LOG_SENSITIVE_DATA').lower() == 'true'
        
        return env_config
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration dictionaries recursively"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _validate_config(self, config_dict: Dict[str, Any]):
        """Validate configuration has required fields"""
        api_config = config_dict.get('api', {})
        
        if not api_config.get('target_api_key'):
            raise ValueError("target_api_key is required. Set it in config file or PI_TARGET_API_KEY environment variable.")
        
        if not api_config.get('analyzer_api_key'):
            raise ValueError("analyzer_api_key is required. Set it in config file or PI_ANALYZER_API_KEY environment variable.")


def setup_logging(config: LoggingConfig):
    """Setup logging based on configuration"""
    from logging.handlers import RotatingFileHandler
    
    # Set logging level
    level = getattr(logging, config.level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(config.format)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        config.log_file,
        maxBytes=config.max_file_size,
        backupCount=config.backup_count
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    root_logger.addHandler(file_handler)
    
    # Console handler
    if config.console_output:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)


def create_sample_config():
    """Create a sample configuration file"""
    config_manager = ConfigManager()
    sample_config = PromptInjectorConfig.from_dict(config_manager._get_default_config())
    
    # Add sample API keys (to be replaced by user)
    sample_config.api.target_api_key = "your-target-model-api-key-here"
    sample_config.api.analyzer_api_key = "your-analyzer-model-api-key-here"
    
    config_manager.save_config(sample_config)
    print(f"Sample configuration created: {config_manager.config_file}")
    print("Please edit the file to add your actual API keys.")


def validate_api_keys(config: PromptInjectorConfig) -> bool:
    """Validate that API keys are properly configured"""
    if not config.api.target_api_key or config.api.target_api_key == "your-target-model-api-key-here":
        print("ERROR: Target API key not configured")
        return False
    
    if not config.api.analyzer_api_key or config.api.analyzer_api_key == "your-analyzer-model-api-key-here":
        print("ERROR: Analyzer API key not configured")
        return False
    
    return True


# Example usage and configuration loading
def load_configuration(config_file: Optional[str] = None) -> PromptInjectorConfig:
    """Load and validate configuration"""
    config_manager = ConfigManager(config_file)
    
    try:
        config = config_manager.load_config()
        
        # Setup logging
        setup_logging(config.logging)
        
        # Validate API keys
        if not validate_api_keys(config):
            raise ValueError("API keys not properly configured")
        
        logging.info("Configuration loaded successfully")
        return config
        
    except FileNotFoundError:
        print(f"Configuration file not found: {config_manager.config_file}")
        print("Creating sample configuration...")
        create_sample_config()
        raise
    
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        raise


if __name__ == "__main__":
    # Create sample configuration if run directly
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "create-sample":
        create_sample_config()
    else:
        try:
            config = load_configuration()
            print("Configuration loaded successfully!")
            print(f"Target model: {config.models.target_model}")
            print(f"Analyzer model: {config.models.analyzer_model}")
            print(f"Log level: {config.logging.level}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)