#!/usr/bin/env python3
"""
Setup script for PromptInjector - helps users configure the tool quickly.
"""

import json
import os
import sys
from pathlib import Path


def create_config_interactive():
    """Interactive configuration setup"""
    
    print("🔒 PromptInjector Setup - Model-Agnostic AI Security Testing")
    print("=" * 60)
    print()
    
    # Choose configuration type
    print("Select your configuration type:")
    print("1. OpenAI (GPT-3.5/GPT-4)")
    print("2. Anthropic (Claude)")
    print("3. Local Ollama")
    print("4. Mixed Providers")
    print("5. Custom HTTP API")
    
    choice = input("\nEnter your choice (1-5): ").strip()
    
    config = {
        "models": {
            "max_tokens": 500,
            "temperature": 0.7
        },
        "testing": {
            "concurrent_tests": 3,
            "rate_limit_delay": 1.0,
            "default_static_tests": 100,
            "default_adaptive_tests": 50
        },
        "logging": {
            "level": "INFO",
            "log_file": "prompt_injector.log",
            "console_output": True
        },
        "security": {
            "enable_content_filtering": True,
            "log_sensitive_data": False,
            "anonymize_results": True
        }
    }
    
    if choice == "1":
        config.update(setup_openai())
    elif choice == "2":
        config.update(setup_anthropic())
    elif choice == "3":
        config.update(setup_ollama())
    elif choice == "4":
        config.update(setup_mixed())
    elif choice == "5":
        config.update(setup_custom_http())
    else:
        print("Invalid choice. Defaulting to OpenAI configuration.")
        config.update(setup_openai())
    
    return config


def setup_openai():
    """Setup OpenAI configuration"""
    print("\n🤖 OpenAI Configuration")
    print("-" * 30)
    
    target_key = input("Enter your target model API key: ").strip()
    analyzer_key = input("Enter your analyzer model API key (or press Enter to use same): ").strip()
    if not analyzer_key:
        analyzer_key = target_key
    
    target_model = input("Target model (default: gpt-3.5-turbo): ").strip() or "gpt-3.5-turbo"
    analyzer_model = input("Analyzer model (default: gpt-4): ").strip() or "gpt-4"
    
    return {
        "api": {
            "target": {
                "type": "openai",
                "endpoint_url": "https://api.openai.com/v1/chat/completions",
                "api_key": target_key,
                "model": target_model,
                "timeout": 30,
                "max_retries": 3
            },
            "analyzer": {
                "type": "openai",
                "endpoint_url": "https://api.openai.com/v1/chat/completions",
                "api_key": analyzer_key,
                "model": analyzer_model,
                "timeout": 30,
                "max_retries": 3
            }
        },
        "models": {
            "target_model": target_model,
            "analyzer_model": analyzer_model
        }
    }


def setup_anthropic():
    """Setup Anthropic configuration"""
    print("\n🧠 Anthropic Claude Configuration")
    print("-" * 35)
    
    api_key = input("Enter your Anthropic API key: ").strip()
    target_model = input("Target model (default: claude-3-sonnet-20240229): ").strip() or "claude-3-sonnet-20240229"
    analyzer_model = input("Analyzer model (default: claude-3-opus-20240229): ").strip() or "claude-3-opus-20240229"
    
    return {
        "api": {
            "target": {
                "type": "anthropic",
                "endpoint_url": "https://api.anthropic.com/v1/messages",
                "api_key": api_key,
                "model": target_model,
                "timeout": 30,
                "max_retries": 3
            },
            "analyzer": {
                "type": "anthropic", 
                "endpoint_url": "https://api.anthropic.com/v1/messages",
                "api_key": api_key,
                "model": analyzer_model,
                "timeout": 30,
                "max_retries": 3
            }
        },
        "models": {
            "target_model": target_model,
            "analyzer_model": analyzer_model
        }
    }


def setup_ollama():
    """Setup Ollama local configuration"""
    print("\n🏠 Local Ollama Configuration")
    print("-" * 30)
    
    base_url = input("Ollama base URL (default: http://localhost:11434): ").strip() or "http://localhost:11434"
    target_model = input("Target model (default: llama2): ").strip() or "llama2"
    analyzer_model = input("Analyzer model (default: codellama): ").strip() or "codellama"
    
    return {
        "api": {
            "target": {
                "type": "ollama",
                "endpoint_url": f"{base_url}/api/generate",
                "api_key": "",
                "model": target_model,
                "timeout": 60,
                "max_retries": 3
            },
            "analyzer": {
                "type": "ollama",
                "endpoint_url": f"{base_url}/api/generate",
                "api_key": "",
                "model": analyzer_model,
                "timeout": 60,
                "max_retries": 3
            }
        },
        "models": {
            "target_model": target_model,
            "analyzer_model": analyzer_model
        }
    }


def setup_mixed():
    """Setup mixed providers configuration"""
    print("\n🔀 Mixed Providers Configuration")
    print("-" * 35)
    
    print("\nTarget Model Configuration:")
    target_type = input("Target type (openai/anthropic/ollama/http): ").strip().lower()
    target_url = input("Target endpoint URL: ").strip()
    target_key = input("Target API key (or press Enter if not needed): ").strip()
    target_model = input("Target model name: ").strip()
    
    print("\nAnalyzer Model Configuration:")
    analyzer_type = input("Analyzer type (openai/anthropic/ollama/http): ").strip().lower()
    analyzer_url = input("Analyzer endpoint URL: ").strip()
    analyzer_key = input("Analyzer API key (or press Enter if not needed): ").strip()
    analyzer_model = input("Analyzer model name: ").strip()
    
    return {
        "api": {
            "target": {
                "type": target_type,
                "endpoint_url": target_url,
                "api_key": target_key,
                "model": target_model,
                "timeout": 30,
                "max_retries": 3
            },
            "analyzer": {
                "type": analyzer_type,
                "endpoint_url": analyzer_url,
                "api_key": analyzer_key,
                "model": analyzer_model,
                "timeout": 30,
                "max_retries": 3
            }
        },
        "models": {
            "target_model": target_model,
            "analyzer_model": analyzer_model
        }
    }


def setup_custom_http():
    """Setup custom HTTP API configuration"""
    print("\n🌐 Custom HTTP API Configuration")
    print("-" * 35)
    
    target_url = input("Target API endpoint URL: ").strip()
    target_key = input("Target API key: ").strip()
    target_model = input("Target model name: ").strip()
    
    analyzer_url = input("Analyzer API endpoint URL: ").strip()
    analyzer_key = input("Analyzer API key: ").strip()
    analyzer_model = input("Analyzer model name: ").strip()
    
    # Optional custom headers
    print("\nOptional: Add custom headers (press Enter to skip)")
    headers = {}
    while True:
        header_name = input("Header name (or press Enter to finish): ").strip()
        if not header_name:
            break
        header_value = input(f"Value for '{header_name}': ").strip()
        headers[header_name] = header_value
    
    target_config = {
        "type": "http",
        "endpoint_url": target_url,
        "api_key": target_key,
        "model": target_model,
        "timeout": 30,
        "max_retries": 3
    }
    
    analyzer_config = {
        "type": "http",
        "endpoint_url": analyzer_url,
        "api_key": analyzer_key,
        "model": analyzer_model,
        "timeout": 30,
        "max_retries": 3
    }
    
    if headers:
        target_config["headers"] = headers
        analyzer_config["headers"] = headers
    
    return {
        "api": {
            "target": target_config,
            "analyzer": analyzer_config
        },
        "models": {
            "target_model": target_model,
            "analyzer_model": analyzer_model
        }
    }


def save_config(config, filename="prompt_injector_config.json"):
    """Save configuration to file"""
    try:
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"\n✅ Configuration saved to {filename}")
        return True
    except Exception as e:
        print(f"\n❌ Error saving configuration: {e}")
        return False


def show_next_steps():
    """Show next steps after configuration"""
    print("\n🎉 Setup Complete!")
    print("=" * 40)
    print("\nNext steps:")
    print("1. Test your configuration:")
    print("   python main.py --quick")
    print()
    print("2. Run a full security audit:")
    print("   python main.py --full")
    print()
    print("3. Start the MCP server:")
    print("   python mcp_server.py --stdio")
    print()
    print("4. View example configurations:")
    print("   ls example_configs/")
    print()
    print("For more options, see the README.md file.")


def main():
    """Main setup function"""
    try:
        # Check if config already exists
        if os.path.exists("prompt_injector_config.json"):
            overwrite = input("Configuration file already exists. Overwrite? (y/N): ").strip().lower()
            if overwrite != 'y':
                print("Setup cancelled.")
                return
        
        # Create configuration
        config = create_config_interactive()
        
        # Save configuration
        if save_config(config):
            show_next_steps()
        
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()