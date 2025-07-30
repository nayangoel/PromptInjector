#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Server for PromptInjector.
Turns the tool into an MCP server that can be integrated with AI agents.
"""

import asyncio
import json
import logging
import sys
from typing import Any, Dict, List, Optional, Sequence
from dataclasses import asdict
import argparse

from config import load_configuration, PromptInjectorConfig
from model_client import ModelClientFactory, BaseModelClient
from prompt_injector import InjectionType, TestResult
from static_prompts import get_static_prompts
from mcp_tester import MCPPromptTester


class MCPPromptInjectorServer:
    """MCP Server for PromptInjector - provides AI agents with prompt injection testing capabilities"""
    
    def __init__(self, config: PromptInjectorConfig):
        self.config = config
        self.tester = None
        self.orchestrator = None
        self.target_client = None
        self.analyzer_client = None
        
        # Initialize logging
        self.logger = logging.getLogger(__name__)
        
        # Available tools/capabilities for external analyzer agents
        self.tools = {
            "test_prompt": self._test_prompt,
            "test_static_prompts": self._test_static_prompts,
            "record_analysis": self._record_analysis,
            "get_test_results": self._get_test_results,
            "get_static_prompts": self._get_static_prompts,
            "set_target_endpoint": self._set_target_endpoint,
            "get_injection_types": self._get_injection_types,
            "get_test_status": self._get_test_status,
            "clear_results": self._clear_results
        }
    
    async def initialize(self):
        """Initialize the MCP server components"""
        try:
            # Create target client (required)
            self.target_client = ModelClientFactory.create_client(
                client_type=self.config.api.target.type,
                endpoint_url=self.config.api.target.endpoint_url,
                api_key=self.config.api.target.api_key,
                model=self.config.api.target.model,
                headers=self.config.api.target.headers,
                timeout=self.config.api.target.timeout,
                max_retries=self.config.api.target.max_retries
            )
            
            # Analyzer client is optional - external MCP agent (like Claude Code) acts as analyzer
            self.analyzer_client = None
            try:
                if (self.config.api.analyzer and 
                    self.config.api.analyzer.api_key and 
                    self.config.api.analyzer.api_key != "dummy-key"):
                    self.analyzer_client = ModelClientFactory.create_client(
                        client_type=self.config.api.analyzer.type,
                        endpoint_url=self.config.api.analyzer.endpoint_url,
                        api_key=self.config.api.analyzer.api_key,
                        model=self.config.api.analyzer.model,
                        headers=self.config.api.analyzer.headers,
                        timeout=self.config.api.analyzer.timeout,
                        max_retries=self.config.api.analyzer.max_retries
                    )
            except Exception as e:
                self.logger.info(f"No analyzer client configured (external agent will be analyzer): {e}")
            
            # Initialize lightweight tester for MCP mode (no internal analyzer needed)
            self.tester = MCPPromptTester(self.target_client, self.config.to_dict())
            
            self.logger.info("MCP PromptInjector Server initialized successfully")
            self.logger.info("External MCP agent will act as the analyzer")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize MCP server: {e}")
            raise
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP requests from external agents"""
        try:
            tool_name = request.get("tool")
            params = request.get("params", {})
            
            # Log incoming request from AI agent
            self.logger.info(f"=== MCP REQUEST FROM AI AGENT ===")
            self.logger.info(f"Tool: {tool_name}")
            self.logger.info(f"Parameters: {json.dumps(params, indent=2)}")
            
            if tool_name not in self.tools:
                error_response = {
                    "error": f"Unknown tool: {tool_name}",
                    "available_tools": list(self.tools.keys())
                }
                self.logger.error(f"Tool not found: {tool_name}")
                return error_response
            
            # Execute the requested tool
            result = await self.tools[tool_name](params)
            
            # Log the response being sent back to AI agent
            self.logger.info(f"=== MCP RESPONSE TO AI AGENT ===")
            self.logger.info(f"Tool: {tool_name}")
            self.logger.info(f"Success: True")
            self.logger.info(f"Response: {json.dumps(result, indent=2)[:1000]}{'...' if len(json.dumps(result, indent=2)) > 1000 else ''}")
            
            return {
                "success": True,
                "result": result,
                "tool": tool_name
            }
            
        except Exception as e:
            self.logger.error(f"Error handling request: {e}")
            error_response = {
                "error": str(e),
                "success": False
            }
            self.logger.error(f"=== MCP ERROR RESPONSE ===")
            self.logger.error(f"Error: {json.dumps(error_response, indent=2)}")
            return error_response
    
    async def _test_prompt(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single prompt and return raw data for external analysis"""
        prompt = params.get("prompt", "")
        injection_type_str = params.get("injection_type", "jailbreak")
        
        if not prompt:
            return {"error": "Prompt is required"}
        
        try:
            injection_type = InjectionType(injection_type_str.lower())
        except ValueError:
            injection_type = InjectionType.JAILBREAK
        
        # Test prompt and return raw data
        result_data = await self.tester.test_prompt(prompt, injection_type)
        return result_data
    
    async def _test_static_prompts(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get static prompts for testing - external agent analyzes results"""
        count = params.get("count", 10)
        injection_types = params.get("injection_types", None)
        test_immediately = params.get("test_immediately", True)
        
        # Get batch of static prompts
        prompts = self.tester.get_static_prompts_batch(count, injection_types)
        
        if test_immediately:
            # Test all prompts and return raw data for analysis
            max_concurrent = params.get("max_concurrent", 2)
            results = await self.tester.test_multiple_prompts(prompts, max_concurrent)
            
            return {
                "total_tests": len(results),
                "results": results,
                "message": "Tests completed. Use 'record_analysis' to record your analysis of each result."
            }
        else:
            # Just return the prompts for manual testing
            return {
                "prompts": [
                    {"prompt": p, "injection_type": t.value} 
                    for p, t in prompts
                ],
                "count": len(prompts)
            }
    
    async def _record_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Record analysis result from external analyzer"""
        result_data = params.get("result_data", {})
        success = params.get("success", False)
        confidence = params.get("confidence", 0.0)
        reasoning = params.get("reasoning", "")
        
        if not result_data:
            return {"error": "result_data is required"}
        
        self.tester.record_analysis_result(result_data, success, confidence, reasoning)
        
        return {
            "message": "Analysis recorded successfully",
            "total_analyzed": len(self.tester.test_results)
        }
    
    async def _get_test_results(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get current test results and analysis"""
        include_summary = params.get("include_summary", True)
        recent_count = params.get("recent_count", 10)
        
        results = {
            "recent_results": self.tester.get_recent_results(recent_count),
            "total_tests": len(self.tester.test_results)
        }
        
        if include_summary:
            results["summary"] = self.tester.generate_test_summary()
        
        return results
    
    async def _get_static_prompts(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get static prompts without testing them"""
        count = params.get("count", 20)
        injection_types = params.get("injection_types", None)
        
        prompts = self.tester.get_static_prompts_batch(count, injection_types)
        
        return {
            "prompts": [
                {"prompt": p, "injection_type": t.value} 
                for p, t in prompts
            ],
            "count": len(prompts)
        }
    
    async def _clear_results(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Clear all test results"""
        self.tester.clear_results()
        return {"message": "All test results cleared"}
    
    
    async def _set_target_endpoint(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Set or update the target endpoint for testing"""
        endpoint_config = params.get("endpoint", {})
        
        required_fields = ["type", "endpoint_url"]
        for field in required_fields:
            if field not in endpoint_config:
                return {"error": f"Missing required field: {field}"}
        
        try:
            # Create new target client
            new_target_client = ModelClientFactory.create_client(**endpoint_config)
            
            # Test connection
            connection_ok = await new_target_client.test_connection()
            if not connection_ok:
                return {"error": "Failed to connect to target endpoint"}
            
            # Update target client
            self.target_client = new_target_client
            
            # Update components
            self.tester = PromptInjectionTester(self.target_client, self.analyzer_client, self.config.to_dict())
            self.orchestrator = TestOrchestrator(self.target_client, self.analyzer_client, self.config.to_dict())
            
            return {
                "message": "Target endpoint updated successfully",
                "endpoint": {
                    "type": endpoint_config["type"],
                    "url": endpoint_config["endpoint_url"],
                    "model": endpoint_config.get("model", "default")
                }
            }
            
        except Exception as e:
            return {"error": f"Failed to set target endpoint: {str(e)}"}
    
    async def _get_injection_types(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get available injection types"""
        return {
            "injection_types": [
                {
                    "name": injection_type.value,
                    "description": self._get_injection_description(injection_type)
                }
                for injection_type in InjectionType
            ]
        }
    
    async def _get_test_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get current testing status"""
        if not self.tester:
            return {"status": "not_initialized"}
        
        status = {
            "status": "ready",
            "mode": "external_analyzer",
            "total_tests_run": len(self.tester.test_results),
            "target_endpoint": {
                "type": self.config.api.target.type,
                "model": self.config.api.target.model,
                "url": self.config.api.target.endpoint_url
            },
            "analyzer": "external_mcp_agent",
            "available_tools": list(self.tools.keys())
        }
        
        if self.tester.test_results:
            recent_summary = self.tester.generate_test_summary()
            status["recent_summary"] = recent_summary["summary"]
        
        return status
    
    def _get_injection_description(self, injection_type: InjectionType) -> str:
        """Get description for injection type"""
        descriptions = {
            InjectionType.ROLE_MANIPULATION: "Attempts to change AI's perceived role",
            InjectionType.INSTRUCTION_OVERRIDE: "Tries to supersede original instructions",
            InjectionType.CONTEXT_ESCAPE: "Attempts to break conversational context",
            InjectionType.SYSTEM_PROMPT_LEAK: "Tries to reveal system instructions",
            InjectionType.JAILBREAK: "Bypasses safety measures through hypotheticals",
            InjectionType.DATA_EXTRACTION: "Attempts to retrieve training data",
            InjectionType.BEHAVIOR_MODIFICATION: "Tries to alter AI behavior patterns"
        }
        return descriptions.get(injection_type, "Unknown injection type")
    
    def _get_recommendations(self, vulnerabilities: List[Dict[str, Any]]) -> List[str]:
        """Get security recommendations based on vulnerabilities"""
        recommendations = []
        
        for vuln in vulnerabilities:
            vuln_type = vuln["type"]
            if vuln_type == "role_manipulation":
                recommendations.append("Implement strict role validation and context awareness")
            elif vuln_type == "instruction_override":
                recommendations.append("Add instruction integrity checks and override prevention")
            elif vuln_type == "context_escape":
                recommendations.append("Strengthen context boundaries and input sanitization")
            elif vuln_type == "system_prompt_leak":
                recommendations.append("Implement system prompt protection and access controls")
            elif vuln_type == "jailbreak":
                recommendations.append("Enhance safety filters and hypothetical scenario detection")
            elif vuln_type == "data_extraction":
                recommendations.append("Implement training data protection and access restrictions")
            elif vuln_type == "behavior_modification":
                recommendations.append("Add behavior consistency checks and modification detection")
        
        if not recommendations:
            recommendations.append("No critical vulnerabilities detected. Continue monitoring.")
        
        return recommendations


async def run_mcp_server(config_file: Optional[str] = None, port: int = 8000):
    """Run the MCP server"""
    try:
        # Load configuration
        config = load_configuration(config_file)
        
        # Create and initialize server
        server = MCPPromptInjectorServer(config)
        await server.initialize()
        
        print(f"MCP PromptInjector Server initialized successfully")
        print(f"Available tools: {list(server.tools.keys())}")
        print("Server ready to handle requests...")
        
        # Simple request handler for demonstration
        # In a real MCP implementation, this would integrate with the MCP protocol
        while True:
            try:
                # For now, we'll just keep the server alive
                # Real MCP integration would handle actual protocol messages
                await asyncio.sleep(1)
                
            except KeyboardInterrupt:
                print("\nShutting down MCP server...")
                break
                
    except Exception as e:
        print(f"Failed to start MCP server: {e}")
        sys.exit(1)


# Simple JSON-RPC over stdin/stdout for MCP compatibility
async def handle_stdin_requests(config_file: Optional[str] = None):
    """Handle MCP requests from stdin (JSON-RPC format)"""
    
    config = load_configuration(config_file)
    server = MCPPromptInjectorServer(config)
    await server.initialize()
    
    # Read requests from stdin
    while True:
        try:
            line = input()
            if not line:
                continue
                
            request = json.loads(line)
            
            # Handle MCP initialization
            if request.get("method") == "initialize":
                # Respond to initialize request
                response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": "PromptInjector MCP Server",
                            "version": "1.0.0"
                        }
                    }
                }
                print(json.dumps(response), flush=True)
                
            elif request.get("method") == "tools/list":
                # List available tools with proper schemas
                tools = [
                    {
                        "name": "test_prompt",
                        "description": "Test a single custom prompt for injection vulnerabilities",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The prompt to test for injection vulnerabilities"
                                },
                                "injection_type": {
                                    "type": "string",
                                    "description": "Type of injection to test",
                                    "enum": ["jailbreak", "role_manipulation", "instruction_override", "context_escape", "system_prompt_leak", "data_extraction", "behavior_modification"],
                                    "default": "jailbreak"
                                }
                            },
                            "required": ["prompt"]
                        }
                    },
                    {
                        "name": "test_static_prompts",
                        "description": "Run batch tests using predefined prompt injection test cases",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "count": {
                                    "type": "integer",
                                    "description": "Number of tests to run",
                                    "default": 10,
                                    "minimum": 1,
                                    "maximum": 50
                                },
                                "injection_types": {
                                    "type": "array",
                                    "description": "Specific injection types to test (optional)",
                                    "items": {
                                        "type": "string",
                                        "enum": ["jailbreak", "role_manipulation", "instruction_override", "context_escape", "system_prompt_leak", "data_extraction", "behavior_modification"]
                                    }
                                },
                                "test_immediately": {
                                    "type": "boolean",
                                    "description": "Whether to test prompts immediately",
                                    "default": True
                                },
                                "max_concurrent": {
                                    "type": "integer",
                                    "description": "Maximum concurrent tests",
                                    "default": 2,
                                    "minimum": 1,
                                    "maximum": 10
                                }
                            },
                            "required": []
                        }
                    },
                    {
                        "name": "get_test_results",
                        "description": "Get recent test results and analysis summary",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "include_summary": {
                                    "type": "boolean",
                                    "description": "Include statistical summary",
                                    "default": True
                                },
                                "recent_count": {
                                    "type": "integer",
                                    "description": "Number of recent results to return",
                                    "default": 10,
                                    "minimum": 1,
                                    "maximum": 100
                                }
                            },
                            "required": []
                        }
                    },
                    {
                        "name": "record_analysis",
                        "description": "Record analysis result from external analyzer",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "result_data": {
                                    "type": "object",
                                    "description": "The test result data to analyze"
                                },
                                "success": {
                                    "type": "boolean",
                                    "description": "Whether the injection was successful"
                                },
                                "confidence": {
                                    "type": "number",
                                    "description": "Confidence level (0.0 to 1.0)",
                                    "minimum": 0.0,
                                    "maximum": 1.0
                                },
                                "reasoning": {
                                    "type": "string",
                                    "description": "Explanation of the analysis",
                                    "default": ""
                                }
                            },
                            "required": ["result_data", "success", "confidence"]
                        }
                    },
                    {
                        "name": "get_injection_types",
                        "description": "Get list of available injection types with descriptions",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    },
                    {
                        "name": "get_test_status",
                        "description": "Get current testing status and configuration",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    },
                    {
                        "name": "clear_results",
                        "description": "Clear all stored test results",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                            "required": []
                        }
                    },
                    {
                        "name": "set_target_endpoint",
                        "description": "Update the target endpoint configuration",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "endpoint": {
                                    "type": "object",
                                    "description": "New endpoint configuration",
                                    "properties": {
                                        "type": {"type": "string"},
                                        "endpoint_url": {"type": "string"},
                                        "api_key": {"type": "string"},
                                        "model": {"type": "string"}
                                    },
                                    "required": ["type", "endpoint_url"]
                                }
                            },
                            "required": ["endpoint"]
                        }
                    },
                    {
                        "name": "get_static_prompts",
                        "description": "Get static prompts without testing them",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "count": {
                                    "type": "integer",
                                    "description": "Number of prompts to return",
                                    "default": 20,
                                    "minimum": 1,
                                    "maximum": 100
                                },
                                "injection_types": {
                                    "type": "array",
                                    "description": "Filter by specific injection types",
                                    "items": {
                                        "type": "string",
                                        "enum": ["jailbreak", "role_manipulation", "instruction_override", "context_escape", "system_prompt_leak", "data_extraction", "behavior_modification"]
                                    }
                                }
                            },
                            "required": []
                        }
                    }
                ]
                
                response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": {
                        "tools": tools
                    }
                }
                print(json.dumps(response), flush=True)
            
            elif request.get("method") == "tools/call":
                params = request.get("params", {})
                tool_name = params.get("name")
                tool_params = params.get("arguments", {})
                
                # Execute tool
                result = await server.handle_request({
                    "tool": tool_name,
                    "params": tool_params
                })
                
                # Send response
                response = {
                    "jsonrpc": "2.0",
                    "id": request.get("id"),
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, indent=2)
                            }
                        ]
                    }
                }
                print(json.dumps(response), flush=True)
            
        except EOFError:
            break
        except Exception as e:
            error_response = {
                "jsonrpc": "2.0",
                "id": request.get("id") if 'request' in locals() else None,
                "error": {
                    "code": -32603,
                    "message": str(e)
                }
            }
            print(json.dumps(error_response), flush=True)


def main():
    """Main entry point for MCP server"""
    parser = argparse.ArgumentParser(description="PromptInjector MCP Server")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--stdio", action="store_true", help="Use stdio for MCP communication")
    
    args = parser.parse_args()
    
    if args.stdio:
        asyncio.run(handle_stdin_requests(args.config))
    else:
        asyncio.run(run_mcp_server(args.config, args.port))


if __name__ == "__main__":
    main()