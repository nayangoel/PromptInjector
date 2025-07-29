#!/usr/bin/env python3
"""
PromptInjector - Main entry point for the defensive prompt injection testing tool.
A comprehensive security testing framework for AI systems.
"""

import asyncio
import argparse
import sys
import json
from pathlib import Path
from typing import Optional

from openai import OpenAI
from config import load_configuration, create_sample_config, PromptInjectorConfig
from test_orchestrator import TestOrchestrator
from prompt_injector import InjectionType
from static_prompts import get_static_prompts


def create_progress_reporter(verbose: bool = False):
    """Create a progress reporting callback"""
    def progress_callback(progress_data):
        if verbose:
            print(f"[{progress_data['test_index']:3d}/{progress_data['total_tests']:3d}] "
                  f"{progress_data['injection_type']:20s} - "
                  f"Success: {progress_data['success']:<5} "
                  f"Confidence: {progress_data['confidence']:.2f} "
                  f"({'Adaptive' if progress_data.get('is_adaptive') else 'Static':8s})")
        else:
            # Simple progress bar
            progress = progress_data['test_index'] / progress_data['total_tests']
            bar_length = 40
            filled_length = int(bar_length * progress)
            bar = '█' * filled_length + '-' * (bar_length - filled_length)
            print(f"\rProgress: |{bar}| {progress:.1%} Complete", end='', flush=True)
            if progress_data['test_index'] == progress_data['total_tests']:
                print()  # New line at completion
    
    return progress_callback


async def run_quick_test(config: PromptInjectorConfig, verbose: bool = False):
    """Run a quick test with a small subset of prompts"""
    print("Running quick test (10 static + 5 adaptive prompts)...")
    
    # Initialize clients
    target_client = OpenAI(
        api_key=config.api.target_api_key,
        base_url=config.api.target_base_url
    )
    analyzer_client = OpenAI(
        api_key=config.api.analyzer_api_key,
        base_url=config.api.analyzer_base_url
    )
    
    # Create orchestrator
    orchestrator = TestOrchestrator(target_client, analyzer_client, config.to_dict())
    orchestrator.add_progress_callback(create_progress_reporter(verbose))
    
    # Create and run campaign
    campaign = await orchestrator.create_campaign(
        name="quick_test",
        description="Quick security test with minimal prompts",
        static_tests=10,
        adaptive_tests=5,
        target_model=config.models.target_model
    )
    
    completed_campaign = await orchestrator.run_campaign(
        campaign,
        concurrent_tests=min(2, config.testing.concurrent_tests),
        rate_limit_delay=config.testing.rate_limit_delay
    )
    
    # Generate and display report
    report = orchestrator.generate_campaign_report(completed_campaign)
    print_summary_report(report)
    
    return completed_campaign


async def run_full_test(config: PromptInjectorConfig, 
                       static_tests: int,
                       adaptive_tests: int,
                       verbose: bool = False):
    """Run a comprehensive test campaign"""
    print(f"Running full test ({static_tests} static + {adaptive_tests} adaptive prompts)...")
    
    # Initialize clients
    target_client = OpenAI(
        api_key=config.api.target_api_key,
        base_url=config.api.target_base_url
    )
    analyzer_client = OpenAI(
        api_key=config.api.analyzer_api_key,
        base_url=config.api.analyzer_base_url
    )
    
    # Create orchestrator
    orchestrator = TestOrchestrator(target_client, analyzer_client, config.to_dict())
    orchestrator.add_progress_callback(create_progress_reporter(verbose))
    
    # Create and run campaign
    campaign = await orchestrator.create_campaign(
        name="full_security_audit",
        description="Comprehensive security audit with static and adaptive testing",
        static_tests=static_tests,
        adaptive_tests=adaptive_tests,
        target_model=config.models.target_model
    )
    
    completed_campaign = await orchestrator.run_campaign(
        campaign,
        concurrent_tests=config.testing.concurrent_tests,
        rate_limit_delay=config.testing.rate_limit_delay
    )
    
    # Generate and save report
    report = orchestrator.generate_campaign_report(completed_campaign)
    print_detailed_report(report)
    
    # Save results
    orchestrator.save_campaign(completed_campaign)
    
    return completed_campaign


def print_summary_report(report: dict):
    """Print a concise summary report"""
    print("\n" + "="*60)
    print("SECURITY TEST SUMMARY")
    print("="*60)
    
    summary = report['summary']
    print(f"Target Model: {report['campaign_info']['target_model']}")
    print(f"Total Tests: {summary['total_tests']}")
    print(f"Successful Injections: {summary['successful_injections']}")
    print(f"Success Rate: {summary['overall_success_rate']:.1%}")
    print(f"Average Confidence: {summary['avg_confidence']:.2f}")
    
    print("\nResults by Injection Type:")
    print("-" * 60)
    for injection_type, stats in report['by_injection_type'].items():
        print(f"{injection_type:25s}: {stats['successful']:3d}/{stats['total']:3d} "
              f"({stats['success_rate']:.1%}) - Confidence: {stats['avg_confidence']:.2f}")
    
    if report.get('vulnerabilities_found'):
        print(f"\nVulnerabilities Found: {len(report['vulnerabilities_found'])}")
        for vuln in report['vulnerabilities_found']:
            severity = vuln['severity']
            color = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(severity, '⚪')
            print(f"  {color} {vuln['injection_type']:20s} - {severity:8s} ({vuln['success_rate']:.1%})")


def print_detailed_report(report: dict):
    """Print a detailed security report"""
    print("\n" + "="*80)
    print("DETAILED SECURITY ASSESSMENT REPORT")
    print("="*80)
    
    # Campaign information
    campaign_info = report['campaign_info']
    print(f"Campaign: {campaign_info['name']}")
    print(f"Description: {campaign_info['description']}")
    print(f"Target Model: {campaign_info['target_model']}")
    print(f"Status: {campaign_info['status']}")
    if campaign_info.get('duration_seconds'):
        print(f"Duration: {campaign_info['duration_seconds']:.1f} seconds")
    
    # Summary statistics
    print("\nSUMMARY STATISTICS")
    print("-" * 40)
    summary = report['summary']
    print(f"Total Tests Executed: {summary['total_tests']}")
    print(f"Successful Injections: {summary['successful_injections']}")
    print(f"Overall Success Rate: {summary['overall_success_rate']:.1%}")
    print(f"Average Response Time: {summary['avg_response_time']:.2f}s")
    print(f"Average Confidence: {summary['avg_confidence']:.2f}")
    
    # Adaptive vs Static comparison
    if report.get('adaptive_vs_static'):
        print("\nADAPTIVE vs STATIC COMPARISON")
        print("-" * 40)
        comparison = report['adaptive_vs_static']
        if 'adaptive' in comparison and 'static' in comparison:
            adaptive = comparison['adaptive']
            static = comparison['static']
            print(f"Static Tests:   {static['count']:3d} tests, {static['success_rate']:.1%} success rate")
            print(f"Adaptive Tests: {adaptive['count']:3d} tests, {adaptive['success_rate']:.1%} success rate")
            
            if adaptive['success_rate'] > static['success_rate']:
                print("📈 Adaptive prompts were more effective than static prompts")
            else:
                print("📊 Static prompts performed as well as or better than adaptive prompts")
    
    # Detailed results by injection type
    print("\nDETAILED RESULTS BY INJECTION TYPE")
    print("-" * 80)
    for injection_type, stats in report['by_injection_type'].items():
        print(f"\n{injection_type.upper().replace('_', ' ')}")
        print(f"  Tests: {stats['total']:3d} | Successful: {stats['successful']:3d} | "
              f"Success Rate: {stats['success_rate']:.1%}")
        print(f"  Avg Confidence: {stats['avg_confidence']:.2f} | "
              f"Avg Response Time: {stats['avg_response_time']:.2f}s")
    
    # Vulnerability assessment
    if report.get('vulnerabilities_found'):
        print("\nVULNERABILITY ASSESSMENT")
        print("-" * 80)
        for vuln in report['vulnerabilities_found']:
            severity_colors = {
                'CRITICAL': '🔴 CRITICAL',
                'HIGH': '🟠 HIGH',
                'MEDIUM': '🟡 MEDIUM',
                'LOW': '🟢 LOW'
            }
            severity_display = severity_colors.get(vuln['severity'], vuln['severity'])
            
            print(f"\n{severity_display}: {vuln['injection_type'].replace('_', ' ').title()}")
            print(f"  Success Rate: {vuln['success_rate']:.1%}")
            print(f"  Recommendation: {vuln['recommendation']}")
    
    # Top successful prompts
    if report.get('top_successful_prompts'):
        print("\nTOP SUCCESSFUL INJECTION ATTEMPTS")
        print("-" * 80)
        for i, prompt_info in enumerate(report['top_successful_prompts'][:5], 1):
            prompt_type = "Adaptive" if prompt_info['is_adaptive'] else "Static"
            print(f"\n{i}. [{prompt_type}] {prompt_info['injection_type'].replace('_', ' ').title()}")
            print(f"   Confidence: {prompt_info['confidence']:.2f}")
            print(f"   Prompt: {prompt_info['prompt']}")


async def run_custom_prompts(config: PromptInjectorConfig, prompts_file: str, verbose: bool = False):
    """Run tests with custom prompts from a file"""
    try:
        with open(prompts_file, 'r') as f:
            custom_prompts = json.load(f)
    except Exception as e:
        print(f"Error loading custom prompts: {e}")
        return None
    
    print(f"Running custom prompts test ({len(custom_prompts)} prompts)...")
    
    # Initialize clients
    target_client = OpenAI(
        api_key=config.api.target_api_key,
        base_url=config.api.target_base_url
    )
    analyzer_client = OpenAI(
        api_key=config.api.analyzer_api_key,
        base_url=config.api.analyzer_base_url
    )
    
    # Create tester directly
    from prompt_injector import PromptInjectionTester
    tester = PromptInjectionTester(target_client, analyzer_client, config.to_dict())
    
    # Run custom prompts
    for i, prompt_data in enumerate(custom_prompts):
        prompt = prompt_data.get('prompt', '')
        injection_type_str = prompt_data.get('type', 'jailbreak')
        
        try:
            injection_type = InjectionType(injection_type_str.lower())
        except ValueError:
            injection_type = InjectionType.JAILBREAK
        
        result = await tester.test_prompt(prompt, injection_type)
        
        if verbose:
            print(f"[{i+1:3d}/{len(custom_prompts):3d}] "
                  f"{injection_type.value:20s} - "
                  f"Success: {result.success:<5} "
                  f"Confidence: {result.confidence:.2f}")
    
    # Generate simple report
    report = tester.generate_report()
    print_summary_report({'campaign_info': {'target_model': config.models.target_model}, 
                         'summary': report['summary'], 
                         'by_injection_type': report['by_injection_type'],
                         'vulnerabilities_found': []})
    
    return tester.test_results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="PromptInjector - Defensive AI Security Testing Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --quick                          # Quick test with 15 prompts
  python main.py --full                           # Full test with default settings
  python main.py --static 50 --adaptive 25       # Custom test sizes
  python main.py --config my_config.json         # Use custom config file
  python main.py --custom prompts.json           # Test custom prompts
  python main.py --create-config                 # Create sample config file
        """
    )
    
    # Test mode options
    test_group = parser.add_mutually_exclusive_group(required=True)
    test_group.add_argument('--quick', action='store_true',
                           help='Run quick test (10 static + 5 adaptive prompts)')
    test_group.add_argument('--full', action='store_true',
                           help='Run full test with default settings')
    test_group.add_argument('--custom', metavar='FILE',
                           help='Run custom prompts from JSON file')
    test_group.add_argument('--create-config', action='store_true',
                           help='Create sample configuration file')
    
    # Test configuration
    parser.add_argument('--static', type=int, default=100,
                       help='Number of static prompts (default: 100)')
    parser.add_argument('--adaptive', type=int, default=50,
                       help='Number of adaptive prompts (default: 50)')
    parser.add_argument('--config', metavar='FILE',
                       help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Handle config creation
    if args.create_config:
        create_sample_config()
        return
    
    # Load configuration
    try:
        config = load_configuration(args.config)
    except Exception as e:
        print(f"Configuration error: {e}")
        print("Use --create-config to create a sample configuration file.")
        sys.exit(1)
    
    # Run appropriate test
    try:
        if args.quick:
            asyncio.run(run_quick_test(config, args.verbose))
        elif args.full:
            asyncio.run(run_full_test(config, args.static, args.adaptive, args.verbose))
        elif args.custom:
            asyncio.run(run_custom_prompts(config, args.custom, args.verbose))
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Test execution error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()