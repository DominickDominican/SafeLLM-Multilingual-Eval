"""
Command-line interface for SafeLLM Multilingual Evaluation.
"""

import os
import sys
import click
import logging
from pathlib import Path
from typing import Optional, List
import json

from .config import ConfigManager
from .evaluator import MultilingualEvaluator
from .visualizer import ResultVisualizer
from .models import ModelFactory


@click.group()
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def cli(ctx, config: Optional[str], verbose: bool):
    """SafeLLM Multilingual Evaluation Framework CLI."""
    ctx.ensure_object(dict)
    
    # Initialize configuration
    config_manager = ConfigManager(config)
    if verbose:
        config_manager.config.evaluation.log_level = "DEBUG"
    
    config_manager.setup_logging()
    
    ctx.obj['config_manager'] = config_manager
    ctx.obj['config'] = config_manager.config


@cli.command()
@click.option('--output', '-o', default='./config_template.yaml', 
              help='Output path for template configuration')
@click.pass_context
def init(ctx, output: str):
    """Initialize a new SafeLLM evaluation project with template configuration."""
    config_manager = ctx.obj['config_manager']
    
    try:
        config_manager.create_template_config(output)
        click.echo(f"✅ Template configuration created at: {output}")
        click.echo("\n📝 Next steps:")
        click.echo("1. Edit the configuration file to add your API keys")
        click.echo("2. Customize model and evaluation settings")
        click.echo("3. Run 'safellm-eval validate' to check your configuration")
        
    except Exception as e:
        click.echo(f"❌ Error creating template: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.pass_context
def validate(ctx):
    """Validate current configuration."""
    config_manager = ctx.obj['config_manager']
    
    click.echo("🔍 Validating configuration...")
    
    issues = config_manager.validate_config()
    
    # Display results
    if issues['errors']:
        click.echo("\n❌ Errors:")
        for error in issues['errors']:
            click.echo(f"  • {error}")
    
    if issues['warnings']:
        click.echo("\n⚠️  Warnings:")
        for warning in issues['warnings']:
            click.echo(f"  • {warning}")
    
    if issues['info']:
        click.echo("\n📊 Configuration Summary:")
        for info in issues['info']:
            click.echo(f"  • {info}")
    
    if issues['errors']:
        click.echo("\n❌ Configuration validation failed")
        sys.exit(1)
    else:
        click.echo("\n✅ Configuration is valid")


@cli.command()
@click.argument('datasets', nargs=-1, required=True)
@click.option('--models', '-m', multiple=True, 
              help='Specific models to evaluate (default: all enabled)')
@click.option('--output', '-o', help='Output directory for results')
@click.option('--format', 'output_format', default='jsonl',
              type=click.Choice(['jsonl', 'csv', 'json']),
              help='Output format for results')
@click.option('--visualize/--no-visualize', default=True,
              help='Generate visualizations')
@click.option('--parallel', '-p', default=5, 
              help='Number of parallel workers')
@click.pass_context
def evaluate(ctx, datasets: tuple, models: tuple, output: Optional[str], 
            output_format: str, visualize: bool, parallel: int):
    """Run multilingual safety evaluation on specified datasets."""
    config = ctx.obj['config']
    config_manager = ctx.obj['config_manager']
    
    click.echo("🚀 Starting SafeLLM Multilingual Evaluation")
    
    # Setup output directory
    if output:
        config.evaluation.output_dir = output
    os.makedirs(config.evaluation.output_dir, exist_ok=True)
    
    # Validate datasets
    for dataset_path in datasets:
        if not os.path.exists(dataset_path):
            click.echo(f"❌ Dataset not found: {dataset_path}", err=True)
            sys.exit(1)
    
    # Setup models
    available_models = config_manager.get_enabled_models()
    if models:
        # Filter to specified models
        model_names = set(models)
        available_models = [m for m in available_models if m.name in model_names]
        
        if not available_models:
            click.echo("❌ No valid models found matching specified names", err=True)
            sys.exit(1)
    
    if not available_models:
        click.echo("❌ No enabled models with valid API keys found", err=True)
        sys.exit(1)
    
    click.echo(f"📋 Using {len(available_models)} models: {[m.name for m in available_models]}")
    click.echo(f"📂 Processing {len(datasets)} datasets")
    
    # Initialize evaluator
    evaluator = MultilingualEvaluator()
    evaluator.config["evaluation"]["max_workers"] = parallel
    
    # Convert model configs to evaluator format
    model_configs = [
        {"name": m.name, "provider": m.provider} 
        for m in available_models
    ]
    
    all_results = []
    
    # Process each dataset
    for dataset_path in datasets:
        click.echo(f"\n📊 Evaluating dataset: {os.path.basename(dataset_path)}")
        
        try:
            results = evaluator.evaluate_dataset(dataset_path, model_configs)
            all_results.extend(results)
            
            click.echo(f"✅ Completed {len(results)} evaluations")
            
        except Exception as e:
            click.echo(f"❌ Error evaluating {dataset_path}: {e}", err=True)
            continue
    
    if not all_results:
        click.echo("❌ No results generated", err=True)
        sys.exit(1)
    
    # Save results
    timestamp = click.DateTime().convert(None, None, None).strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(
        config.evaluation.output_dir, 
        f"evaluation_results_{timestamp}.{output_format}"
    )
    
    evaluator.results = all_results
    evaluator.save_results(results_file, output_format)
    
    click.echo(f"\n💾 Results saved to: {results_file}")
    
    # Generate summary
    summary = evaluator.get_summary_stats()
    click.echo(f"\n📈 Evaluation Summary:")
    click.echo(f"  • Total evaluations: {summary['total_evaluations']}")
    click.echo(f"  • Successful: {summary['successful_evaluations']}")
    click.echo(f"  • Failed: {summary['failed_evaluations']}")
    click.echo(f"  • Average safety score: {summary['average_safety_score']:.3f}")
    
    # Generate visualizations
    if visualize and config.visualization.enabled:
        click.echo("\n🎨 Generating visualizations...")
        
        try:
            visualizer = ResultVisualizer()
            viz_paths = visualizer.create_summary_report(
                [result.__dict__ if hasattr(result, '__dict__') else result for result in all_results],
                config.visualization.output_dir
            )
            
            click.echo("✅ Visualizations generated:")
            for viz_type, path in viz_paths.items():
                click.echo(f"  • {viz_type}: {path}")
                
        except Exception as e:
            click.echo(f"⚠️  Warning: Visualization generation failed: {e}")
    
    click.echo("\n🎉 Evaluation completed successfully!")


@cli.command()
@click.argument('results_file')
@click.option('--output-dir', '-o', default='./visualizations',
              help='Output directory for visualizations')
@click.option('--format', 'viz_format', multiple=True,
              type=click.Choice(['html', 'png', 'svg']),
              help='Visualization formats to generate')
@click.pass_context
def visualize(ctx, results_file: str, output_dir: str, viz_format: tuple):
    """Generate visualizations from existing evaluation results."""
    
    if not os.path.exists(results_file):
        click.echo(f"❌ Results file not found: {results_file}", err=True)
        sys.exit(1)
    
    click.echo(f"📊 Loading results from: {results_file}")
    
    # Load results
    try:
        if results_file.endswith('.jsonl'):
            import jsonlines
            results = []
            with jsonlines.open(results_file) as reader:
                for obj in reader:
                    results.append(obj)
        elif results_file.endswith('.json'):
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
        elif results_file.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(results_file)
            results = df.to_dict('records')
        else:
            click.echo("❌ Unsupported file format. Use .jsonl, .json, or .csv", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error loading results: {e}", err=True)
        sys.exit(1)
    
    click.echo(f"✅ Loaded {len(results)} evaluation results")
    
    # Generate visualizations
    try:
        visualizer = ResultVisualizer()
        viz_paths = visualizer.create_summary_report(results, output_dir)
        
        click.echo("\n🎨 Visualizations generated:")
        for viz_type, path in viz_paths.items():
            click.echo(f"  • {viz_type}: {path}")
            
    except Exception as e:
        click.echo(f"❌ Error generating visualizations: {e}", err=True)
        sys.exit(1)
    
    click.echo("\n✅ Visualization generation completed!")


@cli.command()
@click.pass_context
def list_models(ctx):
    """List available models and their status."""
    config = ctx.obj['config']
    
    click.echo("🤖 Available Models:")
    click.echo("-" * 50)
    
    for model in config.models:
        status = "✅ Enabled" if model.enabled else "❌ Disabled"
        api_key_status = "🔑 API Key Set" if model.api_key else "⚠️  No API Key"
        
        click.echo(f"Model: {model.name}")
        click.echo(f"  Provider: {model.provider}")
        click.echo(f"  Status: {status}")
        click.echo(f"  API Key: {api_key_status}")
        click.echo(f"  Temperature: {model.temperature}")
        click.echo(f"  Max Tokens: {model.max_tokens}")
        click.echo()


@cli.command()
@click.argument('dataset_path')
@click.pass_context
def inspect(ctx, dataset_path: str):
    """Inspect a dataset file and show statistics."""
    
    if not os.path.exists(dataset_path):
        click.echo(f"❌ Dataset not found: {dataset_path}", err=True)
        sys.exit(1)
    
    click.echo(f"🔍 Inspecting dataset: {dataset_path}")
    
    try:
        if dataset_path.endswith('.jsonl'):
            import jsonlines
            data = []
            with jsonlines.open(dataset_path) as reader:
                for obj in reader:
                    data.append(obj)
        else:
            click.echo("❌ Only .jsonl files are supported for inspection", err=True)
            sys.exit(1)
            
    except Exception as e:
        click.echo(f"❌ Error reading dataset: {e}", err=True)
        sys.exit(1)
    
    if not data:
        click.echo("⚠️  Dataset is empty")
        return
    
    # Generate statistics
    import pandas as pd
    df = pd.DataFrame(data)
    
    click.echo(f"\n📊 Dataset Statistics:")
    click.echo(f"  • Total prompts: {len(df)}")
    
    if 'language' in df.columns:
        lang_counts = df['language'].value_counts()
        click.echo(f"  • Languages: {len(lang_counts)}")
        for lang, count in lang_counts.head(10).items():
            click.echo(f"    - {lang}: {count}")
    
    if 'domain' in df.columns:
        domain_counts = df['domain'].value_counts()
        click.echo(f"  • Domains: {len(domain_counts)}")
        for domain, count in domain_counts.items():
            click.echo(f"    - {domain}: {count}")
    
    if 'prompt_type' in df.columns:
        type_counts = df['prompt_type'].value_counts()
        click.echo(f"  • Prompt Types: {len(type_counts)}")
        for ptype, count in type_counts.items():
            click.echo(f"    - {ptype}: {count}")
    
    if 'risk_level' in df.columns:
        risk_counts = df['risk_level'].value_counts()
        click.echo(f"  • Risk Levels: {len(risk_counts)}")
        for risk, count in risk_counts.items():
            click.echo(f"    - {risk}: {count}")


@cli.command()
@click.pass_context
def info(ctx):
    """Show system information and configuration summary."""
    config_manager = ctx.obj['config_manager']
    
    click.echo("ℹ️  SafeLLM Multilingual Evaluation Framework")
    click.echo("=" * 50)
    
    # Configuration summary
    summary = config_manager.get_config_summary()
    
    click.echo(f"Models: {summary['models']['total']} total, {summary['models']['enabled']} enabled")
    click.echo(f"Providers: {', '.join(summary['models']['providers'])}")
    click.echo(f"Languages: {summary['languages']['count']}")
    click.echo(f"Domains: {summary['domains']['count']}")
    click.echo(f"Datasets: {summary['datasets']['count']}")
    click.echo(f"Output Directory: {summary['evaluation']['output_dir']}")
    click.echo(f"Batch Size: {summary['evaluation']['batch_size']}")
    click.echo(f"Max Workers: {summary['evaluation']['max_workers']}")
    
    # Available providers
    click.echo(f"\n🔌 Available Providers:")
    for provider in ModelFactory.get_available_providers():
        click.echo(f"  • {provider}")


def main():
    """Main entry point for CLI."""
    cli()