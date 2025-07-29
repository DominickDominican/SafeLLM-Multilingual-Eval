"""
Test CLI functionality.
"""

import pytest
import os
import tempfile
from click.testing import CliRunner
from unittest.mock import patch, MagicMock

from safellm_eval.cli import cli


class TestCLI:
    """Test command-line interface."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.runner = CliRunner()
    
    def test_cli_help(self):
        """Test CLI help command."""
        result = self.runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'SafeLLM Multilingual Evaluation Framework CLI' in result.output
    
    def test_init_command(self):
        """Test init command."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = os.path.join(temp_dir, 'test_config.yaml')
            
            result = self.runner.invoke(cli, ['init', '--output', config_path])
            
            assert result.exit_code == 0
            assert os.path.exists(config_path)
            assert 'Template configuration created' in result.output
    
    def test_validate_command_success(self):
        """Test validate command with valid config."""
        with patch('safellm_eval.cli.ConfigManager') as mock_config:
            mock_manager = MagicMock()
            mock_manager.validate_config.return_value = {
                'errors': [],
                'warnings': [],
                'info': ['Configuration is valid']
            }
            mock_config.return_value = mock_manager
            
            result = self.runner.invoke(cli, ['validate'])
            
            assert result.exit_code == 0
            assert 'Configuration is valid' in result.output
    
    def test_validate_command_with_errors(self):
        """Test validate command with configuration errors."""
        with patch('safellm_eval.cli.ConfigManager') as mock_config:
            mock_manager = MagicMock()
            mock_manager.validate_config.return_value = {
                'errors': ['No API keys found'],
                'warnings': ['Model disabled'],
                'info': ['1 model configured']
            }
            mock_config.return_value = mock_manager
            
            result = self.runner.invoke(cli, ['validate'])
            
            assert result.exit_code == 1
            assert 'No API keys found' in result.output
    
    def test_list_models_command(self):
        """Test list-models command."""
        with patch('safellm_eval.cli.ConfigManager') as mock_config:
            mock_manager = MagicMock()
            mock_config_obj = MagicMock()
            mock_config_obj.models = [
                MagicMock(
                    name='gpt-4',
                    provider='openai',
                    enabled=True,
                    api_key='test_key',
                    temperature=0.7,
                    max_tokens=1000
                )
            ]
            mock_manager.config = mock_config_obj
            mock_config.return_value = mock_manager
            
            result = self.runner.invoke(cli, ['list-models'])
            
            assert result.exit_code == 0
            assert 'gpt-4' in result.output
            assert 'openai' in result.output
    
    def test_inspect_command(self):
        """Test inspect command."""
        # Create test dataset
        test_data = [
            {"language": "English", "domain": "Test", "prompt_type": "benign", "prompt": "Test prompt 1"},
            {"language": "Spanish", "domain": "Test", "prompt_type": "adversarial", "prompt": "Test prompt 2"}
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as tmp:
            try:
                import jsonlines
                with jsonlines.open(tmp.name, mode='w') as writer:
                    for item in test_data:
                        writer.write(item)
                
                result = self.runner.invoke(cli, ['inspect', tmp.name])
                
                assert result.exit_code == 0
                assert 'Dataset Statistics' in result.output
                assert 'Total prompts: 2' in result.output
                
            finally:
                os.unlink(tmp.name)
    
    def test_inspect_nonexistent_file(self):
        """Test inspect command with non-existent file."""
        result = self.runner.invoke(cli, ['inspect', 'nonexistent.jsonl'])
        
        assert result.exit_code == 1
        assert 'Dataset not found' in result.output
    
    def test_info_command(self):
        """Test info command."""
        with patch('safellm_eval.cli.ConfigManager') as mock_config:
            mock_manager = MagicMock()
            mock_manager.get_config_summary.return_value = {
                'models': {'total': 3, 'enabled': 2, 'providers': ['openai', 'anthropic']},
                'languages': {'count': 12},
                'domains': {'count': 6},
                'datasets': {'count': 2},
                'evaluation': {'output_dir': './results', 'batch_size': 10, 'max_workers': 5}
            }
            mock_config.return_value = mock_manager
            
            result = self.runner.invoke(cli, ['info'])
            
            assert result.exit_code == 0
            assert 'SafeLLM Multilingual Evaluation Framework' in result.output
            assert 'Models: 3 total, 2 enabled' in result.output
    
    @patch('safellm_eval.cli.MultilingualEvaluator')
    @patch('safellm_eval.cli.ResultVisualizer')
    def test_evaluate_command(self, mock_visualizer, mock_evaluator):
        """Test evaluate command."""
        # Create test dataset
        test_data = [{"language": "English", "domain": "Test", "prompt": "Test"}]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as dataset_file:
            try:
                import jsonlines
                with jsonlines.open(dataset_file.name, mode='w') as writer:
                    for item in test_data:
                        writer.write(item)
                
                # Mock evaluator
                mock_eval_instance = MagicMock()
                mock_eval_instance.evaluate_dataset.return_value = [MagicMock()]
                mock_eval_instance.get_summary_stats.return_value = {
                    'total_evaluations': 1,
                    'successful_evaluations': 1,
                    'failed_evaluations': 0,
                    'average_safety_score': 0.8
                }
                mock_evaluator.return_value = mock_eval_instance
                
                # Mock visualizer
                mock_viz_instance = MagicMock()
                mock_viz_instance.create_summary_report.return_value = {
                    'overview': 'test.html'
                }
                mock_visualizer.return_value = mock_viz_instance
                
                # Mock config manager
                with patch('safellm_eval.cli.ConfigManager') as mock_config:
                    mock_manager = MagicMock()
                    mock_manager.get_enabled_models.return_value = [
                        MagicMock(name='gpt-4', provider='openai')
                    ]
                    mock_config.return_value = mock_manager
                    
                    with tempfile.TemporaryDirectory() as temp_dir:
                        result = self.runner.invoke(cli, [
                            'evaluate', 
                            dataset_file.name,
                            '--output', temp_dir,
                            '--format', 'jsonl'
                        ])
                        
                        assert result.exit_code == 0
                        assert 'Evaluation completed successfully' in result.output
                        
            finally:
                os.unlink(dataset_file.name)
    
    def test_evaluate_command_missing_dataset(self):
        """Test evaluate command with missing dataset."""
        result = self.runner.invoke(cli, ['evaluate', 'nonexistent.jsonl'])
        
        assert result.exit_code == 1
        assert 'Dataset not found' in result.output
    
    @patch('safellm_eval.cli.ResultVisualizer')
    def test_visualize_command(self, mock_visualizer):
        """Test visualize command."""
        # Create test results file
        test_results = [
            {
                "language": "English",
                "safety_score": 0.8,
                "model": "gpt-4",
                "domain": "Test"
            }
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as results_file:
            try:
                import json
                json.dump(test_results, results_file)
                results_file.flush()
                
                # Mock visualizer
                mock_viz_instance = MagicMock()
                mock_viz_instance.create_summary_report.return_value = {
                    'overview': 'test.html',
                    'language': 'test.png'
                }
                mock_visualizer.return_value = mock_viz_instance
                
                with tempfile.TemporaryDirectory() as temp_dir:
                    result = self.runner.invoke(cli, [
                        'visualize',
                        results_file.name,
                        '--output-dir', temp_dir
                    ])
                    
                    assert result.exit_code == 0
                    assert 'Visualization generation completed' in result.output
                    
            finally:
                os.unlink(results_file.name)
    
    def test_visualize_command_missing_file(self):
        """Test visualize command with missing results file."""
        result = self.runner.invoke(cli, ['visualize', 'nonexistent.json'])
        
        assert result.exit_code == 1
        assert 'Results file not found' in result.output