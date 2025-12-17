# Test data loading and formatting.

import json
import tempfile
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import (
    load_hf_dataset,
    format_chat_message,
)
from config import data_config

class TestLoadDataset:
    """Tests for HF Dataset loading."""
    
    def test_load_existing_train_file(self):
        """Test loading the actual training file."""
        if not Path(data_config.train_file).exists():
            pytest.skip("Training file not found")
        
        data = load_hf_dataset(data_config.train_file)
        
        assert len(data) > 0
        
        # Check structure of first item
        first = data[0]
        assert "instruction" in first
        assert "input" in first
        assert "output" in first
    
    def test_load_existing_test_file(self):
        """Test loading the actual test file."""
        if not Path(data_config.test_file).exists():
            pytest.skip("Test file not found")
        
        data = load_hf_dataset(data_config.test_file)
        
        assert len(data) > 0
    
    def test_load_with_temp_file(self):
        """Test loading from a temporary JSONL file."""
        test_data = [
            {"instruction": "Rewrite", "input": "Legal text 1", "output": "Plain text 1"},
            {"instruction": "Rewrite", "input": "Legal text 2", "output": "Plain text 2"},
        ]
        
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item) + "\n")
            temp_path = f.name
        
        try:
            loaded = load_hf_dataset(temp_path)
            assert len(loaded) == 2
            assert loaded[0]["input"] == "Legal text 1"
            assert loaded[1]["output"] == "Plain text 2"
        finally:
            Path(temp_path).unlink()
    
    def test_load_nonexistent_file_raises(self):
        """Test that loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_hf_dataset("/nonexistent/path/file.jsonl")

class TestFormatChatMessage:
    """Tests for chat message formatting."""
    
    def test_format_with_output(self):
        """Test formatting with both input and output (training)."""
        messages = format_chat_message(
            input_text="The defendant shall comply.",
            output_text="The defendant must follow the rules.",
            instruction="Rewrite this in plain English."
        )
        
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert "defendant shall comply" in messages[0]["content"]
        assert "must follow the rules" in messages[1]["content"]
    
    def test_format_without_output(self):
        """Test formatting without output (inference)."""
        messages = format_chat_message(
            input_text="The defendant shall comply.",
            output_text=None,
            instruction="Rewrite this in plain English."
        )
        
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
    
    def test_format_with_system_prompt(self):
        """Test formatting with system prompt."""
        messages = format_chat_message(
            input_text="Legal text",
            output_text="Plain text",
            system_prompt="You are a legal simplifier."
        )
        
        assert len(messages) == 3
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a legal simplifier."
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"
    
    def test_format_without_instruction(self):
        """Test formatting without instruction (uses input directly)."""
        messages = format_chat_message(
            input_text="Legal text here.",
            output_text="Plain text here."
        )
        
        assert len(messages) == 2
        assert messages[0]["content"] == "Legal text here."
    
    def test_custom_instruction(self):
        """Test with custom instruction."""
        messages = format_chat_message(
            input_text="Legal text",
            output_text="Plain text",
            instruction="Simplify this legal document."
        )
        
        assert "Simplify this legal document" in messages[0]["content"]

class TestDataQuality:
    """Tests for data quality validation."""
    
    def test_no_empty_inputs(self):
        """Ensure no examples have empty input."""
        if not Path(data_config.train_file).exists():
            pytest.skip("Training file not found")
        
        data = load_hf_dataset(data_config.train_file)
        
        for i, example in enumerate(data):
            assert example["input"].strip(), f"Empty input at index {i}"
    
    def test_no_empty_outputs(self):
        """Ensure no examples have empty output."""
        if not Path(data_config.train_file).exists():
            pytest.skip("Training file not found")
        
        data = load_hf_dataset(data_config.train_file)
        
        for i, example in enumerate(data):
            assert example["output"].strip(), f"Empty output at index {i}"
    
    def test_input_output_different(self):
        """Ensure input and output are not identical."""
        if not Path(data_config.train_file).exists():
            pytest.skip("Training file not found")
        
        data = load_hf_dataset(data_config.train_file)
        
        identical_count = 0
        for example in data:
            if example["input"].strip() == example["output"].strip():
                identical_count += 1
        
        # Allow some identical pairs but flag if too many
        assert identical_count < len(data) * 0.1, \
            f"Too many identical input/output pairs: {identical_count}/{len(data)}"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
