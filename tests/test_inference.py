#test_inference.py
#Unit tests for inference.
#
#Usage: pytest tests/test_inference.py -v -s

import pytest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import data_config, OUTPUT_DIR, MODEL_REGISTRY
from inference import build_messages

class TestBuildMessages:
    #Tests for message formatting
    def test_with_instruction(self):
        messages = build_messages(
            instruction="Rewrite this in plain English.",
            text="The defendant shall comply."
        )
        
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Rewrite this in plain English." in messages[1]["content"]
        assert "The defendant shall comply." in messages[1]["content"]
    
    def test_without_instruction(self):
        messages = build_messages(
            instruction="",
            text="The defendant shall comply."
        )
        
        assert len(messages) == 2
        assert messages[1]["content"] == "The defendant shall comply."
    
    def test_system_prompt_present(self):
        messages = build_messages(
            instruction="",
            text="Test"
        )
        
        assert messages[0]["role"] == "system"
        assert "plain English" in messages[0]["content"]

class TestAdapterExists:
    #Tests that required model files exist
    
    def test_adapter_directory_exists(self):
        adapter_path = OUTPUT_DIR / "final_adapter"
        
        assert adapter_path.exists(), \
            f"Adapter not found at {adapter_path}. Run training first."
    
    def test_adapter_files_present(self):
        adapter_path = OUTPUT_DIR / "final_adapter"
        
        if not adapter_path.exists():
            pytest.skip("Adapter directory not found")
        
        # Check for essential LoRA files
        expected_files = ["adapter_config.json", "adapter_model.safetensors"]
        
        for fname in expected_files:
            # adapter_model could be .bin or .safetensors
            if fname == "adapter_model.safetensors":
                has_safetensors = (adapter_path / "adapter_model.safetensors").exists()
                has_bin = (adapter_path / "adapter_model.bin").exists()
                assert has_safetensors or has_bin, \
                    "Missing adapter weights (adapter_model.safetensors or .bin)"
            else:
                assert (adapter_path / fname).exists(), f"Missing {fname}"
    
    def test_tokenizer_files_present(self):
        adapter_path = OUTPUT_DIR / "final_adapter"
        
        if not adapter_path.exists():
            pytest.skip("Adapter directory not found")
        
        # Check for tokenizer files
        has_tokenizer = (
            (adapter_path / "tokenizer.json").exists() or
            (adapter_path / "tokenizer_config.json").exists()
        )
        
        assert has_tokenizer, "Missing tokenizer files"
            
class TestMergedModelExists:
    #Tests that merged model exists
    
    def test_merged_model_directory_exists(self):
        merged_path = OUTPUT_DIR / "merged_model"
        
        if not merged_path.exists():
            pytest.skip("Merged model not found. Run merge_adapter.py first.")
        
        assert merged_path.is_dir()
    
    def test_merged_model_files_present(self):
        merged_path = OUTPUT_DIR / "merged_model"
        
        if not merged_path.exists():
            pytest.skip("Merged model not found")
        
        # Check for model files (safetensors or bin)
        safetensor_files = list(merged_path.glob("*.safetensors"))
        bin_files = list(merged_path.glob("*.bin"))
        
        assert len(safetensor_files) > 0 or len(bin_files) > 0, \
            "No model weight files found"
        
        # check for config
        assert (merged_path / "config.json").exists(), "Missing config.json"

class TestDataFiles:
    #Tests that test data exists for inference
    
    def test_test_file_exists(self):
        test_path = Path(data_config.test_file)
        
        assert test_path.exists(), \
            f"Test file not found: {test_path}"
    
    def test_test_file_not_empty(self):
        test_path = Path(data_config.test_file)
        
        if not test_path.exists():
            pytest.skip("Test file not found")
        
        with open(test_path) as f:
            lines = f.readlines()
        
        assert len(lines) > 0, "Test file is empty"
        print(f"\nTest file has {len(lines)} examples")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])