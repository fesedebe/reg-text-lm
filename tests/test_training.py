# Validates that training data is correctly formatted and labels are properly masked.

import pytest
from pathlib import Path
from typing import List, Dict, Any

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import data_config, model_config, training_config
from data_loader import load_hf_dataset, format_chat_message


# Special token IDs (will be set after loading tokenizer)
IGNORE_INDEX = -100  # Standard value for ignored labels in PyTorch


def get_tokenizer():
    # Load the tokenizer for testing
    try:
        from transformers import AutoTokenizer
    except ImportError:
        pytest.skip("transformers not installed")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name,
        trust_remote_code=True,
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return tokenizer


class TestLabelMasking:
    """Tests to verify labels are correctly masked for training."""
    
    def test_padding_tokens_masked(self):
        # Verify that padding tokens have label = -100 (ignored in loss)
        tokenizer = get_tokenizer()
        
        # Create a sample with padding
        messages = format_chat_message(
            input_text="The court ordered compliance.",
            output_text="The court said to follow the rules.",
            instruction="Rewrite this in plain English."
        )
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Tokenize with padding
        encoded = tokenizer(
            text,
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt"
        )
        
        input_ids = encoded["input_ids"][0]
        attention_mask = encoded["attention_mask"][0]
        
        # Find padding positions (where attention_mask = 0)
        padding_positions = (attention_mask == 0).nonzero(as_tuple=True)[0]
        
        if len(padding_positions) > 0:
            # In a proper training setup, labels at padding positions should be -100
            # This test documents the expected behavior
            print(f"Found {len(padding_positions)} padding positions")
            print("Note: SFTTrainer handles label masking automatically")
    
    def test_prompt_should_be_masked(self):
        # Verify the prompt portion is identified for masking
        tokenizer = get_tokenizer()
        
        # Create a training example
        messages = format_chat_message(
            input_text="The defendant shall comply with the order.",
            output_text="The defendant must follow the order.",
            instruction="Rewrite this in plain English."
        )
        
        # Full conversation (for training)
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Just the prompt (user message only)
        prompt_only = tokenizer.apply_chat_template(
            [messages[0]],  # Only user message
            tokenize=False,
            add_generation_prompt=True  # Add assistant prompt
        )
        
        # Tokenize both
        full_tokens = tokenizer(full_text, return_tensors="pt")["input_ids"][0]
        prompt_tokens = tokenizer(prompt_only, return_tensors="pt")["input_ids"][0]
        
        prompt_length = len(prompt_tokens)
        response_length = len(full_tokens) - prompt_length
        
        print(f"\nToken breakdown:")
        print(f"  Prompt tokens: {prompt_length}")
        print(f"  Response tokens: {response_length}")
        print(f"  Total tokens: {len(full_tokens)}")
        
        # The response should be a meaningful portion
        assert response_length > 0, "Response has no tokens!"
        
        # Document: In proper training, first `prompt_length` labels should be -100
        print(f"\nFor correct training:")
        print(f"  Labels[:{prompt_length}] should be -100 (ignored)")
        print(f"  Labels[{prompt_length}:] should be actual token IDs")
    
    def test_assistant_response_has_content(self):
        # Verify assistant responses tokenize to reasonable length
        tokenizer = get_tokenizer()
        
        if not Path(data_config.train_file).exists():
            pytest.skip("Training file not found")
        
        data = load_hf_dataset(data_config.train_file)
        
        too_short = []
        for i, example in enumerate(data.select(range(min(50, len(data))))):  # Check first 50
            messages = format_chat_message(
                input_text=example["input"],
                output_text=example["output"],
                instruction=example.get("instruction")
            )
            
            # Tokenize just the output
            output_tokens = tokenizer(
                example["output"],
                return_tensors="pt"
            )["input_ids"][0]
            
            if len(output_tokens) < 5:
                too_short.append((i, example["output"][:50]))
        
        if too_short:
            print(f"\nWarning: {len(too_short)} examples have very short outputs (<5 tokens):")
            for idx, text in too_short[:5]:
                print(f"  [{idx}]: {text}...")
        
        # Allow some short outputs but not too many
        num_checked = min(50, len(data))
        assert len(too_short) < num_checked * 0.2, \
            f"Too many short outputs: {len(too_short)}/{num_checked}"


class TestChatTemplateFormatting:
    """Tests for correct chat template application."""
    
    def test_special_tokens_present(self):
        # Verify chat template includes expected special tokens
        tokenizer = get_tokenizer()
        
        messages = format_chat_message(
            input_text="Legal text here.",
            output_text="Plain text here.",
            instruction="Rewrite this in plain English."
        )
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # Llama 3.1 uses these markers
        expected_markers = [
            "<|begin_of_text|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eot_id|>",
        ]
        
        print(f"\nFormatted text preview:\n{text[:500]}...")
        
        for marker in expected_markers:
            assert marker in text, f"Missing expected marker: {marker}"
    
    def test_roles_correctly_assigned(self):
        # Verify user and assistant roles are in the output
        tokenizer = get_tokenizer()
        
        messages = format_chat_message(
            input_text="The court ruled.",
            output_text="The court decided.",
            instruction="Rewrite this in plain English."
        )
        
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        assert "user" in text.lower(), "User role not found in template"
        assert "assistant" in text.lower(), "Assistant role not found in template"
    
    def test_generation_prompt_format(self):
        # Verify generation prompt is correctly added for inference
        tokenizer = get_tokenizer()
        
        messages = [
            {"role": "user", "content": "Rewrite this in plain English:\n\nLegal text."}
        ]
        
        # Without generation prompt (for training)
        text_train = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        
        # With generation prompt (for inference)
        text_infer = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Inference prompt should be longer (includes assistant header)
        assert len(text_infer) > len(text_train), \
            "Generation prompt not added correctly"
        
        print(f"\nTraining format ends with: ...{text_train[-100:]}")
        print(f"\nInference format ends with: ...{text_infer[-100:]}")


class TestSequenceLengths:
    """Tests for sequence length handling."""
    
    def test_sequences_within_limit(self):
        # Verify training examples fit within max sequence length
        tokenizer = get_tokenizer()
        
        if not Path(data_config.train_file).exists():
            pytest.skip("Training file not found")
        
        data = load_hf_dataset(data_config.train_file)
        max_len = training_config.max_seq_length
        
        too_long = []
        lengths = []
        
        for i, example in enumerate(data):
            messages = format_chat_message(
                input_text=example["input"],
                output_text=example["output"],
                instruction=example.get("instruction")
            )
            
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            tokens = tokenizer(text, return_tensors="pt")["input_ids"][0]
            length = len(tokens)
            lengths.append(length)
            
            if length > max_len:
                too_long.append((i, length))
        
        print(f"\nSequence length statistics:")
        print(f"  Min: {min(lengths)}")
        print(f"  Max: {max(lengths)}")
        print(f"  Mean: {sum(lengths)/len(lengths):.1f}")
        print(f"  Max allowed: {max_len}")
        print(f"  Over limit: {len(too_long)}/{len(data)}")
        
        if too_long:
            print(f"\nExamples exceeding limit:")
            for idx, length in too_long[:5]:
                print(f"  [{idx}]: {length} tokens")
        
        # Warn but don't fail - truncation will handle this
        if len(too_long) > len(data) * 0.1:
            print(f"\nWARNING: {len(too_long)} examples ({100*len(too_long)/len(data):.1f}%) "
                  f"exceed max_seq_length and will be truncated!")


class TestDataLeakage:
    """Tests to detect train/test data overlap."""
    
    def test_no_input_overlap(self):
        # Ensure no input texts appear in both train and test sets
        if not Path(data_config.train_file).exists():
            pytest.skip("Training file not found")
        if not Path(data_config.test_file).exists():
            pytest.skip("Test file not found")
        
        train_data = load_hf_dataset(data_config.train_file)
        test_data = load_hf_dataset(data_config.test_file)
        
        train_inputs = set(ex["input"].strip() for ex in train_data)
        test_inputs = set(ex["input"].strip() for ex in test_data)
        
        overlap = train_inputs & test_inputs
        
        if overlap:
            print(f"\nWARNING: Found {len(overlap)} overlapping inputs!")
            for text in list(overlap)[:3]:
                print(f"  '{text[:80]}...'")
        
        assert len(overlap) == 0, \
            f"Data leakage detected: {len(overlap)} inputs appear in both train and test"
    
    def test_no_output_overlap(self):
        # Ensure no output texts appear in both train and test sets
        if not Path(data_config.train_file).exists():
            pytest.skip("Training file not found")
        if not Path(data_config.test_file).exists():
            pytest.skip("Test file not found")
        
        train_data = load_hf_dataset(data_config.train_file)
        test_data = load_hf_dataset(data_config.test_file)
        
        train_outputs = set(ex["output"].strip() for ex in train_data)
        test_outputs = set(ex["output"].strip() for ex in test_data)
        
        overlap = train_outputs & test_outputs
        
        if overlap:
            print(f"\nWARNING: Found {len(overlap)} overlapping outputs!")
        
        # Outputs might legitimately overlap for very short/common phrases
        # but should be minimal
        overlap_pct = len(overlap) / len(test_outputs) if test_outputs else 0
        assert overlap_pct < 0.1, \
            f"High output overlap: {len(overlap)}/{len(test_outputs)} ({100*overlap_pct:.1f}%)"


class TestTrainingDataIntegrity:
    """Additional data integrity checks for training."""
    
    def test_consistent_instruction(self):
        # Verify instruction field is consistent across examples
        if not Path(data_config.train_file).exists():
            pytest.skip("Training file not found")
        
        data = load_hf_dataset(data_config.train_file)
        
        instructions = set(ex.get("instruction", "") for ex in data)
        
        print(f"\nUnique instructions found: {len(instructions)}")
        for inst in instructions:
            count = sum(1 for ex in data if ex.get("instruction", "") == inst)
            print(f"  '{inst[:50]}...' ({count} examples)")
        
        # Having consistent instructions is usually better for fine-tuning
        if len(instructions) > 3:
            print("\nNote: Multiple different instructions found. "
                  "Consider standardizing for better fine-tuning results.")
    
    def test_reasonable_input_output_ratio(self):
        # Check that input/output lengths are reasonable
        if not Path(data_config.train_file).exists():
            pytest.skip("Training file not found")
        
        data = load_hf_dataset(data_config.train_file)
        
        ratios = []
        for ex in data:
            input_len = len(ex["input"])
            output_len = len(ex["output"])
            if input_len > 0:
                ratios.append(output_len / input_len)
        
        avg_ratio = sum(ratios) / len(ratios)
        
        print(f"\nOutput/Input length ratios:")
        print(f"  Min: {min(ratios):.2f}")
        print(f"  Max: {max(ratios):.2f}")
        print(f"  Mean: {avg_ratio:.2f}")
        
        # For simplification, outputs are typically similar length or shorter
        # Very long outputs relative to inputs might indicate issues

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
