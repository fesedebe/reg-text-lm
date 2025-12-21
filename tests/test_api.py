#test_api.py
#Integration tests for the served model API (requires server to be running: python src/serve.py)
#
#Usage: VLLM_API_KEY=your_api_key pytest tests/test_api.py -v -s

import os
import pytest
from openai import OpenAI
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from config import data_config

# configuration
API_URL = "http://localhost:8000/v1"
API_KEY = os.environ.get("VLLM_API_KEY", "")
MODEL_NAME = "legal-simplifier"

@pytest.fixture
def client():
    #create OpenAI client for API calls
    return OpenAI(base_url=API_URL, api_key=API_KEY or "unused")

@pytest.fixture
def system_prompt():
    #system prompt matching training
    return data_config.system_prompt

class TestServerConnection:
    #tests that server is running and accessible
    
    def test_server_is_running(self, client):
        #Server must be running for these tests
        try:
            models = client.models.list()
            model_ids = [m.id for m in models.data]
            assert MODEL_NAME in model_ids, \
                f"Model '{MODEL_NAME}' not found. Available: {model_ids}"
        except Exception as e:
            pytest.fail(
                f"Cannot connect to server at {API_URL}. "
                f"Start server with: python src/serve.py\n"
                f"Error: {e}"
            )
    
    def test_health_endpoint(self, client):
        #Check server responds to model list
        models = client.models.list()
        assert len(models.data) > 0, "No models available"

class TestBasicInference:
    #tests for basic inference functionality
    
    def test_simple_completion(self, client, system_prompt):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Rewrite this in plain English.\n\nThe defendant shall comply with the order."},
            ],
            temperature=0.7,
            max_tokens=256,
        )
        
        output = response.choices[0].message.content
        
        assert output is not None
        assert len(output) > 0
        print(f"\nInput: The defendant shall comply with the order.")
        print(f"Output: {output}")
    
    def test_response_structure(self, client, system_prompt):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Rewrite this in plain English.\n\nTest input."},
            ],
            max_tokens=64,
        )
        
        assert response.id is not None
        assert response.model == MODEL_NAME
        assert len(response.choices) == 1
        assert response.choices[0].message.role == "assistant"
        assert response.choices[0].finish_reason in ["stop", "length"]
    
    def test_returns_different_output_than_input(self, client, system_prompt):
        input_text = "The Lessee shall indemnify and hold harmless the Lessor."
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Rewrite this in plain English.\n\n{input_text}"},
            ],
            temperature=0.7,
            max_tokens=256,
        )
        
        output = response.choices[0].message.content.strip()
        
        # Output should be different from input (model actually did something)
        assert output.lower() != input_text.lower(), \
            "Model returned same text as input"
        print(f"\nInput: {input_text}")
        print(f"Output: {output}")

class TestLegalSimplification:
    #tests specific to legal text simplification
    
    def test_simplifies_legal_jargon(self, client, system_prompt):
        input_text = "The aforementioned party shall hereinafter be referred to as the Defendant."
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Rewrite this in plain English.\n\n{input_text}"},
            ],
            temperature=0.7,
            max_tokens=256,
        )
        
        output = response.choices[0].message.content.lower()
        
        # Should not contain overly formal legal terms
        formal_terms = ["aforementioned", "hereinafter", "heretofore", "whereas"]
        found_formal = [t for t in formal_terms if t in output]
        
        print(f"\nInput: {input_text}")
        print(f"Output: {response.choices[0].message.content}")
        
        if found_formal:
            print(f"Warning: Output still contains formal terms: {found_formal}")
    
    def test_preserves_key_information(self, client, system_prompt):
        input_text = "On 15 March 2024, the Court of Appeals ruled in favor of Smith v. Jones, case no. 2024-CV-1234."
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Rewrite this in plain English.\n\n{input_text}"},
            ],
            temperature=0.3,  # Lower temp for more consistent output
            max_tokens=256,
        )
        
        output = response.choices[0].message.content
        
        # Key information should be preserved
        assert "2024" in output, "Year not preserved"
        assert "Smith" in output or "Jones" in output, "Party names not preserved"
        
        print(f"\nInput: {input_text}")
        print(f"Output: {output}")
    
    def test_handles_long_input(self, client, system_prompt):
        input_text = (
            "The indemnifying party agrees to defend, indemnify, and hold harmless "
            "the indemnified party, its officers, directors, employees, agents, "
            "successors, and assigns from and against any and all claims, damages, "
            "losses, costs, and expenses, including but not limited to reasonable "
            "attorneys' fees and court costs, arising out of or resulting from any "
            "breach of this Agreement or any negligent or wrongful act or omission "
            "of the indemnifying party in connection with the performance of its "
            "obligations hereunder."
        )
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Rewrite this in plain English.\n\n{input_text}"},
            ],
            temperature=0.7,
            max_tokens=512,
        )
        
        output = response.choices[0].message.content
        
        assert output is not None
        assert len(output) > 50, "Output suspiciously short for long input"
        
        print(f"\nInput length: {len(input_text)} chars")
        print(f"Output length: {len(output)} chars")
        print(f"Output: {output}")

class TestEdgeCases:
    #tests for edge cases and error handling
    
    def test_empty_input(self, client, system_prompt):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Rewrite this in plain English.\n\n"},
            ],
            max_tokens=64,
        )
        
        assert response.choices[0].message.content is not None
    
    def test_non_legal_text(self, client, system_prompt):
        input_text = "The cat sat on the mat."
        
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Rewrite this in plain English.\n\n{input_text}"},
            ],
            max_tokens=64,
        )
        
        output = response.choices[0].message.content
        
        # Should handle non-legal text without crashing
        assert output is not None
        print(f"\nInput: {input_text}")
        print(f"Output: {output}")
    
    def test_temperature_zero(self, client, system_prompt):
        #Deterministic output with temperature=0
        input_text = "The court ruled."
        
        response1 = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Rewrite this in plain English.\n\n{input_text}"},
            ],
            temperature=0,
            max_tokens=64,
        )
        
        response2 = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Rewrite this in plain English.\n\n{input_text}"},
            ],
            temperature=0,
            max_tokens=64,
        )
        
        # With temp=0, outputs should be identical
        assert response1.choices[0].message.content == response2.choices[0].message.content, \
            "Temperature=0 should produce deterministic output"

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])