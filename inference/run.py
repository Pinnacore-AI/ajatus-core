#!/usr/bin/env python3
# Ajatuskumppani — built in Finland, by the free minds of Pinnacore.

"""
AjatusCore Inference Engine

This script provides a simple interface for running inference with the AjatusCore model.
It supports both vLLM (for GPU) and Ollama (for CPU/fallback) backends.
"""

import argparse
import os
from typing import Optional

def run_vllm_inference(model_path: str, prompt: str, max_tokens: int = 512) -> str:
    """
    Run inference using vLLM backend (GPU-optimized).
    
    Args:
        model_path: Path to the model weights
        prompt: Input prompt for the model
        max_tokens: Maximum number of tokens to generate
        
    Returns:
        Generated text completion
    """
    try:
        from vllm import LLM, SamplingParams
        
        llm = LLM(model=model_path)
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=max_tokens
        )
        
        outputs = llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text
        
    except ImportError:
        raise ImportError("vLLM not installed. Install with: pip install vllm")


def run_ollama_inference(model_name: str, prompt: str) -> str:
    """
    Run inference using Ollama backend (CPU-friendly).
    
    Args:
        model_name: Name of the Ollama model
        prompt: Input prompt for the model
        
    Returns:
        Generated text completion
    """
    try:
        import requests
        
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            }
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            raise Exception(f"Ollama API error: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        raise ConnectionError("Cannot connect to Ollama. Make sure it's running.")


def main():
    parser = argparse.ArgumentParser(description="AjatusCore Inference Engine")
    parser.add_argument("--prompt", type=str, required=True, help="Input prompt")
    parser.add_argument("--backend", type=str, choices=["vllm", "ollama"], 
                       default="vllm", help="Inference backend")
    parser.add_argument("--model", type=str, 
                       default="models/ajatuscore-7b",
                       help="Model path (vLLM) or name (Ollama)")
    parser.add_argument("--max-tokens", type=int, default=512,
                       help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    print(f"🧠 AjatusCore Inference Engine")
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model}")
    print(f"\nPrompt: {args.prompt}\n")
    print("-" * 80)
    
    if args.backend == "vllm":
        response = run_vllm_inference(args.model, args.prompt, args.max_tokens)
    else:
        response = run_ollama_inference(args.model, args.prompt)
    
    print(response)
    print("-" * 80)


if __name__ == "__main__":
    main()

