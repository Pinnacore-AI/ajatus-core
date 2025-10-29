"""
Ajatuskumppani AI Inference Engine
Handles AI model loading, inference, and response generation
"""

import os
import json
import torch
from typing import List, Dict, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Message:
    """Chat message structure"""
    role: str  # 'user' or 'assistant'
    content: str


@dataclass
class InferenceConfig:
    """Configuration for inference"""
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2"
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    use_4bit: bool = True  # Use 4-bit quantization for memory efficiency
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class AjatusInferenceEngine:
    """Main inference engine for Ajatuskumppani"""
    
    def __init__(self, config: Optional[InferenceConfig] = None):
        self.config = config or InferenceConfig()
        self.model = None
        self.tokenizer = None
        self.system_prompt = self._load_system_prompt()
        
    def _load_system_prompt(self) -> str:
        """Load Finnish-optimized system prompt"""
        return """Olet Ajatuskumppani, suomalainen avoimen lähdekoodin tekoäly.

Sinun tehtäväsi:
- Auttaa käyttäjiä ystävällisesti ja ammattimaisesti
- Vastata suomeksi, ellei käyttäjä pyydä muuta
- Kunnioittaa käyttäjän yksityisyyttä ja dataa
- Olla rehellinen ja läpinäkyvä
- Myöntää kun et tiedä jotain

Muista: Olet osa hajautettua, käyttäjien omistamaa AI-ekosysteemiä."""

    def load_model(self):
        """Load the AI model with optimizations"""
        logger.info(f"Loading model: {self.config.model_name}")
        
        try:
            # Configure 4-bit quantization if enabled
            quantization_config = None
            if self.config.use_4bit and self.config.device == "cuda":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                trust_remote_code=True
            )
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_name,
                quantization_config=quantization_config,
                device_map="auto" if self.config.device == "cuda" else None,
                trust_remote_code=True,
                torch_dtype=torch.float16 if self.config.device == "cuda" else torch.float32
            )
            
            if self.config.device == "cpu":
                self.model = self.model.to(self.config.device)
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def format_chat_history(self, messages: List[Message]) -> str:
        """Format chat history for the model"""
        formatted = f"<s>[INST] {self.system_prompt}\n\n"
        
        for i, msg in enumerate(messages):
            if msg.role == "user":
                if i == 0:
                    formatted += f"{msg.content} [/INST]"
                else:
                    formatted += f"</s><s>[INST] {msg.content} [/INST]"
            elif msg.role == "assistant":
                formatted += f" {msg.content}"
        
        return formatted
    
    def generate_response(
        self,
        messages: List[Message],
        stream: bool = False
    ) -> str:
        """Generate AI response"""
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Format input
        prompt = self.format_chat_history(messages)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        ).to(self.config.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                repetition_penalty=self.config.repetition_penalty,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return response.strip()
    
    def chat(self, user_message: str, history: List[Dict] = None) -> str:
        """Simple chat interface"""
        if history is None:
            history = []
        
        # Convert history to Message objects
        messages = [Message(role=msg['role'], content=msg['content']) for msg in history]
        messages.append(Message(role='user', content=user_message))
        
        # Generate response
        response = self.generate_response(messages)
        
        return response
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            logger.info("Model unloaded")


# Example usage
if __name__ == "__main__":
    # Initialize engine
    engine = AjatusInferenceEngine()
    
    # Load model
    engine.load_model()
    
    # Test chat
    response = engine.chat("Hei! Kerro itsestäsi.")
    print(f"AI: {response}")
    
    # Chat with history
    history = [
        {"role": "user", "content": "Hei! Kerro itsestäsi."},
        {"role": "assistant", "content": response}
    ]
    
    response2 = engine.chat("Mitä osaat tehdä?", history)
    print(f"AI: {response2}")
    
    # Unload when done
    engine.unload_model()

