"""
Groq LLM Client for ultra-fast inference

This module provides Groq API integration for generating responses
with retrieved context (RAG).
"""

from groq import Groq
from typing import List, Dict, Optional, Generator
from loguru import logger


class GroqClient:
    """
    Client for Groq API with support for RAG and streaming
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        temperature: float = 0.1,
        max_tokens: int = 2048
    ):
        """
        Initialize Groq client
        
        Args:
            api_key: Groq API key
            model: Model name (llama-3.1-70b-versatile, mixtral-8x7b-32768, etc.)
            temperature: Sampling temperature (0-2, lower = more focused)
            max_tokens: Maximum tokens to generate
        """
        if not api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY in .env")
        
        self.client = Groq(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        logger.info(f"GroqClient initialized: model={model}, temp={temperature}")
    
    def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        Generate response from prompt
        
        Args:
            prompt: User prompt/query
            system_message: Optional system message
            temperature: Override default temperature
            max_tokens: Override default max tokens
            stream: Enable streaming response
        
        Returns:
            Generated text response
        """
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=stream
            )
            
            if stream:
                return self._stream_response(response)
            else:
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def generate_with_context(
        self,
        query: str,
        contexts: List[Dict],
        system_message: Optional[str] = None,
        stream: bool = False
    ) -> str:
        """
        Generate response with RAG context
        
        Args:
            query: User query
            contexts: List of context dictionaries with 'text' and metadata
            system_message: Optional system instruction
            stream: Enable streaming
        
        Returns:
            Generated response
        """
        from src.llm.prompts import format_rag_prompt
        
        # Format prompt with contexts
        formatted_prompt = format_rag_prompt(query, contexts)
        
        # Default system message for RAG
        if not system_message:
            system_message = (
                "You are an expert AI research assistant. "
                "Answer questions based on the provided research paper excerpts. "
                "Be accurate, cite sources using [1], [2], etc., and acknowledge when "
                "information is not in the provided context."
            )
        
        return self.generate(
            prompt=formatted_prompt,
            system_message=system_message,
            stream=stream
        )
    
    def _stream_response(self, response) -> Generator[str, None, None]:
        """
        Stream response chunks
        
        Args:
            response: Groq streaming response
        
        Yields:
            Response text chunks
        """
        for chunk in response:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> str:
        """
        Multi-turn chat completion
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Override temperature
            max_tokens: Override max tokens
            stream: Enable streaming
        
        Returns:
            Generated response
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=max_tokens or self.max_tokens,
                stream=stream
            )
            
            if stream:
                return self._stream_response(response)
            else:
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            raise
    
    def get_model_info(self) -> Dict:
        """Get current model configuration"""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }


def main():
    """Example usage"""
    from src.config import settings
    import os
    
    print("\n" + "="*70)
    print("GroqClient Demo")
    print("="*70)
    
    # Check API key
    api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
    if not api_key:
        print("\n❌ GROQ_API_KEY not found!")
        print("Please set it in .env file:")
        print("  GROQ_API_KEY=gsk_...")
        print("\nGet your key from: https://console.groq.com/keys")
        return
    
    # Initialize client
    client = GroqClient(
        api_key=api_key,
        model=settings.GROQ_MODEL,
        temperature=settings.GROQ_TEMPERATURE
    )
    
    print(f"\n✓ Client initialized")
    print(f"  Model: {client.model}")
    print(f"  Temperature: {client.temperature}")
    
    # Test 1: Simple generation
    print("\n1. Simple Generation:")
    print("-"*70)
    
    prompt = "Explain transformer models in 2 sentences."
    print(f"Prompt: {prompt}")
    print()
    
    response = client.generate(prompt)
    print(f"Response: {response}")
    
    # Test 2: RAG with context
    print("\n\n2. RAG with Context:")
    print("-"*70)
    
    query = "What are attention mechanisms?"
    contexts = [
        {
            "text": "Attention mechanisms allow models to focus on relevant parts of the input when making predictions.",
            "source": "Paper 1",
            "score": 0.95
        },
        {
            "text": "Self-attention in transformers enables parallel processing of sequences by computing attention weights.",
            "source": "Paper 2",
            "score": 0.87
        }
    ]
    
    print(f"Query: {query}")
    print(f"Contexts: {len(contexts)}")
    print()
    
    response = client.generate_with_context(query, contexts)
    print(f"Response: {response}")
    
    # Test 3: Streaming
    print("\n\n3. Streaming Response:")
    print("-"*70)
    
    prompt = "List 3 key benefits of transformer models."
    print(f"Prompt: {prompt}")
    print("\nStreaming: ", end="", flush=True)
    
    for chunk in client.generate(prompt, stream=True):
        print(chunk, end="", flush=True)
    
    print("\n\n" + "="*70)
    print("✓ GroqClient working correctly!")
    print("="*70)


if __name__ == "__main__":
    main()

