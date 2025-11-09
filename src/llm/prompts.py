"""
Prompt Templates for RAG

This module provides prompt templates and formatting utilities
for Retrieval-Augmented Generation.
"""

from typing import List, Dict, Optional


def format_rag_prompt(
    query: str,
    contexts: List[Dict],
    include_metadata: bool = True,
    max_contexts: Optional[int] = None
) -> str:
    """
    Format RAG prompt with retrieved contexts
    
    Args:
        query: User query
        contexts: List of context dicts with 'text', 'score', and metadata
        include_metadata: Include paper_id, section, etc. in context
        max_contexts: Limit number of contexts to include
    
    Returns:
        Formatted prompt string
    """
    if max_contexts:
        contexts = contexts[:max_contexts]
    
    # Build context string
    context_parts = []
    
    for i, ctx in enumerate(contexts, 1):
        part = f"[{i}] "
        
        if include_metadata:
            # Add metadata header
            paper_id = ctx.get('metadata', {}).get('paper_id', 'Unknown')
            section = ctx.get('metadata', {}).get('section', 'Unknown')
            score = ctx.get('score', 0)
            
            part += f"**Source:** {paper_id} | **Section:** {section} | **Relevance:** {score:.2f}\n"
        
        # Add text content
        text = ctx.get('text', '')
        part += f"{text}\n"
        
        context_parts.append(part)
    
    context_str = "\n".join(context_parts)
    
    # Build full prompt
    prompt = f"""Based on the following research paper excerpts, please answer the question comprehensively and accurately.

IMPORTANT INSTRUCTIONS:
- Cite sources using [1], [2], etc. when referencing information
- If the answer is not fully covered in the provided context, acknowledge the limitations
- Provide a clear, well-structured response
- If multiple sources discuss the same point, synthesize them
- Use technical language when appropriate for the domain

RESEARCH CONTEXT:
{context_str}

QUESTION: {query}

Please provide a detailed, well-cited answer:"""
    
    return prompt


def format_followup_prompt(
    original_query: str,
    original_response: str,
    followup_query: str,
    new_contexts: Optional[List[Dict]] = None
) -> str:
    """
    Format prompt for follow-up questions with conversation history
    
    Args:
        original_query: Previous query
        original_response: Previous response
        followup_query: New follow-up question
        new_contexts: Optional new retrieved contexts
    
    Returns:
        Formatted follow-up prompt
    """
    prompt = f"""Previous conversation:

Q: {original_query}
A: {original_response}

"""
    
    if new_contexts:
        context_parts = []
        for i, ctx in enumerate(new_contexts, 1):
            text = ctx.get('text', '')
            context_parts.append(f"[{i}] {text}")
        
        context_str = "\n\n".join(context_parts)
        prompt += f"""Additional Context:
{context_str}

"""
    
    prompt += f"""Follow-up Question: {followup_query}

Please answer the follow-up question, taking into account the previous conversation:"""
    
    return prompt


def format_summarization_prompt(contexts: List[Dict], focus: Optional[str] = None) -> str:
    """
    Format prompt for summarizing multiple paper excerpts
    
    Args:
        contexts: List of context dicts
        focus: Optional focus area for summarization
    
    Returns:
        Formatted summarization prompt
    """
    context_parts = []
    
    for i, ctx in enumerate(contexts, 1):
        text = ctx.get('text', '')
        paper_id = ctx.get('metadata', {}).get('paper_id', f'Source {i}')
        context_parts.append(f"[{paper_id}] {text}")
    
    context_str = "\n\n".join(context_parts)
    
    focus_instruction = f" with focus on {focus}" if focus else ""
    
    prompt = f"""Please provide a comprehensive summary of the following research paper excerpts{focus_instruction}.

EXCERPTS:
{context_str}

Please synthesize the key points, methodologies, and findings from these excerpts:"""
    
    return prompt


def format_comparison_prompt(
    topic: str,
    contexts: List[Dict],
    comparison_aspects: Optional[List[str]] = None
) -> str:
    """
    Format prompt for comparing approaches/methods across papers
    
    Args:
        topic: Topic to compare
        contexts: List of context dicts
        comparison_aspects: Optional list of aspects to compare
    
    Returns:
        Formatted comparison prompt
    """
    context_parts = []
    
    for i, ctx in enumerate(contexts, 1):
        text = ctx.get('text', '')
        paper_id = ctx.get('metadata', {}).get('paper_id', f'Paper {i}')
        context_parts.append(f"[{paper_id}] {text}")
    
    context_str = "\n\n".join(context_parts)
    
    aspects_str = ""
    if comparison_aspects:
        aspects_str = f"\n\nPlease compare specifically on these aspects:\n" + "\n".join(
            f"- {aspect}" for aspect in comparison_aspects
        )
    
    prompt = f"""Compare and contrast the different approaches to {topic} across the following research excerpts.

EXCERPTS:
{context_str}
{aspects_str}

Please provide a structured comparison highlighting similarities, differences, and relative strengths:"""
    
    return prompt


def format_citation_prompt(statement: str, contexts: List[Dict]) -> str:
    """
    Format prompt for finding citations for a statement
    
    Args:
        statement: Statement to find citations for
        contexts: List of context dicts
    
    Returns:
        Formatted citation prompt
    """
    context_parts = []
    
    for i, ctx in enumerate(contexts, 1):
        text = ctx.get('text', '')
        paper_id = ctx.get('metadata', {}).get('paper_id', f'Source {i}')
        context_parts.append(f"[{i}] {paper_id}: {text}")
    
    context_str = "\n\n".join(context_parts)
    
    prompt = f"""Given the following statement, identify which sources support it and provide appropriate citations.

STATEMENT: {statement}

AVAILABLE SOURCES:
{context_str}

Please indicate:
1. Which sources support this statement (if any)
2. How well each source supports the statement
3. Any caveats or limitations in the support
4. Proper citations in the format [1], [2], etc."""
    
    return prompt


# System messages for different use cases
SYSTEM_MESSAGES = {
    "default": (
        "You are an expert AI research assistant. "
        "Answer questions based on provided research papers. "
        "Be accurate, cite sources, and acknowledge limitations."
    ),
    
    "technical": (
        "You are an expert technical AI researcher. "
        "Provide detailed, technically accurate answers with appropriate terminology. "
        "Cite sources using [1], [2], etc. and explain complex concepts clearly."
    ),
    
    "beginner_friendly": (
        "You are a helpful AI research assistant. "
        "Explain concepts in an accessible way while maintaining accuracy. "
        "Use analogies when helpful and cite sources."
    ),
    
    "critical": (
        "You are a critical AI research analyst. "
        "Analyze the research objectively, pointing out strengths, limitations, "
        "and areas of uncertainty. Always cite sources."
    ),
    
    "summarizer": (
        "You are an expert research summarizer. "
        "Create concise, accurate summaries that capture key points and findings. "
        "Maintain technical accuracy while being clear and organized."
    )
}


def get_system_message(mode: str = "default") -> str:
    """
    Get system message for specific mode
    
    Args:
        mode: One of: default, technical, beginner_friendly, critical, summarizer
    
    Returns:
        System message string
    """
    return SYSTEM_MESSAGES.get(mode, SYSTEM_MESSAGES["default"])


def main():
    """Example usage"""
    print("\n" + "="*70)
    print("Prompt Templates Demo")
    print("="*70)
    
    # Sample contexts
    contexts = [
        {
            "text": "Transformers use self-attention mechanisms to process input sequences in parallel, enabling efficient training on GPUs.",
            "score": 0.95,
            "metadata": {
                "paper_id": "2301.12345",
                "section": "Methods",
                "has_table": False
            }
        },
        {
            "text": "The attention mechanism computes weighted sums of value vectors, where weights are determined by compatibility between query and key vectors.",
            "score": 0.87,
            "metadata": {
                "paper_id": "2302.67890",
                "section": "Background",
                "has_table": True
            }
        }
    ]
    
    # Test 1: Basic RAG prompt
    print("\n1. Basic RAG Prompt:")
    print("-"*70)
    
    query = "How do transformers process sequences?"
    prompt = format_rag_prompt(query, contexts)
    print(prompt[:500] + "...\n")
    
    # Test 2: Summarization prompt
    print("\n2. Summarization Prompt:")
    print("-"*70)
    
    prompt = format_summarization_prompt(contexts, focus="attention mechanisms")
    print(prompt[:400] + "...\n")
    
    # Test 3: System messages
    print("\n3. System Messages:")
    print("-"*70)
    
    for mode in ["default", "technical", "beginner_friendly"]:
        print(f"\n{mode.upper()}:")
        print(get_system_message(mode))
    
    print("\n" + "="*70)
    print("✓ Prompt templates working correctly!")
    print("="*70)


if __name__ == "__main__":
    main()

