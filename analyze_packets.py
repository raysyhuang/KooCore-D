#!/usr/bin/env python3
"""
Automated LLM analysis of momentum screener packets using OpenAI or Anthropic API.

Usage:
    python analyze_packets.py [--date YYYY-MM-DD] [--provider openai|anthropic] [--model MODEL_NAME]
    
Environment variables:
    OPENAI_API_KEY - OpenAI API key (if using OpenAI)
    ANTHROPIC_API_KEY - Anthropic API key (if using Anthropic)
"""

import os
import sys
import argparse
import datetime as dt
from pathlib import Path

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # python-dotenv not installed, skip (will use environment variables only)
    pass

def load_api_libraries(provider: str):
    """Dynamically load the appropriate API library."""
    if provider == "openai":
        try:
            from openai import OpenAI
            return OpenAI, None
        except ImportError:
            print("Error: openai library not installed. Install with: pip install openai")
            sys.exit(1)
    elif provider == "anthropic":
        try:
            import anthropic
            return None, anthropic
        except ImportError:
            print("Error: anthropic library not installed. Install with: pip install anthropic")
            sys.exit(1)
    else:
        raise ValueError(f"Unknown provider: {provider}")

def read_batch_prompt(prompt_file: str) -> str:
    """Read the batch prompt template."""
    with open(prompt_file, "r", encoding="utf-8") as f:
        return f.read()

def read_packets(packets_file: str) -> str:
    """Read the LLM packets content."""
    with open(packets_file, "r", encoding="utf-8") as f:
        return f.read()

def call_openai(client, prompt: str, model: str = "gpt-4o") -> str:
    """Call OpenAI API."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Lower temperature for more consistent analysis
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        raise

def call_anthropic(client, prompt: str, model: str = "claude-sonnet-4-5-20250929") -> str:
    """Call Anthropic API."""
    try:
        message = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=0.3,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return message.content[0].text
    except Exception as e:
        print(f"Error calling Anthropic API: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Analyze momentum screener packets using LLM API")
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date folder (YYYY-MM-DD). Defaults to today's date."
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=["openai", "anthropic", "both"],
        default="openai",
        help="API provider to use: openai, anthropic, or both (default: openai)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (default: gpt-5.2 for OpenAI, claude-sonnet-4-5-20250929 for Anthropic). Ignored if --provider both."
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key (optional, otherwise uses environment variable)"
    )
    
    args = parser.parse_args()
    
    # Determine date folder
    if args.date:
        date_str = args.date
    else:
        date_str = dt.date.today().strftime("%Y-%m-%d")
    
    # Set default models
    if args.provider == "both":
        # When running both, ignore model argument and use defaults
        gpt_model = "gpt-5.2"
        claude_model = "claude-sonnet-4-5-20250929"
    elif args.model is None:
        if args.provider == "openai":
            args.model = "gpt-5.2"
        else:
            args.model = "claude-sonnet-4-5-20250929"
    
    # Build file paths
    script_dir = Path(__file__).parent
    date_dir = script_dir / date_str
    prompt_file = script_dir / "batch_prompt_gpt.txt"
    packets_file = date_dir / f"llm_packets_{date_str}.txt"
    
    # Validate files exist
    if not prompt_file.exists():
        print(f"Error: Prompt file not found: {prompt_file}")
        sys.exit(1)
    
    if not packets_file.exists():
        print(f"Error: Packets file not found: {packets_file}")
        sys.exit(1)
    
    # Read files (only once, used for both if needed)
    print(f"Reading prompt from: {prompt_file}")
    batch_prompt = read_batch_prompt(str(prompt_file))
    
    print(f"Reading packets from: {packets_file}")
    packets_content = read_packets(str(packets_file))
    
    # Combine prompt and packets
    full_prompt = batch_prompt.rstrip() + "\n\n" + packets_content
    
    # Handle both providers
    if args.provider == "both":
        providers_to_run = [("openai", gpt_model), ("anthropic", claude_model)]
    else:
        providers_to_run = [(args.provider, args.model)]
    
    # Run analyses
    for provider, model in providers_to_run:
        print(f"\n{'='*60}")
        print(f"Running {provider.upper()} analysis (model: {model})...")
        print(f"{'='*60}")
        
        # Load API libraries
        OpenAI_lib, anthropic_lib = load_api_libraries(provider)
        
        # Initialize API client
        if provider == "openai":
            api_key = args.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                print(f"Error: OPENAI_API_KEY not found. Set OPENAI_API_KEY environment variable or use --api-key")
                continue
            client = OpenAI_lib(api_key=api_key)
        else:
            api_key = args.api_key or os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                print(f"Error: ANTHROPIC_API_KEY not found. Set ANTHROPIC_API_KEY environment variable or use --api-key")
                continue
            client = anthropic_lib.Anthropic(api_key=api_key)
        
        # Create output filename
        provider_prefix = "gpt" if provider == "openai" else "claude"
        model_short = model.replace("-", "_").replace(".", "_")
        output_file = date_dir / f"{provider_prefix}_{model_short}_analysis_{date_str}.txt"
        
        # Call API
        print(f"Calling {provider} API...")
        print("This may take a minute...")
        
        try:
            if provider == "openai":
                response = call_openai(client, full_prompt, model)
            else:
                response = call_anthropic(client, full_prompt, model)
            
            # Save output
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(response)
            
            print(f"\n{provider.upper()} analysis complete!")
            print(f"Output saved to: {output_file}\n")
            
        except Exception as e:
            print(f"\nError during {provider} API call: {e}")
            print(f"Skipping {provider} and continuing...\n")
            continue
    
    print("="*60)
    print("All requested analyses completed!")
    print("="*60)

if __name__ == "__main__":
    main()

