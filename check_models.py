#!/usr/bin/env python3
"""
Check available Gemini models
"""

import os
from dotenv import load_dotenv
load_dotenv()

import google.generativeai as genai

def check_available_models():
    """Check what models are available"""
    try:
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        models = genai.list_models()
        print("Available models:")
        for model in models:
            print(f"  - {model.name}")
    except Exception as e:
        print(f"Error checking models: {e}")

if __name__ == "__main__":
    check_available_models()
