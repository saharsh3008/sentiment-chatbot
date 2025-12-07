#!/usr/bin/env python3
"""
Test script to verify environment variables are loaded correctly
"""

import os

try:
    from dotenv import load_dotenv
    load_dotenv()
    print("✅ python-dotenv loaded successfully")
except ImportError:
    print("❌ python-dotenv not installed")

# Check for API keys
gemini_key = os.getenv('GEMINI_API_KEY')
openai_key = os.getenv('OPENAI_API_KEY')

print(f"Gemini API Key: {'✅ Set' if gemini_key else '❌ Not set'}")
print(f"OpenAI API Key: {'✅ Set' if openai_key else '❌ Not set'}")

if gemini_key:
    print(f"Gemini key starts with: {gemini_key[:10]}...")
if openai_key:
    print(f"OpenAI key starts with: {openai_key[:10]}...")
