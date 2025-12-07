#!/usr/bin/env python3
"""
Test script to verify Gemini AI conversational responses
"""

from chatbot import SentimentChatbot

def test_ai_responses():
    """Test AI responses with conversation history"""
    chatbot = SentimentChatbot()

    # Test messages that should trigger AI responses
    test_messages = [
        "Hello, I'm feeling great today!",
        "I love coding with Python",
        "What do you think about AI?",
        "Tell me more about machine learning"
    ]

    print("Testing Gemini AI conversational responses...")
    print("="*60)

    for i, message in enumerate(test_messages, 1):
        print(f"\nTest {i}: '{message}'")
        bot_response, sentiment = chatbot.process_message(message)

        print(f"Sentiment: {sentiment.emoji} {sentiment.sentiment} ({sentiment.confidence}%)")
        print(f"Bot Response: {bot_response}")

        # Check if response looks AI-generated (not from fallback list)
        fallback_responses = [
            "I'm glad to hear that! How else can I assist you today?",
            "That's wonderful! Is there anything else you'd like to discuss?",
            "Great to hear positive feedback! What else can I help with?",
            "I appreciate your kind words! How can I continue to help?",
            "Thank you for sharing! What would you like to explore next?",
            "I'm sorry to hear that. Let me help address your concerns.",
            "I understand your frustration. How can I make this better for you?",
            "I apologize for the inconvenience. What can I do to assist you?",
            "Your feedback is important. Let me see how I can help resolve this.",
            "I'm here to help. Can you tell me more about what went wrong?",
            "I understand. How can I assist you further?",
            "Thank you for sharing. What else would you like to know?",
            "I'm here to help. What would you like to discuss?",
            "Got it. Is there anything specific I can help you with?",
            "I'm listening. How can I support you today?"
        ]

        is_ai_response = bot_response not in fallback_responses
        print(f"AI Generated: {'✅ Yes' if is_ai_response else '❌ No (fallback)'}")

    print("\n" + "="*60)
    print("Conversation History:")
    for msg in chatbot.messages:
        if msg.role == 'user':
            print(f"User: {msg.content}")
        else:
            print(f"Bot: {msg.content}")

if __name__ == "__main__":
    test_ai_responses()
