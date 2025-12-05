"""
API Extension for Sentiment Chatbot (BONUS FEATURE)
Provides REST API endpoints for integration with other applications

Usage:
    python api_chatbot.py

Then access:
    http://localhost:5000/chat (POST) - Send a message
    http://localhost:5000/analysis (GET) - Get conversation analysis
    http://localhost:5000/reset (POST) - Reset conversation

Note: Requires Flask (pip install flask)
This is an OPTIONAL bonus feature demonstrating production-ready deployment
"""

try:
    from flask import Flask, request, jsonify
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("Flask not installed. Install with: pip install flask flask-cors")
    print("This is optional - the core chatbot works without it.")

from chatbot import SentimentChatbot
import json


if FLASK_AVAILABLE:
    app = Flask(__name__)
    CORS(app)  # Enable CORS for web interface
    
    # Global chatbot instance (in production, use session management)
    chatbot = SentimentChatbot()
    
    
    @app.route('/')
    def home():
        """API documentation"""
        return jsonify({
            'name': 'Sentiment Analysis Chatbot API',
            'version': '1.0',
            'endpoints': {
                'POST /chat': 'Send a message and get response with sentiment',
                'GET /analysis': 'Get full conversation analysis',
                'POST /reset': 'Reset conversation history',
                'GET /health': 'Health check'
            },
            'example_request': {
                'url': '/chat',
                'method': 'POST',
                'body': {'message': 'Your service is amazing!'}
            }
        })
    
    
    @app.route('/health')
    def health():
        """Health check endpoint"""
        return jsonify({
            'status': 'healthy',
            'service': 'sentiment-chatbot',
            'messages_count': len(chatbot.messages)
        })
    
    
    @app.route('/chat', methods=['POST'])
    def chat():
        """
        Process a chat message
        
        Request body:
            {
                "message": "Your text here"
            }
        
        Response:
            {
                "user_message": "Your text here",
                "bot_response": "Response...",
                "sentiment": {
                    "sentiment": "Positive",
                    "score": 0.75,
                    "confidence": 85.5,
                    "emoji": "😊"
                }
            }
        """
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'Missing message in request'}), 400
        
        user_message = data['message'].strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Process message
        bot_response, sentiment_result = chatbot.process_message(user_message)
        
        return jsonify({
            'user_message': user_message,
            'bot_response': bot_response,
            'sentiment': {
                'sentiment': sentiment_result.sentiment,
                'score': round(sentiment_result.score, 3),
                'confidence': sentiment_result.confidence,
                'emoji': sentiment_result.emoji,
                'positive_count': sentiment_result.positive_count,
                'negative_count': sentiment_result.negative_count
            },
            'timestamp': chatbot.messages[-1].timestamp
        })
    
    
    @app.route('/analysis', methods=['GET'])
    def analysis():
        """
        Get full conversation analysis
        
        Response:
            {
                "overall_sentiment": "Positive",
                "average_score": 0.45,
                "trend": "Improving",
                "message_count": 5,
                ...
            }
        """
        analysis_result = chatbot.get_conversation_analysis()
        
        if not analysis_result:
            return jsonify({
                'message': 'No conversation yet',
                'message_count': 0
            })
        
        return jsonify(analysis_result)
    
    
    @app.route('/reset', methods=['POST'])
    def reset():
        """Reset conversation history"""
        global chatbot
        chatbot = SentimentChatbot()
        
        return jsonify({
            'message': 'Conversation reset successfully',
            'status': 'success'
        })
    
    
    @app.route('/messages', methods=['GET'])
    def get_messages():
        """Get all messages in the conversation"""
        messages = []
        for msg in chatbot.messages:
            message_dict = {
                'role': msg.role,
                'content': msg.content,
                'timestamp': msg.timestamp
            }
            if msg.sentiment_data:
                message_dict['sentiment'] = {
                    'sentiment': msg.sentiment_data.sentiment,
                    'score': round(msg.sentiment_data.score, 3),
                    'confidence': msg.sentiment_data.confidence,
                    'emoji': msg.sentiment_data.emoji
                }
            messages.append(message_dict)
        
        return jsonify({
            'messages': messages,
            'count': len(messages)
        })
    
    
    @app.errorhandler(404)
    def not_found(e):
        return jsonify({'error': 'Endpoint not found'}), 404
    
    
    @app.errorhandler(500)
    def server_error(e):
        return jsonify({'error': 'Internal server error'}), 500
    
    
def run_api(host='0.0.0.0', port=5000, debug=False):
    """Run the API server"""
    print("="*60)
    print("🚀 SENTIMENT CHATBOT API")
    print("="*60)
    print(f"Server running on http://{host}:{port}")
    print("\nAvailable endpoints:")
    print("  POST   /chat      - Send a message")
    print("  GET    /analysis  - Get conversation analysis")
    print("  GET    /messages  - Get all messages")
    print("  POST   /reset     - Reset conversation")
    print("  GET    /health    - Health check")
    print("\nExample usage:")
    print('  curl -X POST http://localhost:5000/chat \\')
    print("       -H 'Content-Type: application/json' \\")
    print("       -d '{\"message\": \"Hello!\"}'")
    print("="*60)

    # Use environment variables for production
    import os
    host = os.environ.get('HOST', host)
    port = int(os.environ.get('PORT', port))
    debug = os.environ.get('FLASK_ENV') == 'development'

    app.run(host=host, port=port, debug=debug)


def main():
    """Main entry point"""
    if not FLASK_AVAILABLE:
        print("\n❌ Flask is not installed.")
        print("This is an OPTIONAL bonus feature for API deployment.")
        print("\nTo install: pip install flask flask-cors")
        print("Or use the main chatbot: python chatbot.py")
        return
    
    # Run API server
    run_api(debug=True)


if __name__ == "__main__":
    main()