"""
Advanced Features for Sentiment Chatbot (BONUS)
Includes: Emotion detection, sarcasm detection, export functionality
"""

from chatbot import SentimentChatbot, SentimentAnalyzer
import json
import csv
from datetime import datetime
from typing import Dict, List


class EmotionDetector:
    """Detect specific emotions beyond basic sentiment"""
    
    def __init__(self):
        self.emotions = {
            'joy': ['happy', 'excited', 'delighted', 'thrilled', 'elated', 'cheerRwhich one is test_chatbot ?'],
'anger': ['angry', 'furious', 'mad', 'rage', 'irritated', 'annoyed'],
'fear': ['scared', 'afraid', 'worried', 'anxious', 'nervous', 'terrified'],
'sadness': ['sad', 'depressed', 'miserable', 'unhappy', 'down', 'blue'],
'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned'],
'disgust': ['disgusting', 'revolting', 'gross', 'nasty', 'awful']
}
def detect_emotion(self, text: str) -> Dict[str, int]:
    """Detect emotions in text"""
    text_lower = text.lower()
    emotion_counts = {emotion: 0 for emotion in self.emotions}
    
    for emotion, words in self.emotions.items():
        for word in words:
            if word in text_lower:
                emotion_counts[emotion] += 1
    
    return emotion_counts

def get_primary_emotion(self, text: str) -> str:
    """Get the primary emotion"""
    counts = self.detect_emotion(text)
    if sum(counts.values()) == 0:
        return 'neutral'
    return max(counts, key=counts.get)
class SarcasmDetector:
  """Basic sarcasm detection"""
@staticmethod
def detect_sarcasm(text: str, sentiment_score: float) -> bool:
    """
    Detect potential sarcasm
    Rules: Positive words + exclamation + negative context
    """
    has_exclamation = '!' in text
    has_negative_words = any(word in text.lower() for word in 
                            ['sure', 'right', 'yeah', 'great', 'wonderful'])
    
    # If positive sentiment but suspicious markers
    if sentiment_score > 0 and has_exclamation and has_negative_words:
        return True
    
    return False
class ConversationExporter:
     """Export conversations in various formats"""
@staticmethod
def export_to_json(chatbot: SentimentChatbot, filename: str = None):
    """Export conversation to JSON"""
    if filename is None:
        filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    data = {
        'timestamp': datetime.now().isoformat(),
        'messages': [],
        'analysis': chatbot.get_conversation_analysis()
    }
    
    for msg in chatbot.messages:
        msg_dict = {
            'role': msg.role,
            'content': msg.content,
            'timestamp': msg.timestamp
        }
        if msg.sentiment_data:
            msg_dict['sentiment'] = {
                'sentiment': msg.sentiment_data.sentiment,
                'score': msg.sentiment_data.score,
                'confidence': msg.sentiment_data.confidence
            }
        data['messages'].append(msg_dict)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Conversation exported to {filename}")
    return filename

@staticmethod
def export_to_csv(chatbot: SentimentChatbot, filename: str = None):
    """Export conversation to CSV"""
    if filename is None:
        filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Role', 'Content', 'Timestamp', 'Sentiment', 'Score', 'Confidence'])
        
        for msg in chatbot.messages:
            if msg.role == 'user' and msg.sentiment_data:
                writer.writerow([
                    msg.role,
                    msg.content,
                    msg.timestamp,
                    msg.sentiment_data.sentiment,
                    f"{msg.sentiment_data.score:.3f}",
                    f"{msg.sentiment_data.confidence:.2f}"
                ])
            else:
                writer.writerow([msg.role, msg.content, msg.timestamp, '', '', ''])
    
    print(f"✅ Conversation exported to {filename}")
    return filename

@staticmethod
def export_to_text(chatbot: SentimentChatbot, filename: str = None):
    """Export conversation to readable text format"""
    if filename is None:
        filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("SENTIMENT ANALYSIS CHATBOT - CONVERSATION EXPORT\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*60 + "\n\n")
        
        for msg in chatbot.messages:
            if msg.role == 'user':
                f.write(f"[{msg.timestamp}] USER: {msg.content}\n")
                if msg.sentiment_data:
                    f.write(f"    → Sentiment: {msg.sentiment_data.emoji} {msg.sentiment_data.sentiment} ")
                    f.write(f"(Confidence: {msg.sentiment_data.confidence}%, Score: {msg.sentiment_data.score:.3f})\n")
            else:
                f.write(f"[{msg.timestamp}] BOT: {msg.content}\n")
            f.write("\n")
        
        # Add summary
        analysis = chatbot.get_conversation_analysis()
        if analysis:
            f.write("\n" + "="*60 + "\n")
            f.write("CONVERSATION SUMMARY\n")
            f.write("="*60 + "\n")
            f.write(f"Overall Sentiment: {analysis['emoji']} {analysis['overall_sentiment']}\n")
            f.write(f"Average Score: {analysis['average_score']}\n")
            f.write(f"Confidence: {analysis['average_confidence']}%\n")
            f.write(f"Mood Trend: {analysis['trend']}\n")
            f.write(f"\nSentiment Breakdown:\n")
            for sent, count in analysis['sentiment_counts'].items():
                f.write(f"  {sent}: {count} messages\n")
    
    print(f"✅ Conversation exported to {filename}")
    return filename
class AdvancedChatbot(SentimentChatbot):
    """Extended chatbot with advanced features"""
def __init__(self):
    super().__init__()
    self.emotion_detector = EmotionDetector()
    self.sarcasm_detector = SarcasmDetector()
    self.exporter = ConversationExporter()

def process_message_advanced(self, user_input: str):
    """Process message with advanced analysis"""
    # Get basic sentiment
    bot_response, sentiment_result = self.process_message(user_input)
    
    # Detect emotion
    primary_emotion = self.emotion_detector.get_primary_emotion(user_input)
    
    # Detect sarcasm
    is_sarcastic = self.sarcasm_detector.detect_sarcasm(user_input, sentiment_result.score)
    
    return {
        'bot_response': bot_response,
        'sentiment': sentiment_result.sentiment,
        'score': sentiment_result.score,
        'confidence': sentiment_result.confidence,
        'emotion': primary_emotion,
        'sarcasm': is_sarcastic
    }

def run_advanced_interactive(self):
    """Run with advanced features"""
    print("="*60)
    print("🤖 ADVANCED SENTIMENT ANALYSIS CHATBOT")
    print("="*60)
    print("Commands:")
    print("  'analyze' - Show conversation analysis")
    print("  'export json' - Export to JSON")
    print("  'export csv' - Export to CSV")
    print("  'export txt' - Export to text")
    print("  'quit' - Exit")
    print("="*60)
    
    while True:
        user_input = input("\n💬 You: ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() == 'quit':
            print("\nEnding conversation...")
            break
        
        if user_input.lower() == 'analyze':
            self.display_conversation_summary()
            continue
        
        if user_input.lower().startswith('export'):
            parts = user_input.lower().split()
            if len(parts) == 2:
                format_type = parts[1]
                if format_type == 'json':
                    self.exporter.export_to_json(self)
                elif format_type == 'csv':
                    self.exporter.export_to_csv(self)
                elif format_type == 'txt':
                    self.exporter.export_to_text(self)
                else:
                    print("Unknown format. Use: json, csv, or txt")
            continue
        
        # Process with advanced features
        result = self.process_message_advanced(user_input)
        
        print(f"\n{'='*60}")
        print(f"💬 You: {user_input}")
        print(f"→ Sentiment: {result['sentiment']} (Confidence: {result['confidence']}%)")
        print(f"→ Primary Emotion: {result['emotion']}")
        if result['sarcasm']:
            print(f"→ ⚠️  Possible sarcasm detected")
        print(f"\n🤖 Chatbot: {result['bot_response']}")
    
    # Final analysis
    self.display_conversation_summary()
def demo_advanced_features():
    """Demonstrate advanced features"""
print("🎯 ADVANCED FEATURES DEMO\n")
chatbot = AdvancedChatbot()

# Test messages
test_messages = [
    "I'm absolutely thrilled with your service!",
    "This makes me so angry and frustrated",
    "Yeah, right, this is just great...",  # Sarcasm
    "I'm worried about the results"
]

for msg in test_messages:
    result = chatbot.process_message_advanced(msg)
    print(f"\nMessage: {msg}")
    print(f"  Sentiment: {result['sentiment']}")
    print(f"  Emotion: {result['emotion']}")
    print(f"  Sarcasm: {'Yes ⚠️' if result['sarcasm'] else 'No'}")
    print(f"  Response: {result['bot_response']}")

# Export examples
print("\n" + "="*60)
print("EXPORT DEMO")
print("="*60)
chatbot.exporter.export_to_json(chatbot)
chatbot.exporter.export_to_csv(chatbot)
chatbot.exporter.export_to_text(chatbot)
if name == "main":
# Run demo
  demo_advanced_features()
# Or run advanced interactive mode
# chatbot = AdvancedChatbot()
# chatbot.run_advanced_interactive()