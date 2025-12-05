"""
Sentiment Analysis Chatbot - Main Application
A production-ready chatbot with advanced sentiment analysis capabilities
"""

import re
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import json


@dataclass
class SentimentResult:
    """Data class for sentiment analysis results"""
    sentiment: str
    score: float
    confidence: float
    positive_count: int
    negative_count: int
    emoji: str


@dataclass
class Message:
    """Data class for chat messages"""
    role: str
    content: str
    timestamp: str
    sentiment_data: SentimentResult = None


class SentimentAnalyzer:
    """Advanced sentiment analyzer with lexicon-based approach"""
    
    def __init__(self):
        # Expanded sentiment lexicons
        self.positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
            'love', 'happy', 'pleased', 'satisfied', 'perfect', 'awesome', 
            'brilliant', 'better', 'best', 'outstanding', 'superb', 'delighted',
            'impressed', 'thank', 'thanks', 'appreciate', 'helpful', 'nice',
            'glad', 'enjoy', 'enjoyed', 'beautiful', 'terrific', 'splendid',
            'fabulous', 'remarkable', 'exceptional', 'positive', 'joy', 'success'
        }
        
        self.negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'disappointing', 'disappoints',
            'disappointed', 'poor', 'worst', 'hate', 'angry', 'upset', 'frustrated',
            'frustrating', 'annoyed', 'annoying', 'useless', 'pathetic', 'disgusting',
            'unacceptable', 'failed', 'failure', 'broken', 'issue', 'problem',
            'slow', 'difficult', 'confusing', 'complaint', 'unhappy', 'sad',
            'dislike', 'worse', 'never', 'rubbish', 'garbage', 'stupid',
            'ridiculous', 'waste', 'negative', 'disaster', 'sorry', 'wrong'
        }
        
        self.intensifiers = {'very', 'extremely', 'really', 'absolutely', 'totally', 'completely'}
        self.negations = {'not', 'no', 'never', 'neither', 'nobody', 'nothing', "n't", 'dont', "don't"}
    
    def analyze(self, text: str) -> SentimentResult:
        """
        Perform sentiment analysis on the given text
        
        Args:
            text: Input text to analyze
            
        Returns:
            SentimentResult with detailed sentiment information
        """
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        score = 0
        positive_count = 0
        negative_count = 0
        
        for i, word in enumerate(words):
            prev_word = words[i - 1] if i > 0 else ''
            is_negated = prev_word in self.negations
            is_intensified = prev_word in self.intensifiers
            multiplier = 1.5 if is_intensified else 1.0
            
            if word in self.positive_words:
                if is_negated:
                    score -= 1 * multiplier
                    negative_count += 1
                else:
                    score += 1 * multiplier
                    positive_count += 1
            
            if word in self.negative_words:
                if is_negated:
                    score += 0.5 * multiplier
                    positive_count += 1
                else:
                    score -= 1 * multiplier
                    negative_count += 1
        
        # Normalize score
        total_words = positive_count + negative_count
        normalized_score = score / total_words if total_words > 0 else 0
        
        # Determine sentiment with confidence
        if normalized_score > 0.3:
            sentiment = 'Positive'
            confidence = min(95, 60 + normalized_score * 50)
            emoji = '😊'
        elif normalized_score < -0.3:
            sentiment = 'Negative'
            confidence = min(95, 60 + abs(normalized_score) * 50)
            emoji = '😔'
        else:
            sentiment = 'Neutral'
            confidence = max(50, 70 - abs(normalized_score) * 50)
            emoji = '😐'
        
        return SentimentResult(
            sentiment=sentiment,
            score=normalized_score,
            confidence=round(confidence, 2),
            positive_count=positive_count,
            negative_count=negative_count,
            emoji=emoji
        )


class ResponseGenerator:
    """Generate contextual chatbot responses based on sentiment"""
    
    def __init__(self):
        self.responses = {
            'Positive': [
                "I'm glad to hear that! How else can I assist you today?",
                "That's wonderful! Is there anything else you'd like to discuss?",
                "Great to hear positive feedback! What else can I help with?",
                "I appreciate your kind words! How can I continue to help?",
                "Thank you for sharing! What would you like to explore next?"
            ],
            'Negative': [
                "I'm sorry to hear that. Let me help address your concerns.",
                "I understand your frustration. How can I make this better for you?",
                "I apologize for the inconvenience. What can I do to assist you?",
                "Your feedback is important. Let me see how I can help resolve this.",
                "I'm here to help. Can you tell me more about what went wrong?"
            ],
            'Neutral': [
                "I understand. How can I assist you further?",
                "Thank you for sharing. What else would you like to know?",
                "I'm here to help. What would you like to discuss?",
                "Got it. Is there anything specific I can help you with?",
                "I'm listening. How can I support you today?"
            ]
        }
    
    def generate(self, user_message: str, sentiment: str) -> str:
        """Generate appropriate response based on sentiment"""
        import random
        responses = self.responses.get(sentiment, self.responses['Neutral'])
        return random.choice(responses)


class ConversationAnalyzer:
    """Analyze entire conversation for trends and overall sentiment"""
    
    @staticmethod
    def analyze_conversation(messages: List[Message]) -> Dict:
        """
        Analyze the complete conversation for overall sentiment and trends
        
        Args:
            messages: List of Message objects
            
        Returns:
            Dictionary with comprehensive conversation analysis
        """
        user_messages = [msg for msg in messages if msg.role == 'user' and msg.sentiment_data]
        
        if not user_messages:
            return None
        
        total_score = sum(msg.sentiment_data.score for msg in user_messages)
        total_confidence = sum(msg.sentiment_data.confidence for msg in user_messages)
        
        sentiment_counts = {'Positive': 0, 'Negative': 0, 'Neutral': 0}
        score_history = []
        
        for msg in user_messages:
            sentiment_counts[msg.sentiment_data.sentiment] += 1
            score_history.append(msg.sentiment_data.score)
        
        avg_score = total_score / len(user_messages)
        avg_confidence = total_confidence / len(user_messages)
        
        # Detect trend
        trend = 'Stable'
        if len(score_history) >= 3:
            mid_point = len(score_history) // 2
            first_half = score_history[:mid_point]
            second_half = score_history[mid_point:]
            
            first_avg = sum(first_half) / len(first_half)
            second_avg = sum(second_half) / len(second_half)
            
            if second_avg - first_avg > 0.2:
                trend = 'Improving'
            elif first_avg - second_avg > 0.2:
                trend = 'Declining'
        
        # Determine overall sentiment
        if avg_score > 0.2:
            overall_sentiment = 'Positive'
            emoji = '😊'
        elif avg_score < -0.2:
            overall_sentiment = 'Negative'
            emoji = '😔'
        else:
            overall_sentiment = 'Neutral'
            emoji = '😐'
        
        return {
            'overall_sentiment': overall_sentiment,
            'emoji': emoji,
            'average_score': round(avg_score, 3),
            'average_confidence': round(avg_confidence, 2),
            'sentiment_counts': sentiment_counts,
            'trend': trend,
            'message_count': len(user_messages),
            'score_history': [round(s, 3) for s in score_history]
        }


class SentimentChatbot:
    """Main chatbot class integrating all components"""
    
    def __init__(self):
        self.messages: List[Message] = []
        self.sentiment_analyzer = SentimentAnalyzer()
        self.response_generator = ResponseGenerator()
        self.conversation_analyzer = ConversationAnalyzer()
    
    def process_message(self, user_input: str) -> Tuple[str, SentimentResult]:
        """
        Process user message and generate response with sentiment analysis
        
        Args:
            user_input: User's message
            
        Returns:
            Tuple of (bot_response, sentiment_result)
        """
        # Analyze sentiment
        sentiment_result = self.sentiment_analyzer.analyze(user_input)
        
        # Store user message
        user_message = Message(
            role='user',
            content=user_input,
            timestamp=datetime.now().strftime('%H:%M:%S'),
            sentiment_data=sentiment_result
        )
        self.messages.append(user_message)
        
        # Generate bot response
        bot_response = self.response_generator.generate(user_input, sentiment_result.sentiment)
        
        # Store bot message
        bot_message = Message(
            role='bot',
            content=bot_response,
            timestamp=datetime.now().strftime('%H:%M:%S')
        )
        self.messages.append(bot_message)
        
        return bot_response, sentiment_result
    
    def get_conversation_analysis(self) -> Dict:
        """Get comprehensive analysis of the entire conversation"""
        return self.conversation_analyzer.analyze_conversation(self.messages)
    
    def display_message_with_sentiment(self, message: Message):
        """Display a message with its sentiment analysis (Tier 2)"""
        if message.role == 'user':
            print(f"\n{'='*60}")
            print(f"User: {message.content}")
            if message.sentiment_data:
                print(f"→ Sentiment: {message.sentiment_data.emoji} {message.sentiment_data.sentiment} "
                      f"(Confidence: {message.sentiment_data.confidence}%)")
                print(f"  Score: {message.sentiment_data.score:.3f} | "
                      f"Positive words: {message.sentiment_data.positive_count} | "
                      f"Negative words: {message.sentiment_data.negative_count}")
        else:
            print(f"\nChatbot: {message.content}")
    
    def display_conversation_summary(self):
        """Display comprehensive conversation analysis (Tier 1 + Tier 2 bonus)"""
        analysis = self.get_conversation_analysis()
        
        if not analysis:
            print("\nNo conversation to analyze yet.")
            return
        
        print(f"\n{'='*60}")
        print("CONVERSATION SUMMARY - SENTIMENT ANALYSIS")
        print(f"{'='*60}")
        
        print(f"\n📊 OVERALL SENTIMENT: {analysis['emoji']} {analysis['overall_sentiment'].upper()}")
        print(f"   Average Score: {analysis['average_score']}")
        print(f"   Confidence: {analysis['average_confidence']}%")
        
        print(f"\n📈 MOOD TREND: {analysis['trend']}")
        
        print(f"\n📉 SENTIMENT BREAKDOWN:")
        for sentiment, count in analysis['sentiment_counts'].items():
            emoji = '😊' if sentiment == 'Positive' else '😔' if sentiment == 'Negative' else '😐'
            print(f"   {emoji} {sentiment}: {count} messages")
        
        print(f"\n📝 STATISTICS:")
        print(f"   Total messages analyzed: {analysis['message_count']}")
        print(f"   Score progression: {' → '.join([f'{s:.2f}' for s in analysis['score_history']])}")
        
        print(f"\n{'='*60}")
    
    def run_interactive(self):
        """Run the chatbot in interactive CLI mode"""
        print("="*60)
        print("🤖 SENTIMENT ANALYSIS CHATBOT")
        print("="*60)
        print("Type your messages to chat. Commands:")
        print("  'analyze' - Show conversation analysis")
        print("  'quit' - Exit and show final analysis")
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
            
            # Process message
            bot_response, sentiment_result = self.process_message(user_input)
            
            # Display with sentiment (Tier 2)
            print(f"\n{'='*60}")
            print(f"💬 You: {user_input}")
            print(f"→ Sentiment: {sentiment_result.emoji} {sentiment_result.sentiment} "
                  f"(Confidence: {sentiment_result.confidence}%)")
            print(f"\n🤖 Chatbot: {bot_response}")
        
        # Final analysis
        self.display_conversation_summary()


def main():
    """Main entry point"""
    chatbot = SentimentChatbot()
    chatbot.run_interactive()


if __name__ == "__main__":
    main()