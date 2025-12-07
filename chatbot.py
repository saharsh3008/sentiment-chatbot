"""
Sentiment Analysis Chatbot - Main Application
A production-ready chatbot with advanced sentiment analysis capabilities
"""

import re
import os
from datetime import datetime
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
import json

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("python-dotenv not installed. Install with: pip install python-dotenv")
    print("Environment variables will be read from system environment.")

# Check for AI availability
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Google Generative AI library not installed. Install with: pip install google-generativeai")
    print("AI responses will use intelligent fallback mode.")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI library not installed. Install with: pip install openai")
    print("AI responses will use intelligent fallback mode.")


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
            'fabulous', 'remarkable', 'exceptional', 'positive', 'joy', 'success',
            'joyful', 'smiling', 'smile', 'laughing', 'laugh', 'cheerful', 'cheer',
            'excited', 'exciting', 'thrilled', 'thrilling', 'ecstatic', 'blissful', 'bliss',
            'content', 'contented', 'pleasure', 'pleasant', 'fun', 'funny', 'hilarious',
            'amusing', 'entertaining', 'relaxed', 'relaxing', 'peaceful', 'calm', 'serene',
            'tranquil', 'comfortable', 'cozy', 'warm', 'loving', 'affectionate', 'caring',
            'kind', 'generous', 'grateful', 'gratitude', 'blessed', 'fortunate', 'lucky',
            'proud', 'accomplished', 'achieved', 'successful', 'victorious', 'triumphant',
            'confident', 'optimistic', 'hopeful', 'encouraging', 'inspiring', 'motivated',
            'enthusiastic', 'passionate', 'vibrant', 'energetic', 'lively', 'vital',
            'radiant', 'glowing', 'sparkling', 'shining', 'magnificent',
            'spectacular', 'marvelous', 'super', 'incredible', 'unbelievable', 'phenomenal'
        }
        
        self.negative_words = {
            # Strong negative words
            'kill', 'murder', 'death', 'die', 'dead', 'hate', 'hated', 'hating',
            'destroy', 'destruction', 'ruin', 'ruined', 'damn', 'damned', 'hell',
            'suck', 'sucks', 'sucked', 'fuck', 'fucking', 'shit', 'shitty',
            'asshole', 'bastard', 'bitch', 'crap', 'craps', 'bullshit',

            # Emotional negative words
            'angry', 'rage', 'furious', 'enraged', 'infuriated', 'outraged',
            'frustrated', 'frustrating', 'annoyed', 'annoying', 'irritated',
            'irritating', 'upset', 'disturbed', 'distressed', 'worried',
            'anxious', 'nervous', 'scared', 'afraid', 'terrified', 'frightened',
            'horrified', 'shocked', 'disgusted', 'repulsed', 'revolted',

            # Quality negative words
            'bad', 'terrible', 'awful', 'horrible', 'atrocious', 'abominable',
            'dreadful', 'vile', 'wretched', 'miserable', 'pathetic', 'pitiful',
            'lousy', 'crappy', 'shoddy', 'inferior', 'substandard', 'defective',
            'broken', 'damaged', 'ruined', 'spoiled', 'rotten', 'stale',

            # Experience negative words
            'disappointing', 'disappointed', 'disappointment', 'letdown',
            'failed', 'failure', 'failing', 'unsuccessful', 'useless',
            'worthless', 'pointless', 'meaningless', 'stupid', 'dumb',
            'idiotic', 'ridiculous', 'absurd', 'nonsense', 'crazy',

            # Problem negative words
            'problem', 'problems', 'issue', 'issues', 'trouble', 'troubles',
            'difficulty', 'difficulties', 'complication', 'complications',
            'obstacle', 'obstacles', 'barrier', 'barriers', 'hurdle', 'hurdles',
            'setback', 'setbacks', 'drawback', 'drawbacks', 'disadvantage',

            # Complaint words
            'complaint', 'complaints', 'unhappy', 'dissatisfied', 'unsatisfied',
            'displeased', 'discontent', 'discontented', 'grievance', 'grievances',
            'beef', 'gripe', 'griping', 'whining', 'complaining',

            # Mild negative words
            'poor', 'worst', 'worse', 'sad', 'sorrow', 'sorrowful', 'grief',
            'grieving', 'mourn', 'mourning', 'depressed', 'depressing', 'gloomy',
            'dreary', 'dismal', 'bleak', 'dark', 'negative', 'pessimistic',

            # Additional negative words
            'never', 'no', 'none', 'nothing', 'nobody', 'nowhere', 'rejected',
            'rejection', 'denied', 'denial', 'refused', 'refusal', 'banned',
            'ban', 'forbidden', 'prohibited', 'illegal', 'unlawful', 'wrong',
            'mistake', 'error', 'fault', 'blame', 'guilt', 'guilty', 'ashamed',
            'embarrassed', 'humiliated', 'insulted', 'offended', 'hurt',
            'pain', 'painful', 'suffering', 'suffer', 'agony', 'torment',
            'torture', 'cruel', 'cruelty', 'brutal', 'brutality', 'violent',
            'violence', 'abuse', 'abused', 'victim', 'victimized', 'oppressed',
            'oppression', 'injustice', 'unfair', 'unfairness', 'corrupt',
            'corruption', 'scam', 'scammed', 'fraud', 'fraudulent', 'deceit',
            'deceitful', 'lie', 'lied', 'lying', 'false', 'falsely', 'fake',
            'phony', 'counterfeit', 'bogus', 'spurious'
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

        # Adjust thresholds based on word counts for better accuracy
        positive_threshold = 0.2
        negative_threshold = -0.2

        # If we have strong negative words, be more sensitive
        strong_negative_words = {'kill', 'murder', 'death', 'hate', 'fuck', 'shit', 'damn', 'hell'}
        has_strong_negative = any(word in strong_negative_words for word in words)

        if has_strong_negative:
            negative_threshold = -0.1  # More sensitive to strong negative words

        # If message is short and contains negative words, classify as negative
        if len(words) <= 3 and negative_count > 0 and positive_count == 0:
            negative_threshold = 0  # Even single negative words in short messages should be negative

        # If message is short and contains positive words, classify as positive
        if len(words) <= 3 and positive_count > 0 and negative_count == 0:
            positive_threshold = 0  # Even single positive words in short messages should be positive

        # If we have strong positive words, be more sensitive
        strong_positive_words = {'amazing', 'wonderful', 'fantastic', 'excellent', 'perfect', 'awesome',
                                'love', 'joyful', 'smiling', 'laughing', 'thrilled', 'ecstatic',
                                'blissful', 'spectacular', 'phenomenal', 'incredible'}
        has_strong_positive = any(word in strong_positive_words for word in words)

        if has_strong_positive:
            positive_threshold = 0.1  # More sensitive to strong positive words

        # Determine sentiment with confidence
        if normalized_score > positive_threshold:
            sentiment = 'Positive'
            confidence = min(95, 60 + normalized_score * 50)
            emoji = '😊'
        elif normalized_score < negative_threshold:
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
    """Generate contextual chatbot responses based on sentiment with optional AI enhancement"""

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

        # AI enhancement setup - Gemini only
        self.use_gemini = GEMINI_AVAILABLE and os.getenv('GEMINI_API_KEY')

        if self.use_gemini:
            genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
            # Use the latest stable model
            self.model = genai.GenerativeModel('models/gemini-flash-latest')
            print("🤖 AI responses enabled! Using Google Gemini Flash Latest for more natural conversations.")
        else:
            print("💬 Using intelligent fallback responses. Set GEMINI_API_KEY for AI-enhanced responses.")

    def generate(self, user_message: str, sentiment: str, sentiment_data: Dict = None, conversation_history: List = None) -> str:
        """Generate appropriate response based on sentiment, with optional AI enhancement"""
        if (self.use_gemini or self.use_openai) and sentiment_data:
            return self._generate_ai_response(user_message, sentiment, sentiment_data, conversation_history)
        else:
            # Fallback to intelligent sentiment-based responses
            import random
            responses = self.responses.get(sentiment, self.responses['Neutral'])
            return random.choice(responses)

    def _generate_ai_response(self, user_message: str, sentiment: str, sentiment_data: Dict, conversation_history: List = None) -> str:
        """Generate AI-powered response using Gemini or OpenAI with conversation history"""
        try:
            # Prepare conversation history (last 5 messages for context)
            history_context = ""
            if conversation_history:
                recent_messages = [msg for msg in conversation_history[-10:] if msg.role in ['user', 'bot']]  # Last 10 messages, but only user/bot
                if len(recent_messages) > 1:  # Need at least 2 messages for context
                    history_context = "\nRecent conversation:\n"
                    for msg in recent_messages[-6:]:  # Last 6 messages (3 exchanges)
                        role = "User" if msg.role == 'user' else "Assistant"
                        history_context += f"{role}: {msg.content}\n"

            # Create context-aware prompt
            sentiment_context = f"""
            User sentiment: {sentiment}
            Sentiment score: {sentiment_data['score']:.3f}
            Confidence: {sentiment_data['confidence']:.1f}%
            Positive words: {sentiment_data['positive_count']}
            Negative words: {sentiment_data['negative_count']}
            """

            prompt = f"""
            You are a helpful, empathetic chatbot with sentiment analysis capabilities.
            The user just said: "{user_message}"

            {sentiment_context}
            {history_context}

            Respond naturally and helpfully. Keep responses conversational and engaging.
            Build on the conversation history if provided - reference previous topics or maintain context.
            If the user is expressing negative sentiment, show empathy and offer assistance.
            If positive, acknowledge their feedback and continue the conversation.
            If neutral, keep the dialogue flowing naturally.

            Keep your response under 100 words and be friendly.
            """

            if self.use_gemini:
                # Use Google Gemini
                response = self.model.generate_content(prompt)
                return response.text.strip()
            elif self.use_openai:
                # Use OpenAI as fallback
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful chatbot with sentiment awareness."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=150,
                    temperature=0.7
                )
                return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"AI response failed: {e}. Using fallback response.")
            # Fallback to sentiment-based responses
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

        # Prepare sentiment data for AI response
        sentiment_data = {
            'score': sentiment_result.score,
            'confidence': sentiment_result.confidence,
            'positive_count': sentiment_result.positive_count,
            'negative_count': sentiment_result.negative_count
        }

        # Store user message
        user_message = Message(
            role='user',
            content=user_input,
            timestamp=datetime.now().strftime('%H:%M:%S'),
            sentiment_data=sentiment_result
        )
        self.messages.append(user_message)

        # Generate bot response (now with AI enhancement)
        bot_response = self.response_generator.generate(user_input, sentiment_result.sentiment, sentiment_data, self.messages)

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