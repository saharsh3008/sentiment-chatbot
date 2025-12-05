"""
Unit tests for Sentiment Analysis Chatbot
Comprehensive test coverage for all components
"""

import unittest
from chatbot import (
    SentimentAnalyzer, 
    ResponseGenerator, 
    ConversationAnalyzer,
    SentimentChatbot,
    Message,
    SentimentResult
)


class TestSentimentAnalyzer(unittest.TestCase):
    """Test cases for SentimentAnalyzer"""
    
    def setUp(self):
        self.analyzer = SentimentAnalyzer()
    
    def test_positive_sentiment(self):
        """Test positive sentiment detection"""
        result = self.analyzer.analyze("This is amazing and wonderful!")
        self.assertEqual(result.sentiment, 'Positive')
        self.assertGreater(result.score, 0)
        self.assertGreater(result.positive_count, 0)
    
    def test_negative_sentiment(self):
        """Test negative sentiment detection"""
        result = self.analyzer.analyze("This is terrible and disappointing")
        self.assertEqual(result.sentiment, 'Negative')
        self.assertLess(result.score, 0)
        self.assertGreater(result.negative_count, 0)
    
    def test_neutral_sentiment(self):
        """Test neutral sentiment detection"""
        result = self.analyzer.analyze("I went to the store today")
        self.assertEqual(result.sentiment, 'Neutral')
        self.assertAlmostEqual(result.score, 0, places=1)
    
    def test_negation_handling(self):
        """Test handling of negations"""
        result = self.analyzer.analyze("This is not good")
        self.assertEqual(result.sentiment, 'Negative')
        self.assertLess(result.score, 0)
    
    def test_intensifier_handling(self):
        """Test handling of intensifiers"""
        result1 = self.analyzer.analyze("This is good")
        result2 = self.analyzer.analyze("This is very good")
        self.assertGreater(result2.score, result1.score)
    
    def test_empty_text(self):
        """Test handling of empty text"""
        result = self.analyzer.analyze("")
        self.assertEqual(result.sentiment, 'Neutral')
        self.assertEqual(result.score, 0)
    
    def test_confidence_calculation(self):
        """Test confidence score is within valid range"""
        result = self.analyzer.analyze("This is absolutely fantastic!")
        self.assertGreaterEqual(result.confidence, 0)
        self.assertLessEqual(result.confidence, 100)
    
    def test_emoji_assignment(self):
        """Test correct emoji assignment"""
        positive = self.analyzer.analyze("Great job!")
        negative = self.analyzer.analyze("Very disappointing")
        neutral = self.analyzer.analyze("The meeting is at 3pm")
        
        self.assertEqual(positive.emoji, '😊')
        self.assertEqual(negative.emoji, '😔')
        self.assertEqual(neutral.emoji, '😐')
    
    def test_multiple_positive_words(self):
        """Test handling of multiple positive words"""
        result = self.analyzer.analyze("This is great, amazing, and wonderful!")
        self.assertEqual(result.sentiment, 'Positive')
        self.assertGreater(result.positive_count, 2)
    
    def test_multiple_negative_words(self):
        """Test handling of multiple negative words"""
        result = self.analyzer.analyze("This is bad, terrible, and awful")
        self.assertEqual(result.sentiment, 'Negative')
        self.assertGreater(result.negative_count, 2)
    
    def test_mixed_sentiment(self):
        """Test handling of mixed positive and negative words"""
        result = self.analyzer.analyze("The product is good but the service is bad")
        # Should have both positive and negative words
        self.assertGreater(result.positive_count, 0)
        self.assertGreater(result.negative_count, 0)


class TestResponseGenerator(unittest.TestCase):
    """Test cases for ResponseGenerator"""
    
    def setUp(self):
        self.generator = ResponseGenerator()
    
    def test_positive_response(self):
        """Test response generation for positive sentiment"""
        response = self.generator.generate("Great service!", 'Positive')
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
    
    def test_negative_response(self):
        """Test response generation for negative sentiment"""
        response = self.generator.generate("Poor experience", 'Negative')
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
    
    def test_neutral_response(self):
        """Test response generation for neutral sentiment"""
        response = self.generator.generate("What time is it?", 'Neutral')
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
    
    def test_response_variety(self):
        """Test that responses vary"""
        responses = set()
        for _ in range(10):
            response = self.generator.generate("Test", 'Positive')
            responses.add(response)
        
        # Should have some variety (not all identical)
        self.assertGreater(len(responses), 1)
    
    def test_response_not_empty(self):
        """Test that responses are never empty"""
        for sentiment in ['Positive', 'Negative', 'Neutral']:
            response = self.generator.generate("Test message", sentiment)
            self.assertTrue(len(response) > 0)


class TestConversationAnalyzer(unittest.TestCase):
    """Test cases for ConversationAnalyzer"""
    
    def setUp(self):
        self.analyzer = ConversationAnalyzer()
    
    def create_test_message(self, content, sentiment, score):
        """Helper to create test messages"""
        sentiment_data = SentimentResult(
            sentiment=sentiment,
            score=score,
            confidence=80.0,
            positive_count=1,
            negative_count=0,
            emoji='😊' if sentiment == 'Positive' else '😔' if sentiment == 'Negative' else '😐'
        )
        return Message(
            role='user',
            content=content,
            timestamp='12:00:00',
            sentiment_data=sentiment_data
        )
    
    def test_overall_positive_sentiment(self):
        """Test detection of overall positive conversation"""
        messages = [
            self.create_test_message("Great!", 'Positive', 0.8),
            self.create_test_message("Amazing!", 'Positive', 0.9),
        ]
        
        analysis = self.analyzer.analyze_conversation(messages)
        self.assertEqual(analysis['overall_sentiment'], 'Positive')
    
    def test_overall_negative_sentiment(self):
        """Test detection of overall negative conversation"""
        messages = [
            self.create_test_message("Bad", 'Negative', -0.7),
            self.create_test_message("Terrible", 'Negative', -0.8),
        ]
        
        analysis = self.analyzer.analyze_conversation(messages)
        self.assertEqual(analysis['overall_sentiment'], 'Negative')
    
    def test_overall_neutral_sentiment(self):
        """Test detection of overall neutral conversation"""
        messages = [
            self.create_test_message("Okay", 'Neutral', 0.1),
            self.create_test_message("Fine", 'Neutral', -0.1),
        ]
        
        analysis = self.analyzer.analyze_conversation(messages)
        self.assertEqual(analysis['overall_sentiment'], 'Neutral')
    
    def test_trend_detection_improving(self):
        """Test detection of improving trend"""
        messages = [
            self.create_test_message("Bad start", 'Negative', -0.6),
            self.create_test_message("Getting better", 'Positive', 0.3),
            self.create_test_message("Much better!", 'Positive', 0.7),
        ]
        
        analysis = self.analyzer.analyze_conversation(messages)
        self.assertEqual(analysis['trend'], 'Improving')
    
    def test_trend_detection_declining(self):
        """Test detection of declining trend"""
        messages = [
            self.create_test_message("Great start!", 'Positive', 0.8),
            self.create_test_message("Getting worse", 'Negative', -0.2),
            self.create_test_message("Very bad", 'Negative', -0.7),
        ]
        
        analysis = self.analyzer.analyze_conversation(messages)
        self.assertEqual(analysis['trend'], 'Declining')
    
    def test_trend_detection_stable(self):
        """Test detection of stable trend"""
        messages = [
            self.create_test_message("Good", 'Positive', 0.5),
            self.create_test_message("Good again", 'Positive', 0.5),
            self.create_test_message("Still good", 'Positive', 0.5),
        ]
        
        analysis = self.analyzer.analyze_conversation(messages)
        self.assertEqual(analysis['trend'], 'Stable')
    
    def test_empty_conversation(self):
        """Test handling of empty conversation"""
        analysis = self.analyzer.analyze_conversation([])
        self.assertIsNone(analysis)
    
    def test_sentiment_counts(self):
        """Test sentiment counting"""
        messages = [
            self.create_test_message("Good", 'Positive', 0.5),
            self.create_test_message("Bad", 'Negative', -0.5),
            self.create_test_message("Okay", 'Neutral', 0.0),
        ]
        
        analysis = self.analyzer.analyze_conversation(messages)
        self.assertEqual(analysis['sentiment_counts']['Positive'], 1)
        self.assertEqual(analysis['sentiment_counts']['Negative'], 1)
        self.assertEqual(analysis['sentiment_counts']['Neutral'], 1)
    
    def test_average_score_calculation(self):
        """Test average score calculation"""
        messages = [
            self.create_test_message("Test1", 'Positive', 0.6),
            self.create_test_message("Test2", 'Negative', -0.4),
        ]
        
        analysis = self.analyzer.analyze_conversation(messages)
        expected_avg = (0.6 + (-0.4)) / 2
        self.assertAlmostEqual(analysis['average_score'], expected_avg, places=2)
    
    def test_message_count(self):
        """Test message counting"""
        messages = [
            self.create_test_message("Test1", 'Positive', 0.5),
            self.create_test_message("Test2", 'Positive', 0.6),
            self.create_test_message("Test3", 'Negative', -0.3),
        ]
        
        analysis = self.analyzer.analyze_conversation(messages)
        self.assertEqual(analysis['message_count'], 3)


class TestSentimentChatbot(unittest.TestCase):
    """Integration tests for SentimentChatbot"""
    
    def setUp(self):
        self.chatbot = SentimentChatbot()
    
    def test_process_message(self):
        """Test message processing"""
        response, sentiment = self.chatbot.process_message("This is great!")
        
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        self.assertIsInstance(sentiment, SentimentResult)
        self.assertEqual(sentiment.sentiment, 'Positive')
    
    def test_conversation_history(self):
        """Test conversation history is maintained"""
        self.chatbot.process_message("Hello")
        self.chatbot.process_message("How are you?")
        
        self.assertEqual(len(self.chatbot.messages), 4)  # 2 user + 2 bot messages
    
    def test_conversation_analysis(self):
        """Test conversation analysis"""
        self.chatbot.process_message("This is wonderful!")
        self.chatbot.process_message("Great service!")
        
        analysis = self.chatbot.get_conversation_analysis()
        
        self.assertIsNotNone(analysis)
        self.assertIn('overall_sentiment', analysis)
        self.assertIn('trend', analysis)
        self.assertIn('sentiment_counts', analysis)
    
    def test_multiple_messages(self):
        """Test handling multiple messages"""
        messages = [
            "I love this product!",
            "It works perfectly",
            "Very satisfied with the service"
        ]
        
        for msg in messages:
            self.chatbot.process_message(msg)
        
        analysis = self.chatbot.get_conversation_analysis()
        self.assertEqual(analysis['overall_sentiment'], 'Positive')
        self.assertEqual(analysis['message_count'], 3)
    
    def test_mixed_sentiment_conversation(self):
        """Test conversation with mixed sentiments"""
        self.chatbot.process_message("Great product!")
        self.chatbot.process_message("But terrible customer service")
        self.chatbot.process_message("Overall, I'm satisfied")
        
        analysis = self.chatbot.get_conversation_analysis()
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis['message_count'], 3)
    
    def test_user_messages_have_sentiment(self):
        """Test that user messages have sentiment data"""
        self.chatbot.process_message("This is amazing!")
        
        user_messages = [m for m in self.chatbot.messages if m.role == 'user']
        self.assertEqual(len(user_messages), 1)
        self.assertIsNotNone(user_messages[0].sentiment_data)
    
    def test_bot_messages_no_sentiment(self):
        """Test that bot messages don't have sentiment data"""
        self.chatbot.process_message("Hello")
        
        bot_messages = [m for m in self.chatbot.messages if m.role == 'bot']
        self.assertEqual(len(bot_messages), 1)
        self.assertIsNone(bot_messages[0].sentiment_data)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios"""
    
    def setUp(self):
        self.analyzer = SentimentAnalyzer()
    
    def test_special_characters(self):
        """Test handling of special characters"""
        result = self.analyzer.analyze("This is great!!! @#$%")
        self.assertEqual(result.sentiment, 'Positive')
    
    def test_mixed_case(self):
        """Test case insensitivity"""
        result1 = self.analyzer.analyze("GREAT")
        result2 = self.analyzer.analyze("great")
        self.assertEqual(result1.sentiment, result2.sentiment)
    
    def test_very_long_text(self):
        """Test handling of long text"""
        long_text = "This is good. " * 100
        result = self.analyzer.analyze(long_text)
        self.assertEqual(result.sentiment, 'Positive')
    
    def test_numbers_in_text(self):
        """Test text with numbers"""
        result = self.analyzer.analyze("I rate this 10/10, absolutely amazing!")
        self.assertEqual(result.sentiment, 'Positive')
    
    def test_only_punctuation(self):
        """Test text with only punctuation"""
        result = self.analyzer.analyze("!!! ??? ...")
        self.assertEqual(result.sentiment, 'Neutral')
    
    def test_repeated_words(self):
        """Test handling of repeated words"""
        result = self.analyzer.analyze("good good good good good")
        self.assertEqual(result.sentiment, 'Positive')
    
    def test_contradictory_statements(self):
        """Test contradictory positive and negative statements"""
        result = self.analyzer.analyze("This is good but also bad")
        # Should detect both sentiments
        self.assertGreater(result.positive_count, 0)
        self.assertGreater(result.negative_count, 0)
    
    def test_only_negations(self):
        """Test text with only negations"""
        result = self.analyzer.analyze("not not not")
        self.assertEqual(result.sentiment, 'Neutral')
    
    def test_unicode_characters(self):
        """Test handling of unicode characters"""
        result = self.analyzer.analyze("This is great! 😊 👍")
        self.assertEqual(result.sentiment, 'Positive')
    
    def test_whitespace_heavy(self):
        """Test text with lots of whitespace"""
        result = self.analyzer.analyze("This    is     great    !")
        self.assertEqual(result.sentiment, 'Positive')


class TestSentimentResult(unittest.TestCase):
    """Test SentimentResult dataclass"""
    
    def test_sentiment_result_creation(self):
        """Test creating SentimentResult"""
        result = SentimentResult(
            sentiment='Positive',
            score=0.75,
            confidence=85.5,
            positive_count=3,
            negative_count=0,
            emoji='😊'
        )
        
        self.assertEqual(result.sentiment, 'Positive')
        self.assertEqual(result.score, 0.75)
        self.assertEqual(result.confidence, 85.5)
        self.assertEqual(result.positive_count, 3)
        self.assertEqual(result.negative_count, 0)
        self.assertEqual(result.emoji, '😊')


class TestMessage(unittest.TestCase):
    """Test Message dataclass"""
    
    def test_message_creation(self):
        """Test creating Message"""
        msg = Message(
            role='user',
            content='Hello',
            timestamp='12:00:00'
        )
        
        self.assertEqual(msg.role, 'user')
        self.assertEqual(msg.content, 'Hello')
        self.assertEqual(msg.timestamp, '12:00:00')
        self.assertIsNone(msg.sentiment_data)
    
    def test_message_with_sentiment(self):
        """Test Message with sentiment data"""
        sentiment = SentimentResult(
            sentiment='Positive',
            score=0.5,
            confidence=75.0,
            positive_count=1,
            negative_count=0,
            emoji='😊'
        )
        
        msg = Message(
            role='user',
            content='Great!',
            timestamp='12:00:00',
            sentiment_data=sentiment
        )
        
        self.assertIsNotNone(msg.sentiment_data)
        self.assertEqual(msg.sentiment_data.sentiment, 'Positive')


def run_tests():
    """Run all tests with detailed output"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestSentimentAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestResponseGenerator))
    suite.addTests(loader.loadTestsFromTestCase(TestConversationAnalyzer))
    suite.addTests(loader.loadTestsFromTestCase(TestSentimentChatbot))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestSentimentResult))
    suite.addTests(loader.loadTestsFromTestCase(TestMessage))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*60)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)