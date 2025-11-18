import unittest
import Code.backend as backend

class TestreadTextFile(unittest.TestCase):
    def testReadFile(self):
        self.assertEqual(backend.readTextFile('Code\inputTextFiles\oneLineNoPunct.txt'),
                         'In the afternoon we saw what was supposed to be a rock')

class TestProcessTextFile(unittest.TestCase):
    def testEmptyStringFP(self):
        self.assertEqual(backend.processTextFile('', False, backend.allPunctuation, True), [])
    
    def testEmptyStringTP(self):
        self.assertEqual(backend.processTextFile('', True, backend.allPunctuation, True), [])
    
    def testOneWordFP(self):
        self.assertEqual(backend.processTextFile('word', False, backend.allPunctuation, True), ['word'])
    
    def testOneWordTP(self):
        self.assertEqual(backend.processTextFile('word', True, backend.allPunctuation, True), ['word'])

    def testSentenceNoPunctFP(self):
        self.assertEqual(backend.processTextFile('In the afternoon we saw what was supposed to be a rock', False, backend.allPunctuation, True),
                         ['in', 'the', 'afternoon', 'we', 'saw', 'what', 'was', 'supposed', 'to', 'be', 'a', 'rock'])
    
    def testSentenceNoPunctTP(self):
        self.assertEqual(backend.processTextFile('In the afternoon we saw what was supposed to be a rock', True, backend.allPunctuation, True),
                         ['in', 'the', 'afternoon', 'we', 'saw', 'what', 'was', 'supposed', 'to', 'be', 'a', 'rock'])
    
    def testIgnorePunctFP(self):
        self.assertEqual(backend.processTextFile('"Hi, How are: you?"', False, backend.allPunctuation, True), 
                         ['hi', 'how', 'are', 'you'])
    
    def testDotTP(self):
        self.assertEqual(backend.processTextFile('TesTinG. TeStIng.', True, backend.allPunctuation, True), ['testing', '.', 'testing', '.'])
    
    def testCommaTP(self):
        self.assertEqual(backend.processTextFile('TesTinG, TeStIng', True, backend.allPunctuation, True), ['testing', ',', 'testing'])
    
    def testQuestionMarkTP(self):
        self.assertEqual(backend.processTextFile('QuestionMark? QuestionMark?', True, backend.allPunctuation, True),
                         ['questionmark', '?', 'questionmark', '?'])
    
    def testExclamationMarkTP(self):
        self.assertEqual(backend.processTextFile('ExclamationMark! ExclamationMark!', True, backend.allPunctuation, True),
                         ['exclamationmark', '!', 'exclamationmark', '!'])
    
    def testSemicolonTP(self):
        self.assertEqual(backend.processTextFile('Semicolon; Semicolon;', True, backend.allPunctuation, True), 
                         ['semicolon', ';', 'semicolon', ';'])
    
    def testColonTP(self):
        self.assertEqual(backend.processTextFile('Colon: Colon:', True, backend.allPunctuation, True), ['colon', ':', 'colon', ':'])
    
    def testParenthesesTP(self):
        self.assertEqual(backend.processTextFile('Parentheses (Parentheses)', True, backend.allPunctuation, True), 
                         ['parentheses', '(', 'parentheses', ')'])
    
    def testThreeDotsTP(self):
        self.assertEqual(backend.processTextFile('ThreeDots... ThreeDots...', True, backend.allPunctuation, True), 
                         ['threedots', '...', 'threedots', '...'])
    
    def testUnderscoreTP(self):
        self.assertEqual(backend.processTextFile('Under_score Under_score', True, backend.allPunctuation, True), 
                         ['under', '_', 'score', 'under', '_', 'score'])
    
    def testHyphenTP(self):
        self.assertEqual(backend.processTextFile('Hyphen-hyphen Hyphen-hyphen', True, backend.allPunctuation, True), 
                         ['hyphen', '-', 'hyphen', 'hyphen', '-', 'hyphen'])
    
    def testApostropheTP(self):
        self.assertEqual(backend.processTextFile("Apostrophe's Apostrophe's", True, backend.allPunctuation, True), 
                         ['apostrophe', "'", 's', 'apostrophe', "'", 's'])
    
    def testQuotationMarksTP(self):
        self.assertEqual(backend.processTextFile('"QuotationMarks", "QuotationMarks"', True, backend.allPunctuation, True), 
                         ['"', 'quotationmarks', '"', ',', '"', 'quotationmarks', '"'])
    
    def testNumbersCommaTP(self):
        self.assertEqual(backend.processTextFile('Number 1,20375', True, backend.allPunctuation, True), 
                         ['number', '1', ',', '20375'])
    
    def testNumbersDotTP(self):
        self.assertEqual(backend.processTextFile('Number 723.97', True, backend.allPunctuation, True), 
                         ['number', '723', '.', '97'])
    
    def testNumbersSlashTP(self):
        self.assertEqual(backend.processTextFile('Number 13/21', True, backend.allPunctuation, True), 
                         ['number', '13', '/', '21'])

    def testMultiplePunct1(self):
        self.assertEqual(backend.processTextFile('"Wait... did you see that?" she asked.', True, backend.allPunctuation, True), 
                         ['"', 'wait', '...', 'did', 'you', 'see', 'that', '?', '"', 'she', 'asked', '.'])
    
    def testMultiplePunct2(self):
        self.assertEqual(backend.processTextFile('Also, file paths use slashes, like "C:/Users/John_Doe/".', True, backend.allPunctuation, True), 
                         ['also', ',', 'file', 'paths', 'use', 'slashes', ',', 'like', '"', 'c', ':', '/', 'users', '/', 'john', '_', 'doe', '/', '"', '.'])
    
    def testCaseSensitiveIgnorePunct(self):
        self.assertEqual(backend.processTextFile('Also, file paths use slashes, like "C:/Users/John_Doe/".', False, backend.allPunctuation, False), 
                         ['Also', 'file', 'paths', 'use', 'slashes', 'like', 'C', 'Users', 'John', 'Doe'])
    
    def testPunctuationSelection1(self):
        self.assertEqual(backend.processTextFile('[brackets], \{braces\}, and (parentheses)—all in one place!', True, ['brackets', 'comma'], True),
                         ['[', 'brackets', ']', ',', 'braces', ',', 'and', 'parentheses', 'all', 'in', 'one', 'place'])
    
    def testPunctuationSelection2(self):
        self.assertEqual(backend.processTextFile('[brackets], \{braces\}, and (parentheses)—all in one place!', True, ['braces', 'emDash'], True),
                         ['brackets', '{', 'braces', '}', 'and', 'parentheses', '—', 'all', 'in', 'one', 'place'])
    

if __name__ == '__main__':
    unittest.main()