import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parent / 'mainApp_udpipe.py'
spec = importlib.util.spec_from_file_location('mainApp_udpipe', MODULE_PATH)
mainApp_udpipe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mainApp_udpipe)


def make_test_app(language='English', shuffle=''):
    app = mainApp_udpipe.App.__new__(mainApp_udpipe.App)
    app.selectedLanguage = language
    app.selectedShuffle = shuffle
    app.shuffleSeed = 42
    app.udpipeModels = {}
    app.udpipePipelines = {}
    app.modelPaths = {
        'English': app.resolveModelPath('english-ewt-ud-2.5-191206.udpipe'),
        'German': app.resolveModelPath('german-gsd-ud-2.5-191206.udpipe'),
        'Slovak': app.resolveModelPath('slovak-snk-ud-2.5-191206.udpipe'),
    }
    app.punctuationTokens = {
        'period': {'.'},
        'comma': {','},
        'exclamation': {'!'},
        'question': {'?'},
        'semicolon': {';'},
        'colon': {':'},
        'quotation': {'"', '“', '”', '„', '«', '»'},
        'apostrophe': {"'", '’', '‘'},
        'underscore': {'_'},
        'hyphen': {'-'},
        'enDash': {'–'},
        'emDash': {'—'},
        'ellipsis': {'...', '…'},
        'slash': {'/'},
        'parenthesis': {'(', ')'},
        'brackets': {'[', ']'},
        'braces': {'{', '}'},
    }
    app.punctuationNormalization = {
        '“': '"',
        '”': '"',
        '„': '"',
        '«': '"',
        '»': '"',
        '’': "'",
        '‘': "'",
        '…': '...',
    }
    return app


class ReversingRandom:
    def shuffle(self, items):
        items.reverse()


TOKENIZATION_CASES = {
    'English': {
        'text': 'Yesterday I bought books.',
        'expected': ['yesterday', 'i', 'buy', 'book'],
    },
    'German': {
        'text': 'Gestern kaufte ich Bücher.',
        'expected': ['gestern', 'kaufen', 'ich', 'buch'],
    },
    'Slovak': {
        'text': 'Včera som kúpil knihy.',
        'expected': ['včera', 'byť', 'kúpiť', 'kniha'],
    },
}

PUNCTUATION_CASES = {
    'English': {
        'text': 'Hello, world! Really.',
        'expected': ['hello', ',', 'world', 'really', '.'],
    },
    'German': {
        'text': 'Hallo, Welt! Wirklich.',
        'expected': ['hallo', ',', 'welt', 'wirklich', '.'],
    },
    'Slovak': {
        'text': 'Ahoj, svet! Naozaj.',
        'expected': ['ahoj', ',', 'svet', 'naozaj', '.'],
    },
}

SENTENCE_SHUFFLE_CASES = {
    'English': {
        'text': 'Cats sleep. Dogs run.',
        'expected_reversed': ['dog', 'run', 'cat', 'sleep'],
    },
    'German': {
        'text': 'Katzen schlafen. Hunde laufen.',
        'expected_reversed': ['hund', 'laufen', 'katze', 'schlafen'],
    },
    'Slovak': {
        'text': 'Mačky spia. Psy bežia.',
        'expected_reversed': ['pes', 'bežia', 'mačka', 'spia'],
    },
}

WORD_SHUFFLE_CASES = {
    'English': {
        'text': 'One, two, three.',
        'expected_reversed': ['three', ',', 'two', ',', 'one'],
    },
    'German': {
        'text': 'Eins, zwei, drei.',
        'expected_reversed': ['drei', ',', 'zwei', ',', 'ein'],
    },
    'Slovak': {
        'text': 'Raz, dva, tri.',
        'expected_reversed': ['tri', ',', 'dva', ',', 'raz'],
    },
}


class TestUdpipePreprocessing(unittest.TestCase):

    def tokens_from_text(self, app, text, selected_punctuation=None):
        sentences = app.parseTextWithUdpipe(text)
        sentences = app.applySentenceShuffle(sentences)
        return app.filterUdpipeTokens(sentences, selected_punctuation)

    def test_text_is_tokenized_and_lemmatized_without_punctuation_for_all_languages(self):
        for language, case in TOKENIZATION_CASES.items():
            with self.subTest(language=language):
                app = make_test_app(language)

                tokens = self.tokens_from_text(app, case['text'])

                self.assertEqual(tokens, case['expected'])

    def test_selected_punctuation_is_kept_and_unselected_punctuation_is_filtered_for_all_languages(self):
        for language, case in PUNCTUATION_CASES.items():
            with self.subTest(language=language):
                app = make_test_app(language)

                tokens = self.tokens_from_text(
                    app,
                    case['text'],
                    {'comma': ',', 'period': '.'}
                )

                self.assertEqual(tokens, case['expected'])
                self.assertNotIn('!', tokens)

    def test_bracket_selection_does_not_keep_comma(self):
        app = make_test_app('English')

        tokens = self.tokens_from_text(app, 'One [two], three.', {'brackets': '[]'})

        self.assertIn('[', tokens)
        self.assertIn(']', tokens)
        self.assertNotIn(',', tokens)

    def test_slovak_quotation_marks_are_normalized(self):
        app = make_test_app('Slovak')

        tokens = self.tokens_from_text(
            app,
            '„Ahoj,“ povedal.',
            {'quotation': '"..."', 'comma': ',', 'period': '.'}
        )

        self.assertEqual(tokens.count('"'), 2)
        self.assertIn(',', tokens)
        self.assertIn('.', tokens)
        self.assertNotIn('„', tokens)
        self.assertNotIn('“', tokens)

    def test_punctuation_lemma_is_filtered_when_no_punctuation_is_selected(self):
        app = make_test_app('English')

        class FakePipeline:
            def process(self, text):
                return (
                    '# text = fake\n'
                    '1\tword\tword\tNOUN\tNN\t_\t0\troot\t_\t_\n'
                    '2\t’s\t’\tNOUN\tNN\t_\t1\tdep\t_\t_\n'
                    '3\tend\tend\tNOUN\tNN\t_\t1\tdep\t_\t_\n'
                    '\n'
                )

        app.getUdpipePipeline = lambda: FakePipeline()

        sentences = app.parseTextWithUdpipe('fake')

        self.assertEqual(app.filterUdpipeTokens(sentences), ['word', 'end'])
        self.assertEqual(app.filterUdpipeTokens(sentences, {'apostrophe': "'"}), ['word', "'", 'end'])

    def test_sentence_shuffle_shuffles_whole_sentences_before_flattening_for_all_languages(self):
        for language, case in SENTENCE_SHUFFLE_CASES.items():
            with self.subTest(language=language):
                app = make_test_app(language, 'Sentences')
                app.getShuffleRandom = lambda: ReversingRandom()

                tokens = self.tokens_from_text(app, case['text'])

                self.assertEqual(tokens, case['expected_reversed'])

    def test_word_shuffle_shuffles_filtered_tokens_for_all_languages(self):
        for language, case in WORD_SHUFFLE_CASES.items():
            with self.subTest(language=language):
                app = make_test_app(language, 'Words')
                app.inputTextFile = 'unused.txt'
                app.readTextFile = lambda path, text=case['text']: text
                app.getShuffleRandom = lambda: ReversingRandom()

                tokens = app.processTextFile({'comma': ','})

                self.assertEqual(tokens, case['expected_reversed'])

    def test_same_seed_gives_same_word_shuffle_result(self):
        app = make_test_app('English', 'Words')
        app.inputTextFile = 'unused.txt'
        app.readTextFile = lambda path: 'One, two, three, four.'
        app.shuffleSeed = 123

        first_run = app.processTextFile({'comma': ','})
        second_run = app.processTextFile({'comma': ','})

        self.assertEqual(first_run, second_run)

    def test_different_seed_can_change_word_shuffle_result(self):
        app = make_test_app('English', 'Words')
        app.inputTextFile = 'unused.txt'
        app.readTextFile = lambda path: 'One, two, three, four, five, six.'
        app.shuffleSeed = 123
        first_run = app.processTextFile({'comma': ','})

        app.shuffleSeed = 456
        second_run = app.processTextFile({'comma': ','})

        self.assertNotEqual(first_run, second_run)


class TestGraphData(unittest.TestCase):

    def test_create_graph_data_builds_default_positional_word_network(self):
        app = make_test_app('English')

        graph_data, node_counter = app.createGraphData(['alpha', 'beta', 'alpha'])

        self.assertEqual(graph_data, {
            'alpha': ['beta', 'beta'],
            'beta': ['alpha', 'alpha'],
        })
        self.assertEqual(node_counter['alpha'], 2)
        self.assertEqual(node_counter['beta'], 1)

    def test_create_graph_data_normalizes_quotation_nodes(self):
        app = make_test_app('Slovak')

        graph_data, node_counter = app.createGraphData(['„', 'ahoj', '“'])

        self.assertIn('"', graph_data)
        self.assertNotIn('„', graph_data)
        self.assertNotIn('“', graph_data)
        self.assertEqual(node_counter['"'], 2)


if __name__ == '__main__':
    unittest.main()
