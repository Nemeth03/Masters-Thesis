
import wx
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import random

from pathlib import Path
from collections import Counter
from scipy.stats import linregress
from ufal.udpipe import Model, Pipeline


class App(wx.Frame):

    def __init__(self):
        super().__init__(
            parent=None,
            title='Network Models Analysis',
            size=(700, 700),
            style=wx.DEFAULT_FRAME_STYLE & ~wx.RESIZE_BORDER
        )

        self.inputTextFile = None

        self.languageOptions = ['', 'English', 'German', 'Slovak']
        self.selectedLanguage = ''

        self.shuffleOptions = ['', 'Sentences', 'Words']
        self.selectedShuffle = ''
        self.shuffleSeed = 42

        self.udpipeModels = {}
        self.udpipePipelines = {}

        self.modelPaths = {
            'English': self.resolveModelPath(
                'english-ewt-ud-2.5-191206.udpipe'
            ),
            'German': self.resolveModelPath(
                'german-gsd-ud-2.5-191206.udpipe'
            ),
            'Slovak': self.resolveModelPath(
                'slovak-snk-ud-2.5-191206.udpipe'
            ),
        }

        self.punctuation = {
            'period': '.',
            'comma': ',',
            'exclamation': '!',
            'question': '?',
            'semicolon': ';',
            'colon': ':',
            'quotation': '"..."',
            'apostrophe': "'",
            'underscore': '_',
            'hyphen': '-',
            'enDash': '–',
            'emDash': '—',
            'ellipsis': '...',
            'slash': '/',
            'parenthesis': '()',
            'brackets': '[]',
            'braces': '{}',
        }

        self.selectedPunctuation = {}

        self.punctuationTokens = {
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

        self.punctuationNormalization = {
            '“': '"',
            '”': '"',
            '„': '"',
            '«': '"',
            '»': '"',
            '’': "'",
            '‘': "'",
            '…': '...',
        }

        self.initGUI()


    # Resolve UDPipe model path
    def resolveModelPath(self, modelName):
        appDir = Path(__file__).resolve().parent

        candidates = [
            appDir / modelName,
            appDir.parent / 'UDPipeModels' / modelName,
        ]

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return candidates[0]


    # Initialize GUI
    def initGUI(self):
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)

        # File selection
        self.labelFileSelect = wx.StaticText(
            panel, label='Select a text file:'
        )

        self.labelFileSelectPath = wx.TextCtrl(
            panel, style=wx.TE_READONLY
        )
        self.labelFileSelectPath.SetValue('No file selected.')

        self.buttonSelectFile = wx.Button(
            panel, label='Select File'
        )
        self.buttonSelectFile.Bind(
            wx.EVT_BUTTON, self.selectInputFile
        )

        fileSelectLayout = wx.BoxSizer(wx.HORIZONTAL)
        fileSelectLayout.Add(
            self.labelFileSelectPath,
            proportion=1,
            flag=wx.EXPAND
        )
        fileSelectLayout.Add(
            self.buttonSelectFile,
            flag=wx.LEFT,
            border=10
        )

        # Language selection
        self.languageDropdown = wx.Choice(
            panel, choices=self.languageOptions
        )
        self.languageDropdown.Bind(
            wx.EVT_CHOICE, self.languageChange
        )

        languageLayout = wx.BoxSizer(wx.HORIZONTAL)
        languageLayout.Add(
            wx.StaticText(panel, label='Select Language:')
        )
        languageLayout.Add(
            self.languageDropdown,
            flag=wx.LEFT,
            border=10
        )

        # Shuffle selection
        self.shuffleDropdown = wx.Choice(
            panel, choices=self.shuffleOptions
        )
        self.shuffleDropdown.Bind(
            wx.EVT_CHOICE, self.shuffleChange
        )

        self.shuffleSeedInput = wx.TextCtrl(
            panel,
            value=str(self.shuffleSeed),
            size=(70, -1)
        )
        self.shuffleSeedInput.Bind(
            wx.EVT_TEXT, self.shuffleSeedChange
        )

        additionalOptionsLayout = wx.BoxSizer(wx.HORIZONTAL)
        additionalOptionsLayout.Add(
            wx.StaticText(panel, label='Shuffle:')
        )
        additionalOptionsLayout.Add(
            self.shuffleDropdown,
            flag=wx.LEFT,
            border=10
        )
        additionalOptionsLayout.Add(
            wx.StaticText(panel, label='Seed:'),
            flag=wx.LEFT | wx.ALIGN_CENTER_VERTICAL,
            border=20
        )
        additionalOptionsLayout.Add(
            self.shuffleSeedInput,
            flag=wx.LEFT,
            border=10
        )

        # Punctuation selection
        self.selectAllCheckbox = wx.CheckBox(
            panel, label='Select All'
        )
        self.selectAllCheckbox.Bind(
            wx.EVT_CHECKBOX, self.selectAllPunctuation
        )

        self.punctuationLabel = wx.StaticText(
            panel, label='Select Punctuation:'
        )

        punctuationLayout = wx.GridSizer(5, 5, 5, 10)
        self.punctuationCheckboxes = {}

        for label, symbol in self.punctuation.items():
            checkbox = wx.CheckBox(
                panel, label=f'{label}  {symbol}'
            )
            checkbox.Bind(
                wx.EVT_CHECKBOX,
                self.updateSelectedPunctuation
            )
            punctuationLayout.Add(
                checkbox, flag=wx.EXPAND
            )
            self.punctuationCheckboxes[label] = checkbox

        # Compare punctuation checkbox
        self.comparePunctuationCheckbox = wx.CheckBox(
            panel,
            label='Compare With/Without Punctuation'
        )
        self.comparePunctuationCheckbox.SetValue(False)

        # Log window
        self.logWindow = wx.TextCtrl(
            panel,
            style=wx.TE_MULTILINE | wx.TE_READONLY
        )
        self.logWindow.SetBackgroundColour(
            wx.Colour(240, 240, 240)
        )

        # Analysis buttons
        self.statisticsButton = wx.Button(
            panel, label='Statistical Analysis'
        )
        self.statisticsButton.Bind(
            wx.EVT_BUTTON, self.statisticalAnalysis
        )
        self.statisticsButton.Enable(False)
        
        self.zipfAnalysisButton = wx.Button(
            panel, label='Zipf Model Analysis'
        )
        self.zipfAnalysisButton.Bind(
            wx.EVT_BUTTON, self.zipfModelAnalysis
        )
        self.zipfAnalysisButton.Enable(False)

        self.erdosRenyiButton = wx.Button(
            panel, label='Erdos-Renyi Analysis'
        )
        self.erdosRenyiButton.Bind(
            wx.EVT_BUTTON, self.erdosRenyiAnalysis
        )
        self.erdosRenyiButton.Enable(False)

        self.wattsStrogatzButton = wx.Button(
            panel, label='Watts-Strogatz Analysis'
        )
        self.wattsStrogatzButton.Bind(
            wx.EVT_BUTTON, self.wattsStrogatzAnalysis
        )
        self.wattsStrogatzButton.Enable(False)

        self.barabasiAlbertButton = wx.Button(
            panel, label='Barabasi-Albert Analysis'
        )
        self.barabasiAlbertButton.Bind(
            wx.EVT_BUTTON,
            self.barabasiAlbertAnalysis
        )
        self.barabasiAlbertButton.Enable(False)

        self.dgmButton = wx.Button(
            panel, label='DGM Model Analysis'
        )
        self.dgmButton.Bind(
            wx.EVT_BUTTON,
            self.dgmAnalysis
        )
        self.dgmButton.Enable(False)

        self.holmeKimButton = wx.Button(
            panel, label='Holme-Kim Analysis'
        )
        self.holmeKimButton.Bind(
            wx.EVT_BUTTON,
            self.holmeKimAnalysis
        )
        self.holmeKimButton.Enable(False)

        self.growthGammaButton = wx.Button(
            panel, label='Growth Gamma Analysis'
        )   
        self.growthGammaButton.Bind(
            wx.EVT_BUTTON,
            self.growthGammaAnalysis
        )
        self.growthGammaButton.Enable(False)

        # Exit button
        self.exitButton = wx.Button(
            panel, label='Exit'
        )
        self.exitButton.Bind(
            wx.EVT_BUTTON, self.exitApp
        )

        buttonLayout = wx.BoxSizer(wx.VERTICAL)

        rowOne = wx.BoxSizer(wx.HORIZONTAL)
        rowOne.Add(self.statisticsButton)
        rowOne.Add(
            self.zipfAnalysisButton,
            flag=wx.LEFT,
            border=10
        )
        rowOne.Add(
            self.erdosRenyiButton,
            flag=wx.LEFT,
            border=10
        )
        rowOne.Add(
            self.wattsStrogatzButton,
            flag=wx.LEFT,
            border=10
        )

        rowTwo = wx.BoxSizer(wx.HORIZONTAL)
        rowTwo.Add(
            self.barabasiAlbertButton,
            flag=wx.LEFT,
            border=10
        )
        rowTwo.Add(
            self.dgmButton,
            flag=wx.LEFT,
            border=10
        )
        rowTwo.Add(
            self.holmeKimButton,
            flag=wx.LEFT,
            border=10
        )
        rowTwo.Add(
            self.growthGammaButton,
            flag=wx.LEFT,
            border=10
        )

        rowLast = wx.BoxSizer(wx.HORIZONTAL)
        rowLast.Add(self.exitButton)

        buttonLayout.Add(
            rowOne,
            flag=wx.ALIGN_CENTER | wx.TOP,
            border=10
        )
        buttonLayout.Add(
            rowTwo,
            flag=wx.ALIGN_CENTER | wx.TOP,
            border=10
        )
        buttonLayout.Add(
            rowLast,
            flag=wx.ALIGN_CENTER | wx.TOP,
            border=10
        )

        # Main layout
        vbox.Add(
            self.labelFileSelect,
            flag=wx.LEFT | wx.TOP,
            border=10
        )
        vbox.Add(
            fileSelectLayout,
            flag=wx.EXPAND | wx.LEFT | wx.TOP,
            border=10
        )
        vbox.Add(
            languageLayout,
            flag=wx.LEFT | wx.TOP,
            border=10
        )
        vbox.Add(
            additionalOptionsLayout,
            flag=wx.LEFT | wx.TOP,
            border=10
        )
        vbox.Add(
            self.punctuationLabel,
            flag=wx.LEFT | wx.TOP,
            border=10
        )
        vbox.Add(
            self.selectAllCheckbox,
            flag=wx.LEFT | wx.TOP,
            border=10
        )
        vbox.Add(
            punctuationLayout,
            flag=wx.LEFT | wx.TOP,
            border=10
        )
        vbox.Add(
            self.comparePunctuationCheckbox,
            flag=wx.LEFT | wx.TOP,
            border=10
        )
        vbox.Add(
            self.logWindow,
            proportion=1,
            flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP,
            border=10
        )
        vbox.Add(
            buttonLayout,
            flag=wx.ALIGN_CENTER | wx.TOP,
            border=10
        )

        panel.SetSizer(vbox)
        self.Centre()


    # File selection
    def selectInputFile(self, event):
        with wx.FileDialog(
            self,
            'Open Text File',
            wildcard='Text files (*.txt)|*.txt',
            style=wx.FD_OPEN
        ) as fileDialog:

            if fileDialog.ShowModal() == wx.ID_OK:
                self.inputTextFile = fileDialog.GetPath()
                self.labelFileSelectPath.SetValue(
                    fileDialog.GetFilename()
                )
                self.validateInputData()


    # Language selection
    def languageChange(self, event):
        self.selectedLanguage = self.languageOptions[
            self.languageDropdown.GetSelection()
        ]
        self.validateInputData()


    # Shuffle selection
    def shuffleChange(self, event):
        self.selectedShuffle = self.shuffleOptions[
            self.shuffleDropdown.GetSelection()
        ]


    # Shuffle seed
    def shuffleSeedChange(self, event):
        seedValue = self.shuffleSeedInput.GetValue().strip()

        if seedValue:
            try:
                self.shuffleSeed = int(seedValue)
            except ValueError:
                pass


    # Select all punctuation
    def selectAllPunctuation(self, event):
        checked = event.IsChecked()
        self.selectedPunctuation.clear()

        for label, checkbox in self.punctuationCheckboxes.items():
            checkbox.SetValue(checked)

            if checked:
                self.selectedPunctuation[label] = (
                    self.punctuation[label]
                )

        self.validateInputData()


    # Update punctuation selection
    def updateSelectedPunctuation(self, event):
        sender = event.GetEventObject()
        label = sender.GetLabel().split()[0]

        if sender.IsChecked():
            self.selectedPunctuation[label] = self.punctuation[label]
        else:
            self.selectedPunctuation.pop(label, None)

        self.selectAllCheckbox.SetValue(
            all(
                checkbox.IsChecked()
                for checkbox in self.punctuationCheckboxes.values()
            )
        )

        self.validateInputData()


    # Log messages
    def logMessage(self, message):
        self.logWindow.AppendText(f'{message}\n')


    # Enable buttons
    def validateInputData(self):
        valid = bool(
            self.inputTextFile and self.selectedLanguage
        )

        self.erdosRenyiButton.Enable(valid)
        self.zipfAnalysisButton.Enable(valid)
        self.growthGammaButton.Enable(valid)
        self.statisticsButton.Enable(valid)
        self.wattsStrogatzButton.Enable(valid)
        self.barabasiAlbertButton.Enable(valid)
        self.dgmButton.Enable(valid)
        self.holmeKimButton.Enable(valid)


    # Read text
    def readTextFile(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()


    # Load UDPipe pipeline
    def getUdpipePipeline(self):
        if self.selectedLanguage not in self.modelPaths:
            raise ValueError(
                f'Unsupported language: {self.selectedLanguage}'
            )

        if self.selectedLanguage not in self.udpipePipelines:
            modelPath = self.modelPaths[self.selectedLanguage]
            model = Model.load(str(modelPath))

            if not model:
                raise FileNotFoundError(
                    f'UDPipe model could not be loaded: {modelPath}'
                )

            self.udpipeModels[self.selectedLanguage] = model
            self.udpipePipelines[self.selectedLanguage] = Pipeline(
                model,
                'tokenize',
                Pipeline.DEFAULT,
                Pipeline.DEFAULT,
                'conllu'
            )

        return self.udpipePipelines[self.selectedLanguage]


    # Parse text using UDPipe
    def parseTextWithUdpipe(self, text):
        pipeline = self.getUdpipePipeline()
        processed = pipeline.process(text)

        sentences = []
        currentSentence = []

        for line in processed.splitlines():
            if not line:
                if currentSentence:
                    sentences.append(currentSentence)
                    currentSentence = []
                continue

            if line.startswith('#'):
                continue

            cols = line.split('\t')

            if len(cols) != 10 or '-' in cols[0] or '.' in cols[0]:
                continue

            token = cols[1]
            lemma = cols[2]
            upos = cols[3]

            normalizedToken = self.normalizePunctuation(token)
            normalizedLemma = self.normalizePunctuation(
                lemma if lemma != '_' else token
            )

            lemmaIsPunctuation = self.isKnownPunctuation(
                normalizedLemma
            )

            isPunctuation = (
                upos == 'PUNCT'
                or self.isKnownPunctuation(normalizedToken)
                or lemmaIsPunctuation
            )

            currentSentence.append({
                'token': (
                    normalizedLemma
                    if lemmaIsPunctuation
                    else normalizedToken
                ),
                'lemma': normalizedLemma,
                'isPunctuation': isPunctuation,
            })

        if currentSentence:
            sentences.append(currentSentence)

        return sentences


    # Normalize punctuation
    def normalizePunctuation(self, token):
        return self.punctuationNormalization.get(token, token)


    # Check punctuation
    def isKnownPunctuation(self, token):
        return any(
            token in values
            for values in self.punctuationTokens.values()
        )


    # Filter UDPipe tokens
    def filterUdpipeTokens(self, sentences, selectedPunctuation=None):
        selectedPunctuation = selectedPunctuation or {}

        selectedPunctuationTokens = set()

        for key in selectedPunctuation:
            selectedPunctuationTokens.update(
                self.punctuationTokens.get(key, set())
            )

        selectedPunctuationTokens = {
            self.normalizePunctuation(token)
            for token in selectedPunctuationTokens
        }

        tokens = []

        for sentence in sentences:
            for item in sentence:
                if item['isPunctuation']:
                    if item['token'] in selectedPunctuationTokens:
                        tokens.append(item['token'])
                else:
                    tokens.append(item['lemma'].lower())

        return tokens


    # Shuffle sentences
    def applySentenceShuffle(self, sentences):
        shuffledSentences = list(sentences)

        if self.selectedShuffle == 'Sentences':
            random.Random(self.shuffleSeed).shuffle(
                shuffledSentences
            )

        return shuffledSentences


    # Process input text
    def processTextFile(self, selectedPunctuation=None):
        text = self.readTextFile(self.inputTextFile)
        sentences = self.parseTextWithUdpipe(text)
        sentences = self.applySentenceShuffle(sentences)

        data = self.filterUdpipeTokens(
            sentences, selectedPunctuation
        )

        if self.selectedShuffle == 'Words':
            random.Random(self.shuffleSeed).shuffle(data)

        return data

    

    # Logarithmically bin rank-frequency data
    def logBinning(self, ranks, frequencies, binsPerDecade=10):

        # Convert ranks and frequencies to NumPy arrays
        ranks = np.asarray(ranks)
        frequencies = np.asarray(frequencies)

        # Return empty arrays if no ranks are available
        if len(ranks) == 0:
            return np.array([]), np.array([])

        # Create logarithmically spaced bin boundaries
        binEdges = np.logspace(
            0,
            np.log10(ranks[-1] + 1),
            int(np.ceil(np.log10(ranks[-1] + 1) * binsPerDecade)) + 1
        )

        # Initialize lists for binned ranks and frequencies
        binnedRanks = []
        binnedFrequencies = []

        # Process each logarithmic bin
        for i in range(len(binEdges) - 1):

            # Select ranks within the current bin
            mask = (
                (ranks >= binEdges[i]) &
                (ranks < binEdges[i + 1])
            )

            # Process only bins containing observations
            if np.any(mask):

                # Calculate the average rank within the bin
                binnedRanks.append(np.mean(ranks[mask]))

                # Calculate the average frequency within the bin
                binnedFrequencies.append(
                    np.mean(frequencies[mask])
                )

        # Return binned ranks and frequencies as NumPy arrays
        return (
            np.array(binnedRanks),
            np.array(binnedFrequencies)
        )


    # Logarithmically bin the network degree distribution
    def logBinDegrees(self, degrees, binsPerDecade=10):

        # Convert node degrees to a NumPy array
        degrees = np.asarray(degrees)

        # Remove zero-degree nodes from the distribution
        degrees = degrees[degrees > 0]

        # Return empty arrays if no positive degrees exist
        if len(degrees) == 0:
            return np.array([]), np.array([])

        # Find the maximum observed degree
        maxDegree = degrees.max()

        # Create logarithmically spaced degree-bin boundaries
        binEdges = np.logspace(
            0,
            np.log10(maxDegree + 1),
            int(np.ceil(
                np.log10(maxDegree + 1) * binsPerDecade
            )) + 1
        )

        # Count the number of nodes in each degree bin
        counts, edges = np.histogram(
            degrees,
            bins=binEdges
        )

        # Calculate the width of each logarithmic bin
        widths = np.diff(edges)

        # Calculate geometric centers of the bins
        centers = np.sqrt(edges[:-1] * edges[1:])

        # Normalize counts to obtain probability density per unit degree
        density = counts / (len(degrees) * widths)

        # Select only bins containing at least one node
        mask = counts > 0

        # Return nonempty bin centers and their probability densities
        return centers[mask], density[mask]

    
    # Calculate power-law statistics from the degree distribution
    def calculatePowerLawStatistics(self, degrees):

        # Group node degrees into logarithmic bins
        binnedDegrees, binnedDensity = self.logBinDegrees(
            degrees
        )

        # Select bins with positive degrees and probability densities
        mask = (
            (binnedDegrees > 0) &
            (binnedDensity > 0)
        )

        # Extract valid degree values
        x = binnedDegrees[mask]

        # Extract corresponding probability densities
        y = binnedDensity[mask]

        # Require at least three valid points for regression
        if len(x) < 3:
            return None

        # Fit a straight line to the log-log degree distribution
        result = linregress(
            np.log10(x),
            np.log10(y)
        )

        # Return the estimated exponent and regression statistics
        return {
            'gamma': -result.slope,
            'r2': result.rvalue ** 2,
            'intercept': result.intercept,
            'degrees': x,
            'density': y
        }
    
    
    # Generate the deterministic DGM network
    def generateDGM(self, iterations):
        # Start with a triangle
        G = nx.complete_graph(3)

        # Expand every edge from the preceding growth stage
        for _ in range(iterations):
            oldEdges = list(G.edges())

            # Add a new node connected to both endpoints of each old edge
            for a, b in oldEdges:
                newNode = G.number_of_nodes()
                G.add_edge(newNode, a)
                G.add_edge(newNode, b)

        return G


    # Calculate Zipf statistics for a token sequence
    def calculateZipfStatistics(self, tokens):

        # Count token occurrences and sort frequencies in descending order
        frequencies = np.array(
            sorted(Counter(tokens).values(), reverse=True)
        )

        # Require at least two unique tokens
        if len(frequencies) < 2:
            return None

        # Assign ranks based on token frequency
        ranks = np.arange(1, len(frequencies) + 1)

        # Fit a straight line to the log-log rank-frequency distribution
        result = linregress(
            np.log10(ranks),
            np.log10(frequencies)
        )

        # Create an undirected and unweighted word-adjacency network
        G = nx.Graph()
        G.add_nodes_from(tokens)

        # Connect adjacent tokens and exclude self-loops
        for i in range(len(tokens) - 1):
            if tokens[i] != tokens[i + 1]:
                G.add_edge(tokens[i], tokens[i + 1])

        # Return network, frequency, and Zipf statistics
        return {
            'ranks': ranks,
            'frequencies': frequencies,
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'tokens': len(tokens),
            'alpha': -result.slope,
            'r2': result.rvalue ** 2
        }


    # Log Zipf statistics
    def logZipfStatistics(self, title, stats):

        # Display the analysis title
        self.logMessage(f'\n--- {title} ---')

        # Log network size and total tokens
        self.logMessage(
            f"Number of nodes: {stats['nodes']}"
        )
        self.logMessage(
            f"Number of edges: {stats['edges']}"
        )
        self.logMessage(
            f"Total tokens: {stats['tokens']}"
        )

        # Log token frequency statistics
        self.logMessage(
            f"Max frequency: {stats['frequencies'].max()}"
        )
        self.logMessage(
            f"Min frequency: {stats['frequencies'].min()}"
        )
        self.logMessage(
            f"Average frequency: "
            f"{stats['frequencies'].mean():.5f}"
        )

        # Log estimated Zipf exponent and regression fit
        self.logMessage(
            f"Zipf alpha: {stats['alpha']:.5f}"
        )
        self.logMessage(
            f"R-squared: {stats['r2']:.5f}"
        )


    # ZIPF MODEL ANALYSIS
    def zipfModelAnalysis(self, event):
        self.logMessage('\nZipf Model Analysis...')

        # Check whether punctuation comparison is enabled
        comparePunctuation = (
            self.comparePunctuationCheckbox.GetValue()
        )

        # Read and parse the input text only once
        text = self.readTextFile(self.inputTextFile)
        sentences = self.parseTextWithUdpipe(text)

        # Shuffle sentence order if selected
        sentences = self.applySentenceShuffle(sentences)

        # Create token sequence with selected punctuation
        tokensWith = self.filterUdpipeTokens(
            sentences, self.selectedPunctuation
        )

        # Initialize token sequence without punctuation
        tokensWithout = None

        # Create the second token sequence if comparison is enabled
        if comparePunctuation:
            tokensWithout = self.filterUdpipeTokens(
                sentences, {}
            )

        # Shuffle words using the selected random seed
        if self.selectedShuffle == 'Words':
            random.Random(self.shuffleSeed).shuffle(tokensWith)

            if comparePunctuation:
                random.Random(self.shuffleSeed).shuffle(tokensWithout)

        # Calculate Zipf statistics for the selected real text
        realWith = self.calculateZipfStatistics(tokensWith)

        # Stop if there are too few unique tokens
        if realWith is None:
            self.logMessage(
                'Not enough unique words for Zipf analysis.'
            )
            return

        # Initialize statistics for text without punctuation
        realWithout = None

        # Calculate statistics without punctuation if comparison is enabled
        if comparePunctuation:
            realWithout = self.calculateZipfStatistics(
                tokensWithout
            )

            if realWithout is None:
                self.logMessage(
                    'Not enough unique words without punctuation.'
                )
                return

        # Set parameters for the ideal Zipf model
        N = realWith['nodes']
        T = realWith['tokens']
        alpha = 1.0

        # Generate ranks for the ideal model
        ranks = np.arange(1, N + 1)

        # Calculate and normalize ideal Zipf probabilities
        probabilities = 1 / ranks ** alpha
        probabilities /= probabilities.sum()

        # Scale probabilities to the real text's token count
        zipfFrequencies = T * probabilities

        # Fit a log-log regression to the ideal Zipf distribution
        idealResult = linregress(
            np.log10(ranks),
            np.log10(zipfFrequencies)
        )

        # Calculate the ideal exponent and regression R-squared
        idealAlpha = -idealResult.slope
        idealR2 = idealResult.rvalue ** 2

        # Log real text statistics according to punctuation settings
        if comparePunctuation:
            self.logZipfStatistics(
                'REAL TEXT WITH PUNCTUATION',
                realWith
            )
            self.logZipfStatistics(
                'REAL TEXT WITHOUT PUNCTUATION',
                realWithout
            )
        else:
            self.logZipfStatistics(
                'REAL TEXT',
                realWith
            )

        # Log ideal Zipf model statistics
        self.logMessage('\n--- IDEAL ZIPF MODEL ---')
        self.logMessage(f'Number of word ranks: {N}')
        self.logMessage('Number of edges: N/A')
        self.logMessage(
            f'Expected total tokens: {zipfFrequencies.sum():.0f}'
        )
        self.logMessage(
            f'Max expected frequency: {zipfFrequencies.max():.5f}'
        )
        self.logMessage(
            f'Min expected frequency: {zipfFrequencies.min():.5f}'
        )
        self.logMessage(
            f'Average expected frequency: '
            f'{zipfFrequencies.mean():.5f}'
        )
        self.logMessage(
            f'Zipf alpha: {idealAlpha:.5f}'
        )
        self.logMessage(
            f'R-squared: {idealR2:.5f}'
        )
        self.logMessage('')

        # CLASSIC ZIPF VISUALIZATION
        plt.figure(figsize=(8, 6))

        # Plot both real text distributions if comparison is enabled
        if comparePunctuation:

            # Plot rank-frequency distribution with punctuation
            plt.loglog(
                realWith['ranks'],
                realWith['frequencies'],
                'go',
                markersize=3,
                label=(
                    f"Real With Punctuation "
                    f"(alpha={realWith['alpha']:.3f})"
                )
            )

            # Plot rank-frequency distribution without punctuation
            plt.loglog(
                realWithout['ranks'],
                realWithout['frequencies'],
                'bo',
                markersize=3,
                label=(
                    f"Real Without Punctuation "
                    f"(alpha={realWithout['alpha']:.3f})"
                )
            )

        else:
            # Plot the selected real text distribution
            plt.loglog(
                realWith['ranks'],
                realWith['frequencies'],
                'go' if self.selectedPunctuation else 'bo',
                markersize=3,
                label=(
                    f"Real Text "
                    f"(alpha={realWith['alpha']:.3f})"
                )
            )

        # Plot the ideal Zipf rank-frequency distribution
        plt.loglog(
            ranks,
            zipfFrequencies,
            'r-',
            linewidth=2,
            label=f'Ideal Zipf (alpha={idealAlpha:.3f})'
        )

        # Set classic plot labels and title
        plt.xlabel('Rank')
        plt.ylabel('Word Frequency')
        plt.title('Zipf Law - Real Text vs Ideal Model')
        plt.legend()
        plt.grid(False)
        plt.show()

        # LOG-BINNED ZIPF VISUALIZATION
        plt.figure(figsize=(8, 6))

        # Plot both log-binned distributions if comparison is enabled
        if comparePunctuation:

            # Calculate log-binned frequencies with punctuation
            binnedRanksWith, binnedFreqWith = self.logBinning(
                realWith['ranks'],
                realWith['frequencies']
            )

            # Plot log-binned distribution with punctuation
            plt.loglog(
                binnedRanksWith,
                binnedFreqWith,
                'g-',
                markersize=4,
                label=(
                    f"Real With Punctuation "
                    f"(alpha={realWith['alpha']:.3f})"
                )
            )

            # Calculate log-binned frequencies without punctuation
            binnedRanksWithout, binnedFreqWithout = self.logBinning(
                realWithout['ranks'],
                realWithout['frequencies']
            )

            # Plot log-binned distribution without punctuation
            plt.loglog(
                binnedRanksWithout,
                binnedFreqWithout,
                'b-',
                markersize=4,
                label=(
                    f"Real Without Punctuation "
                    f"(alpha={realWithout['alpha']:.3f})"
                )
            )

        else:
            # Calculate log-binned frequencies for the selected real text
            binnedRanksWith, binnedFreqWith = self.logBinning(
                realWith['ranks'],
                realWith['frequencies']
            )

            # Plot the selected log-binned real text distribution
            plt.loglog(
                binnedRanksWith,
                binnedFreqWith,
                'g-' if self.selectedPunctuation else 'b-',
                markersize=4,
                label=(
                    f"Real Text "
                    f"(alpha={realWith['alpha']:.3f})"
                )
            )

        # Calculate log-binned frequencies for the ideal Zipf model
        binnedRanksIdeal, binnedFreqIdeal = self.logBinning(
            ranks,
            zipfFrequencies
        )

        # Plot the ideal log-binned Zipf distribution
        plt.loglog(
            binnedRanksIdeal,
            binnedFreqIdeal,
            'r-',
            linewidth=2,
            label=f'Ideal Zipf (alpha={idealAlpha:.3f})'
        )

        # Set log-binned plot labels and title
        plt.xlabel('Rank')
        plt.ylabel('Average Word Frequency per Bin')
        plt.title('Zipf Law - Log-Binned Rank-Frequency Distribution')
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        plt.show()

    
    # STATISTICAL ANALYSIS - GRAPH AND LANGUAGE
    def statisticalAnalysis(self, event):
        self.logMessage('\nStatistical Analysis...')

        comparePunctuation = self.comparePunctuationCheckbox.GetValue()

        # Read and parse text once
        text = self.readTextFile(self.inputTextFile)
        sentences = self.parseTextWithUdpipe(text)
        sentences = self.applySentenceShuffle(sentences)

        # Create token sequences
        tokensWith = self.filterUdpipeTokens(
            sentences, self.selectedPunctuation
        )

        tokensWithout = None

        if comparePunctuation:
            tokensWithout = self.filterUdpipeTokens(sentences, {})

        # Shuffle words if selected
        if self.selectedShuffle == 'Words':
            random.Random(self.shuffleSeed).shuffle(tokensWith)

            if comparePunctuation:
                random.Random(self.shuffleSeed).shuffle(tokensWithout)

        # Analyze the real network and language
        def analyzeNetwork(tokens, title):
            self.logMessage(f'\n========== {title} ==========')

            if not tokens:
                self.logMessage('No tokens to analyze.')
                return

            # Create undirected, unweighted word adjacency network
            G = nx.Graph()
            G.add_nodes_from(tokens)

            for a, b in zip(tokens, tokens[1:]):
                if a != b:
                    G.add_edge(a, b)

            # GRAPH ANALYSIS
            self.logMessage('\n--- GRAPH ANALYSIS ---')

            N = G.number_of_nodes()
            E = G.number_of_edges()

            degrees = sorted(
                G.degree(),
                key=lambda x: x[1],
                reverse=True
            )

            degreeValues = [degree for _, degree in G.degree()]

            self.logMessage(f'Number of nodes: {N}')
            self.logMessage(f'Number of edges: {E}')

            self.logMessage(
                f'Max degree: {degrees[0]}'
            )
            self.logMessage(
                f'Min degree: {degrees[-1]}'
            )
            self.logMessage(
                f'Average degree: {sum(degreeValues) / N:.5f}'
            )

            self.logMessage(
                f'Network density: {nx.density(G):.5f}'
            )

            self.logMessage(
                f'Average clustering coefficient: '
                f'{nx.average_clustering(G):.5f}'
            )

            # Connected components
            components = list(nx.connected_components(G))

            self.logMessage(
                f'Connected components: {len(components)}'
            )

            largestNodes = max(components, key=len)
            largestComponent = G.subgraph(largestNodes)

            self.logMessage(
                f'Largest component size: '
                f'{largestComponent.number_of_nodes()}'
            )

            # # Shortest paths and diameter
            # # These require a connected graph
            # if nx.is_connected(G):
            #     self.logMessage(
            #         f'Average shortest path length: '
            #         f'{nx.average_shortest_path_length(G):.5f}'
            #     )
            #     self.logMessage(
            #         f'Diameter: {nx.diameter(G)}'
            #     )
            # else:
            #     self.logMessage(
            #         'Average shortest path length: N/A '
            #         '(disconnected graph)'
            #     )
            #     self.logMessage(
            #         'Diameter: N/A (disconnected graph)'
            #     )

            #     # Report these properties for the largest component
            #     if largestComponent.number_of_nodes() > 1:
            #         self.logMessage(
            #             f'Largest component average shortest path: '
            #             f'{nx.average_shortest_path_length(largestComponent):.5f}'
            #         )
            #         self.logMessage(
            #             f'Largest component diameter: '
            #             f'{nx.diameter(largestComponent)}'
            #         )

            # LANGUAGE ANALYSIS
            self.logMessage('\n--- LANGUAGE ANALYSIS ---')

            # Word occurrences, excluding punctuation
            words = [
                token for token in tokens
                if not self.isKnownPunctuation(token)
            ]

            wordCounts = Counter(words)
            wordLengths = [len(word) for word in wordCounts]

            # Sentence lengths from UDPipe sentence boundaries
            sentenceLengths = []

            for sentence in sentences:
                sentenceWords = [
                    item for item in sentence
                    if not item['isPunctuation']
                ]

                if sentenceWords:
                    sentenceLengths.append(len(sentenceWords))

            # Bigrams and trigrams
            bigramCounter = Counter(
                zip(tokens, tokens[1:])
            )

            trigramCounter = Counter(
                zip(tokens, tokens[1:], tokens[2:])
            )

            bigramFrequencies = list(bigramCounter.values())
            trigramFrequencies = list(trigramCounter.values())

            # Punctuation frequencies
            punctuationCounts = Counter(
                token for token in tokens
                if self.isKnownPunctuation(token)
            )

            self.logMessage(
                f'Number of punctuation marks: '
                f'{sum(punctuationCounts.values())}'
            )

            self.logMessage(
                f'Most common punctuation: '
                f'{punctuationCounts.most_common(1)[0] if punctuationCounts else "None"}'
            )

            self.logMessage(
                f'Least common punctuation: '
                f'{punctuationCounts.most_common()[-1] if punctuationCounts else "None"}'
            )

            # Word statistics
            self.logMessage(f'Total words: {len(words)}')
            self.logMessage(f'Number of unique words: {len(wordCounts)}')

            if wordLengths:
                self.logMessage(
                    f'Max word length: {max(wordLengths)}'
                )
                self.logMessage(
                    f'Min word length: {min(wordLengths)}'
                )
                self.logMessage(
                    f'Average word length: '
                    f'{sum(wordLengths) / len(wordLengths):.5f}'
                )

            # Sentence statistics
            self.logMessage(
                f'Number of sentences: {len(sentenceLengths)}'
            )

            if sentenceLengths:
                self.logMessage(
                    f'Max sentence length: {max(sentenceLengths)}'
                )
                self.logMessage(
                    f'Min sentence length: {min(sentenceLengths)}'
                )
                self.logMessage(
                    f'Average sentence length: '
                    f'{sum(sentenceLengths) / len(sentenceLengths):.5f}'
                )

            # Bigram statistics
            self.logMessage(
                f'Number of unique bigrams: {len(bigramCounter)}'
            )

            if bigramFrequencies:
                self.logMessage(
                    f'Max bigram frequency: {max(bigramFrequencies)}'
                )
                self.logMessage(
                    f'Min bigram frequency: {min(bigramFrequencies)}'
                )
                self.logMessage(
                    f'Average bigram frequency: '
                    f'{sum(bigramFrequencies) / len(bigramFrequencies):.5f}'
                )

            # Trigram statistics
            self.logMessage(
                f'Number of unique trigrams: {len(trigramCounter)}'
            )

            if trigramFrequencies:
                self.logMessage(
                    f'Max trigram frequency: {max(trigramFrequencies)}'
                )
                self.logMessage(
                    f'Min trigram frequency: {min(trigramFrequencies)}'
                )
                self.logMessage(
                    f'Average trigram frequency: '
                    f'{sum(trigramFrequencies) / len(trigramFrequencies):.5f}'
                )

        # Run analysis according to checkbox
        if comparePunctuation:
            analyzeNetwork(
                tokensWith,
                'REAL NETWORK WITH PUNCTUATION'
            )

            analyzeNetwork(
                tokensWithout,
                'REAL NETWORK WITHOUT PUNCTUATION'
            )
        else:
            analyzeNetwork(
                tokensWith,
                'REAL NETWORK'
            )

        self.logMessage('\nStatistical analysis completed.\n')

        
    # ERDOS-RENYI MODEL ANALYSIS
    def erdosRenyiAnalysis(self, event):
        self.logMessage('\nErdos-Renyi Model Analysis...')

        # Check whether to compare networks with and without punctuation
        comparePunctuation = (
            self.comparePunctuationCheckbox.GetValue()
        )

        # Read and parse text only once
        text = self.readTextFile(self.inputTextFile)
        sentences = self.parseTextWithUdpipe(text)

        # Shuffle sentence order if selected
        sentences = self.applySentenceShuffle(sentences)

        # Create token sequence with selected punctuation
        tokensWith = self.filterUdpipeTokens(
            sentences, self.selectedPunctuation
        )

        # Initialize token sequence without punctuation
        tokensWithout = None

        # Create a second token sequence if comparison is enabled
        if comparePunctuation:
            tokensWithout = self.filterUdpipeTokens(
                sentences, {}
            )

        # Shuffle words using the selected random seed
        if self.selectedShuffle == 'Words':
            random.Random(self.shuffleSeed).shuffle(tokensWith)

            if comparePunctuation:
                random.Random(self.shuffleSeed).shuffle(
                    tokensWithout
                )

        # Create separate figures for classic and log-binned distributions
        figClassic, axClassic = plt.subplots(figsize=(9, 6))
        figBinned, axBinned = plt.subplots(figsize=(9, 6))

        # Analyze the real network and optionally generate the ER model
        def analyzeER(tokens, title, color, simulateER=False):

            # Skip analysis if the token sequence is empty
            if not tokens:
                self.logMessage(f'\n--- {title} ---')
                self.logMessage('No tokens to analyze.')
                return

            # Create an undirected and unweighted graph
            G = nx.Graph()

            # Add unique tokens as graph nodes
            G.add_nodes_from(tokens)

            # Connect adjacent tokens and exclude self-loops
            for a, b in zip(tokens, tokens[1:]):
                if a != b:
                    G.add_edge(a, b)

            # Get the number of nodes and edges
            N = G.number_of_nodes()
            E = G.number_of_edges()

            # Calculate node degrees
            realDegrees = [
                degree for _, degree in G.degree()
            ]

            # Calculate clustering coefficient and network density
            realClustering = nx.average_clustering(G)
            realDensity = nx.density(G)

            # Log real network statistics
            self.logMessage(f'\n========== {title} ==========')

            self.logMessage('\n--- REAL NETWORK ---')
            self.logMessage(f'Number of nodes: {N}')
            self.logMessage(f'Number of edges: {E}')
            self.logMessage(
                f'Average degree: {sum(realDegrees) / N:.5f}'
            )
            self.logMessage(
                f'Maximum degree: {max(realDegrees)}'
            )
            self.logMessage(
                f'Network density: {realDensity:.5f}'
            )
            self.logMessage(
                f'Average clustering: {realClustering:.5f}'
            )

            # Count how many nodes have each degree
            realUnique, realCounts = np.unique(
                realDegrees,
                return_counts=True
            )

            # Plot the classic real-network degree distribution
            axClassic.loglog(
                realUnique[realUnique > 0],
                realCounts[realUnique > 0],
                'o',
                color=color,
                markersize=4,
                label=f'Real - {title}'
            )

            # Calculate the log-binned real-network degree distribution
            binnedDegrees, binnedDensity = (
                self.logBinDegrees(realDegrees, binsPerDecade=5)
            )

            # Plot the log-binned real-network degree distribution
            axBinned.loglog(
                binnedDegrees,
                binnedDensity,
                '-',
                color=color,
                markersize=4,
                label=f'Real - {title}'
            )

            # Generate the ER model only for the selected real network
            if simulateER:

                # Generate a random graph with the same nodes and edges
                ER = nx.gnm_random_graph(
                    N,
                    E,
                    seed=self.shuffleSeed
                )

                # Calculate degrees of ER model nodes
                erDegrees = [
                    degree for _, degree in ER.degree()
                ]

                # Calculate ER clustering coefficient and density
                erClustering = nx.average_clustering(ER)
                erDensity = nx.density(ER)

                # Log ER model statistics
                self.logMessage('\n--- ERDOS-RENYI MODEL ---')
                self.logMessage(
                    f'Number of nodes: {ER.number_of_nodes()}'
                )
                self.logMessage(
                    f'Number of edges: {ER.number_of_edges()}'
                )
                self.logMessage(
                    f'Average degree: '
                    f'{sum(erDegrees) / N:.5f}'
                )
                self.logMessage(
                    f'Maximum degree: {max(erDegrees)}'
                )
                self.logMessage(
                    f'Network density: {erDensity:.5f}'
                )
                self.logMessage(
                    f'Average clustering: {erClustering:.5f}'
                )

                # Count how many ER nodes have each degree
                erUnique, erCounts = np.unique(
                    erDegrees,
                    return_counts=True
                )

                # Plot the classic ER degree distribution
                axClassic.loglog(
                    erUnique[erUnique > 0],
                    erCounts[erUnique > 0],
                    'o',
                    color='red',
                    markersize=4,
                    label='Erdos-Renyi Model'
                )

                # Calculate the log-binned ER degree distribution
                erBinnedDegrees, erBinnedDensity = (
                    self.logBinDegrees(erDegrees, binsPerDecade=5)
                )

                # Plot the log-binned ER degree distribution
                axBinned.loglog(
                    erBinnedDegrees,
                    erBinnedDensity,
                    '-',
                    color='red',
                    markersize=4,
                    label='Erdos-Renyi Model'
                )

        # Check whether punctuation comparison is enabled
        if comparePunctuation:

            # Count unique nodes in both real networks
            nodesWith = len(set(tokensWith))
            nodesWithout = len(set(tokensWithout))

            # Select the larger network for ER model generation
            simulateWith = nodesWith >= nodesWithout

            # Analyze the real network with punctuation
            analyzeER(
                tokensWith,
                'With Punctuation',
                'green',
                simulateER=simulateWith
            )

            # Analyze the real network without punctuation
            analyzeER(
                tokensWithout,
                'Without Punctuation',
                'blue',
                simulateER=not simulateWith
            )

        else:
            # Analyze one real network and generate its ER model
            analyzeER(
                tokensWith,
                'Real Text',
                'green' if self.selectedPunctuation else 'blue',
                simulateER=True
            )

        # Set labels and title for the classic degree distribution
        axClassic.set_xlabel('Degree (k)')
        axClassic.set_ylabel('Number of Nodes')
        axClassic.set_title(
            'Degree Distribution: Real Network vs Erdos-Renyi'
        )

        # Display the classic distribution legend
        axClassic.legend()
        axClassic.grid(False)
        figClassic.tight_layout()

        # Set labels and title for the log-binned distribution
        axBinned.set_xlabel('Degree (k)')
        axBinned.set_ylabel('Probability Density P(k)')
        axBinned.set_title(
            'Log-Binned Degree Distribution: '
            'Real Network vs Erdos-Renyi'
        )

        # Display the log-binned distribution legend
        axBinned.legend()
        axBinned.grid(False)
        figBinned.tight_layout()

        # Display both graph windows
        plt.show()

        # Log completion message
        self.logMessage('\nErdos-Renyi analysis completed.\n')

    
    # WATTS-STROGATZ MODEL ANALYSIS
    def wattsStrogatzAnalysis(self, event):
        self.logMessage('\nWatts-Strogatz Model Analysis...')

        # Check whether punctuation comparison is enabled
        comparePunctuation = (
            self.comparePunctuationCheckbox.GetValue()
        )

        # Read and parse text only once
        text = self.readTextFile(self.inputTextFile)
        sentences = self.parseTextWithUdpipe(text)

        # Shuffle sentence order if selected
        sentences = self.applySentenceShuffle(sentences)

        # Create token sequence with selected punctuation
        tokensWith = self.filterUdpipeTokens(
            sentences, self.selectedPunctuation
        )

        # Initialize token sequence without punctuation
        tokensWithout = None

        # Create second token sequence if comparison is enabled
        if comparePunctuation:
            tokensWithout = self.filterUdpipeTokens(
                sentences, {}
            )

        # Shuffle words using the selected random seed
        if self.selectedShuffle == 'Words':
            random.Random(self.shuffleSeed).shuffle(tokensWith)

            if comparePunctuation:
                random.Random(self.shuffleSeed).shuffle(
                    tokensWithout
                )

        # Create separate figures for classic and log-binned distributions
        figClassic, axClassic = plt.subplots(figsize=(9, 6))
        figBinned, axBinned = plt.subplots(figsize=(9, 6))

        # Analyze a real network and optionally generate the WS model
        def analyzeWS(tokens, title, color, simulateWS=False):

            # Skip analysis if the token sequence is empty
            if not tokens:
                self.logMessage(f'\n--- {title} ---')
                self.logMessage('No tokens to analyze.')
                return

            # Create an undirected and unweighted word-adjacency network
            G = nx.Graph()

            # Add unique tokens as graph nodes
            G.add_nodes_from(tokens)

            # Connect adjacent tokens and exclude self-loops
            for a, b in zip(tokens, tokens[1:]):
                if a != b:
                    G.add_edge(a, b)

            # Get the number of nodes and edges
            N = G.number_of_nodes()
            E = G.number_of_edges()

            # Calculate node degrees
            realDegrees = [
                degree for _, degree in G.degree()
            ]

            # Calculate clustering coefficient and network density
            realClustering = nx.average_clustering(G)
            realDensity = nx.density(G)

            # Log real network statistics
            self.logMessage(f'\n========== {title} ==========')

            self.logMessage('\n--- REAL NETWORK ---')
            self.logMessage(f'Number of nodes: {N}')
            self.logMessage(f'Number of edges: {E}')
            self.logMessage(
                f'Average degree: {sum(realDegrees) / N:.5f}'
            )
            self.logMessage(
                f'Maximum degree: {max(realDegrees)}'
            )
            self.logMessage(
                f'Network density: {realDensity:.5f}'
            )
            self.logMessage(
                f'Average clustering: {realClustering:.5f}'
            )

            # Count nodes with each degree
            realUnique, realCounts = np.unique(
                realDegrees,
                return_counts=True
            )

            # Plot the classic real-network degree distribution
            axClassic.loglog(
                realUnique[realUnique > 0],
                realCounts[realUnique > 0],
                'o',
                color=color,
                markersize=4,
                label=f'Real - {title}'
            )

            # Calculate the log-binned real-network degree distribution
            binnedDegrees, binnedDensity = (
                self.logBinDegrees(realDegrees, binsPerDecade=5)
            )

            # Plot the log-binned real-network degree distribution
            axBinned.loglog(
                binnedDegrees,
                binnedDensity,
                'o',
                color=color,
                markersize=4,
                label=f'Real - {title}'
            )

            # Generate the WS model only for the selected network
            if simulateWS:

                # Calculate the real network's average degree
                averageDegree = 2 * E / N

                # Approximate average degree with an even integer
                k = 2 * round(averageDegree / 2)
                k = max(2, min(k, N - 1))

                # Ensure k is even and smaller than N
                if k % 2 != 0:
                    k -= 1

                # Set the probability of rewiring each lattice edge
                p = 0.1

                # Generate the Watts-Strogatz small-world network
                WS = nx.watts_strogatz_graph(
                    N,
                    k,
                    p,
                    seed=self.shuffleSeed
                )

                # Calculate WS node degrees
                wsDegrees = [
                    degree for _, degree in WS.degree()
                ]

                # Calculate WS clustering coefficient and density
                wsClustering = nx.average_clustering(WS)
                wsDensity = nx.density(WS)

                # Log WS model statistics
                self.logMessage('\n--- WATTS-STROGATZ MODEL ---')
                self.logMessage(f'Number of nodes: {N}')
                self.logMessage(
                    f'Number of edges: {WS.number_of_edges()}'
                )
                self.logMessage(f'Parameter k: {k}')
                self.logMessage(f'Rewiring probability p: {p}')
                self.logMessage(
                    f'Average degree: '
                    f'{sum(wsDegrees) / N:.5f}'
                )
                self.logMessage(
                    f'Maximum degree: {max(wsDegrees)}'
                )
                self.logMessage(
                    f'Network density: {wsDensity:.5f}'
                )
                self.logMessage(
                    f'Average clustering: {wsClustering:.5f}'
                )

                # Count WS nodes with each degree
                wsUnique, wsCounts = np.unique(
                    wsDegrees,
                    return_counts=True
                )

                # Plot the classic WS degree distribution
                axClassic.loglog(
                    wsUnique[wsUnique > 0],
                    wsCounts[wsUnique > 0],
                    'o',
                    color='red',
                    markersize=4,
                    label='Watts-Strogatz Model'
                )

                # Calculate the log-binned WS degree distribution
                wsBinnedDegrees, wsBinnedDensity = (
                    self.logBinDegrees(wsDegrees, binsPerDecade=5)
                )

                # Plot the log-binned WS degree distribution
                axBinned.loglog(
                    wsBinnedDegrees,
                    wsBinnedDensity,
                    'o',
                    color='red',
                    markersize=4,
                    label='Watts-Strogatz Model'
                )

        # Check whether punctuation comparison is enabled
        if comparePunctuation:

            # Count unique nodes in both real networks
            nodesWith = len(set(tokensWith))
            nodesWithout = len(set(tokensWithout))

            # Select the larger network for WS model generation
            simulateWith = nodesWith >= nodesWithout

            # Analyze the real network with punctuation
            analyzeWS(
                tokensWith,
                'With Punctuation',
                'green',
                simulateWS=simulateWith
            )

            # Analyze the real network without punctuation
            analyzeWS(
                tokensWithout,
                'Without Punctuation',
                'blue',
                simulateWS=not simulateWith
            )

        else:
            # Analyze one real network and generate its WS model
            analyzeWS(
                tokensWith,
                'Real Text',
                'green' if self.selectedPunctuation else 'blue',
                simulateWS=True
            )

        # Set labels and title for the classic degree distribution
        axClassic.set_xlabel('Degree (k)')
        axClassic.set_ylabel('Number of Nodes')
        axClassic.set_title(
            'Degree Distribution: Real Network vs Watts-Strogatz'
        )

        # Display the classic distribution legend
        axClassic.legend()
        axClassic.grid(False)
        figClassic.tight_layout()

        # Set labels and title for the log-binned degree distribution
        axBinned.set_xlabel('Degree (k)')
        axBinned.set_ylabel('Probability Density P(k)')
        axBinned.set_title(
            'Log-Binned Degree Distribution: '
            'Real Network vs Watts-Strogatz'
        )

        # Display the log-binned distribution legend
        axBinned.legend()
        axBinned.grid(False)
        figBinned.tight_layout()

        # Display both graph windows
        plt.show()

        # Log completion message
        self.logMessage(
            '\nWatts-Strogatz analysis completed.\n'
        )
    
    # POWER-LAW STATISTICS FROM UNBINNED DEGREES
    def calculatePowerLawStatistics(self, degrees, minTailSize=50):
        degrees = np.asarray(degrees, dtype=int)
        degrees = degrees[degrees > 0]

        if len(degrees) < minTailSize:
            return None

        bestFit = None

        # Search for the best minimum degree
        for kMin in np.unique(degrees):
            tail = degrees[degrees >= kMin]
            n = len(tail)

            if n < minTailSize:
                break

            # Approximate discrete maximum likelihood estimation
            denominator = np.sum(
                np.log(tail / (kMin - 0.5))
            )

            if denominator <= 0:
                continue

            gamma = 1 + n / denominator

            # Empirical cumulative distribution
            uniqueDegrees, counts = np.unique(
                tail, return_counts=True
            )

            empiricalCDF = np.cumsum(counts) / n

            # Approximate theoretical cumulative distribution
            modelCDF = 1 - (
                uniqueDegrees / (kMin - 0.5)
            ) ** (1 - gamma)

            ks = np.max(
                np.abs(empiricalCDF - modelCDF)
            )

            if bestFit is None or ks < bestFit['ks']:
                bestFit = {
                    'gamma': gamma,
                    'kMin': int(kMin),
                    'ks': ks,
                    'tailSize': n
                }

        return bestFit


    # BARABASI-ALBERT MODEL ANALYSIS
    def barabasiAlbertAnalysis(self, event):
        self.logMessage('\nBarabasi-Albert Model Analysis...')

        # Check whether punctuation comparison is enabled
        comparePunctuation = (
            self.comparePunctuationCheckbox.GetValue()
        )

        # Read and process text once
        text = self.readTextFile(self.inputTextFile)
        sentences = self.parseTextWithUdpipe(text)
        # Shuffle sentence order if selected
        sentences = self.applySentenceShuffle(sentences)

        # Tokens with selected punctuation
        tokensWith = self.filterUdpipeTokens(
            sentences, self.selectedPunctuation
        )

        # Tokens without punctuation
        tokensWithout = None

        # Create or analyze the punctuation-free comparison when enabled
        if comparePunctuation:
            tokensWithout = self.filterUdpipeTokens(
                sentences, {}
            )

        # Apply word shuffling if selected
        if self.selectedShuffle == 'Words':
            random.Random(self.shuffleSeed).shuffle(tokensWith)

            # Create or analyze the punctuation-free comparison when enabled
            if comparePunctuation:
                random.Random(self.shuffleSeed).shuffle(
                    tokensWithout
                )

        # Create two separate graph windows
        # Prepare the classic and log-binned figures
        figClassic, axClassic = plt.subplots(figsize=(9, 6))
        figBinned, axBinned = plt.subplots(figsize=(9, 6))

        # Analyze real network and optionally generate BA model
        def analyzeBA(tokens, title, color, simulateBA=False):
            # Skip empty token sequences
            if not tokens:
                self.logMessage(f'\n--- {title} ---')
                self.logMessage('No tokens to analyze.')
                return

            # Create real word-adjacency network
            G = nx.Graph()
            # Add unique tokens as nodes
            G.add_nodes_from(tokens)

            # Connect consecutive tokens without self-loops
            for a, b in zip(tokens, tokens[1:]):
                if a != b:
                    G.add_edge(a, b)

            # Get the real network size
            N = G.number_of_nodes()
            E = G.number_of_edges()

            # Real network statistics
            realDegrees = [
                degree for _, degree in G.degree()
            ]

            # Measure clustering and network density
            realClustering = nx.average_clustering(G)
            realDensity = nx.density(G)

            # Estimate power-law exponent from unbinned degrees
            realPowerLaw = self.calculatePowerLawStatistics(
                realDegrees
            )

            # Log real network statistics
            self.logMessage(f'\n========== {title} ==========')

            self.logMessage('\n--- REAL NETWORK ---')
            self.logMessage(f'Number of nodes: {N}')
            self.logMessage(f'Number of edges: {E}')
            self.logMessage(
                f'Average degree: {sum(realDegrees) / N:.5f}'
            )
            self.logMessage(
                f'Maximum degree: {max(realDegrees)}'
            )
            self.logMessage(
                f'Network density: {realDensity:.5f}'
            )
            self.logMessage(
                f'Average clustering: {realClustering:.5f}'
            )

            # Log real power-law statistics
            self.logMessage('\n--- REAL POWER-LAW FIT ---')

            if realPowerLaw:
                self.logMessage(
                    f"Estimated gamma: "
                    f"{realPowerLaw['gamma']:.5f}"
                )
                self.logMessage(
                    f"Minimum degree kMin: "
                    f"{realPowerLaw['kMin']}"
                )
                self.logMessage(
                    f"Tail nodes: "
                    f"{realPowerLaw['tailSize']}"
                )
                self.logMessage(
                    f"KS distance: "
                    f"{realPowerLaw['ks']:.5f}"
                )
            else:
                self.logMessage(
                    'Not enough data for power-law estimation.'
                )

            # Classic degree distribution
            # Count real nodes at each degree
            realUnique, realCounts = np.unique(
                realDegrees,
                return_counts=True
            )

            # Plot the classic degree distribution
            axClassic.loglog(
                realUnique[realUnique > 0],
                realCounts[realUnique > 0],
                'o',
                color=color,
                markersize=4,
                label=f'Real - {title}'
            )

            # Log-binned degree distribution
            binnedDegrees, binnedDensity = (
                self.logBinDegrees(
                    realDegrees,
                    binsPerDecade=3
                )
            )

            # Plot the log-binned degree distribution or fitted tail
            axBinned.loglog(
                binnedDegrees,
                binnedDensity,
                'o-',
                color=color,
                markersize=2,
                label=f'Real - {title}'
            )

            # Real network fitted power-law tail
            if realPowerLaw:
                kMin = realPowerLaw['kMin']
                gamma = realPowerLaw['gamma']

                # Generate degree values for the fitted tail line
                fitX = np.logspace(
                    np.log10(kMin),
                    np.log10(max(realDegrees)),
                    100
                )

                # Align the line with the binned distribution
                tailMask = binnedDegrees >= kMin

                if np.any(tailMask):
                    firstIndex = np.where(tailMask)[0][0]

                    referenceX = binnedDegrees[firstIndex]
                    referenceY = binnedDensity[firstIndex]

                    # Scale the fitted line to the observed density
                    fitY = referenceY * (
                        fitX / referenceX
                    ) ** (-gamma)

                    # Plot the log-binned degree distribution or fitted tail
                    axBinned.loglog(
                        fitX,
                        fitY,
                        '--',
                        color=color,
                        linewidth=1.5,
                        label=(
                            f"Real Power Law "
                            f"(gamma={gamma:.3f}, kMin={kMin})"
                        )
                    )

            # Generate BA model only for selected network
            if simulateBA:
                # Match real average degree approximately
                averageDegree = 2 * E / N
                m = round(averageDegree / 2)

                # BA requires 1 <= m < N
                m = max(1, min(m, N - 1))

                # Generate a preferential-attachment network
                BA = nx.barabasi_albert_graph(
                    N,
                    m,
                    seed=self.shuffleSeed
                )

                # Calculate BA node degrees
                baDegrees = [
                    degree for _, degree in BA.degree()
                ]

                # Measure BA clustering and network density
                baClustering = nx.average_clustering(BA)
                baDensity = nx.density(BA)

                # Estimate BA power-law from unbinned degrees
                baPowerLaw = self.calculatePowerLawStatistics(
                    baDegrees
                )

                # Log BA statistics
                self.logMessage(
                    '\n--- BARABASI-ALBERT MODEL ---'
                )
                self.logMessage(f'Number of nodes: {N}')
                self.logMessage(
                    f'Number of edges: {BA.number_of_edges()}'
                )
                self.logMessage(f'Parameter m: {m}')
                self.logMessage(
                    f'Average degree: '
                    f'{sum(baDegrees) / N:.5f}'
                )
                self.logMessage(
                    f'Maximum degree: {max(baDegrees)}'
                )
                self.logMessage(
                    f'Network density: {baDensity:.5f}'
                )
                self.logMessage(
                    f'Average clustering: {baClustering:.5f}'
                )

                # Log BA power-law statistics
                self.logMessage('\n--- BA POWER-LAW FIT ---')
                self.logMessage(
                    'Theoretical gamma: 3.00000'
                )

                if baPowerLaw:
                    self.logMessage(
                        f"Estimated gamma: "
                        f"{baPowerLaw['gamma']:.5f}"
                    )
                    self.logMessage(
                        f"Minimum degree kMin: "
                        f"{baPowerLaw['kMin']}"
                    )
                    self.logMessage(
                        f"Tail nodes: "
                        f"{baPowerLaw['tailSize']}"
                    )
                    self.logMessage(
                        f"KS distance: "
                        f"{baPowerLaw['ks']:.5f}"
                    )
                else:
                    self.logMessage(
                        'Not enough data for power-law estimation.'
                    )

                # Classic BA degree distribution
                # Count BA nodes at each degree
                baUnique, baCounts = np.unique(
                    baDegrees,
                    return_counts=True
                )

                # Plot the classic degree distribution
                axClassic.loglog(
                    baUnique[baUnique > 0],
                    baCounts[baUnique > 0],
                    'o',
                    color='red',
                    markersize=4,
                    label='Barabasi-Albert Model'
                )

                # Log-binned BA degree distribution
                # Calculate log-binned BA degree density
                baBinnedDegrees, baBinnedDensity = (
                    self.logBinDegrees(
                        baDegrees,
                        binsPerDecade=3
                    )
                )

                # Plot the log-binned degree distribution or fitted tail
                axBinned.loglog(
                    baBinnedDegrees,
                    baBinnedDensity,
                    'o-',
                    color='red',
                    markersize=2,
                    label='Barabasi-Albert Model'
                )

                # BA model fitted power-law tail
                if baPowerLaw:
                    kMin = baPowerLaw['kMin']
                    gamma = baPowerLaw['gamma']

                    # Generate degree values for the fitted tail line
                    fitX = np.logspace(
                        np.log10(kMin),
                        np.log10(max(baDegrees)),
                        100
                    )

                    # Align the line with the binned distribution
                    tailMask = baBinnedDegrees >= kMin

                    if np.any(tailMask):
                        firstIndex = np.where(tailMask)[0][0]

                        referenceX = baBinnedDegrees[firstIndex]
                        referenceY = baBinnedDensity[firstIndex]

                        # Scale the fitted line to the observed density
                        fitY = referenceY * (
                            fitX / referenceX
                        ) ** (-gamma)

                        # Plot the log-binned degree distribution or fitted tail
                        axBinned.loglog(
                            fitX,
                            fitY,
                            '--',
                            color='red',
                            linewidth=1.5,
                            label=(
                                f"BA Power Law "
                                f"(gamma={gamma:.3f}, kMin={kMin})"
                            )
                        )

        # Analyze according to punctuation checkbox
        # Create or analyze the punctuation-free comparison when enabled
        if comparePunctuation:
            # Count unique nodes for both punctuation settings
            nodesWith = len(set(tokensWith))
            nodesWithout = len(set(tokensWithout))

            # Generate one BA model for the larger real network
            simulateWith = nodesWith >= nodesWithout

            analyzeBA(
                tokensWith,
                'With Punctuation',
                'green',
                simulateBA=simulateWith
            )

            analyzeBA(
                tokensWithout,
                'Without Punctuation',
                'blue',
                simulateBA=not simulateWith
            )

        else:
            analyzeBA(
                tokensWith,
                'Real Text',
                'green' if self.selectedPunctuation else 'blue',
                simulateBA=True
            )

        # Classic graph settings
        axClassic.set_xlabel('Degree (k)')
        axClassic.set_ylabel('Number of Nodes')
        axClassic.set_title(
            'Degree Distribution: Real Network vs Barabasi-Albert'
        )
        # Finalize the classic degree plot
        axClassic.legend()
        axClassic.grid(False)
        figClassic.tight_layout()

        # Log-binned graph settings
        axBinned.set_xlabel('Degree (k)')
        axBinned.set_ylabel('Probability Density P(k)')
        axBinned.set_title(
            'Log-Binned Degree Distribution: '
            'Real Network vs Barabasi-Albert'
        )
        # Finalize the log-binned degree plot
        axBinned.legend()
        axBinned.grid(False)
        figBinned.tight_layout()

        # Show both graph windows
        plt.show()

        self.logMessage(
            '\nBarabasi-Albert analysis completed.\n'
        )

    
    # DOROGOVTSEV-GOLTSEV-MENDES MODEL ANALYSIS
    def dgmAnalysis(self, event):
        self.logMessage('\nDGM Model Analysis...')

        # Check whether punctuation comparison is enabled
        comparePunctuation = (
            self.comparePunctuationCheckbox.GetValue()
        )

        # Read and process text once
        # Read and parse the source text
        text = self.readTextFile(self.inputTextFile)
        sentences = self.parseTextWithUdpipe(text)
        # Apply sentence shuffling if selected
        sentences = self.applySentenceShuffle(sentences)

        # Tokens with selected punctuation
        tokensWith = self.filterUdpipeTokens(
            sentences, self.selectedPunctuation
        )

        # Tokens without punctuation
        tokensWithout = None

        # Prepare a second sequence without punctuation if requested
        if comparePunctuation:
            tokensWithout = self.filterUdpipeTokens(
                sentences, {}
            )

        # Apply word shuffling if selected
        if self.selectedShuffle == 'Words':
            random.Random(self.shuffleSeed).shuffle(tokensWith)

            if comparePunctuation:
                random.Random(self.shuffleSeed).shuffle(
                    tokensWithout
                )

        # Create two separate graph windows
        figClassic, axClassic = plt.subplots(figsize=(9, 6))
        figBinned, axBinned = plt.subplots(figsize=(9, 6))

        # Analyze real network and optionally generate DGM
        def analyzeDGM(tokens, title, color, simulateDGM=False):
            # Skip empty token sequences
            if not tokens:
                self.logMessage(f'\n--- {title} ---')
                self.logMessage('No tokens to analyze.')
                return

            # Create real word-adjacency network
            G = nx.Graph()
            # Add unique tokens as nodes
            G.add_nodes_from(tokens)

            # Connect consecutive tokens without self-loops
            for a, b in zip(tokens, tokens[1:]):
                if a != b:
                    G.add_edge(a, b)

            # Record network size
            N = G.number_of_nodes()
            E = G.number_of_edges()

            # Real network statistics
            realDegrees = [
                degree for _, degree in G.degree()
            ]

            # Measure local clustering and density
            realClustering = nx.average_clustering(G)
            realDensity = nx.density(G)

            # Estimate power law from unbinned degrees
            realPowerLaw = self.calculatePowerLawStatistics(
                realDegrees
            )

            # Log real network statistics
            self.logMessage(f'\n========== {title} ==========')

            self.logMessage('\n--- REAL NETWORK ---')
            self.logMessage(f'Number of nodes: {N}')
            self.logMessage(f'Number of edges: {E}')
            self.logMessage(
                f'Average degree: {sum(realDegrees) / N:.5f}'
            )
            self.logMessage(
                f'Maximum degree: {max(realDegrees)}'
            )
            self.logMessage(
                f'Network density: {realDensity:.5f}'
            )
            self.logMessage(
                f'Average clustering: {realClustering:.5f}'
            )

            # Log real power-law statistics
            self.logMessage('\n--- REAL POWER-LAW FIT ---')

            if realPowerLaw:
                self.logMessage(
                    f"Estimated gamma: "
                    f"{realPowerLaw['gamma']:.5f}"
                )
                self.logMessage(
                    f"Minimum degree kMin: "
                    f"{realPowerLaw['kMin']}"
                )
                self.logMessage(
                    f"Tail nodes: "
                    f"{realPowerLaw['tailSize']}"
                )
                self.logMessage(
                    f"KS distance: "
                    f"{realPowerLaw['ks']:.5f}"
                )
            else:
                self.logMessage(
                    'Not enough data for power-law estimation.'
                )

            # Classic degree distribution
            # Count nodes at each observed degree
            realUnique, realCounts = np.unique(
                realDegrees,
                return_counts=True
            )

            # Plot degree counts on logarithmic axes
            axClassic.loglog(
                realUnique[realUnique > 0],
                realCounts[realUnique > 0],
                'o',
                color=color,
                markersize=4,
                label=f'Real - {title}'
            )

            # Log-binned degree distribution
            binnedDegrees, binnedDensity = (
                self.logBinDegrees(
                    realDegrees,
                    binsPerDecade=3
                )
            )

            # Plot log-binned degree density
            axBinned.loglog(
                binnedDegrees,
                binnedDensity,
                'o-',
                color=color,
                markersize=2,
                label=f'Real - {title}'
            )

            # Real fitted power-law tail
            if realPowerLaw:
                # Retrieve the fitted tail cutoff and exponent
                kMin = realPowerLaw['kMin']
                gamma = realPowerLaw['gamma']

                # Create degrees for the fitted power-law guide
                fitX = np.logspace(
                    np.log10(kMin),
                    np.log10(max(realDegrees)),
                    100
                )

                # Find a binned tail point to position the fitted line
                tailMask = binnedDegrees >= kMin

                if np.any(tailMask):
                    firstIndex = np.where(tailMask)[0][0]

                    referenceX = binnedDegrees[firstIndex]
                    referenceY = binnedDensity[firstIndex]

                    fitY = referenceY * (
                        fitX / referenceX
                    ) ** (-gamma)

                    axBinned.loglog(
                        fitX,
                        fitY,
                        '--',
                        color=color,
                        linewidth=1.5,
                        label=(
                            f"Real Power Law "
                            f"(gamma={gamma:.3f}, kMin={kMin})"
                        )
                    )

            # Generate DGM model only for selected network
            if simulateDGM:
                # Find iteration with closest number of nodes
                iterations = 0

                while True:
                    currentN = (3 ** (iterations + 1) + 3) // 2
                    nextN = (3 ** (iterations + 2) + 3) // 2

                    if abs(nextN - N) >= abs(currentN - N):
                        break

                    iterations += 1

                # Build the deterministic DGM network
                DGM = self.generateDGM(iterations)

                # Calculate degrees in the generated model
                dgmDegrees = [
                    degree for _, degree in DGM.degree()
                ]

                # Measure DGM clustering and density
                dgmClustering = nx.average_clustering(DGM)
                dgmDensity = nx.density(DGM)

                # Estimate DGM power law
                dgmPowerLaw = self.calculatePowerLawStatistics(
                    dgmDegrees
                )

                # Log DGM statistics
                self.logMessage('\n--- DGM MODEL ---')
                self.logMessage(
                    f'Iterations: {iterations}'
                )
                self.logMessage(
                    f'Number of nodes: {DGM.number_of_nodes()}'
                )
                self.logMessage(
                    f'Number of edges: {DGM.number_of_edges()}'
                )
                self.logMessage(
                    f'Average degree: '
                    f'{sum(dgmDegrees) / len(dgmDegrees):.5f}'
                )
                self.logMessage(
                    f'Maximum degree: {max(dgmDegrees)}'
                )
                self.logMessage(
                    f'Network density: {dgmDensity:.5f}'
                )
                self.logMessage(
                    f'Average clustering: {dgmClustering:.5f}'
                )

                # Log DGM power-law statistics
                self.logMessage('\n--- DGM POWER-LAW FIT ---')
                self.logMessage(
                    'Theoretical gamma: '
                    f'{1 + np.log(3) / np.log(2):.5f}'
                )

                if dgmPowerLaw:
                    self.logMessage(
                        f"Estimated gamma: "
                        f"{dgmPowerLaw['gamma']:.5f}"
                    )
                    self.logMessage(
                        f"Minimum degree kMin: "
                        f"{dgmPowerLaw['kMin']}"
                    )
                    self.logMessage(
                        f"Tail nodes: "
                        f"{dgmPowerLaw['tailSize']}"
                    )
                    self.logMessage(
                        f"KS distance: "
                        f"{dgmPowerLaw['ks']:.5f}"
                    )
                else:
                    self.logMessage(
                        'Not enough data for power-law estimation.'
                    )

                # Classic DGM degree distribution
                # Count DGM nodes at each degree
                dgmUnique, dgmCounts = np.unique(
                    dgmDegrees,
                    return_counts=True
                )

                axClassic.loglog(
                    dgmUnique[dgmUnique > 0],
                    dgmCounts[dgmUnique > 0],
                    'o',
                    color='red',
                    markersize=4,
                    label='DGM Model'
                )

                # Log-binned DGM degree distribution
                # Calculate the log-binned DGM degree distribution
                dgmBinnedDegrees, dgmBinnedDensity = (
                    self.logBinDegrees(
                        dgmDegrees,
                        binsPerDecade=3
                    )
                )

                axBinned.loglog(
                    dgmBinnedDegrees,
                    dgmBinnedDensity,
                    'o-',
                    color='red',
                    markersize=2,
                    label='DGM Model'
                )

                # DGM fitted power-law tail
                if dgmPowerLaw:
                    # Retrieve the DGM fitted tail parameters
                    kMin = dgmPowerLaw['kMin']
                    gamma = dgmPowerLaw['gamma']

                    fitX = np.logspace(
                        np.log10(kMin),
                        np.log10(max(dgmDegrees)),
                        100
                    )

                    # Find a binned tail point to position the fitted line
                    tailMask = dgmBinnedDegrees >= kMin

                    if np.any(tailMask):
                        firstIndex = np.where(tailMask)[0][0]

                        referenceX = dgmBinnedDegrees[firstIndex]
                        referenceY = dgmBinnedDensity[firstIndex]

                        fitY = referenceY * (
                            fitX / referenceX
                        ) ** (-gamma)

                        axBinned.loglog(
                            fitX,
                            fitY,
                            '--',
                            color='red',
                            linewidth=1.5,
                            label=(
                                f"DGM Power Law "
                                f"(gamma={gamma:.3f}, kMin={kMin})"
                            )
                        )

        # Analyze according to punctuation checkbox
        if comparePunctuation:
            nodesWith = len(set(tokensWith))
            nodesWithout = len(set(tokensWithout))

            # Generate DGM for the real network with more nodes
            simulateWith = nodesWith >= nodesWithout

            analyzeDGM(
                tokensWith,
                'With Punctuation',
                'green',
                simulateDGM=simulateWith
            )

            analyzeDGM(
                tokensWithout,
                'Without Punctuation',
                'blue',
                simulateDGM=not simulateWith
            )

        else:
            analyzeDGM(
                tokensWith,
                'Real Text',
                'green' if self.selectedPunctuation else 'blue',
                simulateDGM=True
            )

        # Classic graph settings
        axClassic.set_xlabel('Degree (k)')
        axClassic.set_ylabel('Number of Nodes')
        axClassic.set_title(
            'Degree Distribution: Real Network vs DGM'
        )
        axClassic.legend()
        axClassic.grid(False)
        figClassic.tight_layout()

        # Log-binned graph settings
        axBinned.set_xlabel('Degree (k)')
        axBinned.set_ylabel('Probability Density P(k)')
        axBinned.set_title(
            'Log-Binned Degree Distribution: Real Network vs DGM'
        )
        axBinned.legend()
        axBinned.grid(False)
        figBinned.tight_layout()

        # Show both graph windows
        # Display both degree-distribution figures
        plt.show()

        self.logMessage('\nDGM analysis completed.\n')

    
    # HOLME-KIM MODEL ANALYSIS
    def holmeKimAnalysis(self, event):
        self.logMessage('\nHolme-Kim Model Analysis...')

        # Check whether punctuation comparison is enabled
        comparePunctuation = (
            self.comparePunctuationCheckbox.GetValue()
        )

        # Triad formation probability
        p = 0.85

        # Read and process text once
        text = self.readTextFile(self.inputTextFile)
        sentences = self.parseTextWithUdpipe(text)
        # Apply sentence shuffling if selected
        sentences = self.applySentenceShuffle(sentences)

        # Tokens with selected punctuation
        tokensWith = self.filterUdpipeTokens(
            sentences, self.selectedPunctuation
        )

        # Tokens without punctuation
        # Initialize the punctuation-free token sequence
        tokensWithout = None

        if comparePunctuation:
            tokensWithout = self.filterUdpipeTokens(
                sentences, {}
            )

        # Apply word shuffling
        if self.selectedShuffle == 'Words':
            random.Random(self.shuffleSeed).shuffle(tokensWith)

            if comparePunctuation:
                random.Random(self.shuffleSeed).shuffle(
                    tokensWithout
                )

        # Two separate graph windows
        figClassic, axClassic = plt.subplots(figsize=(9, 6))
        figBinned, axBinned = plt.subplots(figsize=(9, 6))

        # Analyze a real network and optionally generate HK
        # Analyze a real network and optionally generate HK
        def analyzeHK(tokens, title, color, simulateHK=False):
            if not tokens:
                self.logMessage(f'\n--- {title} ---')
                self.logMessage('No tokens to analyze.')
                return

            # Create real word-adjacency network
            G = nx.Graph()
            # Add unique tokens as graph nodes
            G.add_nodes_from(tokens)

            # Connect consecutive distinct tokens
            for a, b in zip(tokens, tokens[1:]):
                if a != b:
                    G.add_edge(a, b)

            # Record the number of real-network nodes
            N = G.number_of_nodes()
            E = G.number_of_edges()

            # Collect real-network degrees
            realDegrees = [
                degree for _, degree in G.degree()
            ]

            # Calculate real-network clustering and density
            realClustering = nx.average_clustering(G)
            realDensity = nx.density(G)

            # Power-law fit from unbinned degrees
            realPowerLaw = self.calculatePowerLawStatistics(
                realDegrees
            )

            # Log real network statistics
            self.logMessage(f'\n========== {title} ==========')
            self.logMessage('\n--- REAL NETWORK ---')
            self.logMessage(f'Number of nodes: {N}')
            self.logMessage(f'Number of edges: {E}')
            self.logMessage(
                f'Average degree: {sum(realDegrees) / N:.5f}'
            )
            self.logMessage(
                f'Maximum degree: {max(realDegrees)}'
            )
            self.logMessage(
                f'Network density: {realDensity:.5f}'
            )
            self.logMessage(
                f'Average clustering: {realClustering:.5f}'
            )

            self.logMessage('\n--- REAL POWER-LAW FIT ---')

            if realPowerLaw:
                self.logMessage(
                    f"Estimated gamma: "
                    f"{realPowerLaw['gamma']:.5f}"
                )
                self.logMessage(
                    f"Minimum degree kMin: "
                    f"{realPowerLaw['kMin']}"
                )
                self.logMessage(
                    f"Tail nodes: "
                    f"{realPowerLaw['tailSize']}"
                )
                self.logMessage(
                    f"KS distance: "
                    f"{realPowerLaw['ks']:.5f}"
                )
            else:
                self.logMessage(
                    'Not enough data for power-law estimation.'
                )

            # Classic real degree distribution
            # Count real nodes at each degree
            realUnique, realCounts = np.unique(
                realDegrees,
                return_counts=True
            )

            axClassic.loglog(
                realUnique[realUnique > 0],
                realCounts[realUnique > 0],
                'o',
                color=color,
                markersize=4,
                label=f'Real - {title}'
            )

            # Log-binned real degree distribution
            # Calculate log-binned real degree density
            binnedDegrees, binnedDensity = (
                self.logBinDegrees(
                    realDegrees,
                    binsPerDecade=3
                )
            )

            axBinned.loglog(
                binnedDegrees,
                binnedDensity,
                'o-',
                color=color,
                markersize=2,
                label=f'Real - {title}'
            )

            # Real fitted power-law tail
            if realPowerLaw:
                kMin = realPowerLaw['kMin']
                gamma = realPowerLaw['gamma']

                # Generate x coordinates for the fitted tail
                fitX = np.logspace(
                    np.log10(kMin),
                    np.log10(max(realDegrees)),
                    100
                )

                # Select bins belonging to the fitted tail
                tailMask = (
                    (binnedDegrees >= kMin) &
                    (binnedDensity > 0)
                )

                if np.any(tailMask):
                    firstIndex = np.where(tailMask)[0][0]

                    referenceX = binnedDegrees[firstIndex]
                    referenceY = binnedDensity[firstIndex]

                    # Scale the fitted line to the first eligible bin
                    fitY = referenceY * (
                        fitX / referenceX
                    ) ** (-gamma)

                    axBinned.loglog(
                        fitX,
                        fitY,
                        '--',
                        color=color,
                        linewidth=1.5,
                        label=(
                            f"Real Power Law "
                            f"(gamma={gamma:.3f}, kMin={kMin})"
                        )
                    )

            # Generate Holme-Kim model only once
            if simulateHK:
                if N < 3:
                    self.logMessage(
                        'Not enough nodes for Holme-Kim model.'
                    )
                    return

                # Match real average degree approximately
                averageDegree = 2 * E / N
                m = round(averageDegree / 2)
                # Keep m within the valid range
                m = max(1, min(m, N - 1))

                # Generate Holme-Kim network
                HK = nx.powerlaw_cluster_graph(
                    N,
                    m,
                    p,
                    seed=self.shuffleSeed
                )

                # Collect generated HK degrees
                hkDegrees = [
                    degree for _, degree in HK.degree()
                ]

                # Calculate HK clustering and density
                hkClustering = nx.average_clustering(HK)
                hkDensity = nx.density(HK)

                # Power-law fit from unbinned degrees
                hkPowerLaw = self.calculatePowerLawStatistics(
                    hkDegrees
                )

                # Log Holme-Kim statistics
                self.logMessage('\n--- HOLME-KIM MODEL ---')
                self.logMessage(
                    f'Number of nodes: {HK.number_of_nodes()}'
                )
                self.logMessage(
                    f'Number of edges: {HK.number_of_edges()}'
                )
                self.logMessage(f'Parameter m: {m}')
                self.logMessage(
                    f'Triad formation probability p: {p}'
                )
                self.logMessage(
                    f'Average degree: '
                    f'{sum(hkDegrees) / N:.5f}'
                )
                self.logMessage(
                    f'Maximum degree: {max(hkDegrees)}'
                )
                self.logMessage(
                    f'Network density: {hkDensity:.5f}'
                )
                self.logMessage(
                    f'Average clustering: {hkClustering:.5f}'
                )

                self.logMessage('\n--- HK POWER-LAW FIT ---')
                self.logMessage(
                    'Theoretical asymptotic gamma: 3.00000'
                )

                if hkPowerLaw:
                    self.logMessage(
                        f"Estimated gamma: "
                        f"{hkPowerLaw['gamma']:.5f}"
                    )
                    self.logMessage(
                        f"Minimum degree kMin: "
                        f"{hkPowerLaw['kMin']}"
                    )
                    self.logMessage(
                        f"Tail nodes: "
                        f"{hkPowerLaw['tailSize']}"
                    )
                    self.logMessage(
                        f"KS distance: "
                        f"{hkPowerLaw['ks']:.5f}"
                    )
                else:
                    self.logMessage(
                        'Not enough data for power-law estimation.'
                    )

                # Classic HK degree distribution
                # Count HK nodes at each degree
                hkUnique, hkCounts = np.unique(
                    hkDegrees,
                    return_counts=True
                )

                axClassic.loglog(
                    hkUnique[hkUnique > 0],
                    hkCounts[hkUnique > 0],
                    'o',
                    color='red',
                    markersize=4,
                    label='Holme-Kim Model'
                )

                # Log-binned HK degree distribution
                # Calculate log-binned HK degree density
                hkBinnedDegrees, hkBinnedDensity = (
                    self.logBinDegrees(
                        hkDegrees,
                        binsPerDecade=3
                    )
                )

                axBinned.loglog(
                    hkBinnedDegrees,
                    hkBinnedDensity,
                    'o-',
                    color='red',
                    markersize=2,
                    label='Holme-Kim Model'
                )

                # HK fitted power-law tail
                if hkPowerLaw:
                    kMin = hkPowerLaw['kMin']
                    gamma = hkPowerLaw['gamma']

                    # Generate x coordinates for the fitted tail
                    fitX = np.logspace(
                        np.log10(kMin),
                        np.log10(max(hkDegrees)),
                        100
                    )

                    # Select bins belonging to the fitted tail
                    tailMask = (
                        (hkBinnedDegrees >= kMin) &
                        (hkBinnedDensity > 0)
                    )

                    if np.any(tailMask):
                        firstIndex = np.where(tailMask)[0][0]

                        referenceX = hkBinnedDegrees[firstIndex]
                        referenceY = hkBinnedDensity[firstIndex]

                        # Scale the fitted line to the first eligible bin
                        fitY = referenceY * (
                            fitX / referenceX
                        ) ** (-gamma)

                        axBinned.loglog(
                            fitX,
                            fitY,
                            '--',
                            color='red',
                            linewidth=1.5,
                            label=(
                                f"HK Power Law "
                                f"(gamma={gamma:.3f}, kMin={kMin})"
                            )
                        )

        # Analyze according to punctuation checkbox
        if comparePunctuation:
            nodesWith = len(set(tokensWith))
            nodesWithout = len(set(tokensWithout))

            # Choose the larger real network for HK simulation
            simulateWith = nodesWith >= nodesWithout

            analyzeHK(
                tokensWith,
                'With Punctuation',
                'green',
                simulateHK=simulateWith
            )

            analyzeHK(
                tokensWithout,
                'Without Punctuation',
                'blue',
                simulateHK=not simulateWith
            )

        else:
            analyzeHK(
                tokensWith,
                'Real Text',
                'green' if self.selectedPunctuation else 'blue',
                simulateHK=True
            )

        # Classic graph settings
        axClassic.set_xlabel('Degree (k)')
        axClassic.set_ylabel('Number of Nodes')
        axClassic.set_title(
            'Degree Distribution: Real Network vs Holme-Kim'
        )
        axClassic.legend()
        axClassic.grid(False)
        figClassic.tight_layout()

        # Log-binned graph settings
        axBinned.set_xlabel('Degree (k)')
        axBinned.set_ylabel('Probability Density P(k)')
        axBinned.set_title(
            'Log-Binned Degree Distribution: '
            'Real Network vs Holme-Kim'
        )
        axBinned.legend()
        axBinned.grid(False)
        figBinned.tight_layout()

        plt.show()

        self.logMessage(
            '\nHolme-Kim analysis completed.\n'
        )

    
    # GROWTH GAMMA ANALYSIS
    def growthGammaAnalysis(self, event):
        self.logMessage('\nGrowth Gamma Analysis...')

        # Check whether punctuation comparison is enabled
        comparePunctuation = (
            self.comparePunctuationCheckbox.GetValue()
        )

        # Read and parse the input text
        text = self.readTextFile(self.inputTextFile)
        sentences = self.parseTextWithUdpipe(text)

        # Shuffle sentence order if selected
        sentences = self.applySentenceShuffle(sentences)

        # Create token sequence with selected punctuation
        tokensWith = self.filterUdpipeTokens(
            sentences, self.selectedPunctuation
        )

        # Initialize token sequence without punctuation
        tokensWithout = None

        # Create second token sequence if comparison is enabled
        if comparePunctuation:
            tokensWithout = self.filterUdpipeTokens(
                sentences, {}
            )

        # Shuffle words using the selected random seed
        if self.selectedShuffle == 'Words':
            random.Random(self.shuffleSeed).shuffle(tokensWith)

            if comparePunctuation:
                random.Random(self.shuffleSeed).shuffle(
                    tokensWithout
                )

        # Set the number of growth stages
        numberOfSteps = 100

        # Analyze how the power-law exponent changes during network growth
        def analyzeGrowth(tokens, title, color):

            # Skip analysis if the token sequence is empty
            if not tokens:
                self.logMessage(f'\n--- {title} ---')
                self.logMessage('No tokens to analyze.')
                return

            # Log the analyzed network title
            self.logMessage(f'\n========== {title} ==========')

            # Count the total number of tokens
            totalTokens = len(tokens)

            # Initialize an empty word-adjacency network
            G = nx.Graph()

            # Track the previous token and processing progress
            previousToken = None
            processedTokens = 0

            # Store network sizes and estimated gamma values
            numberOfNodes = []
            gammas = []

            # Calculate evenly spaced token checkpoints
            checkpoints = sorted(set(
                max(1, round(totalTokens * i / numberOfSteps))
                for i in range(1, numberOfSteps + 1)
            ))

            # Process the token sequence incrementally
            for checkpoint in checkpoints:

                # Add tokens between the previous and current checkpoint
                for token in tokens[processedTokens:checkpoint]:

                    # Add the token as a node if it does not already exist
                    G.add_node(token)

                    # Connect consecutive tokens without creating self-loops
                    if previousToken is not None:
                        if previousToken != token:
                            G.add_edge(previousToken, token)

                    # Remember the current token for the next connection
                    previousToken = token

                # Update the number of processed tokens
                processedTokens = checkpoint

                # Get the degrees of all nodes in the current network
                degrees = [
                    degree for _, degree in G.degree()
                ]

                # Estimate the power-law exponent from unbinned degrees
                powerLaw = self.calculatePowerLawStatistics(
                    degrees
                )

                # Store results only when power-law estimation succeeds
                if powerLaw:

                    # Record the current number of unique nodes
                    numberOfNodes.append(
                        G.number_of_nodes()
                    )

                    # Record the estimated power-law exponent
                    gammas.append(
                        powerLaw['gamma']
                    )

            # Log the total number of processed tokens
            self.logMessage(
                f'Total tokens: {totalTokens}'
            )

            # Log the final number of nodes
            self.logMessage(
                f'Final number of nodes: {G.number_of_nodes()}'
            )

            # Log the final number of edges
            self.logMessage(
                f'Final number of edges: {G.number_of_edges()}'
            )

            # Log the number of growth checkpoints
            self.logMessage(
                f'Growth stages: {len(checkpoints)}'
            )

            # Log the number of successful gamma estimates
            self.logMessage(
                f'Valid gamma estimates: {len(gammas)}'
            )

            # Display gamma statistics if valid estimates exist
            if gammas:

                # Log the first valid gamma estimate
                self.logMessage(
                    f'Initial estimated gamma: {gammas[0]:.5f}'
                )

                # Log the last valid gamma estimate
                self.logMessage(
                    f'Final estimated gamma: {gammas[-1]:.5f}'
                )

                # Log the minimum estimated gamma
                self.logMessage(
                    f'Minimum estimated gamma: {min(gammas):.5f}'
                )

                # Log the maximum estimated gamma
                self.logMessage(
                    f'Maximum estimated gamma: {max(gammas):.5f}'
                )

                # Plot gamma evolution against the number of nodes
                plt.plot(
                    numberOfNodes,
                    gammas,
                    'o-',
                    color=color,
                    markersize=3,
                    linewidth=1.5,
                    label=title
                )

            else:
                # Report when no valid power-law estimates are available
                self.logMessage(
                    'Not enough data for power-law estimation.'
                )

        # Create a figure for the gamma evolution plot
        plt.figure(figsize=(9, 6))

        # Analyze both token sequences if comparison is enabled
        if comparePunctuation:

            # Analyze network growth with punctuation
            analyzeGrowth(
                tokensWith,
                'With Punctuation',
                'green'
            )

            # Analyze network growth without punctuation
            analyzeGrowth(
                tokensWithout,
                'Without Punctuation',
                'blue'
            )

        else:
            # Analyze one network using the selected punctuation settings
            analyzeGrowth(
                tokensWith,
                'Real Text',
                'green' if self.selectedPunctuation else 'blue'
            )

        # Set the horizontal axis label
        plt.xlabel('Number of Nodes')

        # Set the vertical axis label
        plt.ylabel('Estimated Power-Law Exponent (gamma)')

        # Set the graph title
        plt.title('Growth Gamma Analysis')

        # Get the current plot axes
        ax = plt.gca()

        # Get the current vertical axis limits
        yMin, yMax = ax.get_ylim()

        # Calculate the vertical axis range
        yRange = yMax - yMin

        # Add vertical padding around the gamma values
        ax.set_ylim(
            yMin - yRange * 0.3,
            yMax + yRange * 0.3
        )

        # Display the plot legend
        plt.legend()

        # Display a dashed grid
        plt.grid(True, linestyle='--', alpha=0.7)

        # Adjust spacing and display the graph
        plt.tight_layout()
        plt.show()

        # Log completion message
        self.logMessage(
            '\nGrowth gamma analysis completed.\n'
        )


    # Exit application
    def exitApp(self, event):
        self.logMessage('Exiting application...')
        wx.CallLater(500, self.Close)


if __name__ == '__main__':
    app = wx.App(False)
    frame = App()
    frame.Show()
    app.MainLoop()
