import wx
import matplotlib.pyplot as plt
import networkx as nx
import re
from collections import Counter
import numpy as np
from scipy.stats import linregress

class App(wx.Frame):

    def __init__(self):
        super().__init__(parent=None, title='Punctuation Marks Analysis', size=(700, 700), \
                         style=wx.DEFAULT_FRAME_STYLE & ~wx.RESIZE_BORDER)
        
        self.inputTextFile = None
        self.languageOptions = ['', 'English', 'German']
        self.selectedLanguage = ''
        self.punctuation = {'period': '.',
                            'comma': ',',
                            'exclamation': '!',
                            'question': '?', 
                            'semicolon': ';',
                            'colon': ':', 
                            'quotation': '"..."',
                            'apostrophe': '\'',
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

        self.regexDictEng = {'wordsNumbers': r'[a-zA-Z0-9]+',
                            'ellipsis': r'\.{3}',
                            'underscore': r'_',
                            'period': r'\.',
                            'comma': r',',
                            'exclamation': r'!',
                            'question': r'\?',
                            'semicolon': r';',
                            'colon': r':',
                            'parenthesis': r'[()]',
                            'brackets': r'[\[\]]',
                            'braces': r'[{}]',
                            'quotation': r'["“”]',
                            'apostrophe': r'[\'’‘]',
                            'slash': r'/',
                            'hyphen': r'-',
                            'enDash': r'–',
                            'emDash': r'—'
                            }
        self.regexDictGer = self.regexDictEng
        self.regexDictGer['wordsNumbers'] = r'[a-zA-ZäöüÄÖÜß0-9]+'
        self.regexDictGer['quotation'] = r'["„“«»]'
        self.regexDictGer['apostrophe'] = r'[\'’‘]'
        self.initGUI()


    # Initialize the GUI components 
    def initGUI(self):
        panel = wx.Panel(self)
        vbox = wx.BoxSizer(wx.VERTICAL)
        
        # File Selection
        self.labelFileSelect = wx.StaticText(panel, label='Select a text file:')
        self.labelFileSelectPath = wx.TextCtrl(panel, style=wx.TE_READONLY)
        self.labelFileSelectPath.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.labelFileSelectPath.SetForegroundColour(wx.Colour(0, 0, 0))
        self.labelFileSelectPath.SetValue('No file selected.')
        self.buttonSelectFile = wx.Button(panel, label='Select File')
        self.buttonSelectFile.Bind(wx.EVT_BUTTON, self.selectInputFile)
        
        fileSelectLayout = wx.BoxSizer(wx.HORIZONTAL)
        fileSelectLayout.Add(self.labelFileSelectPath, proportion=1, flag=wx.EXPAND | wx.ALIGN_LEFT)
        fileSelectLayout.Add(self.buttonSelectFile, flag=wx.LEFT, border=10)
        
        # Language Selection
        self.languageDropdown = wx.Choice(panel, choices=self.languageOptions)
        self.languageDropdown.Bind(wx.EVT_CHOICE, self.languageChange)
        languageLayout = wx.BoxSizer(wx.HORIZONTAL)
        languageLayout.Add(wx.StaticText(panel, label='Select Language:'), flag=wx.ALIGN_LEFT)
        languageLayout.Add(self.languageDropdown, flag=wx.LEFT, border=10)

        # Punctuation Selection
        self.selectAllCheckbox = wx.CheckBox(panel, label='Select All')
        self.selectAllCheckbox.Bind(wx.EVT_CHECKBOX, self.selectAllPunctuation)
        self.punctuationLabel = wx.StaticText(panel, label='Select Punctuation:')
        punctuationLayout = wx.GridSizer(5, 5, 5, 10)
        self.punctuationCheckboxes = {}
        for label, symbol in self.punctuation.items():
            checkbox = wx.CheckBox(panel, label=f'{label}  {symbol}')
            checkbox.Bind(wx.EVT_CHECKBOX, self.updateSelectedPunctuation)
            punctuationLayout.Add(checkbox, flag=wx.EXPAND)
            self.punctuationCheckboxes[label] = checkbox

        # Log window
        self.logWindow = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY)
        self.logWindow.SetBackgroundColour(wx.Colour(240, 240, 240))

        # Buttons
        self.exitButton = wx.Button(panel, label='Exit')
        self.exitButton.Bind(wx.EVT_BUTTON, self.exitApp)

        self.saveNetworkButton = wx.Button(panel, label='Save Network')
        self.saveNetworkButton.Bind(wx.EVT_BUTTON, self.saveNetwork)
        self.saveNetworkButton.Enable(False)

        self.distributionAnalysisButton = wx.Button(panel, label='Power-Law Analysis')
        self.distributionAnalysisButton.Bind(wx.EVT_BUTTON, self.distributionAnalysis)
        self.distributionAnalysisButton.Enable(False)

        self.compareDistributionsButton = wx.Button(panel, label='Compare Distributions')
        self.compareDistributionsButton.Bind(wx.EVT_BUTTON, self.compareDistributions)
        self.compareDistributionsButton.Enable(False)

        self.outputValues = wx.Button(panel, label='Calculate Analysis')
        self.outputValues.Bind(wx.EVT_BUTTON, self.logOutputValues)
        self.outputValues.Enable(False)

        self.growthGamma = wx.Button(panel, label='Growth Gamma')
        self.growthGamma.Bind(wx.EVT_BUTTON, self.growthGammaPlot)
        self.growthGamma.Enable(False)

        self.growthComparison = wx.Button(panel, label='Growth Comparison')
        self.growthComparison.Bind(wx.EVT_BUTTON, self.growthComparisonPlot)
        self.growthComparison.Enable(False)

        # Buttons Layout with 3 rows
        buttonLayout = wx.BoxSizer(wx.VERTICAL)

        rowOne = wx.BoxSizer(wx.HORIZONTAL)
        rowOne.Add(self.saveNetworkButton, flag=wx.LEFT, border=10)
        rowOne.Add(self.distributionAnalysisButton, flag=wx.LEFT, border=10)
        rowOne.Add(self.compareDistributionsButton, flag=wx.LEFT, border=10)

        rowTwo = wx.BoxSizer(wx.HORIZONTAL)
        rowTwo.Add(self.growthGamma, flag=wx.LEFT, border=10)
        rowTwo.Add(self.growthComparison, flag=wx.LEFT, border=10)
        rowTwo.Add(self.outputValues, flag=wx.LEFT, border=10)

        rowThree = wx.BoxSizer(wx.HORIZONTAL)
        rowThree.Add(self.exitButton, flag=wx.LEFT, border=10)

        buttonLayout.Add(rowOne, flag=wx.ALIGN_CENTER | wx.TOP, border=10)
        buttonLayout.Add(rowTwo, flag=wx.ALIGN_CENTER | wx.TOP, border=10)
        buttonLayout.Add(rowThree, flag=wx.ALIGN_CENTER | wx.TOP, border=10)

        # Layout
        vbox.Add(self.labelFileSelect, flag=wx.EXPAND | wx.LEFT | wx.TOP, border=10)
        vbox.Add(fileSelectLayout, flag=wx.EXPAND | wx.LEFT | wx.TOP, border=10)
        vbox.Add(languageLayout, flag=wx.EXPAND | wx.LEFT | wx.TOP, border=10)
        vbox.Add(self.punctuationLabel, flag=wx.LEFT | wx.TOP, border=10)
        vbox.Add(self.selectAllCheckbox, flag=wx.LEFT | wx.TOP, border=10)
        vbox.Add(punctuationLayout, flag=wx.LEFT | wx.TOP, border=10)
        vbox.Add(self.logWindow, proportion=1, flag=wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, border=10)        
        vbox.Add(buttonLayout, flag=wx.ALIGN_CENTER | wx.TOP, border=10)

        panel.SetSizer(vbox)
        self.Centre()


    # Input file selection event handler and validate data
    def selectInputFile(self, event):
        fileDialog = wx.FileDialog(self, 'Open Text File', wildcard='Text files (*.txt)|*.txt', style=wx.FD_OPEN)
        if fileDialog.ShowModal() == wx.ID_OK:
            self.inputTextFile = fileDialog.GetPath()
            self.labelFileSelectPath.SetValue(self.inputTextFile)
            self.validateInputData()

    
    # Language selection event handler and validate data
    def languageChange(self, event):
        self.selectedLanguage = self.languageOptions[self.languageDropdown.GetSelection()]
        self.validateInputData()


    # Select or deselect all punctuation checkboxes and validate data
    def selectAllPunctuation(self, event):
        checked = event.IsChecked()
        self.selectedPunctuation.clear()
        for label, checkbox in self.punctuationCheckboxes.items():
            checkbox.SetValue(checked)
            if checked:
                self.selectedPunctuation[label] = self.punctuation[label]
        self.validateInputData()


    # Update selected punctuation based on checkbox state and validate data
    def updateSelectedPunctuation(self, event):
        sender = event.GetEventObject()
        punctuationCheckboxLabel = sender.GetLabel().split()[0]
        if sender.IsChecked():
            self.selectedPunctuation[punctuationCheckboxLabel] = self.punctuation[punctuationCheckboxLabel]
        else:
            if punctuationCheckboxLabel in self.selectedPunctuation:
                del self.selectedPunctuation[punctuationCheckboxLabel]
        self.selectAllCheckbox.SetValue(False) if any([not checkbox.IsChecked() \
                                    for checkbox in self.punctuationCheckboxes.values()]) else self.selectAllCheckbox.SetValue(True)
        self.validateInputData()


    # Log messages to the log window
    def logMessage(self, message):
        self.logWindow.AppendText(f'{message}\n')

    # Enable or disable buttons based on input data
    def validateInputData(self):
        self.saveNetworkButton.Enable(bool(self.inputTextFile and self.selectedLanguage))
        self.distributionAnalysisButton.Enable(bool(self.inputTextFile and self.selectedLanguage))
        self.compareDistributionsButton.Enable(bool(self.inputTextFile and self.selectedLanguage))
        self.outputValues.Enable(bool(self.inputTextFile and self.selectedLanguage))
        self.growthGamma.Enable(bool(self.inputTextFile and self.selectedLanguage))
        self.growthComparison.Enable(bool(self.inputTextFile and self.selectedLanguage))


    # Read text file and return its content
    def readTextFile(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
        

    # Tokenize text file and return a list
    def processTextFile(self, selectedPunctuation={}):
        if self.selectedLanguage == 'English':
            regexDict = self.regexDictEng
        if self.selectedLanguage == 'German':
            regexDict = self.regexDictGer
        if not selectedPunctuation:
            regexPattern = regexDict['wordsNumbers']
        else:
            usePunctuation = [regexDict[key] for key in selectedPunctuation.keys()]
            regexPattern = '|'.join([regexDict['wordsNumbers']] + usePunctuation)
        data = re.findall(regexPattern, self.readTextFile(self.inputTextFile))
        if not data:
            return []
        return [word.lower() for word in data]


    # Create dict, word adjacency network and count word occurrences
    def createGraphData(self, data):
        if self.selectedLanguage == 'English':
            regexDict = self.regexDictEng
        if self.selectedLanguage == 'German':
            regexDict = self.regexDictGer
        graphDataDict = {}
        nodeCounter = Counter()
        previousWord = None
        for element in data:
            if element in regexDict['quotation']:
                element = '"'
            if element in regexDict['apostrophe']:
                element = '\''
            if element not in graphDataDict:
                graphDataDict[element] = []
            if previousWord is not None:
                graphDataDict[previousWord].append(element)
                graphDataDict[element].append(previousWord)
            previousWord = element
            nodeCounter[element] += 1
        return graphDataDict, nodeCounter


    # Log input data info and data analysis values and validate data
    def logOutputValues(self, event):
        self.logMessage(self.collectInputDataInfo())
        self.logMessage(self.calculateValues(*self.createGraphData(self.processTextFile(self.selectedPunctuation))))
        self.logMessage('\n')


    # Save the network as a graphml file
    def saveNetwork(self, event):
        self.logMessage(self.collectInputDataInfo())
        self.logMessage('Saving network...\n')

        N = nx.Graph(self.createGraphData(self.processTextFile(self.selectedPunctuation))[0])
        if self.selectedPunctuation:
            graphName = f'{self.labelFileSelectPath.GetValue().split("/")[-1].split(".")[0]}_graphYesPunct.graphml'
            self.logMessage(f'Saving graph to {graphName}')
            nx.write_graphml(N, f'{graphName}')
        else:
            graphName = f'{self.labelFileSelectPath.GetValue().split("/")[-1].split(".")[0]}_graphNoPunct.graphml'
            self.logMessage(f'Saving graph to {graphName}')
            nx.write_graphml(N, graphName)


    # Distribution analysis and visualization, comparison to Dorogovtsev-Goltsev-Mendes model
    def distributionAnalysis(self, event):
        self.logMessage(self.collectInputDataInfo())
        self.logMessage('Visualizing Analysis...')

        graphData, occurrenceData = self.createGraphData(self.processTextFile(self.selectedPunctuation))
        # create graph from graphData
        G = nx.Graph()
        for node, neighbors in graphData.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
        degrees = [G.degree(n) for n in G.nodes()]
        unique, counts = np.unique(degrees, return_counts=True)

        # create Dorogovtsev-Goltsev-Mendes model, with same number of nodes as my data
        DGM_G = nx.dorogovtsev_goltsev_mendes_graph(10)
        dgmDegrees = [DGM_G.degree(n) for n in DGM_G.nodes()]
        dgmUnique, dgmCounts = np.unique(dgmDegrees, return_counts=True)

        # my data, log binning, selecting only the longest decreasing slice
        binCenters, binValues = self.calculateLogBin(np.array(degrees), 20)
        start, end = self.longestDecreasingSlice(binValues)
        binValues = binValues[start: end+1]
        binCenters = binCenters[start: end+1]

        # simulated dmg model, log binning, selecting only the longest decreasing slice
        dgmBinCenters, dgmBinValues = self.calculateLogBin(np.array(dgmDegrees), 20)
        # dgmStart, dgmEnd = self.longestDecreasingSlice(dgmBinValues)
        # dgmBinValues = dgmBinValues[dgmStart: dgmEnd+1]
        # dgmBinCenters = dgmBinCenters[dgmStart: dgmEnd+1]

        # calculate slopes
        slope = self.calculateLogLogSlope(binCenters, binValues)

        # calculate Zipf's law on my data and slope
        wordFrequencies = sorted(occurrenceData.values(), reverse=True)
        ranks = np.arange(1, len(wordFrequencies) + 1)
        zipfSlope = self.calculateLogLogSlope(ranks, wordFrequencies)

        # Ensure both distributions are of the same length by padding with zeros
        wordDegrees = degrees.copy()
        dmDegrees = dgmDegrees.copy()
        maxLen = max(len(wordDegrees), len(dmDegrees))
        wordDegrees += [0] * (maxLen - len(wordDegrees))
        dmDegrees += [0] * (maxLen - len(dmDegrees))

        wordDegreesNormalized = np.array(wordDegrees) / max(wordDegrees)
        dmDegreesNormalized = np.array(dmDegrees) / max(dmDegrees)
        mseNormalized = np.mean((wordDegreesNormalized - dmDegreesNormalized)**2)
        self.logMessage(f'Normalized Degree Distribution MSE: {mseNormalized:.5f}')

        # normalize dgm raw degree distribution
        scalingFactor = max(counts) / max(dgmCounts)
        dgmCountsScaled = dgmCounts * scalingFactor

        plt.figure(figsize=(8, 6))
        plt.loglog(unique, counts, 'bo', markersize=4, label='Vytvorená sieť')
        plt.loglog(dgmUnique, dgmCountsScaled, 'go', markersize=4, label=f'DGM Model')
        plt.xlabel('Stupne')
        plt.ylabel('Frekvencia')
        plt.title('Názov... \n Rozdelenie stupňov vrcholov')
        plt.legend()
        
        plt.figure(figsize=(8, 6))
        plt.loglog(binCenters, binValues, 'x', color='black', alpha=0.9)
        plt.loglog(binCenters, binValues, '-', color='blue', alpha=0.8, label=f'Vytvorená sieť, gamma={slope:.5f}')
        plt.loglog(dgmBinCenters, dgmBinValues, 'x', color='black', alpha=0.9)
        plt.loglog(dgmBinCenters, dgmBinValues, '-', color='green', alpha=0.8, label=f'DGM Model')
        plt.xlabel('Stupne')
        plt.ylabel('Frekvencia')
        plt.title('Názov... \n Rozdelenie stupňov vrcholov s logaritmickým zoskupovaním')
        plt.legend()

        plt.figure(figsize=(8, 6))
        plt.loglog(ranks, wordFrequencies, 'x', color='black', alpha=0.9)
        plt.loglog(ranks, wordFrequencies, '-', color='black', alpha=0.8, label=f'Zipfov zákon {zipfSlope:.5f}')
        plt.xlabel('Rank')
        plt.ylabel('Frekvencia')
        plt.title('Názov... \n Zipfov zákon')
        plt.legend()

        plt.show()
        self.logMessage('\n')


    # Compare distributions of words and words + punctuation
    def compareDistributions(self, event):
        self.logMessage(self.collectInputDataInfo())
        graphDataOnlyWords, occurrenceDataOnlyWords = self.createGraphData(self.processTextFile())
        graphDataCombined, occurrenceDataCombined = self.createGraphData(self.processTextFile(self.selectedPunctuation))
        self.logMessage('Plotting degree distribution comparison histogram...\n')

        # graph without punctuation
        G = nx.Graph()
        for node, neighbors in graphDataOnlyWords.items():
            for neighbor in neighbors:
                G.add_edge(node, neighbor)
        gDegrees = [G.degree(n) for n in G.nodes()]
        gUnique, gCounts = np.unique(gDegrees, return_counts=True)

        # graph with punctuation
        M = nx.Graph()
        for node, neighbors in graphDataCombined.items():
            for neighbor in neighbors:
                M.add_edge(node, neighbor)
        mDegrees = [M.degree(n) for n in M.nodes()]
        mUnique, mCounts = np.unique(mDegrees, return_counts=True)

        # create Dorogovtsev-Goltsev-Mendes model, with same number of nodes as my data
        n = 1
        DGM_G = nx.dorogovtsev_goltsev_mendes_graph(n)
        while DGM_G.number_of_nodes() < len(gDegrees):
            n += 1
            DGM_G = nx.dorogovtsev_goltsev_mendes_graph(n)
        dgmDegrees = [DGM_G.degree(n) for n in DGM_G.nodes()]
        dgmUnique, dgmCounts = np.unique(dgmDegrees, return_counts=True)

        # log binning
        gBinCenters, gBinValues = self.calculateLogBin(np.array(gDegrees), 20)
        mBinCenters, mBinValues = self.calculateLogBin(np.array(mDegrees), 20)
        dgmBinCenters, dgmBinValues = self.calculateLogBin(np.array(dgmDegrees), 20)

        # selecting longest decreasing slice, same range
        gStart, gEnd = self.longestDecreasingSlice(gBinValues)
        mStart, mEnd = self.longestDecreasingSlice(mBinValues)
        gBinValues = gBinValues[gStart: gEnd]
        gBinCenters = gBinCenters[gStart: gEnd]
        mBinValues = mBinValues[mStart: mEnd]
        mBinCenters = mBinCenters[mStart: mEnd]
        dgmBinValues = dgmBinValues[:]
        dgmBinCenters = dgmBinCenters[:]

        # calculate slopes
        gSlope = self.calculateLogLogSlope(gBinCenters, gBinValues)
        mSlope = self.calculateLogLogSlope(mBinCenters, mBinValues)

        # normalize dgm raw degree distribution
        scalingFactor = max(gCounts) / max(dgmCounts)
        dgmCountsScaled = dgmCounts * scalingFactor

        plt.figure(figsize=(8, 6))
        plt.loglog(gUnique, gCounts, 'bo', markersize=4, label='Slová')
        plt.loglog(mUnique, mCounts, 'ro', markersize=4, label='Slová + Interpunkcia')
        plt.loglog(dgmUnique, dgmCountsScaled, 'go', markersize=4, label='DGM Model')
        plt.xlabel('Stupne')
        plt.ylabel('Frekvencia')
        plt.title('Názov... \n Rozdelenie stupňov vrcholov')
        plt.legend()

        plt.figure(figsize=(8, 6))
        plt.loglog(gBinCenters, gBinValues, '-', color='red', alpha=0.8, label=f'Slová, gamma={gSlope:.5f}')
        plt.loglog(mBinCenters, mBinValues, '-', color='blue', alpha=0.8, label=f'Slová + Interpunkcia, gamma={mSlope:.5f}')
        plt.loglog(dgmBinCenters, dgmBinValues, '-', color='green', alpha=0.8, label='DGM Model')
        plt.loglog(gBinCenters, gBinValues, 'x', alpha=0.9, color='black')
        plt.loglog(mBinCenters, mBinValues, 'x', alpha=0.9, color='black')
        plt.loglog(dgmBinCenters, dgmBinValues, 'x', alpha=0.9, color='black')
        plt.xlabel('Stupne')
        plt.ylabel('Frekvencia')
        plt.title('Názov... \n Rozdelenie stupňov vrcholov s logaritmickým zoskupovaním')
        plt.legend()

        plt.show()

    
    # Plot growth gamma analysis
    def growthGammaPlot(self, event):
        self.logMessage(self.collectInputDataInfo())
        self.logMessage('Plotting growth gamma...')

        tokens = self.processTextFile(self.selectedPunctuation)

        sliceSize = len(tokens) // 100
        slicedTokens = [tokens[:(i + 1) * sliceSize] for i in range(100)]
        if len(tokens) % 100 != 0:
            slicedTokens[-1] = tokens

        graphObjects = []
        for slice in slicedTokens:
            G = nx.Graph()
            for node, neighbors in self.createGraphData(slice)[0].items():
                for neighbor in neighbors:
                    G.add_edge(node, neighbor)
            graphObjects.append(G)
            
        gammas = []
        numberOfNodes = []
        for G in graphObjects:
            degrees = [deg for _, deg in G.degree()]
            binCenters, binValues = [], []
            if len(degrees) > 1:
                binCenters, binValues = self.calculateLogBin(np.array(degrees), 20)
            if len(binCenters) > 1 and len(binValues) > 1:
                gStart, gEnd = self.longestDecreasingSlice(binValues)
                binValues = binValues[gStart: gEnd]
                binCenters = binCenters[gStart: gEnd]
                slope = self.calculateLogLogSlope(binCenters, binValues)
                gammas.append(-slope)
            else:
                gammas.append(0)
            numberOfNodes.append(G.number_of_nodes())

        
        # # Save graph at index 72 as a GraphML file
        # graph72 = graphObjects[12]
        # graph72_name = f'{self.labelFileSelectPath.GetValue().split('/')[-1].split('.')[0]}_graph72.graphml'
        # nx.write_graphml(graph72, graph72_name)
        # self.logMessage(f'Graph at index 72 saved as {graph72_name}')

        # # Save the corresponding slice of tokens as a text file
        # slice72 = slicedTokens[12]
        # slice72_name = f'{self.labelFileSelectPath.GetValue().split('/')[-1].split('.')[0]}_selectionNoPunct.txt'
        # with open(slice72_name, 'w', encoding='utf-8') as f:
        #     f.write('\n'.join(slice72))
        # self.logMessage(f'Slice at index 72 saved as {slice72_name}')

        # for i in range(100):
        #     self.logMessage(f'Slice {i+1}: {len(slicedTokens[i])} words, {numberOfNodes[i]} nodes, gamma={gammas[i]:.5f}')


        plt.figure(figsize=(8, 6))
        plt.plot(numberOfNodes, gammas, marker='o', linestyle='-', color='black', alpha=0.8, label='s interpunkciou')
        plt.xlim(min(numberOfNodes), min(max(numberOfNodes)+500, 5000))
        plt.ylim(min(gammas)-0.5, max(gammas)+0.5)
        plt.xlabel('Počet vrcholov')
        plt.ylabel('Gamma (Exponent mocninového rozdelenia)')
        plt.title('Názov...')
        plt.grid(True)
        plt.legend()
        plt.show()


    # Plot growth comparison of network with and without punctuation
    def growthComparisonPlot(self, event):
        self.logMessage(self.collectInputDataInfo())
        self.logMessage('Plotting growth comparison...')

        tokensWithPunct = self.processTextFile(self.selectedPunctuation)
        tokensWithoutPunct = self.processTextFile()

        sliceWithPunct = len(tokensWithPunct) // 100
        sliceWithoutPunct = len(tokensWithoutPunct) // 100
        slicedTokensWithPunct = [tokensWithPunct[:(i + 1) * sliceWithPunct] for i in range(100)]
        slicedTokensWithoutPunct = [tokensWithoutPunct[:(i + 1) * sliceWithoutPunct] for i in range(100)]
        if len(tokensWithPunct) % 100 != 0:
            slicedTokensWithPunct[-1] = tokensWithPunct
        if len(tokensWithoutPunct) % 100 != 0:
            slicedTokensWithoutPunct[-1] = tokensWithoutPunct

        graphObjectsWithPunct = []
        for slice in slicedTokensWithPunct:
            G = nx.Graph()
            for node, neighbors in self.createGraphData(slice)[0].items():
                for neighbor in neighbors:
                    G.add_edge(node, neighbor)
            graphObjectsWithPunct.append(G)

        graphObjectsWithoutPunct = []
        for slice in slicedTokensWithoutPunct:
            G = nx.Graph()
            for node, neighbors in self.createGraphData(slice)[0].items():
                for neighbor in neighbors:
                    G.add_edge(node, neighbor)
            graphObjectsWithoutPunct.append(G)

        gammasWithPunct = []
        numberOfNodesWithPunct = []
        for G in graphObjectsWithPunct:
            degrees = [deg for _, deg in G.degree()]
            binCenters, binValues = [], []
            if len(degrees) > 1:
                binCenters, binValues = self.calculateLogBin(np.array(degrees), 20)
            if len(binCenters) > 1 and len(binValues) > 1:
                gStart, gEnd = self.longestDecreasingSlice(binValues)
                binValues = binValues[gStart: gEnd]
                binCenters = binCenters[gStart: gEnd]
                slope = self.calculateLogLogSlope(binCenters, binValues)
                gammasWithPunct.append(-slope)
            else:
                gammasWithPunct.append(0)
            numberOfNodesWithPunct.append(G.number_of_nodes())

        gammasWithoutPunct = []
        numberOfNodesWithoutPunct = []
        for G in graphObjectsWithoutPunct:
            degrees = [deg for _, deg in G.degree()]
            binCenters, binValues = [], []
            if len(degrees) > 1:
                binCenters, binValues = self.calculateLogBin(np.array(degrees), 20)
            if len(binCenters) > 1 and len(binValues) > 1:
                gStart, gEnd = self.longestDecreasingSlice(binValues)
                binValues = binValues[gStart: gEnd]
                binCenters = binCenters[gStart: gEnd]
                slope = self.calculateLogLogSlope(binCenters, binValues)
                gammasWithoutPunct.append(-slope)
            else:
                gammasWithoutPunct.append(0)
            numberOfNodesWithoutPunct.append(G.number_of_nodes())
        
        plt.figure(figsize=(8, 6))
        plt.plot(numberOfNodesWithPunct, gammasWithPunct, marker='o', linestyle='-', color='blue', alpha=0.8, label='s interpunkciou')
        plt.plot(numberOfNodesWithoutPunct, gammasWithoutPunct, marker='o', linestyle='-', color='red', alpha=0.8, label='bez interpunkcie')
        plt.xlim(min(min(numberOfNodesWithPunct), min(numberOfNodesWithoutPunct)), 
                 min(max(numberOfNodesWithPunct), max(numberOfNodesWithoutPunct), 5000))
        plt.ylim(min(min(gammasWithPunct), min(gammasWithoutPunct))-0.5,
                    max(max(gammasWithPunct), max(gammasWithoutPunct))+0.5)
        plt.xlabel('Počet vrcholov')
        plt.ylabel('Gamma (Exponent mocninového rozdelenia)')
        plt.title('Názov...')
        plt.grid(True)
        plt.legend()
        plt.show()
        

    # Calculate log binning for degree distribution
    def calculateLogBin(self, degrees, binCount):
        minDegree = max(1, min(degrees))
        maxDegree = degrees.max()
        bins = np.logspace(np.log10(minDegree), np.log10(maxDegree), num=binCount)
        hist, binEdges = np.histogram(degrees, bins=bins, density=True)
        binCenters = (binEdges[:-1] + binEdges[1:]) / 2
        nonzero = hist > 0
        return binCenters[nonzero], hist[nonzero]
    

    # Calculate log-log slope for the given x and y values
    def calculateLogLogSlope(self, x, y):
        logDegrees = np.log10(x)
        logCounts = np.log10(y)
        slope, intercept, r_value, p_value, std_err = linregress(logDegrees, logCounts)
        return slope
    

    # Calculate the longest decreasing slice of the data
    def longestDecreasingSlice(self, data):
        sliceStart, sliceEnd = 0, 0
        currentStart = 0
        for i in range(1, len(data)):
            if data[i] >= data[i-1]:
                currentStart = i
            elif i - currentStart > sliceEnd - sliceStart:
                sliceStart, sliceEnd = currentStart, i
        return sliceStart, sliceEnd


    # Calculate values for the graph and language analysis
    def calculateValues(self, graphData, occurrenceData):
        result = ['Grafová analýza...']
        G = nx.Graph(graphData)
        degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        degValues = [deg for _, deg in G.degree()]
        # correlationDegreeCloseness = np.corrcoef(list(nx.degree_centrality(G).values()), list(nx.closeness_centrality(G).values()))[0, 1]
        # correlationDegreeBetweenness = np.corrcoef(list(nx.degree_centrality(G).values()), list(nx.betweenness_centrality(G).values()))[0, 1]
        # correlationClosenessBetweenness = np.corrcoef(list(nx.closeness_centrality(G).values()), list(nx.betweenness_centrality(G).values()))[0, 1]
        
        result.append(f'Number of nodes: {G.number_of_nodes()}')
        result.append(f'Number of edges: {G.number_of_edges()}')
        result.append(f'Max degree: {degrees[0]}')
        result.append(f'Min degree: {degrees[-1]}')
        result.append(f'Average degree: {sum(degValues)/len(degValues):.5f}')
        result.append(f'Network density: {nx.density(G):.5f}')
        # result.append(f'Correlation (Degree vs Closeness): {correlationDegreeCloseness:.5f}')
        # result.append(f'Correlation (Degree vs Betweenness): {correlationDegreeBetweenness:.5f}')
        # result.append(f'Correlation (Closeness vs Betweenness): {correlationClosenessBetweenness:.5f}')
        result.append(f'Average clustering coefficient: {nx.average_clustering(G):.5f}')
        result.append(f'Average shortest path length: {nx.average_shortest_path_length(G):.5f}')
        result.append(f'Diameter: {nx.diameter(G)}')

        result.append('Jazyková analýza...')
        wordLengths = [len(word) for word in occurrenceData.keys()]
        processedSentences = self.processTextFile({'period': '.', 'exclamation': '!', 
                                              'question': '?', 'ellipsis': '...', 'apostrophe': '\'’‘'})
        i = 0
        while i < len(processedSentences):
            if processedSentences[i] in ['\'', '’', '‘'] and i > 0 and i < len(processedSentences)-1:
                processedSentences[i-1:i+2] = [''.join(processedSentences[i-1:i+2])]
            else:
                i += 1
        sentences = []
        subList = []
        sentencelengths = []
        for element in processedSentences:
            if element in ['.', '!', '?', '...'] and subList:
                sentences.append(subList)
                sentencelengths.append(len(subList))
                subList = []
            else:
                subList.append(element)
        data = self.processTextFile(self.selectedPunctuation)
        bigramCounter = Counter(zip(data[:-1], data[1:]))
        bigramFrequencies = [count for _, count in bigramCounter.items()]
        trigramCounter = Counter(zip(data[:-2], data[1:-1], data[2:]))
        trigramFrequencies = [count for _, count in trigramCounter.items()]

        punctuationCounts = Counter([char for char in data if char in self.punctuation.values()])
        result.append(f'Number of punctuation marks: {sum(punctuationCounts.values())}')
        result.append(f'Most common punctuation: {punctuationCounts.most_common(1)[0] if punctuationCounts else "None"}')
        result.append(f'Least common punctuation: {punctuationCounts.most_common()[-1] if punctuationCounts else "None"}')
        result.append(f'Number of words: {len(wordLengths)}')
        result.append(f'Max word length: {max(wordLengths)}')
        result.append(f'Min word length: {min(wordLengths)}')
        result.append(f'Average word length: {(sum(wordLengths)/len(wordLengths)):.5f}')
        result.append(f'Number of sentences: {len(sentences)}')
        result.append(f'Max sentence length: {max(sentencelengths)}')
        result.append(f'Min sentence length: {min(sentencelengths)}')
        result.append(f'Average sentence length: {(sum(sentencelengths)/len(sentencelengths)):.5f}')
        result.append(f'Number of bigrams: {len(bigramCounter)}')
        result.append(f'Max bigram frequency: {max(bigramFrequencies)}')
        result.append(f'Min bigram frequency: {min(bigramFrequencies)}')
        result.append(f'Average bigram frequency: {sum(bigramFrequencies)/len(bigramFrequencies):.5f}')
        result.append(f'Number of trigrams: {len(trigramCounter)}')
        result.append(f'Max trigram frequency: {max(trigramFrequencies)}')
        result.append(f'Min trigram frequency: {min(trigramFrequencies)}')
        result.append(f'Average trigram frequency: {sum(trigramFrequencies)/len(trigramFrequencies):.5f}')

        return '\n'.join(result)
    

    # Collect input data info
    def collectInputDataInfo(self):
        return f'Input Data...\nFile selected: {self.labelFileSelectPath.GetValue()}\nLanguage: {self.selectedLanguage} \
            \nSelected Punctuation: {", ".join(self.selectedPunctuation.keys())}'


    # Exit the application
    def exitApp(self, event):
        self.logMessage('Exiting application...')
        wx.CallLater(500, self.Close)


if __name__ == '__main__':
    app = wx.App(False)
    frame = App()
    frame.Show()
    app.MainLoop()