from collections import Counter
from unidecode import unidecode
from itertools import combinations
import matplotlib.pyplot as plt, math, random, re

# HELP FUNCTIONS
def handleLang(input):
    string = open(input, "rt", encoding="utf8").read().lower()
    string = re.sub("\d+", " ", string)
    string = re.sub(' +', ' ', string)
    for ch in ["\n", ",", "."]:
        if ch == "\n":
            string = string.replace(ch, " ")
        elif ch == ",":
            string = string.replace(ch, "")
        elif ch == ".":
            string = string.replace(ch, "")
    return string

def average(lst):
    return sum(lst) / len(lst)

def checkIfEqual(lst1,lst2):
    if len(lst1) != len(lst2):
        return False
    for ls1, ls2 in zip(lst1,lst2):
        if set(ls1) != set(ls2):
            return False
    return True

def drawHistogram(lst):
    plt.hist(lst)
    plt.ylabel('Frequency');
    plt.title('Histogram')
    plt.show()

# K-MEANS CLUSTERING
class kmeans:
    # INITIALIZE DATA
    def __init__(self):

        self.corpus = {
            "Dutch": handleLang("../langfiles/dut.txt"), "German": handleLang("../langfiles/ger.txt"),
            "Swedish": handleLang("../langfiles/swd.txt"), "Danish": handleLang("../langfiles/dns.txt"),
            "English": handleLang("../langfiles/eng.txt"), "French": handleLang("../langfiles/frn.txt"),
            "Italian": handleLang("../langfiles/itn.txt"), "Romanian": handleLang("../langfiles/rum.txt"),
            "Spanish": handleLang("../langfiles/spn.txt"), "Portuguese": handleLang("../langfiles/por.txt"),
            "Bosnian": handleLang("../langfiles/src1.txt"), "Serbian": handleLang("../langfiles/src3.txt"),
            "Slovakian": handleLang("../langfiles/slo.txt"), "Slovenian": handleLang("../langfiles/slv.txt"),
            "Russian": unidecode(handleLang("../langfiles/rus.txt")), "Greek": unidecode(handleLang("../langfiles/grk.txt")),
            "Czech": handleLang("../langfiles/czc.txt"), "Bulgarian": unidecode(handleLang("../langfiles/blg.txt")),
            "Luxembourgish": handleLang("../langfiles/lux.txt"), "Polish": handleLang("../langfiles/pql.txt")
        }

        self.data = {key: [] for key in self.corpus.keys()}
        for key in self.corpus.keys():
            self.data[key] = dict(Counter(self.kmers(self.corpus[key], 3)))
        self.languages = list(self.data.keys())
        self.cosineSimilarities = {}
        self.textSimilarities = {}
        for k1,k2 in combinations(self.languages, r=2):
            self.calculateCosineSimilarity(k1,k2)

    # RUN
    def __call__(self, k):
        self.run(k)
        self.run100(k)
        self.findLan("../inputfile.txt")

    def run(self, k):
        medoids = random.sample(self.languages, k)
        result = self.calc(medoids)
        medoids = list(self.newMedoids(result))
        temp = []
        while not checkIfEqual(temp, result):
            temp = list(result)
            result = self.calc(medoids)
            medoids = list(self.newMedoids(result))
        print("{}{}".format("Language groups: ", result))
        print()

    def run100(self, k):
        silhouettes = []
        counter = 0
        while counter != 100:
            medoids = random.sample(self.languages, k)
            result = self.calc(medoids)
            medoids = list(self.newMedoids(result))
            temp = []
            while not checkIfEqual(temp, result):
                temp = list(result)
                result = self.calc(medoids)
                medoids = list(self.newMedoids(result))
            silhouetteValue = self.calcSilhouette(result)
            silhouettes.append(silhouetteValue)
            counter = counter + 1
            print("{}{}".format(counter, ". iteration"))
            print("{}{}".format("Language groups: ", result))
            print("{}{}".format("Silhouette: ", silhouetteValue))
            print()
        drawHistogram(silhouettes)

    def findLan(self, file):
        f = unidecode(handleLang(file))
        self.data["Text"] = dict(Counter(self.kmers(f, 3)))
        for x in self.languages:
            self.calculateCosineSimilarity("Text", x)
        tuple = max(self.textSimilarities, key=self.textSimilarities.get)
        print("{}{}".format("Your language is: ", tuple[1]))
        self.data.pop("Text", None)
        self.textSimilarities = {}

    def calc(self, medoids):
        groups = []
        for medoid in medoids:
            groups.append([medoid])
        for language in self.languages:
            if language in medoids:
                continue
            maxSimilarity = 0
            maxMedoid = ""
            for medoid in medoids:
                similarity = self.findCosineSimilarity(language, medoid)
                if similarity > maxSimilarity:
                    maxSimilarity = similarity
                    maxMedoid = medoid
            for group in groups:
                if maxMedoid in group:
                    group.append(language)
        return groups

    def newMedoids(self, groups):
        newmedoids = []
        for group in groups:
            if len(group) == 1:
                newmedoids.append(group[0])
                continue
            newmedoids.append(self.calculateMedoids(group))
        return newmedoids

    def calculateMedoids(self, group):
        dict = {key: [] for key in group}
        for k1,k2 in combinations(group, r=2):
            similarity = self.findCosineSimilarity(k1,k2)
            dict[k1].append(similarity)
            dict[k2].append(similarity)
        for key, value in dict.items():
            dict[key] = average(value)
        return max(dict, key=dict.get)

    def kmers(self, s, k=3):
        for i in range(len(s)-k+1):
            yield s[i:i+k]

    def findCosineSimilarity(self,k1,k2):
        if (k1,k2) in self.cosineSimilarities:
            return self.cosineSimilarities[(k1,k2)]
        else:
            return self.cosineSimilarities[(k2,k1)]

    def calculateCosineSimilarity(self, k1, k2):
        equalThrees = set(self.data[k1].keys()) & set(self.data[k2].keys())
        temp = []

        for key in equalThrees:
            temp.append(self.data[k1][key]*self.data[k2][key])

        x = sum(temp)
        y1 = math.sqrt(sum([a*a for a in list(self.data[k1].values())]))
        y2 = math.sqrt(sum([a*a for a in list(self.data[k2].values())]))
        rez = x/(y1*y2)

        if k1 == "Text":
            self.textSimilarities[(k1,k2)] = rez
            return None
        self.cosineSimilarities[(k1,k2)] = rez

    def calcSilhouette(self, groups):
        dict = {key: [] for key in self.languages}
        for group in groups:
            for language in self.languages:
                temp = []
                if language in group and len(group) == 1:
                    dict[language].append(1)
                    continue
                elif language in group:
                    for k1, k2 in combinations(group, r=2):
                        if k1 == language or k2 == language:
                            val = self.findCosineSimilarity(k1, k2)
                            temp.append(val)
                    dict[language].append(average(temp))
                else:
                    group.append(language)
                    for k1, k2 in combinations(group, r=2):
                        if k1 == language or k2 == language:
                            val = self.findCosineSimilarity(k1, k2)
                            temp.append(val)
                    dict[language].append(average(temp))
                    group.remove(language)
        for key, value in dict.items():
            value.sort(reverse=True)
            dict[key] = (value[0] - value[1]) / value[0]
        return average(list(dict.values()))

km = kmeans()
km(5)
