# TODO: implement derivational analysis
# going to use a simple rules table based on the affixes in data/prefixes-and-suffixes.json, and the UD lemmas and POS tags
# for example, if we see a word like "անհաջող" (unsuccessful),
# we can check if it starts with "ան" (privative prefix), and if the rest of the word "հաջող"
# is a valid lemma in the lexicon with POS "ADJECTIVE". If so, we can analyze it as "ան" + "հաջող", with the meaning "not successful".

class DerivationalAnalyzer:
    def __init__(self, affix_file):
        ...
    
    def decompose(self, form, lexicon):
        # try prefixes first, then suffixes, then both
        # return list[DerivationalAnalysis] or empty
        ...