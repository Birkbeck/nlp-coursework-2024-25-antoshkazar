#Re-assessment template 2025

# Note: The template functions here and the dataframe format for structuring your solution is a suggested but not mandatory approach. You can use a different approach if you like, as long as you clearly answer the questions and communicate your answers clearly.

import nltk
import spacy
import pandas as pd
from pathlib import Path
from nltk.tokenize import word_tokenize
import string
from collections import Counter
import math


nlp = spacy.load("en_core_web_sm")
nlp.max_length = 2000000



def fk_level(text, d):
    """Returns the Flesch-Kincaid Grade Level of a text (higher grade is more difficult).
    Requires a dictionary of syllables per word.

    Args:
        text (str): The text to analyze.
        d (dict): A dictionary of syllables per word.

    Returns:
        float: The Flesch-Kincaid Grade Level of the text. (higher grade is more difficult)
    """
    tokens = word_tokenize(text)
    words = [t.lower() for t in tokens if t.isalpha()]
    
    if not words:
        return 0
    
    sentences = len([t for t in tokens if t in '.!?'])
    if sentences == 0:
        sentences = 1
    
    syllables = sum(count_syl(word, d) for word in words)
    
    avg_sentence_length = len(words) / sentences
    avg_syllables_per_word = syllables / len(words)
    
    fk_score = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59
    
    return max(0, fk_score)


def count_syl(word, d):
    """Counts the number of syllables in a word given a dictionary of syllables per word.
    if the word is not in the dictionary, syllables are estimated by counting vowel clusters

    Args:
        word (str): The word to count syllables for.
        d (dict): A dictionary of syllables per word.

    Returns:
        int: The number of syllables in the word.
    """
    word = word.lower()
    if word in d:
        pronunciations = d[word]
        if pronunciations:
            pronunciation = pronunciations[0]
            return len([p for p in pronunciation if p[-1].isdigit()])
    
    vowels = 'aeiouy'
    count = 0
    prev_vowel = False
    for char in word:
        if char in vowels:
            if not prev_vowel:
                count += 1
            prev_vowel = True
        else:
            prev_vowel = False
    
    if count == 0:
        count = 1
    return count


def read_novels(path=Path.cwd() / "p1-texts" / "novels"):
    """Reads texts from a directory of .txt files and returns a DataFrame with the text, title,
    author, and year"""
    novels_data = []
    
    for file_path in path.glob("*.txt"):
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        filename = file_path.stem
        parts = filename.split('-')
        
        year = int(parts[-1])
        author = parts[-2]
        title = '-'.join(parts[:-2])
        
        novels_data.append({
            'text': text,
            'title': title,
            'author': author,
            'year': year
        })
    
    df = pd.DataFrame(novels_data)
    df = df.sort_values('year').reset_index(drop=True)
    
    return df


def parse(df, store_path=Path.cwd() / "pickles", out_name="parsed.pickle"):
    """Parses the text of a DataFrame using spaCy, stores the parsed docs as a column and writes 
    the resulting  DataFrame to a pickle file"""
    store_path.mkdir(exist_ok=True)
    
    df['parsed'] = df['text'].apply(lambda x: nlp(x))
    
    pickle_path = store_path / out_name
    df.to_pickle(pickle_path)
    
    return df


def nltk_ttr(text):
    """Calculates the type-token ratio of a text. Text is tokenized using nltk.word_tokenize."""
    tokens = word_tokenize(text)
    tokens = [t.lower() for t in tokens if t.isalpha()]
    if not tokens:
        return 0
    types = set(tokens)
    return len(types) / len(tokens)


def get_ttrs(df):
    """helper function to add ttr to a dataframe"""
    results = {}
    for i, row in df.iterrows():
        results[row["title"]] = nltk_ttr(row["text"])
    return results


def get_fks(df):
    """helper function to add fk scores to a dataframe"""
    results = {}
    cmudict = nltk.corpus.cmudict.dict()
    for i, row in df.iterrows():
        results[row["title"]] = round(fk_level(row["text"], cmudict), 4)
    return results


def subjects_by_verb_pmi(doc, target_verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    subjects = []
    all_tokens = [token.text.lower() for token in doc if token.pos_ in ['NOUN', 'PROPN', 'PRON']]
    verb_tokens = [token.text.lower() for token in doc if token.lemma_.lower() == target_verb.lower() and token.pos_ == 'VERB']
    
    for token in doc:
        if token.lemma_.lower() == target_verb.lower() and token.pos_ == 'VERB':
            for child in token.children:
                if child.dep_ == 'nsubj':
                    subjects.append(child.text.lower())
    
    if not subjects or not verb_tokens:
        return []
    
    subject_counts = Counter(subjects)
    verb_count = len(verb_tokens)
    total_tokens = len(all_tokens)
    
    pmi_scores = []
    for subject, count in subject_counts.items():
        subject_freq = all_tokens.count(subject)
        joint_freq = count
        
        if subject_freq > 0 and joint_freq > 0:
            pmi = math.log((joint_freq * total_tokens) / (subject_freq * verb_count))
            pmi_scores.append((subject, pmi))
    
    return sorted(pmi_scores, key=lambda x: x[1], reverse=True)[:10]


def subjects_by_verb_count(doc, verb):
    """Extracts the most common subjects of a given verb in a parsed document. Returns a list."""
    subjects = []
    
    for token in doc:
        if token.lemma_.lower() == verb.lower() and token.pos_ == 'VERB':
            for child in token.children:
                if child.dep_ == 'nsubj':
                    subjects.append(child.text.lower())
    
    return Counter(subjects).most_common(10)


def adjective_counts(doc):
    """Extracts the most common adjectives in a parsed document. Returns a list of tuples."""
    adjectives = [token.text.lower() for token in doc if token.pos_ == 'ADJ']
    return Counter(adjectives).most_common(10)


if __name__ == "__main__":
    """
    uncomment the following lines to run the functions once you have completed them
    """
    path = Path.cwd() / "p1-texts" / "novels"
    print(path)
    df = read_novels(path) # this line will fail until you have completed the read_novels function above.
    nltk.download("cmudict")
    df = parse(df)
    print(df.head())
    print(get_ttrs(df))
    print(get_fks(df))
    df = pd.read_pickle(Path.cwd() / "pickles" /"parsed.pickle")
    print(adjective_counts(df))

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_count(row["parsed"], "hear"))
        print("\n")

    for i, row in df.iterrows():
        print(row["title"])
        print(subjects_by_verb_pmi(row["parsed"], "hear"))
        print("\n")

