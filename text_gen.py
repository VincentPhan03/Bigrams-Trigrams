"""This program generates random text based on n-grams
calculated from sample text.

Author: Nathan Sprague and Vincent Phan
Date: 1/15/21
Modified: 1/25/24

"""

# Honor code statement (if you received help from an outside source): 
# I used chatgpt to assist me with this pa
#

import random
import string
from typing import Dict, List, Any, Tuple

# Create some type aliases to simplify the type hinting below.
BigramDict = Dict[str, Dict[str, float]]
TrigramDict = Dict[Tuple[str, str], Dict[str, float]]


def text_to_list(file_name: str) -> List[str]:
    """ Converts the provided plain-text file to a list of words.  All
    punctuation will be removed, and all words will be converted to
    lower-case.

    Args:
        file_name: A string containing a file path.

    Returns:
        A list containing the words from the file.
    """
    with open(file_name, 'r') as handle:
        text = handle.read().lower()
        text = text.translate(
            str.maketrans(string.punctuation,
                          " " * len(string.punctuation)))
    return text.split()


def select_random(distribution: Dict[Any, float]):
    """
    Select an item from the the probability distribution
    represented by the provided dictionary.

    Example:
    >>> select_random({'a':.9, 'b':.1})
    'a'
    """

    # Make sure that the probability distribution has a sum close to 1.
    assert abs(sum(distribution.values()) - 1.0) < .000001, \
        "Probability distribution does not sum to 1!"

    r = random.random()
    total = 0.0
    for item in distribution:
        total += distribution[item]
        if r < total:
            return item

    assert False, "Error in select_random!"


def counts_to_probabilities(counts: Dict[Any, int]) -> Dict[Any, float]:
    """ Convert a dictionary of counts to probabilities.

    Args:
       counts: a dictionary mapping from items to integers

    Returns:
       A new dictionary where each count has been divided by the sum
       of all entries in counts.

    Example:

    >>> counts_to_probabilities({'a':9, 'b':1})
    {'a': 0.9, 'b': 0.1}

    """
    probabilities = {}
    total = 0
    for item in counts:
        total += counts[item]
    for item in counts:
        probabilities[item] = counts[item] / float(total)
    return probabilities


def calculate_unigrams(word_list: List[str]) -> Dict[str, float]:
    """Calculates the probability distribution over individual words.

    Args:
       word_list: a list of strings corresponding to the sequence of
           words in a document. Words must be all lower-case with no
           punctuation.

    Returns:
       A dictionary mapping from words to probabilities.

    Example:

    >>> calculate_unigrams(['i', 'think', 'therefore', 'i', 'am'])
    {'i': 0.4, 'am': 0.2, 'think': 0.2, 'therefore': 0.2}

    """
    unigrams = {}
    for word in word_list:
        if word in unigrams:
            unigrams[word] += 1
        else:
            unigrams[word] = 1
    return counts_to_probabilities(unigrams)


def random_unigram_text(unigrams: Dict[str, float], num_words: int) -> str:
    """Generate a random sequence according to the provided probabilities.

    Args:
       unigrams: Probability distribution over words (as returned by
           the calculate_unigrams function).
       num_words: The number of words of random text to generate.

    Returns:
       The random string of words with each subsequent word separated by a
       single space.

    Example:

    >>> u = calculate_unigrams(['i', 'think', 'therefore', 'i', 'am'])
    >>> random_unigram_text(u, 5)
    'think i therefore i i'

    """
    result = ""
    for i in range(num_words):
        next_word = select_random(unigrams)
        result += next_word + " "
    return result.rstrip()


def calculate_bigrams(word_list: List[str]) -> BigramDict:
    """Calculates, for each word in the list, the probability distribution
    over possible subsequent words.

    This function returns a dictionary that maps from words to
    dictionaries that represent probability distributions over
    subsequent words.

    Args:
       word_list: a list of strings corresponding to the sequence of
           words in a document. Words must be all lower-case with no
           punctuation.

    Example:

    >>> b = calculate_bigrams(['i', 'think', 'therefore', 'i', 'am',\
                               'i', 'think', 'i', 'think'])
    >>> print(b)
    {'i':  {'am': 0.25, 'think': 0.75},
     None: {'i': 1.0},
     'am': {'i': 1.0},
     'think': {'i': 0.5, 'therefore': 0.5},
     'therefore': {'i': 1.0}}

    Note that None stands in as the predecessor of the first word in
    the sequence.

    Once the bigram dictionary has been obtained it can be used to
    obtain distributions over subsequent words, or the probability of
    individual words:

    >>> print(b['i'])
    {'am': 0.25, 'think': 0.75}

    >>> print(b['i']['think'])
    0.75

    """
    # YOUR CODE HERE
    bigrams = {}

    if word_list:
        bigrams[None] = {word_list[0]: 1}

    for i in range(len(word_list) - 1):
        current_word = word_list[i]
        next_word = word_list[i + 1]

        if current_word not in bigrams:
            bigrams[current_word] = {}

        if next_word in bigrams[current_word]:
            bigrams[current_word][next_word] += 1
        else:
            bigrams[current_word][next_word] = 1

    for word in bigrams:
        bigrams[word] = counts_to_probabilities(bigrams[word])

    return bigrams


def calculate_trigrams(word_list: List[str]) -> TrigramDict:
    """Calculates, for each adjacent pair of words in the list, the
    probability distribution over possible subsequent words.

    The returned dictionary maps from two-word tuples to dictionaries
    that represent probability distributions over subsequent
    words.

    Example:

    >>> calculate_trigrams(['i', 'think', 'therefore', 'i', 'am',\
                                'i', 'think', 'i', 'think'])
    {('think', 'i'): {'think': 1.0},
    ('i', 'am'): {'i': 1.0},
    (None, None): {'i': 1.0},
    ('therefore', 'i'): {'am': 1.0},
    ('think', 'therefore'): {'i': 1.0},
    ('i', 'think'): {'i': 0.5, 'therefore': 0.5},
    (None, 'i'): {'think': 1.0},
    ('am', 'i'): {'think': 1.0}}
    """
    # YOUR CODE HERE
    trigrams = {}

    if word_list:
        trigrams[None, None] = {word_list[0]: 1}

    for i in range(len(word_list) - 2):
        if i == 0:
            start_pair = (None, word_list[0])
            next_word = word_list[1]
            if start_pair not in trigrams:
                trigrams[start_pair] = {}
            trigrams[start_pair][next_word] = trigrams[start_pair].get(
                next_word, 0) + 1

        current_pair = (word_list[i], word_list[i + 1])
        next_word = word_list[i + 2]

        if current_pair not in trigrams:
            trigrams[current_pair] = {}

        if next_word in trigrams[current_pair]:
            trigrams[current_pair][next_word] += 1
        else:
            trigrams[current_pair][next_word] = 1

    for pair in trigrams:
        trigrams[pair] = counts_to_probabilities(trigrams[pair])

    return trigrams


def random_bigram_text(first_word: str,
                       bigrams: BigramDict,
                       num_words: int) -> str:
    """Generate a random sequence of words following the word pair
    probabilities in the provided distribution.

    Args:
       first_word: This word will be the first word in the generated
           text.
       bigrams: Probability distribution over word pairs (as returned
           by the calculate_bigrams function).
       num_words: Length of the generated text (including the provided
          first word)

    Returns:
       The random string of words with each subsequent word separated by a
       single space.

    Example:
    >>> b = calculate_bigrams(['i', 'think', 'therefore', 'i', 'am',\
                               'i', 'think', 'i', 'think'])
    >>> random_bigram_text('think', b, 5)
    'think i think therefore i'

    >>> random_bigram_text('think', b, 5)
    'think therefore i think therefore'

    """
    result = first_word + " "
    current_word = first_word

    for i in range(1, num_words):
        next_word = select_random(bigrams[current_word])
        result += next_word + " "
        current_word = next_word

    return result.rstrip()


def random_trigram_text(first_word: str, second_word: str,
                        trigrams: TrigramDict,
                        num_words: int) -> str:
    """Generate a random sequence of words according to the provided
    trigram distributions. The first two words provided must
    appear in the trigram distribution.

    Args:
       first_word: The first word in the generated text.
       second_word: The second word in the generated text.
       trigrams: trigram probabilities (as returned by the
           calculate_trigrams function).
       num_words: Length of the generated text (including the provided
           words)

    Returns:
       The random string of words with each subsequent word separated by a
       single space.

    """
    # YOUR CODE HERE
    result = [first_word, second_word]
    for i in range(num_words - 2):
        current_pair = (result[-2], result[-1])
        next_word = select_random(trigrams[current_pair])
        result.append(next_word)

    return ' '.join(result)


def unigram_main():
    """ Generate text from Huck Fin unigrams."""
    words = text_to_list('huck.txt')
    unigrams = calculate_unigrams(words)
    print(random_unigram_text(unigrams, 100))


def bigram_main():
    """ Generate text from Huck Fin bigrams."""
    words = text_to_list('huck.txt')
    bigrams = calculate_bigrams(words)
    print(random_bigram_text('the', bigrams, 100))


def trigram_main():
    """ Generate text from Huck Fin trigrams."""
    words = text_to_list('huck.txt')
    trigrams = calculate_trigrams(words)
    print(random_trigram_text('there', 'is', trigrams, 100))


if __name__ == "__main__":
    # You can insert testing code here, or switch out the main method
    # to try bigrams or trigrams.
    trigram_main()
