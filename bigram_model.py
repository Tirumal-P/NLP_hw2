from collections import defaultdict

def build_bigram_model(corpus):
    """
    Build unigram and bigram counts from a list of sentences.
    Each sentence should already include <s> and </s> tokens.
    """
    unigram_counts = defaultdict(int)
    bigram_counts = defaultdict(lambda: defaultdict(int))

    for sentence in corpus:
        tokens = sentence.split()
        for token in tokens:
            unigram_counts[token] += 1
        for i in range(len(tokens) - 1):
            bigram_counts[tokens[i]][tokens[i + 1]] += 1

    return unigram_counts, bigram_counts


def compute_bigram_probabilities(bigram_counts, unigram_counts):
    """
    Estimate bigram probabilities using Maximum Likelihood Estimation (MLE).
    P(w_i | w_{i-1}) = count(w_{i-1}, w_i) / count(w_{i-1})
    """
    bigram_probs = defaultdict(dict)
    for prev_word, next_words in bigram_counts.items():
        total = unigram_counts[prev_word]
        for next_word, count in next_words.items():
            bigram_probs[prev_word][next_word] = count / total
    return bigram_probs


def sentence_probability(sentence, bigram_probs):
    """
    Compute the probability of a sentence using the bigram model.
    Returns 0 if any bigram has zero probability (unseen bigram).
    """
    tokens = sentence.split()
    prob = 1.0
    steps = []
    for i in range(len(tokens) - 1):
        prev = tokens[i]
        curr = tokens[i + 1]
        p = bigram_probs.get(prev, {}).get(curr, 0)
        steps.append((prev, curr, p))
        prob *= p
    return prob, steps


# ------- Training Corpus -------
corpus = [
    "<s> I love NLP </s>",
    "<s> I love deep learning </s>",
    "<s> deep learning is fun </s>"
]

# ------- Build Model -------
unigram_counts, bigram_counts = build_bigram_model(corpus)

print("=" * 55)
print("Unigram Counts:")
for word, count in sorted(unigram_counts.items()):
    print(f"  {word}: {count}")

print("\nBigram Counts:")
for prev, nexts in sorted(bigram_counts.items()):
    for nxt, cnt in sorted(nexts.items()):
        print(f"  ({prev}, {nxt}): {cnt}")

# ------- Estimate Probabilities -------
bigram_probs = compute_bigram_probabilities(bigram_counts, unigram_counts)

print("\nBigram Probabilities (MLE):")
for prev, nexts in sorted(bigram_probs.items()):
    for nxt, p in sorted(nexts.items()):
        print(f"  P({nxt} | {prev}) = {p:.4f}")

# ------- Evaluate Test Sentences -------
test_sentences = [
    "<s> I love NLP </s>",
    "<s> I love deep learning </s>"
]

print("\n" + "=" * 55)
print("Sentence Probabilities:")
results = []
for sent in test_sentences:
    prob, steps = sentence_probability(sent, bigram_probs)
    results.append((sent, prob, steps))
    print(f"\nSentence: {sent}")
    for prev, curr, p in steps:
        print(f"  P({curr} | {prev}) = {p:.4f}")
    print(f"  --> Total Probability = {prob:.6f}")

print("\n" + "=" * 55)
if results[0][1] > results[1][1]:
    print(f"The model prefers: \"{results[0][0]}\"")
    print(f"  Reason: It has a higher bigram probability ({results[0][1]:.6f} > {results[1][1]:.6f}).")
    print("  'NLP' follows 'love' with prob 1/2, and '</s>' follows 'NLP' with prob 1/1,")
    print("  while S2 has an extra step P(</s>|learning)=1/2, making it half as likely.")
else:
    print(f"The model prefers: \"{results[1][0]}\"")
    print(f"  Reason: It has a higher bigram probability ({results[1][1]:.6f} > {results[0][1]:.6f}).")
