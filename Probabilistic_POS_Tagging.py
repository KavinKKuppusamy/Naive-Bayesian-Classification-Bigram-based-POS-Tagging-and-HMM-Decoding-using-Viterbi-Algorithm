import sys
from collections import Counter, defaultdict

f = open(sys.argv[1], "r")
all_lines = f.readlines()
corpus_words_count = 0
corpus_list = []
corpus_word_list = []
corpus_tag_list = []
corpus_word_tag_list = []
word_to_tag_list = defaultdict(list)
for line in all_lines:
    line = line.strip()
    word_list = []
    tag_list = []
    words_in_lines = line.split(' ')
    for i in range(len(words_in_lines)):
        word, tag = words_in_lines[i].split('_')
        word_to_tag_list[word].append(tag)
        word_list.append(word)
        tag_list.append(tag)
    tag_list.insert(0, '<s>')
    tag_list.append('</s>')
    corpus_words_count += len(words_in_lines)
    corpus_word_list.append(word_list)
    corpus_tag_list.append(tag_list)
    corpus_word_tag_list.append(words_in_lines)

unigram_tag_count = dict(Counter([item for sub_list in corpus_tag_list for item in sub_list]))
unigram_word_count = dict(Counter([item for sub_list in corpus_word_list for item in sub_list]))
unique_words_in_corpus = len(unigram_word_count)
unigram_word_prob = {key: val / corpus_words_count for key, val in unigram_word_count.items()}
word_to_tag_list = {key: list(set(val)) for key, val in word_to_tag_list.items()}

## Bigram Count for word,word ##

bigram_word_count = {}
for each_sentence in corpus_word_list:
    for index in range(1, len(each_sentence)):
        bigram_word_count[each_sentence[index - 1] + ' ' + each_sentence[index]] = bigram_word_count.get(
            each_sentence[index - 1] + ' ' + each_sentence[index], 0) + 1
## Bigram Probability for word,word ##
bigram_word_prob = {}
for each_sentence in corpus_word_list:
    for index in range(1, len(each_sentence)):
        bigram_word_prob[each_sentence[index] + '|' + each_sentence[index - 1]] = bigram_word_count.get(
            each_sentence[index - 1] + ' ' + each_sentence[index]) / unigram_word_count.get(each_sentence[index - 1])

## Bigram Count for tag,tag ##

bigram_tag_count = {}
for each_sentence in corpus_tag_list:
    for index in range(1, len(each_sentence)):
        bigram_tag_count[each_sentence[index - 1] + ' ' + each_sentence[index]] = bigram_tag_count.get(
            each_sentence[index - 1] + ' ' + each_sentence[index], 0) + 1

## Bigram Probability for tag,tag ##

bigram_tag_prob = {}
for each_sentence in corpus_tag_list:
    for index in range(1, len(each_sentence)):
        bigram_tag_prob[each_sentence[index] + '|' + each_sentence[index - 1]] = bigram_tag_count.get(
            each_sentence[index - 1] + ' ' + each_sentence[index]) / unigram_tag_count.get(each_sentence[index - 1])

## Bigram Count for word,tag ##
bigram_word_tag_count = {}
for each_word_tag in corpus_word_tag_list:
    for i in range(0, len(each_word_tag)):
        bigram_word_tag_count[each_word_tag[i]] = bigram_word_tag_count.get(each_word_tag[i], 0) + 1

## Bigram Probability for word,tag ##

bigram_word_tag_prob = {}
for each_word_tag in corpus_word_tag_list:
    for i in range(0, len(each_word_tag)):
        bigram_word_tag_prob[each_word_tag[i].replace('_', '|')] = bigram_word_tag_count.get(
            each_word_tag[i]) / unigram_tag_count.get(each_word_tag[i].split('_')[1])


def naive_bayes_probabilistic_pos_tagging(input_sequence):
    input_sentence.append("</s>")
    pos_tagged_sequence = ''
    prob_seq_dict = dict()
    argmax_prob = 0
    argmax_tag = ''
    prev_argmax_tag = '<s>'
    total_probability = 1
    for i in range(0, len(input_sequence) - 1):
        for each_tag in unigram_tag_count.keys():
            curr_tag_prob = bigram_word_tag_prob.get(input_sequence[i] + '|' + each_tag, 0) * bigram_tag_prob.get(
                each_tag + '|' + prev_argmax_tag, 0)
            if input_sentence[i + 1] == '</s>':
                curr_tag_prob *= bigram_tag_prob.get('</s>' + '|' + each_tag, 0)
            if curr_tag_prob > argmax_prob:
                argmax_prob = curr_tag_prob
                argmax_tag = each_tag

        total_probability *= argmax_prob
        prob_seq_dict[input_sequence[i] + '|' + argmax_tag] = str(
            bigram_word_tag_prob.get(input_sequence[i] + '|' + argmax_tag, 0))
        prob_seq_dict[argmax_tag + '|' + prev_argmax_tag] = str(
            bigram_tag_prob.get(argmax_tag + '|' + prev_argmax_tag, 0))
        pos_tagged_sequence += input_sequence[i] + '_' + argmax_tag + ' '
        prev_argmax_tag = argmax_tag
        argmax_prob = 0
    prob_seq_dict['</s>' + '|' + argmax_tag] = str(bigram_tag_prob.get('</s>' + '|' + argmax_tag, 0))
    return prob_seq_dict, total_probability, pos_tagged_sequence


input_sentence = sys.argv[2].strip().split()
prob_seq_d, total_prob, pos_tagged_seq = naive_bayes_probabilistic_pos_tagging(input_sentence)
print("\nCompleted performing Naïve Bayesian Classification (Bigram) based POS Tagging\n")

for seq_tag, prob_val in prob_seq_d.items():
    print(f"{seq_tag}  :  {prob_val}")
print("\n\nTotal Probability : ", total_prob)
print("\n\nPOS Tagged Sequence : ", pos_tagged_seq)
