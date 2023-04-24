from collections import defaultdict
import sys


def create_corpus(path_input, occurrence_thres):
    corpus = defaultdict(int)
    for line in open(path_input, 'r'):
        if line != '\n':
            ind, vocab, pos = line.split()
            corpus[vocab] += 1

    corpus_filter = {'< unk >': 0}
    for k, v in corpus.items():
        if v > occurrence_thres:
            corpus_filter[k] = v
        else:
            corpus_filter['< unk >'] += v

    return corpus_filter


def format_output(corpus):
    corpus_copy = corpus.copy()
    occ_unk = corpus_copy['< unk >']
    corpus_copy.pop('< unk >', None)
    corpus_sorted = sorted(corpus_copy.items(), key = lambda x: (-x[1], x[0]))
    output = '< unk >\t0\t' + str(occ_unk) + '\n'
    output += '\n'.join(str(item[0]) + '\t' + str(i) + '\t' + str(item[1]) for i, item in enumerate(corpus_sorted, 1))
    return output


def query(corpus, occ_thres):
    # thres = min(corpus.values())
    corpus_size = len(corpus) - 1
    occ_unk = corpus['< unk >']
    print('Threshold of unknown word replacement =', occ_thres)
    print('Total size of vocabulary =', corpus_size)
    print('Total occurrences of special token < unk > after replacement =', occ_unk)


def write_file(corpus, path_output):
    output_formatted = format_output(corpus)
    with open(path_output, 'w') as file:
        file.write(output_formatted)


if __name__ == '__main__':
    path_input = sys.argv[1]
    occ_thres = int(sys.argv[2])
    path_output = sys.argv[3]
    corpus = create_corpus(path_input, occ_thres)
    query(corpus, occ_thres)
    write_file(corpus, path_output)




