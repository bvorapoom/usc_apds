from collections import defaultdict
import sys
import json


def read_file_to_list(path_input):
    output_list = list()
    with open(path_input, 'r') as file:
        data = file.read()
        for sentence in data.split('\n\n'):
            format_sentence = [tuple(line.split('\t')) for line in sentence.strip('\n').split('\n')]
            output_list.append(format_sentence)
    return output_list


def get_emission_transition(data):
    temp_transition = dict()
    temp_emission = dict()
    temp_emission_lower = dict()

    for sentence in data:
        prev_pos = 'start'
        for word_item in sentence:

            word = word_item[1]
            pos = word_item[2]

            # update emission
            if pos in temp_emission.keys():
                temp_emission[pos][word] += 1
            else:
                temp_emission[pos] = defaultdict(int)
                temp_emission[pos][word] += 1

            if pos in temp_emission_lower.keys():
                temp_emission_lower[pos][word.lower()] += 1
            else:
                temp_emission_lower[pos] = defaultdict(int)
                temp_emission_lower[pos][word.lower()] += 1

            # update transition
            if prev_pos in temp_transition.keys():
                temp_transition[prev_pos][pos] += 1
            else:
                temp_transition[prev_pos] = defaultdict(int)
                temp_transition[prev_pos][pos] += 1

            # update prev_pos
            prev_pos = pos


    transition = defaultdict(int)
    emission = defaultdict(int)
    emission_lower = defaultdict(int)

    # get transition parameters
    for prev_pos, pos_dict in temp_transition.items():
        count_prev_pos = sum(pos_dict.values())
        for pos, cnt in pos_dict.items():
            transition[(prev_pos, pos)] = cnt / count_prev_pos

    # get emission parameters
    for pos, word_dict in temp_emission.items():
        count_pos = sum(word_dict.values())
        for word, cnt in word_dict.items():
            emission[(pos, word)] = cnt / count_pos

    for pos, word_dict in temp_emission_lower.items():
        count_pos = sum(word_dict.values())
        for word, cnt in word_dict.items():
            emission_lower[(pos, word)] = cnt / count_pos


    list_pos = list(temp_emission.keys())

    return transition, emission, list_pos, emission_lower


def query(transition, emission):
    print('Number of transition parameters =', len(transition))
    print('Number of emission parameters =', len(emission))


def write_file(transition, emission, path_output):
    output_formatted = {'transition': {str(k): v for k, v in transition.items()}, \
                        'emission': {str(k): v for k, v in emission.items()}}
    with open(path_output, 'w') as file:
        json.dump(output_formatted, file)


if __name__ == '__main__':
    path_input = sys.argv[1]
    path_output = sys.argv[2]

    data = read_file_to_list(path_input)

    transition, emission, list_pos, emission_lower = get_emission_transition(data)
    query(transition, emission)
    write_file(transition, emission, path_output)
    # print(emission)




