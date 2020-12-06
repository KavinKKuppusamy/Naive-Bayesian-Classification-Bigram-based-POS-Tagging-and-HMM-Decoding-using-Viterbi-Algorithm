import sys
import pandas as pd
import numpy as np

print("Viterbi Algorithm - POS Tagging Sequence")
print("\nReading Transition Probabilities")

trans_prob = pd.read_csv('Transition_Probabilities.csv')
trans_prob.set_index('Tags', inplace=True)
print(trans_prob)

print("\nReading Observation Probabilities")
obs_prob = pd.read_csv('Observation_Probabilities.csv')
obs_prob.set_index('Tags', inplace=True)
print(obs_prob)

states = trans_prob.columns
print(f"\nStates(Tags) : {' , '.join(states)}")

input_sentence = sys.argv[1]
observed_seq = input_sentence.strip().split()
print(f"\nObserved Sequence : {' '.join(observed_seq)}")
observed_seq.insert(0, '<s>')


def viterbi_decoding_algorithm(obs_seq):
    viterbi_table = []
    start_state = 0
    for word in obs_seq[1:]:
        trellis = []
        for tag in states:
            ## Initial State ##
            if start_state == 0:
                trellis.append(([trans_prob.loc[obs_seq[0]][tag] * obs_prob.loc[tag][word], -1, obs_seq[0], tag, word]))
            else:
                max_prob_val = -1
                cur_state = []
                for index, prev_state in enumerate(viterbi_table[-1]):
                    current_prob_val = prev_state[0] * trans_prob.loc[states[index]][tag] * obs_prob.loc[tag][word]
                    if current_prob_val > max_prob_val:
                        max_prob_val = current_prob_val
                        cur_state = [current_prob_val, index, states[index], tag, word]
                trellis.append(cur_state)
        start_state += 1
        viterbi_table.append(trellis)

    ## Finding maximum probability from last observation sequence ##

    max_index_prob = np.argmax([last_sub_row[0] for last_sub_row in viterbi_table[-1]])
    sequence_probability = viterbi_table[-1][max_index_prob][0]
    tag_sequence = []
    
    ## Backtrace for finding the sequence ##
    for each_obs_states in reversed(viterbi_table):
        tag_sequence.append(each_obs_states[max_index_prob][4] + '_' + each_obs_states[max_index_prob][3])
        max_index_prob = each_obs_states[max_index_prob][1]
    tag_sequence = tag_sequence[::-1]
    return sequence_probability, tag_sequence


tag_seq_prob, tag_seq =  viterbi_decoding_algorithm(observed_seq)
print("Probability for the Observation sequence : ", tag_seq_prob)
print(f"Most likely Tag Sequence : {' '.join(tag_seq)}")
