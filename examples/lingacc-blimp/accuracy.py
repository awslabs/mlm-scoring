#!/usr/bin/env python3

import logging
from pathlib import Path
import sys

import numpy as np

anaphor_agreement = {
    'anaphor_gender_agreement',
    'anaphor_number_agreement'
}
argument_structure = {
    'animate_subject_passive',
    'animate_subject_trans',
    'causative',
    'drop_argument',
    'inchoative',
    'intransitive',
    'passive_1',
    'passive_2',
    'transitive'
}
binding = {
    'principle_A_c_command',
    'principle_A_case_1',
    'principle_A_case_2',
    'principle_A_domain_1',
    'principle_A_domain_2',
    'principle_A_domain_3',
    'principle_A_reconstruction'
}
control_raising = {
    'existential_there_object_raising',
    'existential_there_subject_raising',
    'expletive_it_object_raising',
    'tough_vs_raising_1',
    'tough_vs_raising_2'
}
determiner_noun_agreement = {
    'determiner_noun_agreement_1',
    'determiner_noun_agreement_2',
    'determiner_noun_agreement_irregular_1',
    'determiner_noun_agreement_irregular_2',
    'determiner_noun_agreement_with_adjective_1',
    'determiner_noun_agreement_with_adj_2',
    'determiner_noun_agreement_with_adj_irregular_1',
    'determiner_noun_agreement_with_adj_irregular_2'
}
ellipsis = {
    'ellipsis_n_bar_1',
    'ellipsis_n_bar_2'
}
filler_gap = {
    'wh_questions_object_gap',
    'wh_questions_subject_gap',
    'wh_questions_subject_gap_long_distance',
    'wh_vs_that_no_gap',
    'wh_vs_that_no_gap_long_distance',
    'wh_vs_that_with_gap',
    'wh_vs_that_with_gap_long_distance'
}
irregular_forms = {
    'irregular_past_participle_adjectives',
    'irregular_past_participle_verbs'
}
island_effects = {
    'adjunct_island',
    'complex_NP_island',
    'coordinate_structure_constraint_complex_left_branch',
    'coordinate_structure_constraint_object_extraction',
    'left_branch_island_echo_question',
    'left_branch_island_simple_question',
    'sentential_subject_island',
    'wh_island'
}
npi_licensing = {
    'matrix_question_npi_licensor_present',
    'npi_present_1',
    'npi_present_2',
    'only_npi_licensor_present',
    'only_npi_scope',
    'sentential_negation_npi_licensor_present',
    'sentential_negation_npi_scope'
}
quantifiers = {
    'existential_there_quantifiers_1',
    'existential_there_quantifiers_2',
    'superlative_quantifiers_1',
    'superlative_quantifiers_2'
}
subject_verb_agreement = {
    'distractor_agreement_relational_noun',
    'distractor_agreement_relative_clause',
    'irregular_plural_subject_verb_agreement_1',
    'irregular_plural_subject_verb_agreement_2',
    'regular_plural_subject_verb_agreement_1',
    'regular_plural_subject_verb_agreement_2'
}

anaphor_agreement_list = []
argument_structure_list = []
binding_list = []
control_raising_list = []
determiner_noun_agreement_list = []
ellipsis_list = []
filler_gap_list = []
irregular_forms_list = []
island_effects_list = []
npi_licensing_list = []
quantifiers_list = []
subject_verb_agreement_list = []


if __name__ == '__main__':

    base_dir = Path(sys.argv[1])

    num_files = 0

    for file in sorted(base_dir.glob('*.lm.json')):

        with file.open('rt') as f:
            scores = f.readlines()
        num_pairs = len(scores) // 2

        # Count # times good triumphed over evil
        count = 0
        for i in range(num_pairs):
            if float(scores[2*i]) > float(scores[2*i+1]):
                count += 1
        # Remove .lm.json
        uid = file.name.split('.')[0]

        if len(scores) == 0:
            logging.error("{} is empty, skipping".format(file))
            continue
        assert num_pairs == 1000

        acc = count / num_pairs
        print("{:.3f} ({}/{}) - {}".format(acc, count, num_pairs, uid))
        if uid in anaphor_agreement:
            anaphor_agreement_list.append(acc)
        elif uid in argument_structure:
            argument_structure_list.append(acc)
        elif uid in binding:
            binding_list.append(acc)
        elif uid in control_raising:
            control_raising_list.append(acc)
        elif uid in determiner_noun_agreement:
            determiner_noun_agreement_list.append(acc)
        elif uid in ellipsis:
            ellipsis_list.append(acc)
        elif uid in filler_gap:
            filler_gap_list.append(acc)
        elif uid in irregular_forms:
            irregular_forms_list.append(acc)
        elif uid in island_effects:
            island_effects_list.append(acc)
        elif uid in npi_licensing:
            npi_licensing_list.append(acc)
        elif uid in quantifiers:
            quantifiers_list.append(acc)
        elif uid in subject_verb_agreement:
            subject_verb_agreement_list.append(acc)
        else:
            logging.error("Unrecognized UID: {}".format(uid))
        num_files += 1

    print("# of files: {}".format(num_files))

    # Since all 67 classes have 1000 pairs, per-class and overall accuracies are the desired (micro)averages

    print("anaphor_agreement: ", np.mean(anaphor_agreement_list))
    print("argument_structure: ", np.mean(argument_structure_list))
    print("binding: ", np.mean(binding_list))
    print("control_raising: ", np.mean(control_raising_list))
    print("determiner: ", np.mean(determiner_noun_agreement_list))
    print("ellipsis: ", np.mean(ellipsis_list))
    print('filler_gap: ', np.mean(filler_gap_list))
    print('irregular_forms: ', np.mean(irregular_forms_list))
    print('island_effects: ', np.mean(island_effects_list))
    print('npi_licensing: ', np.mean(npi_licensing_list))
    print('quantifiers: ', np.mean(quantifiers_list))
    print('subject_verb_agreement: ', np.mean(subject_verb_agreement_list))

    lists = anaphor_agreement_list + argument_structure_list + binding_list + control_raising_list + determiner_noun_agreement_list + ellipsis_list + filler_gap_list + irregular_forms_list + island_effects_list + npi_licensing_list + quantifiers_list + subject_verb_agreement_list

    assert len(lists) == num_files

    print("overall: ", np.mean(lists))
