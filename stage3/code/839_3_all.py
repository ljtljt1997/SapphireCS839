import py_entitymatching as em
import scipy
import numpy as np
import pandas as pd
import re
import time
import random
from scipy.stats import norm
from numpy import sqrt

### Read-In data
table_a = '../data/Table_A'
table_b = '../data/Table_B'
candidate_set = '../data/Candidate_Set'
prediction_set = '../data/Prediction_List'

dfa = pd.read_csv(table_a)
dfb = pd.read_csv(table_b)
dfc = pd.read_csv(candidate_set)
dfp = pd.read_csv(prediction_set)

# We have about 1772 tuples in prediction set, but we have 464670 in candidate set
# To keep TN / Total > 20%, I decide to make the candidate set size less than 5000




### Blocking Rule ###

## First Blocking Rule: The release year should be the same
# Then I find that there are lots of NAs in the release year
# Thus I put NAs in the code, then run the blocking test
# But I find that there are many matchers being filtered out.
# Because the release years sometimes are wrong but other parts are perfectly right
# Thus I decide to first test if the movie name is similar, if not similar,
# then test if release year is not the same.
# The rule of testing name: If one string is the substring of the other one
dfc1 = dfc.copy()
def match_year(col):
    col1, col2 = col
    name1, year1 = dfa.iloc[col1, [2,5]]
    name2, year2 = dfb.iloc[col2, [2,5]]
    return (name1.strip() != '' and name2.strip() != '' and (name1 in name2 or name2 in name1)) or np.isnan(year1) or np.isnan(year2) or year1 == year2

ans = dfc1.apply(match_year, axis=1)
dfc1 = dfc1[ans].reset_index(drop=True)
# candidate set C1 now has 49689 observations
# Now I have to check if this blocking rule remove many TP (True Positive)
# I decide to print out the movie name in those 200 potential matches
# And manually see if there exist matches or not

# I change the run_debug_blocker a little bit
def run_debug_blocker(table_a, table_b, table_a_key, table_b_key, dfcand):
    dfl = em.read_csv_metadata(table_a, key=table_a_key)
    dfr = em.read_csv_metadata(table_b, key=table_b_key)

    # reading the candidate set and adding key
    # dfcand = pd.read_csv(candidate_set)
    dfcand.drop_duplicates(inplace=True)
    dfcand.to_csv('cand_set_with_index.csv', index_label='id')

    dfcset = em.read_csv_metadata('cand_set_with_index.csv', key='id', ltable=dfl,
                                  rtable=dfr, fk_ltable='A_id', fk_rtable='B_id')

    # running debug blocker to identify the records in A x B \ C
    debug_file = em.debug_blocker(dfcset, dfl, dfr)

    return debug_file

debug_file = run_debug_blocker(table_a, table_b, '_id', '_id', dfc1)

# Print out the name
debug_file.iloc[:,[4,9]]

# The result is nice, does not contain many TP. The blocking rule is applicable
# And I also use a keeping ratio to track the blocking
# S1 = how many tuples are in prediction set
# S2 = how many tuples from prediction set are still left in candidate set
# Keeping ratio = S2 / S1 * 100%
def keeping_ratio(dfc_now):
    compare1 = dfp.sort_values(by=['id1', 'id2']).iloc[:,:2]
    compare1 = compare1.reset_index(drop=True).drop_duplicates()
    compare2 = dfc_now.sort_values(by=['A_id', 'B_id'])
    compare2 = compare2.reset_index(drop=True).drop_duplicates()
    tmp_sum = pd.merge(compare1, compare2,left_on= ['id1','id2'], right_on=['A_id','B_id'])
    ratio = round(tmp_sum.shape[0] / compare1.shape[0], 3)
    print('The keeping ratio is: ', ratio)
    return

keeping_ratio(dfc1) # 99.6%



## Second Blocking Rule: Director Name overlapped at least once
dfc2 = dfc1.copy()
def match_director(col):
    col1, col2 = col
    name1, director1 = dfa.iloc[col1, [2, 3]]
    name2, director2 = dfb.iloc[col2, [2, 3]]
    return (name1.strip() != '' and name2.strip() != '' and (name1 in name2 or name2 in name1)) or isinstance(director1,float) or isinstance(director2,float) or (set(director1.split(',')) & set(director2.split(','))) != set()
ans = dfc1.apply(match_director, axis=1)
dfc2 = dfc1[ans].reset_index(drop=True)
# candidate set C2 now has 16085 observations
debug_file = run_debug_blocker(table_a, table_b, '_id', '_id', dfc2)
# At first I did not put name filter in the match_director. I got a poor result
# Print out the name
debug_file.iloc[:,[4,9]]
keeping_ratio(dfc2) # 98.5%



## Third Blocking Rule: Star Name overlapped at least once
dfc3 = dfc2.copy()
def match_star(col):
    col1, col2 = col
    name1, star1 = dfa.iloc[col1, [2, 4]]
    name2, star2 = dfb.iloc[col2, [2, 4]]
    return (name1.strip() != '' and name2.strip() != '' and (name1 in name2 or name2 in name1)) or isinstance(star1,float) or isinstance(star2,float) or (set(star1.split(',')) & set(star2.split(','))) != set()
ans = dfc2.apply(match_star, axis=1)
dfc3 = dfc2[ans].reset_index(drop=True)
# candidate set C3 now has 14007 observations
debug_file = run_debug_blocker(table_a, table_b, '_id', '_id', dfc3)
# At first I did not put name in the match_director. I got a poor result
# Print out the name
debug_file.iloc[:,[4,9]]
keeping_ratio(dfc3) # 98.2%



## Final Blocking Rule: Keep the best match for every item in table A
dfc4 = dfc3.copy()
def cal_sim(col):
    col1, col2 = col
    name1, director1, star1, year1, runtime1 = dfa.iloc[col1, [2, 3, 4, 5, 6]]
    name2, director2, star2, year2, runtime2 = dfb.iloc[col2, [2, 3, 4, 5, 6]]
    score_name = 1 if name1.strip() != '' and name2.strip() != '' and (name1 in name2 or name2 in name1) else 0
    score_director = 1 if (not isinstance(director1,float)) and (not isinstance(director2,float)) and (set(director1.split(',')) & set(director2.split(','))) != set() else 0
    score_star = 1 if (not isinstance(star1,float)) and (not isinstance(star2,float)) and (set(star1.split(',')) & set(star2.split(','))) != set() else 0
    score_year = 1 if year1 == year2 else 0
    score_runtime = 1 if runtime1 == runtime2 else 0
    return score_name + score_director + score_star + score_year + score_runtime

ans = dfc4.apply(cal_sim, axis=1)
dfc4['score'] = ans
dfc4 = dfc4.sort_values(by=['A_id', 'score'])
dfc4.drop_duplicates(subset = ['A_id'], keep = 'last',inplace = True)
# candidate set C4 now has 2306 observations
debug_file = run_debug_blocker(table_a, table_b, '_id', '_id', dfc4)
# Print out the name
debug_file.iloc[:,[4,9]]
keeping_ratio(dfc4) # 97.7%

## Blocking part finish
dfc4.to_csv('dfc_final.csv',index = False) # Save the file



### Random sample for density check
# select 50 data points
random.seed(960512)
idx = random.sample(range(dfc4.shape[0]),50)
df_tmp1, df_tmp2 = dfa.iloc[dfc4.iloc[idx,0],], dfb.iloc[dfc4.iloc[idx,1],]
df_tmp1, df_tmp2 = df_tmp1.reset_index(drop=True), df_tmp2.reset_index(drop=True)
df_tmp1['index'], df_tmp2['index']  = range(50), range(50)
df_check = pd.concat([df_tmp1, df_tmp2], axis = 0).sort_values(by=['index'])
df_check.to_csv('check.csv',index = False)
# Save the label in check_label.csv
check_label = pd.read_csv('../data/Intermidiate_files/check_label.csv')
idx_label = pd.Series([check_label.iloc[_,0] == 1 for _ in range(50)], index=dfc4.iloc[idx,:2].index)
# The code below is for density calculation
compare1 = dfp.sort_values(by=['id1', 'id2']).iloc[:,:2]
compare1 = compare1.reset_index(drop=True).drop_duplicates()
compare2 = dfc4.iloc[idx,:2][idx_label].sort_values(by=['A_id', 'B_id'])
compare2 = compare2.reset_index(drop=True).drop_duplicates()
tmp_sum = pd.merge(compare1, compare2,left_on= ['id1','id2'], right_on=['A_id','B_id'])
print('The Density is ', tmp_sum.shape[0] / 50)



# We can do the final step now.
idx2 = random.sample(range(dfc4.shape[0]),400)
df_tmp1, df_tmp2 = dfa.iloc[dfc4.iloc[idx2,0],], dfb.iloc[dfc4.iloc[idx2,1],]
df_tmp1, df_tmp2 = df_tmp1.reset_index(drop=True), df_tmp2.reset_index(drop=True)
df_tmp1['index'], df_tmp2['index']  = range(400), range(400)
df_check = pd.concat([df_tmp1, df_tmp2], axis = 0).sort_values(by=['index'])
df_check.to_csv('final.csv',index = False)
# Save the label in final_label.csv
final_label = pd.read_csv('../data/final_label.csv')


## Calculate Recall and Precision
delta = .05
Z = norm.ppf(1 - (delta / 2))

def estimate_PR(labeled_pairs, reduced_cands, predicted_matches):
    '''
    labeled_pairs - a pandas dataframe with schema id1,id2,label
                    Note label needs to be Boolean

    reduced_cands - a pandas dataframe with schema id1,id2
    predicted_matches - a pandas dataframe with schema id1,id2

    return:
        ( (recall lower bound, recall upper bound), (precision lower bound, precision upper bound) )
    '''

    labeled_pairs.drop_duplicates(inplace=True)
    labeled_pairs.columns = ['id1', 'id2', 'label']
    reduced_cands.columns = ['id1', 'id2']
    reduced_cand_set = set(zip(reduced_cands.id1, reduced_cands.id2))
    predicted_matches = set(zip(predicted_matches.id1, predicted_matches.id2))

    # estimate the recall
    # number of positives in the labeled sample
    actual_pos = float(labeled_pairs.label.sum())
    # the maximum number of postives in the candidate set
    max_actual_pos = float(actual_pos + len(reduced_cand_set) - len(labeled_pairs))

    # true positives in the labeled sample
    true_pos = float(
        labeled_pairs.apply(lambda x: (x['id1'], x['id2']) in predicted_matches and x['label'], axis=1).sum())
    # estimated recall
    recall = float(true_pos / actual_pos)

    recall_error = Z * sqrt(
        ((recall * (1 - recall)) / (actual_pos)) * ((max_actual_pos - actual_pos) / (max_actual_pos - 1)))

    # estimate Precision
    labeled_set = set(zip(labeled_pairs.id1, labeled_pairs.id2))
    predicted_pos = float(len(labeled_set & predicted_matches))

    predicted_pos_in_reduced_cand_set = float(len(reduced_cand_set & predicted_matches))

    alpha = predicted_pos_in_reduced_cand_set / len(predicted_matches)
    precision = alpha * (true_pos / predicted_pos)

    precision_error = alpha * Z * sqrt(((precision * (1 - precision)) / predicted_pos) * (
                float((len(predicted_matches) - predicted_pos)) / (len(predicted_matches) - 1)))

    return ((recall - recall_error, recall + recall_error),
            (precision - precision_error, precision + precision_error))


# read the labeled pairs file, i.e. the file with the labels
print(estimate_PR(final_label, dfc, dfp))
