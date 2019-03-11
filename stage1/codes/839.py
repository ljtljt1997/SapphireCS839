import re
import pandas as pd
import collections
import nltk
import os
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import ast

# os.chdir('/Users/moran/Google_Drive/Course/CS839/codes')
# nltk.download('averaged_perceptron_tagger')


###### 1. DATA CLEANING ######


### 1.1 Find all combinations of Name ###

# We use <> First Last </> to denote name
# This step means that we will generate First, Last and First Last. (total three names)

def chopoff(name):
    s =  set()
    name = name.split()
    l = len(name)
    k = l - 1
    while k >= 0:
        for i in range(l-k):
            s.add(' '.join(name[i:i + k + 1]))
        k -= 1
    return s

### 1.2 Remove all periods in name tag <>...</> and redo 1.1 ###

# We have this step because for names like <>Donald J. Trump</>
# We will recoginze all names combinations as below:
# Donald / J. / J / Trump / Donald J. / Donald J / J. Trump / J Trump / Donald J. Trump / Donald J Trump

### 1.3 Remove all name tags and remove '-' ###

# The code below combines 1.1, 1.2 and 1.3
def pre_cleaning(text):
    s = set()
    for item in re.findall(r'<>(.*?)</>', text):
        s = s | chopoff(item) # 1.1 step with period
    while re.findall(r'<>[^<>]*?\..*?</>', text): # Remove all period between Name tag <> </>
        text = re.sub(r'(.*?<>[^<>]*?)\.(.*?</>.*)', r'\1 \2', text, 1)
    for item in re.findall(r'<>(.*?)</>', text):
        s = s | chopoff(item) # 1.1 step without period
    text = re.sub(r'<[/]?>', '', text) #
    text = re.sub(r'—', ' ', text)
    return (s, text)



###### 2. FEATURE GENERATION ######

### 2.1 Based on the word's class ###

### 2.1.1 General pattern ###
# In order to run the code more efficiently, we group them.
def get_pos_tag(word):
    word = word.lower()
    pos_nominal_value_map = {
        ',': 'A',
        '.': 'A',
        'CC': 'A',
        'IN': 'B',
        'POS': 'B',
        'NN': 'C',
        'NNP': 'C',
        'JJ': 'D',
        'JJR': 'D',
        'JJS': 'D',
        'VB': 'E',
        'VBD': 'E',
        'VBG': 'E',
        'VBN': 'E',
        'VBP': 'E',
        'VBZ': 'E',
        'PRP': 'F',
        'PRP$': 'F'
    }
    if not word:
        return 0
    tag = nltk.pos_tag([word])
    return pos_nominal_value_map.get(tag[0][1], 0)

### 2.1.2 The former word's class ###
# We select some specific word as special former word,
# and assign the word behind them into a different group
def pre_tag(word):
    s = set(['Mr.','Mrs.','Mr', 'Mrs', 'Miss', 'Ms.', 'Ms.',
     'Mx.', 'Mx', 'Dr', 'Dr.', 'Lady', 'Sir', 'Lord', 'Director','Empire'])
    s = set(list([]))
    if word in s:
        return 'G'
    else:
        return get_pos_tag(word)

### 2.1.3 The latter word's class ###
# Special latter word, and assign the word before them into a different group
def post_tag(word):
    s = set(['Sr.', 'Sr', 'Jr.', 'Jr', 'I', 'II','III','IV', "'s", '"s', 'Prize'])
    if word in s:
        return 'H'
    else:
        return get_pos_tag(word)


###### 3. MORE PRE-PROCESSING RULES ######

# That's our main function and combine all data cleaning into one function
# There are some rules besides the word's class we use in step2 above
### RULES ###
#1 The first letter is uppercase, and the following letters, if exist, must be lowercase.
#   regexp: r'[A-Z][^A-Z]*$'
#2. Pattern: The/An/... + xxxxx + famile/Prize/... We believe it is more likely a name.
#   We use set [cutoff] to denote.
#3. Pattern: The/An/ + xxxxx + ... (without certain word like family/...)
#   We believe it is NOT likely a name.
#4. Pattern: xxxx + School/Hospital/... We believe it is NOT likely a name.
#   We use set [wrong_suf] to denote.
#5. Pattern: When certain words like called/said/told happend, we think
#   the former/latter word is likely a name. we use set [right_pre] and [right_suf] to denote.

def find_feature(word, s):
    # s is a set contains all name
    df = pd.DataFrame() # We pack the data into a dataframe
    # The variables below respectively denote:
    # Word, former word's class, latter word's class, label, word's class, pre-knowledge
    cand, pre_cat, suf_cat, label, self_cat, highly = [], [], [], [], [], []
    cutoff = set(['family', 'Prize', 'siblings', 'Prize-winning'])
    wrong_suf = set(['High','Award-winning','street','Street','school', 'library', 'college', 'alliance', 'hospital', 'building', 'fellowship', 'club', 'university','School', 'Library', 'College', 'Alliance', 'Hospital', 'Building', 'Fellowship', 'Club', 'University'])
    right_pre = set(['called', 'said', 'told', 'defeated'])
    right_suf = set(['called', 'said', 'told', 'defeated', 'named',"'s"])

    # Given start and end index, add one observation to df
    def add_one(idx_start, idx_end, high = 0):
        word_tmp = ' '.join(word[idx_start:idx_end+1])
        self_cat.append(get_pos_tag(word_tmp))
        cand.append(word_tmp)
        label.append((word_tmp in s) + 0)
        highly.append(high)
        pre_cat.append(pre_tag(word[idx_start - 1]) if l - 1 >= idx_start - 1 >= 0 else 'X')
        suf_cat.append(post_tag(word[idx_end + 1]) if 0 <= idx_end + 1 < l else 'Y')

    # Given a list of start and end, add all combination
    def add_range(list_tmp, high = 0):
        if not list_tmp:
            return
        start, end = list_tmp[0], list_tmp[-1]
        for i in range(start, end+1):
            for j in range(start, end+1):
                if i <= j:
                    add_one(i, j, high)

    # The pattern below is also like right_pre, but it's in regexp form
    def reg_pre(word):
        pattern = [r'.*ist', r'.*-year-old']
        return any(re.match(pat, word) for pat in pattern)

    # length, index, candidate index, if we have faith on this word
    l, i, queue, high = len(word), 0, [], 0

    # Combine all rules together, it's kind of messy down there
    while i < l:
        if word[i] in wrong_suf:
            queue = []
            i += 1
            continue
        if word[i] in set(['an', 'a', 'A', 'the', 'The']):
            if queue:
                add_range(queue,1 if word[i] in right_suf else high)
                queue = []
            high = 0
            flag = 0
            while i + 1 < l and word[i + 1] not in cutoff and re.match(r'[A-Z][^A-Z]*$', word[i + 1]):
                queue.append(i + 1)
                i += 1
                flag = 1
            if flag:
                if word[i + 1] in cutoff:
                    add_range(queue, 1)
            queue = []
            i += 2
            continue
        if word[i] in right_pre or reg_pre(word[i]):
            if queue:
                add_range(queue,1 if word[i] in right_suf else high)
                queue = []
            high = 1
            i += 1
            continue
        if re.match(r'[A-Z][^A-Z]*$', word[i]):
            queue.append(i)
        else:
            if queue:
                add_range(queue,1 if word[i] in right_suf else high)
                queue = []
            high = 0
        i += 1
    df['cand'] = cand
    df['pre_cat'] = pre_cat
    df['suf_cat'] = suf_cat
    df['self_cat'] = self_cat
    df['highly'] = highly
    df['label'] = label

    return df


###### 4. MODEL TRAINING ######

### 4.1 Read In Data ###
df = pd.DataFrame()
for i in range(1,301):
    path = os.path.abspath('../Set_I/'+ str(i)+'.txt')
    with open(path, 'r') as f:
        text = f.read()
        text = re.sub('\"', "", text)
        df_this_file = pd.DataFrame()
        s, text = pre_cleaning(text)
        for sent in nltk.sent_tokenize(text):
            df_tmp = find_feature(nltk.word_tokenize(sent), s)
            df_this_file = pd.concat([df_this_file, df_tmp])
    df = pd.concat([df, df_this_file])


### 4.2 Model fitting ###

# Here we compare four models
names = ["Linear Regression","Logistic Regression",
         "Decision Tree", "Random Forest",
         "Linear SVM"]

classifiers = [
    LinearRegression(),
    LogisticRegression(),
    DecisionTreeClassifier(max_depth=8),
    RandomForestClassifier(max_depth=10, n_estimators=15, max_features=2),
    SVC(kernel="linear", C=0.025),
    ]

# Prepare our X and y
df_x, y = df.iloc[:,:df.shape[1] - 1], df['label']
X = pd.get_dummies(df_x.iloc[:,1:])

# Compare results
for name, clf in zip(names, classifiers):
    print(name)
    y_pred = cross_val_predict(clf,X,y, cv=5)
    if (name == 'Linear Regression'):
        mid = 0.55
        for i in range(len(y_pred)):
            if (y_pred[i] >= mid):
                y_pred[i] = int(1)
            else:
                y_pred[i] = int(0)
    train_f1 = f1_score(y,y_pred)
    train_precision = precision_score(y,y_pred)
    train_recall = recall_score(y,y_pred)
    print("Training F1 Score: %.3f%%" % (100*train_f1))
    print("Training Precision: %.3f%%" % (100*train_precision))
    print("Training Recall: %.3f%%" % (100*train_recall))

### 4.3 Final model ###

clf = RandomForestClassifier(max_depth=10, n_estimators=15, max_features=2)
clf.fit(X,y)
y_pred = clf.predict(X)


###### 5. POST-PROCESSING ######

### 5.0 Create tmp_data for further cleaning ###
tmp_data = pd.DataFrame()
tmp_data['name'], tmp_data['pre'], tmp_data['real'] = df_x['cand'], y_pred, y


### 5.1 Create Black list ###

# We create our own blacklists: prefix, month, location
# We put other special words in 'other'
bl, dict_list = set(), ['prefix.txt', 'month.txt', 'other.txt', 'location.txt']
for item in dict_list:
    with open(os.path.abspath('../dict/' + item),'r') as f:
       bl = bl | ast.literal_eval(f.read())

### 5.2 Remove words from blacklist ###

for i in range(tmp_data.shape[0]):
    if tmp_data.iloc[i,1] == 1:
        s_tmp = set(tmp_data.iloc[i,0].split() + [tmp_data.iloc[i,0]])
        if s_tmp & bl :
            tmp_data.iloc[i,1] = 0

### 5.3 Create a name dictionary ###

# If we find a word which is already in our name dictionary,
# We predict it to be a name
idx = []
for i in range(tmp_data.shape[0]):
    if tmp_data.iloc[i,1] * tmp_data.iloc[i,2]:
        idx.append(i)

name_already = set(df.iloc[idx,0])


###### 6. MODEL TESTING ######

### 6.1 Read in test data ###

df_test = pd.DataFrame()
for i in range(301,451):
    path = os.path.abspath('../Set_J/' + str(i) + '.txt')
    with open(path, 'r') as f:
        text = f.read()
        text = re.sub('\"', "", text)
        df_this_file = pd.DataFrame()
        s, text = pre_cleaning(text)
        for sent in nltk.sent_tokenize(text):
            df_tmp = find_feature(nltk.word_tokenize(sent), s)
            df_this_file = pd.concat([df_this_file, df_tmp])
    df_test = pd.concat([df_test, df_this_file])


df_x = df_test.iloc[:,:df_test.shape[1] - 1]
X_test, y_test = pd.get_dummies(df_x.iloc[:,1:]), df_test['label']

### 6.2 Model predicting ###

# Use the model trained above with train data
y_pred = clf.predict(X_test)

def show_result(y_test, y_pred):
    print('Confusion Matrix: ')
    print(confusion_matrix(y_test, y_pred))
    print('Precision: ')
    print(precision_score(y_test, y_pred))
    print('Recall: ')
    print(recall_score(y_test, y_pred))
    print('F1: ')
    print(f1_score(y_test, y_pred))

show_result(y_test, y_pred)


# We mark all words in our name dictionary (from train data) as a name

for i in range(len(y_pred)):
    if y_pred[i] == 0:
        if df_test['cand'].iloc[i] in name_already:
            y_pred[i] = 1

# Remove blacklist words
for i in range(len(y_pred)):
    if y_pred[i] == 1:
        s_tmp = set(df_test['cand'].iloc[i].split() + [df_test['cand'].iloc[i]])
        if s_tmp & bl :
            y_pred[i] = 0

show_result(y_test, y_pred)