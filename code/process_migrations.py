"""
The code used for extracting migration metaphors from media discourse.
Requires the komet corpus of metaphors for training data (https://www.clarin.si/repository/xmlui/handle/11356/1293) and (optionally) the SloIE corpus of idioms (https://www.clarin.si/repository/xmlui/handle/11356/1335)
"""

# DATA_DICT should point to the komet corpus.
DATA_DICT = '../komet.tei'

# SLOIE_DATA_DICT should point to the location of the  SloIE corpus
SLOIE_DATA_DICT = '../sloie'

import json
from transformers import AutoModel, AutoTokenizer, CamembertForSequenceClassification, CamembertTokenizer
import os
import numpy as np
from xml.etree import ElementTree as ET
import tensorflow as tf
import torch
import torch.nn.functional as F
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch.autograd.profiler as profiler
import gc
import sys
from collections import Counter
from old_dataset import process_old_dataset
import nltk



TYPE_DICT = { 'MRWd': 1,  #(direktna metafora)
              'MRWi': 2,  #(implicitna metafora)
              'WIDLI': 3, #(mejni primer) 
              'MFlag': 4,  #(metaforični signalizator)}
              'NoType': 5   # No type found, but was still labelled as seg
              }
              
REVERSE_TYPE_DICT = { 1: 'MRWd',  #(direktna metafora)
                      2: 'MRWi',  #(implicitna metafora)
                      3: 'WIDLI', #(mejni primer) 
                      4: 'MFlag',  #(metaforični signalizator)}
                      5: 'NoType'    # No type found, but was still labelled as seg
                    }


def read_json(filename):
    ret_lines = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            processed_line = json.loads(line)
            #print(processed_line['Content'])
            ret_lines.append(processed_line['Content'])
    return ret_lines
        


def get_correct_type(input_type):
    for t in TYPE_DICT.keys():
        if t in input_type:
            return t
    return 'NoType'


def get_specific_classes(Y, cls_num):
    ret = []
    for sent_classes in Y:
        has_cls = 0
        curr_number_sent_classes = 0
        for word_classes in sent_classes:
            if word_classes != []:
                for c in word_classes:
                    if TYPE_DICT[c] == cls_num:
                        has_cls = 1
        ret.append(has_cls)
    return ret
    
    
def split_dataset_by_mask(X, Y, mask):
    return_x_1 = []
    return_x_2 = []
    return_y_1 = []
    return_y_2 = []
    negative_x = []
    negative_y = []
    for x_row, y_row, mask_item in zip(X, Y, mask):
        if mask_item == 1:
            return_x_1.append(x_row)
            return_y_1.append(y_row)
        elif mask_item == 0:
            return_x_2.append(x_row)
            return_y_2.append(y_row)
        elif mask_item == 2:
            negative_x.append(x_row)
            negative_y.append(y_row)
    return return_x_1, return_y_1, return_x_2, return_y_2, negative_x, negative_y
    
    
def get_classified_word_indices(Y, cls_num):
    ret = []
    for sent_classes in Y:
        indices = []
        #has_cls = 0
        #curr_number_sent_classes = 0
        for wc_index, word_classes in enumerate(sent_classes):
            if word_classes != []:
                for c in word_classes:
                    if TYPE_DICT[c] == cls_num:
                        indices.append(wc_index)
        ret.append(indices)
    return ret


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def get_non_verb_noun(document_metaphor_classes):
    return_list = []
    for sent_metaphor_classes in document_metaphor_classes:
        if sent_metaphor_classes == []:
            return_list.append(2)
        else:
            has_noun = 1 if 'NOUN' in sent_metaphor_classes else 0
            has_verb = 1 if 'VERB' in sent_metaphor_classes else 0
            has_noun_or_verb = has_noun or has_verb
            return_list.append(has_noun_or_verb)
    return return_list


def count_classes_per_sentence(Y):
    ret = []
    for sent_classes in Y:
        curr_number_sent_classes = 0
        for word_classes in sent_classes:
            if word_classes != []:
                curr_number_sent_classes += 1
        ret.append(curr_number_sent_classes)
    return ret
        

class CustomBERTModel(nn.Module):
    def __init__(self):
          super(CustomBERTModel, self).__init__()
          self.bert = AutoModel.from_pretrained('../sloberta2/')
          ### New layers:
          self.linear1 = nn.Linear(768, 256)
          self.linear2 = nn.Linear(256, 2) ## 3 is the number of classes in this example

    def forward(self, input_ids, attention_mask):
          #print('input ids', input_ids)
          #print('attention mask', attention_mask)
          results = self.bert(input_ids, attention_mask)
          
          #print('results', results)
          sequence_output = results[0]
          pooled_output = results[1]
          # sequence_output has the following shape: (batch_size, sequence_length, 768)
          #print('pooled_output', pooled_output)
          #print('sequence output', sequence_output)
          #print('sequence_output[:,0,:]', sequence_output[:,0,:])
          #print('sequence_output[:,0,:].view(-1,768)', sequence_output[:,0,:].view(-1,768))
          linear1_output = self.linear1(sequence_output[:,0,:].view(-1,768)) ## extract the 1st token's embeddings

          linear2_output = self.linear2(linear1_output)

          return linear2_output


    

sentences = []
document_pos_tags = []




for filename in os.listdir(DATA_DICT):
    if filename.endswith(".xml"): 
        tree = ET.parse(os.path.join(DATA_DICT, filename))
        print(filename)
        root = tree.getroot()
        children  = list(root.iter())
        sentence_words = []
        sentence_classes = []
        sentence_pos_tags = []
        tag_next_word = []
        for c in children:
            tag = c.tag.split('}')[1]
            #print(c.tag.split('}')[1])
            if tag == 's' and sentence_words != []:
                sentences.append((sentence_words, sentence_classes))
                document_pos_tags.append(sentence_pos_tags)
                sentence_pos_tags = []
                sentence_words = []
                sentence_classes = []
                #print('---------------')
            if tag == 'w':
                #if c.text == "šoku":
                #    print(c.text, c.attrib['msd'].split('|')[0].split('=')[1])
                    #exit()
                sentence_words.append(c.text)
                sentence_classes.append(tag_next_word)
                tag_next_word = []
                sentence_pos_tags.append(c.attrib['msd'].split('|')[0].split('=')[1])
                #print(c.attrib['msd'].split('|')[0].split('=')[1])
                #print(c.text)
            if tag == 'seg':
                tag_next_word.append(c.attrib)
                #print(c.attrib)
        sentences.append((sentence_words, sentence_classes))
        document_pos_tags.append(sentence_pos_tags)
        sentence_pos_tags = []
        sentence_words = []
        sentence_classes = []
        #break
#exit()
#print(document_pos_tags)



X = []
Y = []

document_classes_mask = []
for s in sentences:
    words = s[0]
    X.append(words)
    classes = s[1]
    #print('classes', classes)
    processed_classes = []
    sent_classes_mask = []
    #print(classes)
    for c in classes:
        if c == []:
            processed_classes.append([])
            sent_classes_mask.append(0)
        else:
            sent_classes_mask.append(1)
            current_classes = []
            for subc in c:
                if 'subtype' in subc.keys():
                    current_classes.append(get_correct_type(subc['subtype']))
            if current_classes == []:
                current_classes = ['#met.no_subclass']
            processed_classes.append(current_classes)
            
    #processed_classes = [0 if c == [] else 1 for c in classes]
    #if 'Channing' in words:
    #    print(words)
    #    print(classes)
    #    print(processed_classes)
    #    print(sent_classes_mask)
    #    exit()
    Y.append(processed_classes)
    document_classes_mask.append(sent_classes_mask)
#print(document_classes_mask)



document_metaphor_classes = []
for sent_words, sent_tags, sent_mask in zip(sentences, document_pos_tags, document_classes_mask):
    words = sent_words[0]
    sent_metaphor_classes = []
    #if "Channing" in words:
    #    print(sent_tags, sent_mask)
    #    exit()
    for word_tag, word_mask in zip(sent_tags, sent_mask):
        if word_mask == 1:
            sent_metaphor_classes.append(word_tag)
    document_metaphor_classes.append(sent_metaphor_classes)
    sent_metaphor_classes = []
#print(document_metaphor_classes)
X_sentences = [' '.join(x) for x in X]

#print(Y)
#exit()
#print(X[:10])
#print(Y[:10])
#print(len(Y))
num_classes = count_classes_per_sentence(Y)
#print('num_classes', num_classes)
#for y, c in zip(Y, num_classes):
#    print(y, c)
    
    

flattened_Y = flatten(flatten(Y))
classes_set = set(flattened_Y)
print('------------')
for y in classes_set:
    print(y)
print('------------')
num_pos = 0


specific_classes_Y = []

for i in range(1, 6):
    specific_classes = get_specific_classes(Y, i)
    print('class', REVERSE_TYPE_DICT[i], 'had', sum(specific_classes), 'positive classes, default CA =', sum(specific_classes) / len(specific_classes))
    num_pos += sum(specific_classes)
    classes_to_append = [i*y for y in specific_classes]
    if specific_classes_Y == []:
        specific_classes_Y = [[y] for y in classes_to_append]
    else:
        for yi in range(len(classes_to_append)):
            if classes_to_append[yi] not in specific_classes_Y[yi] and specific_classes_Y[yi] == [0]:
                specific_classes_Y[yi] = [classes_to_append[yi]]
            elif classes_to_append[yi] not in specific_classes_Y[yi] and specific_classes_Y[yi] != [0]:
                specific_classes_Y[yi].append(classes_to_append[yi])
            else:
                pass
                


# Probajmo z direktnimi + indirektnimi metaforami
specific_classes_direct = get_specific_classes(Y, 1)
specific_classes_indirect = get_specific_classes(Y, 2)
specific_classes_edge_case = get_specific_classes(Y, 3)
specific_classes_mflag = get_specific_classes(Y, 4)
specific_classes_notype = get_specific_classes(Y, 5)
classified_indices_direct = get_classified_word_indices(Y, 1)
classified_indices_indirect = get_classified_word_indices(Y, 2)
#simplified_Y = [ x | y for (x,y) in zip(specific_classes_direct, specific_classes_indirect)]
print(Y)
print('1')
print(sum(specific_classes_direct))
print(sum(specific_classes_indirect))
print(sum(specific_classes_edge_case))
print(sum(specific_classes_mflag))
print(sum(specific_classes_notype))

"""
Start with direct classes.
This is a very small dataset, might need to start with something higher
"""
#simplified_Y = specific_classes_direct
#simplified_Y = specific_classes_indirect
simplified_Y = [a or b for a, b in zip(specific_classes_direct, specific_classes_indirect)]
print('class', 'MRWd+MRWi', 'had', sum(simplified_Y), 'positive classes, default CA =', sum(simplified_Y) / len(simplified_Y))
#print('total had', num_pos, 'positive classes, default CA =', num_pos / len(Y))
#print(specific_classes_Y)
#exit()
#simplified_Y = [1 if y > 0 else 0 for y in num_classes]

X_sentences = [' '.join(x) for x in X]

#for x, y in zip(X_sentences, simplified_Y):
#    print(x, y)

i = 0
for sent, sent_metaphor_classes, sy in zip(X_sentences, document_metaphor_classes, simplified_Y):
    #print(sent, sy, sent_metaphor_classes)
    if sy == 0 and sent_metaphor_classes != []:
        #print(sent, sy, sent_metaphor_classes) 
        document_metaphor_classes[i] = []
    i+= 1

#exit()
# 0 indicates no noun or verb, 1 indicates at least one noun or verb, 2 indicates no metaphors at all
non_verb_noun_list = get_non_verb_noun(document_metaphor_classes)
print(non_verb_noun_list)

print('non_verb_noun_list', Counter(non_verb_noun_list))
print('simplified_Y', Counter(simplified_Y))


"""
If we want to properly evaluate this, we presumably need to split into training and test set before limiting ourselves
to a speficic subset
"""
X_train_komet, X_test_komet, \
Y_train_komet, Y_test_komet, \
non_verb_noun_list_train_komet, non_verb_noun_list_test_komet, \
classified_indices_direct_train, classified_indices_direct_test, \
classified_indices_indirect_train, classified_indices_indirect_test = train_test_split(X_sentences, simplified_Y, non_verb_noun_list, classified_indices_direct, classified_indices_indirect, train_size=0.8)

"""
Overwrite X_test, Y_test with data from the old dataset
"""

# The argument should be the sloie folder
old_x, old_y = process_old_dataset(SLOIE_DATA_DICT)
old_x = [' '.join(x) for x in old_x]
#print(X_test[0])
#print(old_x[0])
#exit()
old_x_train, old_x_test, old_y_train, old_y_test = train_test_split(old_x, old_y, train_size=0.8, shuffle=True)



#X_train = old_x_train
#Y_train = old_y_train


"""
Afterwards, we ignore X_test and Y_test until the final evaluation
"""

# Lets see how this works if we remove metaphors that do not contain a noun or verb
# Keep metaphors with noun/verb and non-metaphors


#keep_noun_verb_mask_train = [1 if n == 1 or n == 2 else 0 for n in non_verb_noun_list_train]
X_nv_train_komet, Y_nv_train_komet, X_nnv_train_komet, Y_nnv_train_komet, negative_x_train_komet, negative_y_train_komet = split_dataset_by_mask(X_train_komet, Y_train_komet, non_verb_noun_list_train_komet)



X_nv_train = X_nv_train_komet + negative_x_train_komet 
Y_nv_train = Y_nv_train_komet + negative_y_train_komet 
#X_nnv_train = X_nv_train_komet + X_nnv_train_komet + negative_x_train_komet
#Y_nnv_train = Y_nv_train_komet + Y_nnv_train_komet + negative_y_train_komet
#X_nv_train = X_train + X_nv_train_komet + X_nnv_train_komet + negative_x_train_komet
#Y_nv_train = Y_train + Y_nv_train_komet + Y_nnv_train_komet + negative_y_train_komet
#X_nnv_train = []
#Y_nnv_train = []

#print(X_nv_train[:10])
#print(Y_nv_train[:10])

#print(old_x_train[:10])
#print(old_y_train)


X_nv_train = X_nv_train[:int(len(X_nv_train)/1)]
Y_nv_train = Y_nv_train[:int(len(Y_nv_train)/1)]

old_x_train = old_x_train[:int(len(X_nv_train)/1)]
old_y_train = old_y_train[:int(len(X_nv_train)/1)]
#X_nv_train = X_nv_train 
X_nv_train = X_nv_train
#Y_nv_train = Y_nv_train 
Y_nv_train = Y_nv_train

X_nnv_train = X_nnv_train_komet
Y_nnv_train = Y_nnv_train_komet

X_nv_train, Y_nv_train = shuffle(X_nv_train, Y_nv_train)
X_nnv_train, Y_nnv_train = shuffle(X_nnv_train, Y_nnv_train)

X_test = X_test_komet
Y_test = Y_test_komet

test_data = read_json('SloNews-immigration.jsonl')

new_test_data = []
for text in test_data:
    sents = nltk.sent_tokenize(text)
    sents = [x.replace('\n', ' ') for x in sents]
    new_test_data += sents
    
#print(new_test_data[:10])
#exit()

test_X_migracije = new_test_data

print(test_X_migracije[:10])

"""
Lets add negative examples to nv_train and nnv_train
half half to each? Should do for a start
TODO - maybe try with a balanced dataset
"""
"""
X_nv_train += negative_x_train[:int(len(negative_x_train)/2)]
Y_nv_train += negative_y_train[:int(len(negative_y_train)/2)]
X_nnv_train += negative_x_train[int(len(negative_x_train)/2):]
Y_nnv_train += negative_y_train[int(len(negative_y_train)/2):]

X_nv_train, Y_nv_train = shuffle(X_nv_train, Y_nv_train)
X_nnv_train, Y_nnv_train = shuffle(X_nnv_train, Y_nnv_train)
"""


# Classified indices, used to see which metaphors are direct and which are not
# Might be worth keeping
"""
TODO - add classified indices back
"""
#ci_nv_train, cd_nv_train, ci_nnv_train, cd_nnv_train = split_dataset_by_mask(classified_indices_indirect_train, classified_indices_direct_train, keep_noun_verb_mask_train)

#print('Y_nv', Counter(Y_nv))
#print('ci_nv', ci_nv)

#print(simplified_Y)




#exit()
print('loading model')
#slo_bert = AutoModel.from_pretrained('./sloberta2/')



#test_sentence = 'To je testna poved, poglejmo kaj se z njo zgodi'

print(Counter(Y_test))
#exit()

# split sentences into batches:
#batches_X = [X_sentences[i:i + 8] for i in range(0, len(X_sentences), 8)]
train_batches_X_nv = [X_nv_train[i:i + 8] for i in range(0, len(X_nv_train), 8)]
train_batches_Y_nv = [Y_nv_train[i:i + 8] for i in range(0, len(Y_nv_train), 8)]
train_batches_X_old = [old_x_train[i:i + 8] for i in range(0, len(old_x_train), 8)]
train_batches_Y_old = [old_y_train[i:i + 8] for i in range(0, len(old_y_train), 8)]
train_batches_X_nnv = [X_nnv_train[i:i + 8] for i in range(0, len(X_nnv_train), 8)]
train_batches_Y_nnv = [Y_nnv_train[i:i + 8] for i in range(0, len(Y_nnv_train), 8)]

test_batches_X = [X_test[i:i + 8] for i in range(0, len(X_test), 8)]
test_batches_Y = [Y_test[i:i + 8] for i in range(0, len(Y_test), 8)]

test_X_migracije_batches = [test_X_migracije[i:i + 8] for i in range(0, len(test_X_migracije), 8)]



print('Y test', Counter(Y_test))
print('Y train', Counter(Y_nv_train))


#batches_classified_indices_direct_train = [cd_nv_train[i:i + 8] for i in range(0, len(cd_nv_train), 8)]
#batches_classified_indices_indirect_train = [ci_nv_train[i:i + 8] for i in range(0, len(ci_nv_train), 8)]
running_loss = 0.0
total_batches = len(X_nv_train)/8
#train_batches_X, test_batches_X, train_batches_Y, test_batches_Y, train_direct_i, test_direct_i, train_indirect_i, test_indirect_i = train_test_split(batches_X, batches_Y, batches_classified_indices_direct, batches_classified_indices_indirect, train_size=0.8, shuffle=True)
print('total batches', total_batches)
i = 0
#for x, direct, indirect in zip(test_batches_X, test_direct_i, test_indirect_i):
#    for xx, dd, ii in zip(x, direct, indirect):
#        print(xx, dd, ii)

#model = CustomBERTModel()
model = CamembertForSequenceClassification.from_pretrained('../sloberta2/', num_labels=2)
model.train()
device = torch.device("cuda")
#device = torch.device("cpu")
model.to(device)
#criterion = nn.CrossEntropyLoss()
#criterion.to(device)
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=1e-5)
#optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
#slo_tokenizer = AutoTokenizer.from_pretrained('./sloberta2/')
slo_tokenizer = CamembertTokenizer.from_pretrained('../sloberta2/')
#slo_tokenizer.to(device)
running_loss = 0
"""
Lets do a test run by letting the model train 3 times
"""

# Train on idioms

print('Training on SloIE')
for epoch in range(3):
    print('epoch', epoch)
    for batch_X, batch_Y in zip(train_batches_X_old, train_batches_Y_old):
        i += 1
        pt_batch = slo_tokenizer(
                    batch_X,
                    padding=True,
                    truncation=True,
                    max_length=64,
                    return_tensors="pt").to(device)          
        optimizer.zero_grad()
        targets = torch.tensor(batch_Y).to(device)
        outputs = model(**pt_batch, labels=targets)
        del pt_batch
        del targets
        loss = outputs.loss
        loss.backward()
        running_loss += loss.item()
        if i % 30 == 0:
            print(i, 'loss', running_loss/30, file=sys.stderr)
            running_loss = 0
        optimizer.step()
        del loss
        gc.collect()
        torch.cuda.ipc_collect()
for current_iteration in range(3):
    print('------------------------------', file=sys.stderr)
    print('ITERATION', current_iteration, file=sys.stderr)
    print('------------------------------', file=sys.stderr)
    # Train on komet
    print('Training on Komet')
    for epoch in range(1):
        print('epoch', epoch)
        for batch_X, batch_Y in zip(train_batches_X_nv, train_batches_Y_nv):
            i += 1
            pt_batch = slo_tokenizer(
                        batch_X,
                        padding=True,
                        truncation=True,
                        max_length=64,
                        return_tensors="pt").to(device)          
            optimizer.zero_grad()
            targets = torch.tensor(batch_Y).to(device)
            outputs = model(**pt_batch, labels=targets)
            del pt_batch
            del targets
            loss = outputs.loss
            loss.backward()
            running_loss += loss.item()
            if i % 30 == 0:
                print(i, 'loss', running_loss/30, file=sys.stderr)
                running_loss = 0
            optimizer.step()
            del loss
            gc.collect()
            torch.cuda.ipc_collect()
            #break
    """
    After training, we want to see which metaphors that were not yet included
    in the training set the model did well on and add those to the training set
    """
    with torch.no_grad():
        CAs = []
        combined_results = []
        for i in range(20):
            loop_results = []
            print(i)
            j = 0
            for test_batch_X, test_batch_Y in zip(train_batches_X_nnv, train_batches_Y_nnv):
                test_encodings = slo_tokenizer(
                                    test_batch_X,
                                    padding=True,
                                    truncation=True,
                                    max_length=512,
                                    return_tensors="pt").to(device)
                preds = model(**test_encodings)
                preds = preds.logits.tolist()
                preds = np.argmax(preds, axis=1)
                #while len(test_batch_Y) < 8:
                #    test_batch_Y.append(2)
                loop_results += list(preds)
                # This drops some batches
                # Should be okay, this intermediate CA doesn't really mean much
                if len(preds) == len( test_batch_Y):
                    CAs.append(sum(preds==test_batch_Y)/len(preds))
                #print(i, j, '/', len(train_batches_X_nnv), file=sys.stderr)
                j += 1
            combined_results.append(loop_results)
            loop_results = []
        """
        Let's see which examples the model is sure in
        Let's first try with defining sure as either > 0.95 or < 0.05
        """
        print('remaining train CAs is', np.mean(CAs))
        train_batches_X_nnv_flat = flatten(train_batches_X_nnv)
        train_batches_Y_nnv_flat = flatten(train_batches_Y_nnv)
        sure_results_pos = 0
        sure_results_neg = 0
        all_average_results = []
        for i in range(len(combined_results[0])):
            average_result = sum([x[i] for x in combined_results])/len(combined_results)
            all_average_results.append(average_result)
        print(len(all_average_results), len(train_batches_X_nnv_flat), Y_nnv_train, train_batches_Y_nnv)
        #exit()
        sure_indices = []
        curr_i = 0
        for x, result in zip(train_batches_X_nnv_flat, all_average_results):
            if result >= 0.99:
                #print(x, result)
                sure_indices.append(curr_i)
                sure_results_pos += 1
            elif result <= 0.01:
                #print(x, result)
                sure_indices.append(curr_i)
                sure_results_neg += 1
            #else:
            #    print(x, result)
            curr_i += 1
        print(len(train_batches_X_nnv_flat))
        print(len(train_batches_Y_nnv_flat))
        sure_X_elements_to_add = [train_batches_X_nnv_flat[i] for i in sure_indices]
        sure_Y_elements_to_add = [train_batches_Y_nnv_flat[i] for i in sure_indices]
        print('sure elements', len(sure_X_elements_to_add), len(sure_Y_elements_to_add), sure_indices)
        remaining_elements_X = [train_batches_X_nnv_flat[i] for i in range(len(train_batches_X_nnv_flat)) if i not in sure_indices]
        remaining_elements_Y = [train_batches_Y_nnv_flat[i] for i in range(len(train_batches_Y_nnv_flat)) if i not in sure_indices]
        if len(remaining_elements_X) != len(remaining_elements_Y):
            print('remaining elements', len(remaining_elements_X), len(remaining_elements_Y))
            exit()
        
        new_train_batches_X_sure = [sure_X_elements_to_add[i:i + 8] for i in range(0, len(sure_X_elements_to_add), 8)]
        new_train_batches_Y_sure = [sure_Y_elements_to_add[i:i + 8] for i in range(0, len(sure_Y_elements_to_add), 8)]
        
        train_batches_X_nv += new_train_batches_X_sure
        train_batches_Y_nv += new_train_batches_Y_sure
        
        train_batches_X_nnv = [remaining_elements_X[i:i + 8] for i in range(0, len(remaining_elements_X), 8)]
        train_batches_Y_nnv = [remaining_elements_Y[i:i + 8] for i in range(0, len(remaining_elements_Y), 8)]
        print('lengths', len(sure_X_elements_to_add), len(remaining_elements_X), len(train_batches_X_nnv_flat))
        print(sure_results_neg + sure_results_pos, 'sure results out of', len(all_average_results))
        print('sure_results_pos', sure_results_pos)
        print('sure_results_neg', sure_results_neg)
        #exit()
            
        
    
    
torch.cuda.empty_cache()
# flatten testing batches:
#flat_test_batches_X = [item for sublist in test_batches_X for item in sublist]
#print('test batches', test_batches_X)
#print('flattened test batches', flat_test_batches_X)
#combined_preds = []
#model.to('cpu')
CAs = []
tp_sentences = []
tn_sentences = []
fp_sentences = []
fn_sentences = []
combined_results = []
final_preds = []
# Lets test if monte-carlo dropout works just by leaving the model in train mode
# If it does, re-running the model multiple times should produce different results
with torch.no_grad():
    for i in range(1):
        loop_results = []
        print(i)
        j = 0
        #for test_batch_X, test_batch_Y, direct_i, indirect_i in zip(test_batches_X, test_batches_Y, test_direct_i, test_indirect_i):
        for test_batch_X, test_batch_Y in zip(test_batches_X, test_batches_Y):
            test_encodings = slo_tokenizer(
                                test_batch_X,
                                padding=True,
                                truncation=True,
                                max_length=512,
                                return_tensors="pt").to(device)
            preds = model(**test_encodings)
            preds = preds.logits.tolist()
            preds = np.argmax(preds, axis=1)
            loop_results += list(preds)
            CAs.append(sum(preds==test_batch_Y)/len(preds))
            #print(i, j, '/', len(test_batches_X), file=sys.stderr)
            j += 1
        combined_results.append(loop_results)
        loop_results = []

"""
print(len(combined_results), len(combined_results[0]))

with open('results_monte_carlo.txt', 'w', encoding='utf-8') as outf:
    for i in range(len(combined_results[0])):
        print(sum([x[i] for x in combined_results])/len(combined_results), test_batches_X[int(i/len(test_batches_X))][int(i%len(test_batches_X))])  
        print(sum([x[i] for x in combined_results])/len(combined_results), test_batches_X[int(i/len(test_batches_X))][int(i%len(test_batches_X))], file=outf)

print([x[0] for x in combined_results], test_batches_X[0][0])
print([x[1] for x in combined_results], test_batches_X[0][1])
print([x[2] for x in combined_results], test_batches_X[0][2])
print([x[3] for x in combined_results], test_batches_X[0][3])
print([x[4] for x in combined_results], test_batches_X[0][4])
print([x[5] for x in combined_results], test_batches_X[0][5])
print([x[6] for x in combined_results], test_batches_X[0][6])
print([x[7] for x in combined_results], test_batches_X[0][6])
"""
all_average_results = []
for i in range(len(combined_results[0])):
    average_result = sum([x[i] for x in combined_results])/len(combined_results)
    all_average_results.append(average_result)

#print(final_preds)
#print(Y_test)
#print('Better final CAs', sum(final_preds == Y_test)/len(Y_test))
print('Y test', Counter(Y_test))
print('Y train', Counter(Y_nv_train))

small_pos = 0
small_neg = 0
big_pos = 0
big_neg = 0
TP = 0
TN = 0
FP = 0
FN = 0
for result, x, y in zip(all_average_results, X_test, Y_test):
    #print(x, result)
    #print(x.encode('ascii', 'ignore'), result)
    binary_result = 1 if result > 0.5 else 0
    if result >= 0.95 or result <= 0.05:
        if result == y:
            small_pos += 1
        else:
            small_neg += 1
    if binary_result == y:
        big_pos += 1
    else:
        big_neg += 1
    if binary_result == 1 and y == 1:
        TP += 1
    if binary_result == 1 and y == 0:
        FP += 1
    if binary_result == 0 and y == 1:
        FN += 1
    if binary_result == 0 and y == 0:
        FP += 1
precision = TP / (TP+FP)
recall = TP / (TP+FN)  
print('small CA', small_pos/(small_pos+small_neg))
print(small_pos, 'out of', len(Y_test))
print('big CA', big_pos/(big_pos+small_neg))
print('Final CAs', sum(CAs)/len(CAs))
print('Default classifier', sum(Y_test)/len(Y_test))
print('precision', precision)
print('recall', recall)
#print('all CAs', CAs)
print('Y test', Counter(Y_test))
print('Y train', Counter(Y_nv_train))

exit()





    
combined_results = []
print('FINAL PREDICTIONS')
with torch.no_grad():
    for i in range(1):
        loop_results = []
        print(i)
        j = 0
        #for test_batch_X, test_batch_Y, direct_i, indirect_i in zip(test_batches_X, test_batches_Y, test_direct_i, test_indirect_i):
        for test_batch_X in test_X_migracije_batches:
            #print(test_batch_X)
            test_encodings = slo_tokenizer(
                                test_batch_X,
                                padding=True,
                                truncation=True,
                                max_length=512,
                                return_tensors="pt").to(device)
            preds = model(**test_encodings)
            preds = preds.logits.tolist()
            preds = np.argmax(preds, axis=1)
            loop_results += list(preds)
            print(i, j, '/', len(test_X_migracije_batches), file=sys.stderr)
            j += 1
        combined_results.append(loop_results)
        loop_results = []    


with open('./final_results_50_sents.txt', 'w', encoding='utf-8') as outf:
    #print(combined_results, file=outf)
    for i, sent in enumerate(new_test_data):
        total_score = 0
        for j in range(len(combined_results)):
            total_score += combined_results[j][i]
        total_score = total_score / len(combined_results)
        print(sent.replace('\n', ' '), total_score, file=outf)


