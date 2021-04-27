import gensim.models
import numpy as np
from tqdm import tqdm
import csv
from scipy.sparse import csr_matrix
import gensim.models.word2vec as w2v
import gensim.models.fasttext as fasttext
import codecs
import struct
import re
import operator
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset
from elmo.elmo import batch_to_ids


class ProcessedIter(object):

    def __init__(self, Y, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename) as f:
            r = csv.reader(f)
            next(r)
            for row in r:
                yield (row[3].split())

def build_vocab(vocab_min, infile, vocab_filename):
    """
        INPUTS:
            vocab_min: how many documents a word must appear in to be kept
            infile: (training) data file to build vocabulary from
            vocab_filename: name for the file to output
    """
    with open(infile, 'r') as csvfile:
        reader = csv.reader(csvfile)
        # header
        next(reader)

        # 0. read in data
        print("reading in data...")
        # holds number of terms in each document
        note_numwords = []
        # indices where notes start
        note_inds = [0]
        # indices of discovered words
        indices = []
        # holds a bunch of ones
        data = []
        # keep track of discovered words
        vocab = {}
        # build lookup table for terms
        num2term = {}
        # preallocate array to hold number of notes each term appears in
        note_occur = np.zeros(400000, dtype=int)
        i = 0
        for row in reader:
            text = row[2]
            numwords = 0
            for term in text.split():
                # put term in vocab if it's not there. else, get the index
                index = vocab.setdefault(term, len(vocab))
                indices.append(index)
                num2term[index] = term
                data.append(1)
                numwords += 1
            # record where the next note starts
            note_inds.append(len(indices))
            indset = set(indices[note_inds[-2]:note_inds[-1]])
            # go thru all the word indices you just added, and add to the note occurrence count for each of them
            for ind in indset:
                note_occur[ind] += 1
            note_numwords.append(numwords)
            i += 1
        # clip trailing zeros
        note_occur = note_occur[note_occur > 0]

        # turn vocab into a list so indexing doesn't get fd up when we drop rows
        vocab_list = np.array([word for word, ind in sorted(vocab.items(), key=operator.itemgetter(1))])

        # 1. create sparse document matrix
        C = csr_matrix((data, indices, note_inds), dtype=int).transpose()
        # also need the numwords array to be a sparse matrix
        note_numwords = csr_matrix(1. / np.array(note_numwords))

        # 2. remove rows with less than 3 total occurrences
        print("removing rare terms")
        # inds holds indices of rows corresponding to terms that occur in < 3 documents
        inds = np.nonzero(note_occur >= vocab_min)[0]
        print(str(len(inds)) + " terms qualify out of " + str(C.shape[0]) + " total")
        # drop those rows
        C = C[inds, :]
        note_occur = note_occur[inds]
        vocab_list = vocab_list[inds]

        print("writing output")
        with open(vocab_filename, 'w') as vocab_file:
            for word in vocab_list:
                vocab_file.write(word + "\n")


def concat_data(labelsfile, notes_file, outfilename):
    """
        INPUTS:
            labelsfile: sorted by hadm id, contains one label per line
            notes_file: sorted by hadm id, contains one note per line
    """
    with open(labelsfile, 'r') as lf:
        print("CONCATENATING")
        with open(notes_file, 'r') as notesfile:

            with open(outfilename, 'w') as outfile:
                w = csv.writer(outfile)
                w.writerow(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS'])

                labels_gen = next_labels(lf)
                notes_gen = next_notes(notesfile)

                for i, (subj_id, text, hadm_id) in enumerate(notes_gen):
                    if i % 10000 == 0:
                        print(str(i) + " done")
                    cur_subj, cur_labels, cur_hadm = next(labels_gen)

                    if cur_hadm == hadm_id:
                        w.writerow([subj_id, str(hadm_id), text, ';'.join(cur_labels)])
                    else:
                        print("couldn't find matching hadm_id. data is probably not sorted correctly")
                        break
    return outfilename

def split_data(labeledfile, base_name, mimic_dir):
    print("SPLITTING")
    #create and write headers for train, dev, test
    train_name = '%s_train_split.csv' % (base_name)
    dev_name = '%s_dev_split.csv' % (base_name)
    test_name = '%s_test_split.csv' % (base_name)
    train_file = open(train_name, 'w')
    dev_file = open(dev_name, 'w')
    test_file = open(test_name, 'w')
    train_file.write(','.join(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS']) + "\n")
    dev_file.write(','.join(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS']) + "\n")
    test_file.write(','.join(['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS']) + "\n")

    hadm_ids = {}

    #read in train, dev, test splits
    for splt in ['train', 'dev', 'test']:
        hadm_ids[splt] = set()
        with open('%s/%s_full_hadm_ids.csv' % (mimic_dir, splt), 'r') as f:
            for line in f:
                hadm_ids[splt].add(line.rstrip())

    with open(labeledfile, 'r') as lf:
        reader = csv.reader(lf)
        next(reader)
        i = 0
        cur_hadm = 0
        for row in reader:
            #filter text, write to file according to train/dev/test split
            if i % 10000 == 0:
                print(str(i) + " read")

            hadm_id = row[1]

            if hadm_id in hadm_ids['train']:
                train_file.write(','.join(row) + "\n")
            elif hadm_id in hadm_ids['dev']:
                dev_file.write(','.join(row) + "\n")
            elif hadm_id in hadm_ids['test']:
                test_file.write(','.join(row) + "\n")

            i += 1

        train_file.close()
        dev_file.close()
        test_file.close()
    return train_name, dev_name, test_name


def next_labels(labelsfile):
    """
        Generator for label sets from the label file
    """
    labels_reader = csv.reader(labelsfile)
    # header
    next(labels_reader)

    first_label_line = next(labels_reader)

    cur_subj = int(first_label_line[0])
    cur_hadm = int(first_label_line[1])
    cur_labels = [first_label_line[2]]

    for row in labels_reader:
        subj_id = int(row[0])
        hadm_id = int(row[1])
        code = row[2]
        # keep reading until you hit a new hadm id
        if hadm_id != cur_hadm or subj_id != cur_subj:
            yield cur_subj, cur_labels, cur_hadm
            cur_labels = [code]
            cur_subj = subj_id
            cur_hadm = hadm_id
        else:
            # add to the labels and move on
            cur_labels.append(code)
    yield cur_subj, cur_labels, cur_hadm


def next_notes(notesfile):
    """
        Generator for notes from the notes file
        This will also concatenate discharge summaries and their addenda, which have the same subject and hadm id
    """
    nr = csv.reader(notesfile)
    # header
    next(nr)

    first_note = next(nr)

    cur_subj = int(first_note[0])
    cur_hadm = int(first_note[1])
    cur_text = first_note[3]

    for row in nr:
        subj_id = int(row[0])
        hadm_id = int(row[1])
        text = row[3]
        # keep reading until you hit a new hadm id
        if hadm_id != cur_hadm or subj_id != cur_subj:
            yield cur_subj, cur_text, cur_hadm
            cur_text = text
            cur_subj = subj_id
            cur_hadm = hadm_id
        else:
            # concatenate to the discharge summary and move on
            cur_text += " " + text
    yield cur_subj, cur_text, cur_hadm


def load_vocab_dict(args, vocab_file):
    """
    Load vocabulary dictionary from file: vocab_file
    """
    vocab = set()
    with open(vocab_file, 'r') as vocabfile:
        for i, line in enumerate(vocabfile):
            line = line.rstrip()
            if line != '':
                vocab.add(line.strip())
    ind2w = {i + 1: w for i, w in enumerate(sorted(vocab))}
    w2ind = {w: i for i, w in ind2w.items()}
    return ind2w, w2ind


def load_full_codes(task, train_path, mimic2_dir, version='mimic3'):
    """
    Load full set of ICD codes
    """
    if version == 'mimic2':
        ind2c = defaultdict(str)
        codes = set()
        with open(mimic2_dir, 'r') as f:
            r = csv.reader(f)
            next(r) # skip header
            for row in r:
                if task == 'ICD9':
                    codes.update(set(row[-2].split(';')))
                else:
                    codes.update(set(row[-1].split(';')))
        codes = set([c for c in codes if c != ''])
        ind2c = defaultdict(str, {i:c for i,c in enumerate(sorted(codes))})
    else:
        codes = set()
        for split in ['train', 'dev', 'test']:
            with open(train_path.replace('train', split), 'r') as f:
                lr = csv.reader(f)
                next(lr)
                for row in lr:
                    if task == 'ICD9':
                        task_row = row[3]
                    else:
                        task_row = row[4]
                    for code in task_row.split(';'):
                        codes.add(code)
        codes = set([c for c in codes if c != ''])
        ind2c = defaultdict(str, {i:c for i,c in enumerate(sorted(codes))})
    return ind2c


def load_full_codes_MTL(train_path, mimic2_dir, version='mimic3'):
    """
    Load full set of ICD codes
    """
    if version == 'mimic2':
        ind2c = defaultdict(str)
        codes = set()
        codes_ccs = set()
        with open(mimic2_dir, 'r') as f:
            r = csv.reader(f)
            next(r) # skip header
            for row in r:
                codes.update(set(row[-2].split(';')))
                codes_ccs.update(set(row[-1].split(';')))
        codes = set([c for c in codes if c != ''])
        codes_ccs = set([c for c in codes_ccs if c != ''])
        ind2c = defaultdict(str, {i:c for i,c in enumerate(sorted(codes))})
        ind2c_ccs = defaultdict(str, {i:c for i,c in enumerate(sorted(codes_ccs))})
    else:
        codes = set()
        codes_ccs = set()
        for split in ['train', 'dev', 'test']:
            with open(train_path.replace('train', split), 'r') as f:
                lr = csv.reader(f)
                next(lr)
                for row in lr:
                    for code in row[3].split(';'):
                        codes.add(code)
                    for code1 in row[4].split(';'):
                        codes_ccs.add(code1)
        codes = set([c for c in codes if c != ''])
        codes_ccs = set([c for c in codes_ccs if c != ''])
        ind2c = defaultdict(str, {i:c for i,c in enumerate(sorted(codes))})
        ind2c_ccs = defaultdict(str, {i:c for i,c in enumerate(sorted(codes_ccs))})
    return ind2c, ind2c_ccs

def load_lookups(args, desc_embed=False):
    """
    Load lookup dictionaries: index2word, word2index, index2code, code2index 
    """
    ind2w, w2ind = load_vocab_dict(args, args.vocab)
    if args.Y == 'full':
        ind2c = load_full_codes(args.task, args.data_path, '%s/proc_dsums_m.csv' % args.MIMIC_2_DIR, version=args.version)
    else:
        codes = set()
        with open("%s/TOP_%s_CODES_m.csv" % (args.MIMIC_3_DIR, str(args.Y)), 'r') as labelfile:
            lr = csv.reader(labelfile)
            for i, row in enumerate(lr):
                if args.task == 'ICD9':
                    codes.add(row[0])
                else:
                    codes.add(row[1])
        ind2c = {i:c for i,c in enumerate(sorted(codes))}
        desc_dict = load_code_descriptions(args)
    c2ind = {c:i for i,c in ind2c.items()}

    if desc_embed:
        dv_dict = load_description_vectors(args, args.Y, version=args.version)
    else:
        dv_dict = None

    dicts = {'ind2w': ind2w, 'w2ind': w2ind, 'ind2c': ind2c, 'c2ind': c2ind, 'desc': desc_dict, 'dv': dv_dict}
    return dicts


def load_lookups_MTL(args, desc_embed=False):
    """
    Load lookup dictionaries: index2word, word2index, index2code, code2index 
    """
    ind2w, w2ind = load_vocab_dict(args, args.vocab)
    if args.Y == 'full':
        ind2c, ind2c_ccs = load_full_codes_MTL(args.data_path, '%s/proc_dsums_m.csv' % args.MIMIC_2_DIR, version=args.version)
    else:
        codes = set()
        codes_ccs = set()
        with open("%s/TOP_%s_CODES_m.csv" % (args.MIMIC_3_DIR, str(args.Y)), 'r') as labelfile:
            lr = csv.reader(labelfile)
            for i, row in enumerate(lr):
                codes.add(row[0])
                codes_ccs.add(row[1])
        ind2c = {i:c for i,c in enumerate(sorted(codes))}
        ind2c_ccs = {i:c for i,c in enumerate(sorted(codes_ccs))}
    c2ind = {c:i for i,c in ind2c.items()}
    c2ind_ccs = {c:i for i,c in ind2c_ccs.items()}
    dicts = {'ind2w': ind2w, 'w2ind': w2ind, 'ind2c': ind2c, 'c2ind': c2ind, 
    'ind2c_ccs': ind2c_ccs, 'c2ind_ccs': c2ind_ccs}
    return dicts

def reformat(code, is_diag):
    """
        Put a period in the right place because the MIMIC-3 data files exclude them.
        Generally, procedure codes have dots after the first two digits, 
        while diagnosis codes have dots after the first three digits.
    """
    code = ''.join(code.split('.'))
    if is_diag:
        if code.startswith('E'):
            if len(code) > 4:
                code = code[:4] + '.' + code[4:]
        else:
            if len(code) > 3:
                code = code[:3] + '.' + code[3:]
    else:
        code = code[:2] + '.' + code[2:]
    return code

def load_code_descriptions(args, version='mimic3'):
    #load description lookup from the appropriate data files
    desc_dict = defaultdict(str)
    if version == 'mimic2':
        with open('%s/MIMIC_ICD9_mapping' % MIMIC_2_DIR, 'r') as f:
            r = csv.reader(f)
            #header
            next(r)
            for row in r:
                desc_dict[str(row[1])] = str(row[2])
    else:
        with open("%s/D_ICD_DIAGNOSES.csv" % args.DATA_DIR, 'r') as descfile:
            r = csv.reader(descfile)
            #header
            next(r)
            for row in r:
                code = row[1]
                desc = row[-1]
                desc_dict[reformat(code, True)] = desc
        with open("%s/D_ICD_PROCEDURES.csv" % args.DATA_DIR, 'r') as descfile:
            r = csv.reader(descfile)
            #header
            next(r)
            for row in r:
                code = row[1]
                desc = row[-1]
                if code not in desc_dict.keys():
                    desc_dict[reformat(code, False)] = desc
        with open('%s/ICD9_descriptions' % args.DATA_DIR, 'r') as labelfile:
            for i,row in enumerate(labelfile):
                row = row.rstrip().split()
                code = row[0]
                if code not in desc_dict.keys():
                    desc_dict[code] = ' '.join(row[1:])
    return desc_dict

def pad_desc_vecs(desc_vecs):
    #pad all description vectors in a batch to have the same length
    desc_len = max([len(dv) for dv in desc_vecs])
    pad_vecs = []
    for vec in desc_vecs:
        if len(vec) < desc_len:
            vec.extend([0] * (desc_len - len(vec)))
        pad_vecs.append(vec)
    return pad_vecs

def prepare_instance(dicts, filename, args, max_length):
    ind2w, w2ind, ind2c, c2ind, dv_dict = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind'], dicts['dv']
    instances = []
    num_labels = len(dicts['ind2c'])
    with open(filename, 'r') as infile:
        r = csv.reader(infile)
        next(r)     # skip header
        for row in r:
            desc_vecs=[]
            text = row[2]
            cur_code_set = set()
            labels_idx = np.zeros(num_labels)
            labelled = False
            if args.task == 'ICD9':
                task_row = row[3]
            else:
                task_row = row[4]
            for l in task_row.split(';'):
                if l in c2ind.keys():
                    code = int(c2ind[l])
                    labels_idx[code] = 1
                    cur_code_set.add(code)
                    labelled = True
            if not labelled:
                continue
            if args.model == 'DR_CAML':
                for code in cur_code_set:
                    l = ind2c[code]
                    if l in dv_dict.keys():
                        #need to copy or description padding will get screwed up
                        desc_vecs.append(dv_dict[l][:])
                    else:
                        desc_vecs.append([len(w2ind)+1])
            tokens_ = text.split()
            tokens = []
            tokens_id = []
            for token in tokens_:
                if token == '[CLS]' or token == '[SEP]':
                    continue
                tokens.append(token)
                token_id = w2ind[token] if token in w2ind else len(w2ind) + 1
                tokens_id.append(token_id)
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
                tokens_id = tokens_id[:max_length]
            dict_instance = {'label': labels_idx, 'tokens': tokens, "tokens_id": tokens_id, "descr": pad_desc_vecs(desc_vecs)}
            instances.append(dict_instance)
    return instances

def prepare_instance_MTL(dicts, filename, args, max_length):
    ind2w, w2ind, ind2c, c2ind, ind2c_ccs, c2ind_ccs = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind'], dicts['ind2c_ccs'], dicts['c2ind_ccs']
    instances = []
    num_labels = len(dicts['ind2c'])
    num_labels_ccs = len(dicts['ind2c_ccs'])
    with open(filename, 'r') as infile:
        r = csv.reader(infile)
        next(r)     # skip header
        for row in r:
            text = row[2]
            labels_idx = np.zeros(num_labels)
            labels_idx_ccs = np.zeros(num_labels_ccs)
            # ICD-9-CM
            labelled = False
            for l in row[3].split(';'):
                if l in c2ind.keys():
                    code = int(c2ind[l])
                    labels_idx[code] = 1
                    labelled = True
            if not labelled:
                continue
            # CCS
            labelled_ccs = False
            for l_ccs in row[4].split(';'):
                if l_ccs in c2ind_ccs.keys():
                    code_ccs = int(c2ind_ccs[l_ccs])
                    labels_idx_ccs[code_ccs] = 1
                    labelled_ccs = True
            if not labelled_ccs:
                continue

            tokens_ = text.split()
            tokens = []
            tokens_id = []
            for token in tokens_:
                if token == '[CLS]' or token == '[SEP]':
                    continue
                tokens.append(token)
                token_id = w2ind[token] if token in w2ind else len(w2ind) + 1
                tokens_id.append(token_id)
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
                tokens_id = tokens_id[:max_length]
            dict_instance = {'label': labels_idx, 'label_ccs': labels_idx_ccs, 'tokens': tokens, "tokens_id": tokens_id}
            instances.append(dict_instance)
    return instances

def prepare_instance_bert_MTL(dicts, filename, args, max_length):
    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']
    instances = []
    num_labels = len(dicts['ind2c'])
    num_labels_ccs = len(dicts['ind2c_ccs'])
    wp_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    with open(filename, 'r') as infile:
        r = csv.reader(infile)
        next(r)
        for row in r:
            text = row[2]
            labels_idx = np.zeros(num_labels)
            labels_idx_ccs = np.zeros(num_labels_ccs)
            # ICD-9-CM
            labelled = False
            for l in row[3].split(';'):
                if l in c2ind.keys():
                    code = int(c2ind[l])
                    labels_idx[code] = 1
                    labelled = True
            if not labelled:
                continue
            # CCS
            labelled_ccs = False
            for l_ccs in row[4].split(';'):
                if l_ccs in c2ind_ccs.keys():
                    code_ccs = int(c2ind_ccs[l_ccs])
                    labels_idx_ccs[code_ccs] = 1
                    labelled_ccs = True
            if not labelled_ccs:
                continue

            tokens_ = text.split()
            tokens = []
            for token in tokens_:
                if token == '[CLS]' or token == '[SEP]':
                    continue
                wps = wp_tokenizer.tokenize(token)
                tokens.extend(wps)
            tokens_max_len = max_length-2 # for CLS SEP
            tokens_max_len_BERT = args.BERT_MAX_LENGTH - 2 # for CLS SEP
            if len(tokens) > tokens_max_len_BERT:
                tokens = tokens[:tokens_max_len] # First truncating from end as with other models
                if len(tokens) > tokens_max_len_BERT: # Checking is lenght still too long after first truncation
                    start = len(tokens) - tokens_max_len_BERT
                    tokens = tokens[start:]
            tokens.insert(0, '[CLS]')
            tokens.append('[SEP]')
            tokens_id = wp_tokenizer.convert_tokens_to_ids(tokens)
            masks = [1] * len(tokens)
            segments = [0] * len(tokens)
            dict_instance = {'label':labels_idx, 'label_ccs': labels_idx_ccs, 'tokens':tokens, "tokens_id":tokens_id, 
                             "segments":segments, "masks":masks}
            instances.append(dict_instance)
    return instances

def prepare_instance_bert(dicts, filename, args, max_length):
    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']
    instances = []
    num_labels = len(dicts['ind2c'])

    wp_tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    with open(filename, 'r') as infile:
        r = csv.reader(infile)
        next(r)
        for row in r:
            text = row[2]
            labels_idx = np.zeros(num_labels)
            labelled = False
            for l in row[3].split(';'):
                if l in c2ind.keys():
                    code = int(c2ind[l])
                    labels_idx[code] = 1
                    labelled = True
            if not labelled:
                continue
            tokens_ = text.split()
            tokens = []
            for token in tokens_:
                if token == '[CLS]' or token == '[SEP]':
                    continue
                wps = wp_tokenizer.tokenize(token)
                tokens.extend(wps)
            tokens_max_len = max_length-2 # for CLS SEP
            tokens_max_len_BERT = args.BERT_MAX_LENGTH - 2 # for CLS SEP
            if len(tokens) > tokens_max_len_BERT:
                tokens = tokens[:tokens_max_len] # First truncating from end as with other models
                if len(tokens) > tokens_max_len_BERT: # Checking is lenght still too long after first truncation
                    start = len(tokens) - tokens_max_len_BERT
                    tokens = tokens[start:]
            tokens.insert(0, '[CLS]')
            tokens.append('[SEP]')
            tokens_id = wp_tokenizer.convert_tokens_to_ids(tokens)
            masks = [1] * len(tokens)
            segments = [0] * len(tokens)
            dict_instance = {'label':labels_idx, 'tokens':tokens, "tokens_id":tokens_id, 
                             "segments":segments, "masks":masks}
            instances.append(dict_instance)
    return instances


class MyDataset(Dataset):

    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


def pad_sequence(x, max_len, type=np.int):
    padded_x = np.zeros((len(x), max_len), dtype=type)
    for i, row in enumerate(x):
        padded_x[i][:len(row)] = row
    return padded_x


def my_collate_MTL(x):
    words = [x_['tokens_id'] for x_ in x]
    seq_len = [len(w) for w in words]
    masks = [[1]*len(w) for w in words]
    max_seq_len = max(seq_len)

    inputs_id = pad_sequence(words, max_seq_len)
    inputs_mask = pad_sequence(masks, max_seq_len)
    labels = [x_['label'] for x_ in x]
    labels_ccs = [x_['label_ccs'] for x_ in x]
    text_inputs = [x_['tokens'] for x_ in x]
    text_inputs = batch_to_ids(text_inputs)
    return inputs_id, labels, labels_ccs, text_inputs, inputs_mask

def my_collate(x):
    words = [x_['tokens_id'] for x_ in x]
    seq_len = [len(w) for w in words]
    masks = [[1]*len(w) for w in words]
    max_seq_len = max(seq_len)

    inputs_id = pad_sequence(words, max_seq_len)
    inputs_mask = pad_sequence(masks, max_seq_len)
    labels = [x_['label'] for x_ in x]
    descs = [x_['descr'] for x_ in x]
    text_inputs = [x_['tokens'] for x_ in x]
    text_inputs = batch_to_ids(text_inputs)
    return inputs_id, labels, text_inputs, inputs_mask, descs


def my_collate_bert_MTL(x):
    words = [x_['tokens_id'] for x_ in x]
    segments = [x_['segments'] for x_ in x]
    masks = [x_['masks'] for x_ in x]
    seq_len = [len(w) for w in words]
    max_seq_len = max(seq_len)    # max of batch

    inputs_id = pad_sequence(words, max_seq_len)
    segments = pad_sequence(segments, max_seq_len)
    masks = pad_sequence(masks, max_seq_len)
    labels = [x_['label'] for x_ in x]
    labels_ccs = [x_['label_ccs'] for x_ in x]
    return inputs_id, segments, masks, labels, labels_ccs

def my_collate_bert(x):
    words = [x_['tokens_id'] for x_ in x]
    segments = [x_['segments'] for x_ in x]
    masks = [x_['masks'] for x_ in x]
    seq_len = [len(w) for w in words]
    max_seq_len = max(seq_len)    # max of batch

    inputs_id = pad_sequence(words, max_seq_len)
    segments = pad_sequence(segments, max_seq_len)
    masks = pad_sequence(masks, max_seq_len)
    labels = [x_['label'] for x_ in x]
    return inputs_id, segments, masks, labels


def _readString(f, code):
    # s = unicode()
    s = str()
    c = f.read(1)
    value = ord(c)

    while value != 10 and value != 32:
        if 0x00 < value < 0xbf:
            continue_to_read = 0
        elif 0xC0 < value < 0xDF:
            continue_to_read = 1
        elif 0xE0 < value < 0xEF:
            continue_to_read = 2
        elif 0xF0 < value < 0xF4:
            continue_to_read = 3
        else:
            raise RuntimeError("not valid utf-8 code")
        i = 0
        # temp = str()
        # temp = temp + c

        temp = bytes()
        temp = temp + c
        while i<continue_to_read:
            temp = temp + f.read(1)
            i += 1
        temp = temp.decode(code)
        s = s + temp
        c = f.read(1)
        value = ord(c)
    return s


def _readFloat(f):
    bytes4 = f.read(4)
    f_num = struct.unpack('f', bytes4)[0]
    return f_num


def norm2one(vec):
    root_sum_square = np.sqrt(np.sum(np.square(vec)))
    return vec/root_sum_square


def pad_sequence_2d(x, max_len, sent_len, type=np.int):
    padded_x = np.zeros((len(x), max_len, sent_len), dtype=type)
    for i, matrix in enumerate(x):
        padded_x[i][:len(matrix),:] = matrix
    return padded_x
    

def prepare_instance_hier(dicts, filename, args, max_length):
    # filename: data/mimic[2/3]/[train/dev/test]_[50/full].csv, e.g., data/mimic3/train_50.csv
    ind2w, w2ind, ind2c, c2ind = dicts['ind2w'], dicts['w2ind'], dicts['ind2c'], dicts['c2ind']
    instances = []
    num_labels = len(dicts['ind2c'])
    with open(filename, 'r') as infile:
        r = csv.reader(infile)
        next(r)     # skip header
        for row in r:
            text = row[2]
            labels_idx = np.zeros(num_labels)
            labelled = False
            for l in row[3].split(';'):
                if l in c2ind.keys():
                    code = int(c2ind[l])
                    labels_idx[code] = 1
                    labelled = True
            if not labelled:
                continue
            tokens_ = text.split('[SEP]')
            tokens = []
            tokens_id = []
            for token in tokens_:
                sent_token, sent_id = [], []
                sent_tokens_ = token.split()
                for t in sent_tokens_:
                    if t == '[CLS]' or t == '[SEP]':
                        continue
                    sent_token.append(t)
                    sent_id.append(w2ind[t] if t in w2ind else len(w2ind) + 1)
                if len(sent_token) == 0:
                    continue
                tokens.append(sent_token if len(sent_token) <= args.len_sent else sent_token[:args.len_sent])
                tokens_id.append(sent_id if len(sent_id) <= args.len_sent else sent_id[:args.len_sent])
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
                tokens_id = tokens_id[:max_length]
            dict_instance = {'label': labels_idx, 'tokens': tokens, "tokens_id": tokens_id}
            instances.append(dict_instance)
    return instances


def collate_hier(x):
    """
    Collate function for hierarchical encoding
    :param x: list of dicts, dict_keys(['label', 'tokens', 'tokens_id'])
    :return inputs_idx: LongTensor (batch_size, padded_doc_length, padded_sent_length)
            labels: FloatTensor (batch_size, n_classes)
            doc_lengths: LongTensor (num_docs)
            sent_lengths: LongTensor (num_docs, max_doc_len)
    """
    docs = [x_['tokens_id'] for x_ in x]
    labels = torch.FloatTensor([x_['label'] for x_ in x])

    doc_len, sent_len = [], []
    for doc in docs:
        doc_len.append(len(doc))
        sent_len.append([len(sent) for sent in doc])
    max_doc_len = max(doc_len)
    max_sent_len = max(max(sent_len))
    docs = [pad_sequence(sentences, max_sent_len) for sentences in docs]
    inputs_idx = torch.LongTensor(pad_sequence_2d(docs, max_doc_len, max_sent_len))
    sent_len = torch.LongTensor(pad_sequence(sent_len, max_doc_len))
    return inputs_idx, labels, torch.LongTensor(doc_len), sent_len

def load_description_vectors(args, Y, version='mimic3'):
    #load description one-hot vectors from file
    dv_dict = {}
    if version == 'mimic2':
        data_dir = MIMIC_2_DIR
    else:
        data_dir = args.MIMIC_3_DIR
    with open("%s/description_vectors.vocab" % data_dir, 'r') as vfile:
        r = csv.reader(vfile, delimiter=" ")
        #header
        next(r)
        for row in r:
            code = row[0]
            vec = [int(x) for x in row[1:]]
            dv_dict[code] = vec
    return dv_dict