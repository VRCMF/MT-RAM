from options import args_parser
import pandas as pd
import re
import os

args = args_parser()

split_string = ['train', 'dev', 'test']

for split in split_string:

    ICD_file_path = '%s/%s_50.csv' % (args.MIMIC_3_DIR, split)
    df=pd.read_csv(ICD_file_path)

    labels_list=df['LABELS'].values

    #Diagnoses CCS coding
    dx_path = '%s/dx2015.csv' % args.MIMIC_3_DIR

    dx=pd.read_csv(dx_path)
    dx_ccs_ICD9_list=[]
    dx_ccs_CCS_list=[]
    #.values[1:] remove the empty row
    for i in dx[dx.columns[0]].values[1:]:
        dx_ccs_ICD9_list.append(re.findall(r'[A-Z]*[0-9]+',i)[0])
    for ii in dx[dx.columns[1]].values[1:]:
        dx_ccs_CCS_list.append(re.findall(r'[0-9]+',ii)[0])

    #Procedures CCS coding
    pr_path = '%s/pr2015.csv' % args.MIMIC_3_DIR

    pr=pd.read_csv(pr_path)
    pr_ccs_ICD9_list=[]
    pr_ccs_CCS_list=[]
    #.values[1:] remove the empty row
    for i in pr[pr.columns[0]].values[1:]:
        code = re.findall(r'[A-Z]*[0-9]+',i)[0]
        code = code[:2] + '.' + code[2:]
        pr_ccs_ICD9_list.append(str(float(code)))
    for ii in pr[pr.columns[1]].values[1:]:
        pr_ccs_CCS_list.append(re.findall(r'[0-9]+',ii)[0])


    trans_CCS_list=[]

    for item in labels_list:
        tmp_ICD_list=[]
        try:
            tmp_ICD_list=item.split(';')
        except AttributeError:
            trans_CCS_list.append('0')
            pass
            continue
        tmp_CCS_list=[]
        for i in tmp_ICD_list:
            if len(i.split('.')[0]) > 2:
                ori_tmp_ICD = "".join(i.split('.'))
                tmp_CCS_list.append(dx_ccs_CCS_list[dx_ccs_ICD9_list.index(ori_tmp_ICD)])
            else:
                try:
                    tmp_CCS_list.append(pr_ccs_CCS_list[pr_ccs_ICD9_list.index(i)])
                except ValueError:
                    code = "".join(i.split('.'))
                    code = code[:1] + '.' + code[1:]
                    try:
                        tmp_CCS_list.append(pr_ccs_CCS_list[pr_ccs_ICD9_list.index(code)])
                    except ValueError:
                        code = "".join(i.split('.'))
                        code = '0' + '.' + code[0]
        tmp_CCS_list=set(tmp_CCS_list)
        trans_CCS_list.append(";".join(tmp_CCS_list))

    df['LABELS_CCS']=trans_CCS_list
    df = df[['SUBJECT_ID', 'HADM_ID', 'TEXT', 'LABELS', 'LABELS_CCS','length']]
    transformed_path = '%s/%s_50_m.csv' % (args.MIMIC_3_DIR, split)
    df.to_csv(transformed_path, index=False)