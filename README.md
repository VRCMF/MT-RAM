# Multitask Recalibrated Aggregation Network for Medical Code Prediction (MT-RAM)
To reproduce the results of the paper [Multitask Recalibrated Aggregation Network](https://arxiv.org/abs/2104.00952), we present this code repository.

## Highlight

- **Code Associations** The multi-task learning scheme to capture the relationship between different medical codes. 
- **Recalibrate Feature** The designed Recalibrated Attention Module (RAM) reduce the effect of noise in clinical documents. Also, RAM could alleviate the lengthy document problem by iterative convolution. 
- **Extensible** The multi-task learning framework could be extended to multiple (>= 3) medical coding task, such as HCC coding task and CPT coding task. 

# Package Dependencies

* allennlp == 0.9.0
* ax-platform == 0.1.12
* gensim == 3.8.3
* plotly == 4.7.1
* pytorch==1.5.1
* spacy == 2.1.9
* tensorboardx == 2.0
* tokenizers == 0.7.0
* numpy == 1.15.1
* nltk == 3.5
* python == 3.6.12
* pytorch-pretrained-bert == 0.6.2
* transformers == 2.9.1

You can use the following command (recommended):
~~~
pip install -r requirements.txt
~~~

## Preprossing 

### Clinical Document

We follow the preproces setting of [MultiResCNN](https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network). The structure of data files can be shown like:
```
data
|   D_ICD_DIAGNOSES.csv
|   D_ICD_PROCEDURES.csv
|   ICD9_descriptions.txt (for DR_CAML)
└───mimic3/
|   |   NOTEEVENTS.csv
|   |   DIAGNOSES_ICD.csv
|   |   PROCEDURES_ICD.csv
|   |   *_hadm_ids.csv (get from CAML)
```
Running the ```python preprocess_mimic3.py``` obtain corresponding ICD code file.

### Obtain CCS dataset

Clinical Classifications Software (CCS) for ICD-9-CM is a tool from HCUP.
Next, download the ```dx2015.csv``` and ```pr2015.csv``` from [web](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/Single_Level_CCS_2015.zip). Place two file in the data, the structure is shown like this:

```
data
|   D_ICD_DIAGNOSES.csv
|   D_ICD_PROCEDURES.csv
|   ICD9_descriptions.txt (for DR_CAML)
└───mimic3/
|   |   dev_50.csv
|   |   train_50.csv
|   |   test_50.csv
|   |   dx2015.csv
|   |   pr2015.csv
|   |   NOTEEVENTS.csv
|   |   DIAGNOSES_ICD.csv
|   |   PROCEDURES_ICD.csv
|   |   *_hadm_ids.csv (get from CAML)
```
use the script ```python ICD2CCS.py``` to obtain CCS labels and attach them on corresponding csv files.

## Training

#### MT-RAM
~~~
python main.py --MAX_LENGTH 2500 --n_epochs 50 --batch_size 16 --model GRU --lr 8e-3 --MTL Yes --loss_weight_CCS 0.3
~~~
#### CAML + MTL + RAM
~~~
python main.py --MAX_LENGTH 2500 --n_epochs 50 --batch_size 16 --model caml --lr 8e-3 --MTL Yes --loss_weight_CCS 0.3
~~~
#### MultiResCNN + MTL + RAM
~~~
python main.py --MAX_LENGTH 2500 --n_epochs 50 --batch_size 16 --model MultiResCNN --lr 8e-3 --MTL Yes --loss_weight_CCS 0.3
~~~

## Main Results (all evaluation results are presented in %)
### MIMIC-III (ICD)

| Models     |  Macro AUC-ROC |  Micro AUC-ROC | Macro F1 | Micro F1 |  Precision at 5 | Model |
|--------------|-----------|-----------|-----------|--------------|-----------------------|-----|
|CAML + MTL + RAM | 91.4 | 93.8 | 62.5 | 68.7 | 65.3 | [CAML](https://drive.google.com/file/d/1f2E3nJO9C9spFujQTWALpRZMFR0hkjxg/view?usp=sharing) |
|MultiResCNN + MTL + RAM | 91.7 | 93.9 | 64.1 | 69.0 | 65.0 | [MultiResCNN](https://drive.google.com/file/d/18JQ740kIX8zbd6lGIeS5HYUNqbaUJdzd/view?usp=sharing) |
|MT-RAM    | 92.1 | 94.3 | 65.1 | 70.6 | 66.4 | [MT-RAM](https://drive.google.com/file/d/1-8TMF0qt2IJhYQnnQC7oCXh8LCdd_-Me/view?usp=sharing) |

### MIMIC-III (CCS)

| Models     |  Macro AUC-ROC |  Micro AUC-ROC | Macro F1 | Micro F1 |  Precision at 5 |Model |
|--------------|-----------|-----------|-----------|--------------|-----------------------|-----|
|CAML + MTL + RAM | 91.5 | 94.2 |	66.9 | 72.8	| 67.5 |[CAML](https://drive.google.com/file/d/1f2E3nJO9C9spFujQTWALpRZMFR0hkjxg/view?usp=sharing) |
|MultiResCNN + MTL + RAM | 91.7	| 94.3 | 67.8 |	72.7 | 67.3 |[MultiResCNN](https://drive.google.com/file/d/18JQ740kIX8zbd6lGIeS5HYUNqbaUJdzd/view?usp=sharing) |
|MT-RAM    |92.2 | 94.6 |	69.3	| 74.3	| 68.3 |[MT-RAM](https://drive.google.com/file/d/1-8TMF0qt2IJhYQnnQC7oCXh8LCdd_-Me/view?usp=sharing) |

## Citation
If you find that our code is helpful, please use the Bibtex citation shown below.

    @article{sun2021multitask,
    title={Multitask Recalibrated Aggregation Network for Medical Code Prediction},
    author={Sun, Wei and Ji, Shaoxiong and Cambria, Erik and Marttinen, Pekka},
    journal={arXiv preprint arXiv:2104.00952},
    year={2021}
    }

## Acknowledgement
We appreciate for all code providers, especially for [MultiResCNN](https://github.com/foxlf823/Multi-Filter-Residual-Convolutional-Neural-Network), [CAML](https://github.com/jamesmullenbach/caml-mimic) and [CCS](https://www.hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp).
