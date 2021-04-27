import argparse
import sys

def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--MODEL_DIR', type=str, default='./saved/models')
    parser.add_argument('--DATA_DIR', type=str, default='./data')
    parser.add_argument('--MIMIC_3_DIR', type=str, default='./data/mimic3')
    parser.add_argument('--MIMIC_2_DIR', type=str, default='./data/mimic2')
    parser.add_argument('--load_model', default='')

    parser.add_argument("--data_path", type=str, default='./data/mimic3/train_50_m.csv')
    parser.add_argument("--vocab", type=str, default='./data/mimic3/vocab.csv')
    parser.add_argument("--Y", type=str, default='50', choices=['full', '50'], help="full ICD code set or top 50 frequent codes")
    parser.add_argument("--version", type=str, choices=['mimic2', 'mimic3'], default='mimic3')
    parser.add_argument("--MAX_LENGTH", type=int, default=20, help="maximum sequence length, usually 2500")
    parser.add_argument("--BERT_MAX_LENGTH", type=int, default=512, help="Maximum sequence length for BERT, last tokens selected after MAX_LENGTH has been applied. Bio_ClinicalBert has max lenght 512.")

    # model
    parser.add_argument("--model", type=str, default='DR_CAML', choices=['caml', 'elmo', 'bert','logistic_regression', 'GRU', 'MultiResCNN', 'DR_CAML'])
    parser.add_argument("--filter_size", type=str, default="10", help='MultiResCNN for 3,5,9,15,19,25 & CAML and DR-CAML for 10')
    parser.add_argument("--num_filter_maps", type=int, default=50)
    parser.add_argument("-conv_layer", type=int, default=1)
    parser.add_argument("--embed_file", type=str, default='./data/mimic3/processed_full.embed')
    parser.add_argument("--test_model", type=str, default=None)
    parser.add_argument("--nhid", type=int, default=300, help="hidden representation dimension")
    parser.add_argument("--bidirectional", action="store_true", help="bidirectional RNN")

    # training
    parser.add_argument("--n_epochs", type=int, default=1, help="usually 50 for top50 codes, 100 for full set, 1-10 for fine tuning")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--patience", type=int, default=10, help="early stopping patience rounds")
    parser.add_argument("--batch_size", type=int, default=2, help="16 (MIMIC-III top-50), 32 (MIMIC-III full), 64 (MIMIC-II full)")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--optimizer", type=str, default="AdamW", choices=['Adam', 'AdaX', 'AdamW'])
    parser.add_argument("--weight_decay", type=float, default=0, help="L2 regularization")
    parser.add_argument("--criterion", type=str, default='f1_macro', choices=['prec_at_8', 'prec_at_15', 'f1_macro', 'f1_micro', 'prec_at_5','loss_dev'])
    parser.add_argument("--use_lr_scheduler", action="store_const", const=True, default=False)
    parser.add_argument("--warm_up", type=float, default=0.1)
    parser.add_argument("--use_lr_layer_decay", action="store_const", const=True, default=False)
    parser.add_argument("--lr_layer_decay", type=float, default=1.0)
    parser.add_argument("--weight_tuning", action="store_const", const=True, default=False)
    parser.add_argument("--lmbda", type=int, default=10)

    # Multi-task
    parser.add_argument("--loss_weight_CCS", type=float, default=0.3)
    parser.add_argument("--MTL", type=str, default='No', choices=['Yes', 'No'], help='Multi-task learning')
    parser.add_argument("--task", type=str, default='CCS', choices=['ICD9', 'CCS'], help='Single-task learning, which from ICD9 or CCS code')
    parser.add_argument("--RAM", action="store_const", const=True, default=False, help="Recalibrated Attention Module")

    # settings
    parser.add_argument('--gpus', default='0', help='use comma for multiple gpus')
    parser.add_argument("--gpu", type=int, default=0, help='-1 if not use gpu, >=0 if use gpu')
    parser.add_argument("--tune_wordemb", action="store_const", const=True, default=False)
    parser.add_argument('--random_seed', type=int, default=1, help='0 if randomly initialize the model, other if fix the seed')
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--use_ext_emb", type=int, default=1)
    parser.add_argument("--debug", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--credential", type=str, default='credential-mt.json', help="Google spreadsheet credential")

    # elmo
    parser.add_argument("--elmo_options_file", type=str, default='./elmo_small/elmo_2x1024_128_2048cnn_1xhighway_options.json')
    parser.add_argument("--elmo_weight_file", type=str, default='./elmo_small/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')
    parser.add_argument("--elmo_tune", action="store_true", help="tune ELMo")
    parser.add_argument("--elmo_dropout", type=float, default=0)
    parser.add_argument("--elmo_gamma", type=float, default=0.1)
    parser.add_argument("--use_elmo", type=int, default=0)

    # bert
    parser.add_argument("--pretrained_model", type=str, default="emilyalsentzer/Bio_ClinicalBERT", choices=["emilyalsentzer/Bio_ClinicalBERT"], help="refer to Huggingface transformers for more")

    args = parser.parse_args()
    command = ' '.join(['python'] + sys.argv)
    args.command = command
    return args