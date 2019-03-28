from pathlib import Path
from config import get_config
from data.data_pipe import load_bin, load_mx_rec
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-r", "--rec_path", help="mxnet record file path",default='faces_emore', type=str)
    args = parser.parse_args()
    conf = get_config()
    rec_path = conf.data_path/args.rec_path
#    load_mx_rec(rec_path)
    
#    bin_files = ['agedb_30', 'cfp_fp', 'lfw', 'calfw', 'cfp_ff', 'cplfw', 'vgg2_fp']
    bin_files = [ 'cfp_fp', 'lfw', 'calfw', 'cfp_ff', 'cplfw', 'vgg2_fp']
    
#    for i in range(len(bin_files)):
#        load_bin(rec_path/(bin_files[i]+'.bin'), rec_path/bin_files[i], conf.test_transform)
#    load_bin('/home/zzg/Datasets/faces_emore/agedb_30.bin', '/home/zzg/Datasets/faces_emore/agedb_30', conf.test_transform)
    load_bin('/home/zzg/Datasets/faces_emore/cfp_fp.bin', '/home/zzg/Datasets/faces_emore/cfp_fp', conf.test_transform)
    load_bin('/home/zzg/Datasets/faces_emore/lfw.bin', '/home/zzg/Datasets/faces_emore/lfw', conf.test_transform)
    load_bin('/home/zzg/Datasets/faces_emore/lfw.bin', '/home/zzg/Datasets/faces_emore/lfw', conf.test_transform)
    load_bin('/home/zzg/Datasets/faces_emore/calfw.bin', '/home/zzg/Datasets/faces_emore/calfw', conf.test_transform)
    load_bin('/home/zzg/Datasets/faces_emore/cfp_ff.bin', '/home/zzg/Datasets/faces_emore/cfp_ff', conf.test_transform)
    load_bin('/home/zzg/Datasets/faces_emore/cplfw.bin', '/home/zzg/Datasets/faces_emore/cplfw', conf.test_transform)
    load_bin('/home/zzg/Datasets/faces_emore/vgg2_fp.bin', '/home/zzg/Datasets/faces_emore/vgg2_fp', conf.test_transform)
