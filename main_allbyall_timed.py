#!/usr/bin/env python3

import argparse
import configparser
import sys
import os
import platform
import numpy as np

from datetime import datetime

# modules
from PCprophet import io_ as io
from PCprophet import generate_features_allbyall_timed as generate_features_allbyall
from PCprophet import plots_allbyall as plots
from PCprophet import validate_input as validate
from PCprophet import fit_model

class ParserHelper(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

# TODO check os
def get_os():
    return platform.system()

def is_running_in_slurm():
    return 'SLURM_JOB_ID' in os.environ


# I reorganized the output directory, such that for each run a new sub-directory is created which holds the tmp folder and all other outputs. 
# Moving the tmp folder ensures uniqueness and resolves an issue where the content of a previous run in tmp would cause an error.
def setup_output_directory(base_output, sid_file):
    """
    Ensure a unique output directory for each SLURM job by including the SLURM job ID.
    """
    array_task_id = os.getenv("SLURM_ARRAY_TASK_ID", "0")

    # Ensure base output folder exists
    if not os.path.exists(base_output):
        os.makedirs(base_output)

    # Extract run name from sid_file (e.g., "test1" from "test/test1.txt")
    run_name = os.path.splitext(os.path.basename(sid_file))[0]

    # Get current datetime
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create run folder (e.g., 20241101_103412_test1_16660190)
    run_folder = f"{current_datetime}_{run_name}_{array_task_id}"

    # Create full run path
    full_run_path = os.path.join(base_output, run_folder)

    # Create tmp folder within run path
    run_temp_folder = os.path.join(full_run_path, 'tmp')

    # Create the unique run folder with tmp inside
    os.makedirs(run_temp_folder, exist_ok=True)

    if is_running_in_slurm():
        slurm_file_path = os.path.join(full_run_path, f"slurm_analysis_dir_{array_task_id}.sh")
        with open(slurm_file_path, "w") as f:
            f.write(f"export ANALYSIS_DIR={full_run_path}\n")

    return full_run_path, run_temp_folder

def move_logs(output_dir, array_task_id):
    slurm_output_file = f"slurm_output_{array_task_id}.log"
    slurm_error_file = f"slurm_error_{array_task_id}.log"
    for file, dest in zip([slurm_output_file, slurm_error_file],
                        ["slurm_output.log", "slurm_error.log"]):
        try:
            os.rename(file, os.path.join(output_dir, dest))
            print(f"Moved {file} to {output_dir}/{dest}")
        except FileNotFoundError:
            print(f"{file} not found; skipping.")
        except Exception as e:
            print(f"Error moving {file}: {e}")


def create_config():
    '''
    parse command line and create .ini file for configuration
    '''
    parser = ParserHelper(description='Protein Complex Prophet argument')
    parser.add_argument(
        '-db',
        help='protein complex database from CORUM or ppi network in STRING format',
        dest='database',
        action='store',
        default='coreComplexes.txt',
    )
    # maybe better to add function for generating a dummy sample id?
    parser.add_argument(
        '-sid',
        help='sample ids file',
        dest='sample_ids',
        default='sample_ids.txt',
        action='store',
    )
    parser.add_argument(
        '-output',
        help='outfile folder path',
        dest='out_folder',
        default=r'./Output',
        action='store',
    )
    parser.add_argument(
        '-run_name',
        help='Custom name for the run folder in Output',
        dest='run_name',
        default='default',
        action='store',
    )
    parser.add_argument(
        '-cal',
        help='calibration file no headers tab delimited fractiosn to mw in KDa',
        dest='calibration',
        default='None',
        action='store',
    )
    parser.add_argument(
        '-mw_uniprot',
        help='Molecular weight from uniprot',
        dest='mwuni',
        default='None',
        action='store',
    )
    parser.add_argument(
        '-is_ppi',
        help='is the -db a protein protein interaction database',
        dest='is_ppi',
        action='store',
        default='False',
        choices=['True', 'False'],
    )
    parser.add_argument(
        '-a',
        help='use all fractions [1,X]',
        dest='all_fract',
        action='store',
        default='all',
    )
    parser.add_argument(
        '-ma',
        help='merge using all complexes or reference only',
        dest='merge',
        action='store',
        choices=['all', 'reference'],
        default='all',
    )
    parser.add_argument(
        '-fdr',
        help='false discovery rate for novel complexes',
        dest='fdr',
        action='store',
        default=0.5,
        type=float,
    )
    parser.add_argument(
        '-co',
        help='collapse mode',
        choices=['GO', 'CAL', 'SUPER', 'PROB', 'NONE'],
        dest='collapse',
        default='GO',
        action='store',
    )
    parser.add_argument(
        '-sc',
        help='score for missing proteins in differential analysis',
        dest='score_missing',
        action='store',
        default=0.5,
        type=float,
    )
    parser.add_argument(
        '-mult',
        help='Number of cores for multi-processing feature generation (default: 8)',
        dest='multi',
        type=int,
        default=8,
    )
    
    parser.add_argument('-w', dest='weight_pred', help='LEGACY', action='store', default=1, type=float)
    parser.add_argument('-v', dest='verbose', help='Verbose', action='store', default=1)
    parser.add_argument('-skip',
                        dest='skip',
                        help='Skip feature generation and complex prediction step',action='store',
                        default=False)
    
    args = parser.parse_args()

    # deal with numpy warnings and so on
    if args.verbose == 0:
        np.seterr(all='ignore')
    else:
        pass
        # print them

    # Call setup_output_directory with parsed arguments
    output_folder, tmp_folder = setup_output_directory(base_output=args.out_folder, sid_file=args.sample_ids)
  

    # create config file
    config = configparser.ConfigParser()
    config['GLOBAL'] = {
        'db': args.database,
        'sid': args.sample_ids,
        'go_obo': io.resource_path('go-basic.obo'),
        'sp_go': io.resource_path('tmp_GO_sp_only.txt'),
        'output': output_folder,
        'cal': args.calibration,
        'mw': args.mwuni,
        'temp': tmp_folder,
        'mult': args.multi,
        'skip': args.skip
    }
    config['PREPROCESS'] = {
        'is_ppi': args.is_ppi,
        'all_fract': args.all_fract,
        'merge': args.merge,
    }
    config['POSTPROCESS'] = {'fdr': args.fdr, 'collapse_mode': args.collapse}
    config['DIFFERENTIAL'] = {
        'score_missing': args.score_missing,
        'weight_pred': args.weight_pred,
        'fold_change': '-5,-2,2,5',
        'correlation': '0.3,0.9',
        'ratio': '-2,-0.5,0.5,2',
        'shift': '-10,-5,5,10',
        'weight_fold_change': 1,
        'weight_correlation': 0.75,
        'weight_ratio': 0.25,
        'weight_shift': 0.5,
    }
    # create config ini file for backup
    with open('ProphetConfig.conf', 'w') as conf:
        config.write(conf)
    return config


def preprocessing(infile, config, tmp_folder):
    #validate.InputTester(infile, 'in').test_file()

    # sample specific folder
    tmp_folder = io.file2folder(infile, tmp_folder=tmp_folder)

    generate_features_allbyall.runner(
        infile=infile,
        tmp_folder=tmp_folder,
        npartitions=config['GLOBAL']['mult'],
        features=['correlation_raw', 
                  'correlation_smooth', 
                  'euclidean_distance_smooth',
                  'max_sliding_window_correlation_raw',
                  'mean_sliding_window_correlation_raw'],
    )
    return True


def main():
    config = create_config()
    validate.InputTester(config['GLOBAL']['db'], 'db').test_file()
    validate.InputTester(config['GLOBAL']['sid'], 'ids').test_file()
    files = io.read_sample_ids(config['GLOBAL']['sid'])
    files = [os.path.abspath(x) for x in files.keys()]

    features = ['correlation_raw',
                'correlation_smooth', 
                'euclidean_distance_smooth',
                'max_sliding_window_correlation_raw',
                'mean_sliding_window_correlation_raw']
    
    # Skip feature generation
    if config['GLOBAL']['skip'] == 'False':
        [preprocessing(infile, config, config['GLOBAL']['temp']) for infile in files]

    # Model fitting and plotting
    fit_model.runner(
        config['GLOBAL']['temp'],
        config['GLOBAL']['db'],
        features,
    )

    plots.runner(
        config['GLOBAL']['temp'],
        config['GLOBAL']['output'],
        features,
    )

    #Maybe write these to global env for access here and in setup_output_directory
    array_task_id = os.getenv("SLURM_ARRAY_TASK_ID", "0")  # Default to "0" for non-array jobs

    if is_running_in_slurm():  # Perform log movement only if running under SLURM
        move_logs(config['GLOBAL']['output'], array_task_id)
    else:
        print("Running locally: skipping log movement.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass