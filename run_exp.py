# import pdb; pdb.set_trace()
import argparse
import torch
import os
import csv

from relcon.utils.utils import printlog,  init_dl_program, count_parameters
from relcon.utils.imports import import_model
from relcon.utils.datasets import load_data

from relcon.experiments.configs.MotifDist_expconfigs import allmotifdist_expconfigs
from relcon.experiments.configs.RelCon_expconfigs import allrelcon_expconfigs

all_expconfigs = {**allmotifdist_expconfigs, **allrelcon_expconfigs}

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="Select specific config from experiments/configs/",
                        type=str)
    parser.add_argument("--retrain", help="WARNING: Retrain model config, overriding existing model directory",
                        action='store_true', default=False)
    parser.add_argument("--retrain_eval", help="WARNING: Retrain eval model config, overriding existing model directory",
                        action='store_true', default=False)
    parser.add_argument("--eval_epoch", help="Epoch to be reloaded for evaluation",type=str,
                        default="best")
    # parser.add_argument("--gputiloff", help="print log the gpu utilizaiton",
    #                     action='store_true', default=False)
    parser.add_argument("--resume_on", help="resume unfinished model training",
                        action='store_true', default=False)
    args = parser.parse_args()


    # selecting config according to arg
    # CONFIGFILE = "24_1_4_ppgdist_stride2_5maskperc50"
    CONFIGFILE = args.config
    config = all_expconfigs[CONFIGFILE]
    config.set_rundir(CONFIGFILE)

    init_dl_program(config=config, device_name=0, max_threads=torch.get_num_threads())

    # Begin training contrastive learner
    train_data, train_labels, val_data, val_labels, test_data, test_labels  = \
        load_data(data_config = config.data_config)

    model = import_model(config, 
                        train_data=train_data, train_labels=train_labels, 
                        val_data=val_data, val_labels=val_labels, 
                        test_data=test_data, test_labels=test_labels, 
                        resume_on = args.resume_on)
    
    table, total_params = count_parameters(model.net)
    print(f"Total Trainable Params: {total_params:,}")
    # import pdb; pdb.set_trace()


    try:
        logpath = os.path.join("relcon/experiments/out", config.run_dir)
        printlog(f"----------------------------------------------------------------------------------- Config: {CONFIGFILE} -----------------------------------------------------------------------------------", logpath)

        # if not args.gputiloff:
        #     rt = RepeatedTimer(5, repeating_func, path=logpath, killafter=30) # it auto-starts, no need of rt.start()

        if (args.retrain == True) or (not os.path.exists(os.path.join("relcon/experiments/out/", 
                                                                config.run_dir, 
                                                                "checkpoint_best.pkl"))):
            model.fit()
        # import sys; sys.exit()

        all_eval_results_title = ["name", "epoch", "notes"]
        all_eval_results = [CONFIGFILE, args.eval_epoch, f"{total_params:,}"]
        for eval_config in config.eval_configs:
            print(f"Doing {eval_config.model_file} evaluation")
            train_data, train_labels, val_data, val_labels, test_data, test_labels, data_normalizer, data_clipping  = \
                load_data(data_config = eval_config.data_config)
            
            eval_config.set_rundir(os.path.join(CONFIGFILE, eval_config.name, eval_config.model_file))
            evalmodel = import_model(eval_config, 
                                    train_data=train_data, train_labels=train_labels, 
                                    val_data=val_data, val_labels=val_labels, 
                                    test_data=test_data, test_labels=test_labels,
                                    #############################
                                    reload_ckpt = False, evalmodel=True)
            
            model = import_model(config)
            model.load(args.eval_epoch)
            evalmodel.setup_eval(trained_net=model.net)

            if (args.retrain_eval == True) or (not os.path.exists(os.path.join(evalmodel.run_dir, "checkpoint_best.pkl"))):
                evalmodel.fit()

            out_test = evalmodel.test() # automatically loads
            printlog(eval_config.name + " " + eval_config.model_file +" ++++++++++++++++++++++++++++++++++++++++", logpath)

            all_eval_results_title.extend(list(out_test.keys()))
            all_eval_results.extend(list(out_test.values()))

        # create csv file that is easy to paste into spreadsheet
        csv_file = os.path.join(logpath, f"{CONFIGFILE}_easy_paste.csv")
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(all_eval_results_title)
            writer.writerow(all_eval_results)
            
    except Exception as e:
        raise  
    finally:
        printlog(f"Config: {CONFIGFILE}", logpath)
        # if not args.gputiloff:
        #     rt.stop() # better in a try/finally block to make sure the program ends!

