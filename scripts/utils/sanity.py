# sanity check
import wandb
from ast import literal_eval
api = wandb.Api()

def print_run_diff(run_id1, run_id2):
    run1 = api.run(f"[author1]/onlineCL-cs1/{run_id1}")
    run2 = api.run(f"[author1]/onlineCL-cs1/{run_id2}")

    conf1 = literal_eval(run1.config['cfg'])
    conf2 = literal_eval(run2.config['cfg'])

    # check run1 and run2 config differences
    def print_dict_diff(d1, d2):
        for k in d1.keys():
            if k not in d2.keys():
                print(k, d1[k], "MISSING")
            elif d1[k] != d2[k]:
                if type(d1[k]) == dict:
                    print_dict_diff(d1[k], d2[k])
                else:
                    print(k, d1[k], d2[k])


        for k in d2.keys():
            if k not in d1.keys():
                print(k, "MISSING", d2[k])

    print_dict_diff(conf1, conf2)

ISL_IWM = "tk1sgelf"
ISL2_IWM = "af8plmyp"
IBEX_IWM = "9ncyp1pk"

print_run_diff(ISL_IWM, ISL2_IWM)
