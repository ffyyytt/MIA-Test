import os

for dataset in ["cifar10", "AID", "UCM"]:
    os.popen(f"mkdir {dataset}").read()
    os.popen(f"sbatch run_cen.sh {dataset}").read()
    os.popen(f"mkdir {dataset}/cen").read()
    for FL in ["FedAvg", "FedProx"]:
        for method in range(3):
            os.popen(f"sbatch run_fed.sh {dataset} {FL} {method}").read()
            os.popen(f"mkdir {dataset}/{FL}{'FT'*method}").read()