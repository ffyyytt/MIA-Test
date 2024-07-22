import os

for dataset in ["cifar10", "AID", "UCM"]:
    os.popen(f"sbatch run_cen.sh {dataset}").read()
    for FL in ["FedAvg", "FedProx"]:
        for method in range(3):
            os.popen("sbatch run_fed.sh {dataset} {FL} {method}").read()