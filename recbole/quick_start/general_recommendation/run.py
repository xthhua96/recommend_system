from pdb import run
from recbole.quick_start import run_recbole

# run_recbole(model="BPR", dataset="ml-100k", config_file_list=["test.yaml"])
# run_recbole(model="NCL", dataset="ml-100k", config_file_list=["test.yaml"])
run_recbole(model="SLIMElastic", dataset="ml-100k")
