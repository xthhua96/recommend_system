from recbole.quick_start import run_recbole

# run_recbole(model="GRU4Rec", dataset="ml-100k", config_file_list=["test.yaml"])
parameter_dict = {
    "train_neg_sample_args": None,
}
run_recbole(model="KSR", dataset="ml-100k", config_dict=parameter_dict)
