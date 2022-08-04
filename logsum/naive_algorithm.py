def naive_algo_variable_id_assignment(log_content,var_index):
    naive_var_index = []
    global_var_idx = 0
    current_var_assign = {}
    for log_id in range(len(log_content)):
        logkey = " ".join(log_content[log_id])
        if logkey not in current_var_assign:
            var_list = []
            for j in range(len(var_index[log_id])):
                var_list.append(global_var_idx if var_index[log_id][j] is not None else None)
                global_var_idx += 1
            current_var_assign[logkey] = var_list
        var_list = current_var_assign[logkey]
        naive_var_index.append(var_list)
    return naive_var_index
