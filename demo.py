from logsum.prefix_graph import PrefixGraph
from logsum.log_summary import LogSummary,register_graph_attributes,get_structured_match_result
from utils.loader import *
from utils.evaluation import evaluate_word_classification,evaluate_variable_mapping,evaluate_header_extraction
from logsum.naive_algorithm import *
import time

def run_log_summary(logs):
	# Run Prefix Graph model
	print("Load",len(logs),"log mesages...")
	print("Building Prefix Graph...")
	begin = time.time()
	graph = PrefixGraph()
	for log in logs:
		graph.insert(log,sep=" ")
	result = graph.build(gamma=0.15,early_stop=10)
	log_templates,counts = graph.template_extract()
	end = time.time()
	print("Number of Log Templates: %d"%len(log_templates))
	print("Elapsed Time: %.4lf"%(end-begin))

	# Off-Line Summarizing
	print("Off-Line Summarizing...")
	begin = time.time()
	logsum = LogSummary()
	register_graph_attributes(graph,logsum)
	logsum.summarize(graph,{"ngram":5})
	end = time.time()
	print("Elapsed Time: %.4lf"%(end-begin))

	# Update Variable Format
	begin = time.time()
	for log in logs:
		get_structured_match_result(logsum,graph,log,update_model=True)
	end = time.time()
	print("Evaluate Structuring Time: %.4lf"%(end-begin))

	return logsum,graph,log_templates


if __name__=="__main__":
	logname = "SSH"
	#train_logs = load_data("./data/%s_2k.log"%logname)
	train_logs = load_data("./data/%s_10k.log"%logname)

	print("Run Log Summary...")
	logsum,graph,log_templates = run_log_summary(train_logs)
	test_logs = load_data("./data/%s_2k.log"%logname)
	logsum_pre_matched_result = []
	sutructured_logs = []
	reserve_keys = ["TemplateID","Frequency","TemplateWord","VariableField","ValueDict"]
	for log in test_logs:
		result = get_structured_match_result(logsum,graph,log,update_model=False)
		logsum_pre_matched_result.append(result)
		sutructured_log = {key:result[key] for key in reserve_keys if key in result}
		sutructured_logs.append(sutructured_log)
	print("\nExample of sutructured log:")
	print(logsum_pre_matched_result[0])
	
	print("\nOutput Structured Logs...")
	write_structured_logs(sutructured_logs,"./result/%s_structured_logs.json"%logname)
	print("Output Log Templates...")
	with open("./result/Template_%s_PreifxGraph.txt"%logname,"w+") as f:
		for line in log_templates:
			f.write(line+"\n")

	generate_template_groundtruth("./label/%s_2k.log_structured.csv"%logname,
								"./result/Template_%s_GroundTruth.txt"%logname)
	print("Finished!")

	print("\nEvaluate Log Parsing...")
	log_content,parsing_truth = load_groundtruth_log_parsing("./label/%s_2k.log_structured.csv"%logname)
	test_logs = load_data("./data/%s_2k.log"%logname)
	match_tree_truth,_ = load_log_templates("./result/Template_%s_GroundTruth.txt"%logname)
	match_tree_test,_ = load_log_templates("./result/Template_%s_PreifxGraph.txt"%logname)
	print(">> Prefix Graph:")
	result = evaluate_word_classification(test_logs,log_content,parsing_truth,
			match_tree_test,match_tree_truth,graph=graph,logsum=None)
	print(">> Log Summary:")
	result,var_index = evaluate_word_classification(test_logs,log_content,parsing_truth,
			match_tree_test,match_tree_truth,graph=graph,logsum=logsum,
			logsum_pre_matched_result=logsum_pre_matched_result)

	print("\n")
	naive_var_index = naive_algo_variable_id_assignment(log_content,var_index)
	print(">> Naive Algorithm:")
	result = evaluate_variable_mapping("./label/%s_map_label.csv"%logname,naive_var_index)
	print(">> Log Summary:")
	result = evaluate_variable_mapping("./label/%s_map_label.csv"%logname,var_index)
	
	print("\nEvaluate Header Extraction...")
	header_groundtruth = load_header_groundtruth("./label/header_groundtruth.json")
	evaluate_header_extraction(graph,logsum,header_groundtruth[logname])



