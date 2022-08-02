from logsum.prefix_graph import PrefixGraph
from logsum.log_summary import LogSummary,register_graph_attributes,get_structured_match_result
from utils.loader import load_data

if __name__=="__main__":
	logs = load_data("./data/BGL_2k.log")
	graph = PrefixGraph()
	for log in logs:
		graph.insert(log,sep=" ")
	result = graph.build(gamma=0.15,early_stop=10)
	templates,counts = graph.template_extract()
	print("Number of Templates:",len(templates))
	logsum = LogSummary()
	register_graph_attributes(graph,logsum)
	logsum.summarize(graph,{"ngram":5})
	# Off-Line
	for log in logs:
		get_structured_match_result(logsum,graph,log,update_model=True)
	# On-Line
	result = get_structured_match_result(logsum,graph,logs[0],update_model=False)
	print(result)


