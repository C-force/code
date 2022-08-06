import re
import numpy as np
import pandas as pd
import sys
sys.path.append("./logsum")
import os
import time
from log_summary import prefix_graph_proprocessor,get_structured_match_result,search_header_by_frequency

class Node:
	def __init__(self):
		self.son = dict()
		self.jump = False
		self.flag = None
		self.label = set()

class MatchTree:
	def __init__(self,wildcard="<*>"):
		self.root = Node()
		self.wildcard = wildcard
		self.num_template = 0
		self.label_pair = []
		self.END_VALUE = "$END$"

	def insert(self,word_vec,tag=None):
		word_vec.append(self.END_VALUE)
		pre = self.root
		for key in word_vec:
			if key==self.wildcard:
				pre.jump = True
				continue
			if key not in pre.son:
				pre.son[key] = Node()
			pre = pre.son[key]
		pre.jump = True
		if tag is not None:
			pre.label.add(tag)

	def construct(self):
		self.num_template = 0
		self.label_pair = []
		def _dfs(node):
			if len(node.son)==0:
				self.num_template += 1
				node.flag = "C"+str(self.num_template)
				for tag in node.label:
					self.label_pair.append([node.flag,tag])
			else:
				for son in node.son.values():
					_dfs(son)
		_dfs(self.root)

	def match(self,word_vec):
		word_vec.append(self.END_VALUE)
		pre = self.root
		for key in word_vec:
			if key in pre.son:
				pre = pre.son[key]
			elif pre.jump==True:
				continue
			else:
				return None
		return pre.flag

	def deep_match(self,word_vec,return_flag=False):
		word_vec.append(self.END_VALUE)
		match_flag = [False]*len(word_vec)
		def _match(node,word_vec,pos=0):
			if pos>=len(word_vec):
				if len(node.son)==0:
					return node.flag
				return None
			key = word_vec[pos]
			if key in node.son:
				match_flag[pos] = True
				flag = _match(node.son[key],word_vec,pos+1)
				if flag is not None:
					return flag
			if node.jump==True:
				match_flag[pos] = False
				return _match(node,word_vec,pos+1)
			return None
		result = _match(self.root,word_vec)
		if return_flag==True:
			return result,match_flag
		return result

	def fit_templates(self,templates):
		for line in templates:
			line = line.replace("\n","")
			word_vec = line.split(" ")
			self.insert(word_vec)
		self.construct()

	def get_match_result(self,test_data,deep_mode=False):
		result = []
		st = time.time()
		for i in range(len(test_data)):
			if deep_mode==False:
				flag = self.match(test_data[i].split(" "))
			else:
				flag = self.deep_match(test_data[i].split(" "))
			result.append([i+1,flag])
		ed = time.time()
		print("Match Time : %.4lf s" % float(ed-st))
		return result


def count_unmatch(result):
	unmatched = 0
	for i in range(len(result)):
		if result[i][1] is None:
			unmatched += 1
			result[i][1] = "UNMATCHED"
	return result,unmatched

def get_accuracy(match_label, match_result):
	if type(match_label)!=pd.Series:
		match_label = np.array(match_label)
		series_groundtruth = pd.Series(match_label[:,1],index=match_label[:,0])
	else:
		series_groundtruth = match_label
	if type(match_result)!=pd.Series:
		match_result = np.array(match_result)
		series_parsedlog = pd.Series(match_result[:,1],index=match_result[:,0])
	else:
		series_parsedlog = match_result
	total_pairs = len(series_groundtruth)*(len(series_groundtruth)-1)/2
	series_groundtruth_valuecounts = series_groundtruth.value_counts()
	real_pairs = 0
	for count in series_groundtruth_valuecounts:
		if count > 1:
			real_pairs += count*(count-1)/2
	series_parsedlog_valuecounts = series_parsedlog.value_counts()
	parsed_pairs = 0
	for count in series_parsedlog_valuecounts:
		if count > 1:
			parsed_pairs += count*(count-1)/2
	accurate_pairs = 0
	accurate_events = 0
	for parsed_eventId in series_parsedlog_valuecounts.index:
		logIds = series_parsedlog[series_parsedlog == parsed_eventId].index
		series_groundtruth_logId_valuecounts = series_groundtruth[logIds].value_counts()
		error_eventIds = (parsed_eventId, series_groundtruth_logId_valuecounts.index.tolist())
		error = True
		if series_groundtruth_logId_valuecounts.size == 1:
			groundtruth_eventId = series_groundtruth_logId_valuecounts.index[0]
			if logIds.size == series_groundtruth[series_groundtruth == groundtruth_eventId].size:
				accurate_events += logIds.size
				error = False
		for count in series_groundtruth_logId_valuecounts:
			if count > 1:
				accurate_pairs += count*(count-1)/2 #scipy.misc.comb(count, 2)

	precision = float(accurate_pairs) / max(parsed_pairs,1)
	recall = float(accurate_pairs) / max(real_pairs,1)
	f_measure = 2 * precision * recall / max(precision + recall, 1e-16)
	rand_index = 1.0 - float(parsed_pairs+real_pairs-2*accurate_pairs) / max(total_pairs,1)
	TP = accurate_pairs
	FP = parsed_pairs - accurate_pairs
	FN = real_pairs - accurate_pairs
	TN = total_pairs - TP - FP - FN
	conf_mtx = [[TN,FP],[FN,TP]]
	return {
	"precision":precision,
	"recall":recall,
	"f1-score":f_measure,
	"rand-index":rand_index,
	"confusion-matrix":conf_mtx,
	}


def evaluate_header_extraction(graph,logsum,header_format):
	total_field = 0
	correct_field = 0
	ground_truth_names,header_label = search_header_by_frequency(graph,logsum,header_format)
	for i in range(len(ground_truth_names)):
		num_field = len(re.findall(r'<\w+>',ground_truth_names[i]))
		if header_label[i]==True:
			correct_field += num_field
		total_field += num_field
	result = {
		"correct":(correct_field,total_field),
	}
	print("Correct / Total : %d / %d"%(result["correct"][0],result["correct"][1]))
	return result


def evaluate_variable_mapping(map_label_filename,var_index):
	truth_var_index_list = []
	predict_var_index_list = []
	var_format = re.compile(r'^<#\d+>$')
	num_of_variable = 0
	column_names = None
	with open(map_label_filename,"r") as f:
		for line in f.readlines():
			line_ = line.strip()
			pos = line_.find(",")
			line_ = line_[:pos],line_[pos+1:]
			if not column_names:
				column_names = line_
				continue
			log_id = int(line_[0])
			log_txt = line_[1]
			word_list = log_txt.split(" ")
			for j in range(min(len(word_list),len(var_index[log_id]))):
				item = word_list[j]
				if re.match(var_format,item):
					num_of_variable += 1
					truth_var_index_list.append(int(item[2:-1]))
					predict_var_index_list.append(var_index[log_id][j])
	
	truth_var_index_list = pd.Series(truth_var_index_list,index=range(len(truth_var_index_list)))
	predict_var_index_list = pd.Series(predict_var_index_list,index=range(len(predict_var_index_list)))
	evaluate_result = get_accuracy(truth_var_index_list,predict_var_index_list)
	print("Evaluation Result of Vairable-Field Mapping:")
	print("\tRandIndex: %.6lf"%evaluate_result["rand-index"])
	return evaluate_result



def evaluate_word_classification(log_data,log_content,parsing_truth,
			match_tree_test,match_tree_truth,graph=None,logsum=None,
			logsum_pre_matched_result=None):
	f1_micro = []
	TP_macro,FP_macro,FN_macro = 0,0,0
	wt_dict = {}
	f1_micro_wt = []
	TP_macro_wt,FP_macro_wt,FN_macro_wt = 0,0,0
	for item in parsing_truth:
		if item[1] not in wt_dict:
			wt_dict[item[1]] = 0
		wt_dict[item[1]] += 1
	for key in wt_dict:
		wt_dict[key] /= len(log_data)
	wt_sum = 0.0

	match_result = []
	match_truth = []
	variable_global_id = []

	for log_id in range(len(log_data)):
		log = log_data[log_id]
		log_ = prefix_graph_proprocessor(log).split(" ")
		matched_event,word_flag = match_tree_test.deep_match(log_,return_flag=True)
		word_flag = word_flag[:-1]
		log_.pop()
		header_length = len(log_)-len(log_content[log_id].split(" "))
		event_id = parsing_truth[log_id][1]
		match_result.append([log_id+1,matched_event])
		match_truth.append([log_id+1,event_id])

		# Log Paring Correction
		matched_variable_list = [None]*len(log_)
		if graph is not None and logsum is not None:
			match_obj = logsum_pre_matched_result[log_id]
			if match_obj["Unmatched"]==False:
				for i in range(header_length,len(log_)):
					var = match_obj["FieldList"][i] if i<len(match_obj["FieldList"]) else None
					if var is None:
						continue
					conn_id = logsum.uf.find(var.id)-1
					matched_variable_list[i] = (var.id,conn_id) 
					if word_flag[i]==False and len(var.value_set)<=1:
						word_flag[i] = True
		for i in range(header_length,len(log_)):
			if "#" in log_[i]:
				word_flag[i] = False
		
		variable_global_id.append(matched_variable_list[header_length:])
		y_pred = np.array([1 if word_flag[i] else 0 for i in range(header_length,len(log_))])
		
		matched_event0,word_flag0 = match_tree_truth.deep_match(log_,return_flag=True)
		word_flag0 = word_flag0[:-1]
		log_.pop()
		y_true = np.array([1 if word_flag0[i] else 0 for i in range(header_length,len(log_))])

		tp = np.sum(y_pred*y_true)
		fp = np.sum(y_pred*(1-y_true))
		fn = np.sum((1-y_pred)*y_true)
		if 2*tp+fp+fn==0:
			continue
		TP_macro += tp
		FP_macro += fp
		FN_macro += fn
		f1_micro.append((2*tp)/(2*tp+fp+fn))
		TP_macro_wt += tp/wt_dict[event_id]
		FP_macro_wt += fp/wt_dict[event_id]
		FN_macro_wt += fn/wt_dict[event_id]
		f1_micro_wt.append((2*tp)/(2*tp+fp+fn)/wt_dict[event_id])
		wt_sum += 1.0/wt_dict[event_id]
		
	match_result,unmatched = count_unmatch(match_result)
	total = len(match_result)
	print("Unmatched : %d/%d" % (unmatched,total))
	match_result = get_accuracy(match_truth,match_result)
	
	evaluate_result = {
		"RandIndex":match_result["rand-index"],
		"F1_Macro_W":np.mean(f1_micro),
		"F1_Micro_W":(2*TP_macro)/(2*TP_macro+FP_macro+FN_macro),
		"F1_Macro_B":np.sum(f1_micro_wt)/wt_sum,
		"F1_Micro_B":(2*TP_macro_wt)/(2*TP_macro_wt+FP_macro_wt+FN_macro_wt),
	}
	print("Evaluation Result of Template Word Extraction:")
	print("\tRandIndex: %.6lf"%evaluate_result["RandIndex"])
	print("\tF1_Macro_W: %.6lf"%evaluate_result["F1_Macro_W"])
	print("\tF1_Micro_W: %.6lf"%evaluate_result["F1_Micro_W"])
	print("\tF1_Macro_B: %.6lf"%evaluate_result["F1_Macro_B"])
	print("\tF1_Micro_B: %.6lf"%evaluate_result["F1_Micro_B"])
	if graph is not None and logsum is not None:
		return evaluate_result,variable_global_id
	return evaluate_result
