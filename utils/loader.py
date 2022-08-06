import csv
import os
import re
import json
import sys
sys.path.append("./utils")
from evaluation import MatchTree

def read_csv(filename,encoding=None):
	with open(filename,"r",encoding=encoding) as f:
		data = list(csv.reader(f))
	return data

def write_csv(data,filename,encoding=None):
	with open(filename,"w+",encoding=encoding,newline='') as f:
		csv.writer(f).writerows(data)
	return 

def load_data(logfile_name,encoding=None):
	logs = []
	with open(logfile_name,"r",encoding=encoding) as f:
		for line in f.readlines():
			line_ = line.replace("\n","") #.replace("|","")
			line_ = re.sub(r'\s+'," ",line_)
			logs.append(line_)
	return logs

def load_log_templates(template_name):
	all_templates = load_data(template_name,encoding="utf-8")
	match_tree = MatchTree("<*>")
	match_tree.fit_templates(all_templates)
	return match_tree,all_templates

def generate_template_groundtruth(filename_structured_log,filename_output_template):
	data = read_csv(filename_structured_log)
	data = data[1:]
	log_templates = set()
	for line in data:
		template_word = ["<*>"]
		raw_template = re.sub(r'\s+',' ',line[-1].strip())
		for item in raw_template.split(" "):
			if "<*>" not in item:
				template_word.append(item)
			elif template_word[-1]!="<*>":
				template_word.append("<*>")
		if template_word[-1]!="<*>":
			template_word.append("<*>")
		log_templates.add(" ".join(template_word))
	sorted_templates = sorted(list(log_templates))
	with open(filename_output_template,"w+",encoding="utf-8") as f:
		for line in sorted_templates:
			f.write(line+"\n")
	return sorted_templates

def load_groundtruth_log_parsing(filename):
	data = read_csv(filename)
	data = data[1:]
	log_content = []
	parsing_truth = []
	for line_id in range(len(data)):
		line = data[line_id]
		content = re.sub(r'\s+',' ',line[-3].strip())
		eventid = re.sub(r'\s+',' ',line[-2].strip())
		log_content.append(content)
		parsing_truth.append([line_id,eventid])
	return log_content,parsing_truth

def load_header_groundtruth(filename):
	with open(filename,"r") as f:
		data = json.load(f)
	return data

def write_structured_logs(data,filename):
	json_str = json.dumps(data)
	with open(filename,"w+") as f:
		#json.dump(f,data)
		f.write(json_str)
	return
