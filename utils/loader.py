def load_data(filename):
	logs = []
	with open(filename,"r") as f:
		for line in f.readlines():
			logs.append(line.replace("\n",""))
	return logs

