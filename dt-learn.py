import sys
from scipy.io import arff
from cStringIO import StringIO
import numpy as num
import pandas as pd
import math
import operator
from collections import OrderedDict

sp = " "
true_count = 0
tab ="\t"

feature_type = OrderedDict()	

def most_common(data):
	pos = data[class1][data[class1] == 'positive'].count()
	neg = data[class1][data[class1] == 'negative'].count()
	if pos > neg:
		return 'positive'
	else:
		return 'negative'

def compute_entropy(data):
	data_count = data[class1].count()
	pos = data[class1][data[class1] == 'positive'].count()
	neg = data[class1][data[class1] == 'negative'].count()
	if pos == 0 or neg ==0 or data_count == 0:
		return 0
	else:
		if data_count > 0 and pos > 0 and neg>0:
			p =  pos / float(data_count)
			q =  neg / float(data_count)
			entropy = - ( p * math.log(p,2) ) - ((q) * math.log(q,2))

			return entropy

class1 = "class"

def parse_input(train_file):
	(data,metadata) = arff.loadarff(open(train_file, 'r'))
	global feature_type
	global feature_range
	feature_range = {}
	feature_used = {} #initially defining usage of all features as false
	feature_names = []
	for name in metadata.names():
		feature_names.append(name)

	for name in metadata.names():
		feature_type[name] = metadata[name][0]
	del feature_type['class']

	for name in metadata.names():
		if name != "class":
			if feature_type[name] == 'nominal':
				feature_range[name] = metadata[name][1]

	return data, metadata, feature_range, feature_type


def data_split_numerical_util2(data,attribute):
	candidates = []
	attribute_values_sorted = data.sort_values(by=attribute)
	attribute_values = attribute_values_sorted[attribute].unique()
	#print attribute_values
	len1 = len(attribute_values)
	if(len1 == 1):
		candidates.append(attribute_values[0])
		return candidates
	#print len1
	for i in range(0,len1-1):
		set1 = data[data[attribute] == attribute_values[i]] 
		set2 = data[data[attribute] == attribute_values[i+1]] 
		set3 = set1[class1].tolist() + set2[class1].tolist()
		if(len(set(set3)) == 2):
			candidates.append((attribute_values[i]+attribute_values[i+1])/2.0)
	return candidates




def data_split_numerical(data,attribute):
	candidates = data_split_numerical_util2(data,attribute)
	entropy_value = {}
	for candidate in candidates:

		lower_subset = data[data[attribute] <= candidate ]
		higher_subset = data[data[attribute] > candidate ]
		entropy_value[candidate] = ((len(lower_subset.index))/float(len(data.index)))*compute_entropy(lower_subset) + ((len(higher_subset.index))/float(len(data.index)))*compute_entropy(higher_subset)
		#info_gain_value[candidate] = compute_entropy(data) - ((len(lower_subset.index))/float(len(data.index)))*compute_entropy(lower_subset) + ((len(higher_subset.index))/float(len(data.index)))*compute_entropy(higher_subset)

	min_entropy_threshold = min(entropy_value, key=entropy_value.get)
  	return [min_entropy_threshold,  entropy_value[min_entropy_threshold]] #returns the threshold value and entropy value of threshold 




def data_split_nominal(data,item):
	v_dict = {}
	for index,row in data.iterrows():
		if(v_dict.has_key(row[item])):
			v_dict[row[item]] += 1.0
		else:
			v_dict[row[item]] = 1.0
	subset_entropy = 0
	for ele in v_dict.keys():
		coeff = v_dict[ele] / sum(v_dict.values())
		subset_data = data[data[item] == ele]
		subset_entropy += coeff * compute_entropy(subset_data)
	res = subset_entropy
	return res

def get_info_gain(data, feature_type):
	
	min_entropy = {}
	for i,attribute in enumerate(feature_type):
		if feature_type[attribute] == 'numeric':
			min_entropy[attribute] = data_split_numerical_utility(data,attribute) #sending the index to 
		if feature_type[attribute] == 'nominal':
			min_entropy[attribute] = data_split_nominal(data ,attribute)

	min_entropy_value = min(min_entropy.values())
	min_entropy_key = []
	for key in min_entropy.keys():
		if min_entropy[key] == min_entropy_value:
			min_entropy_key.append(key)
	if len(min_entropy_key) == 1: #only one key found,so return
		return min_entropy_key[0]
	for item in feature_type.keys():
		for key in min_entropy_key:
			if (key == item):
				return key



def find_best_split(data,feature_type):
	return get_info_gain(data,feature_type)

def data_split_numerical_utility(data,attribute):
	result = data_split_numerical(data,attribute)
	return result[1]

#returns the best split candidate
def data_split_numerical_threshold(data,attribute):
	result = data_split_numerical(data,attribute)
	return result[0]




class MakeTreeNode:
	def __init__(self,label = "",isleaf = False,attribute_name="",operator = "" ,operand = "",type = "",pos="",neg = ""):
		self.attribute_name =attribute_name
		self.label = label
		self.isleaf = isleaf
		self.node = []
		self.operator = operator
		self.operand = operand
		self.type = type
		self.pos = pos
		self.neg = neg





def get_trimmed(feature_type1,best_split_attribute):
	trimmed_featured_type = OrderedDict()
	for key,value in feature_type1.items():
		if key != best_split_attribute:
			trimmed_featured_type[key] = value
	return trimmed_featured_type

	#first time all the attributes are passed
def MakeTree(data,attribute,m,parent_data=None):
	Node = []
	#checking if all the class values are positive
	if (data.empty or len(attribute.keys()) <= 0):
		p = data[class1][data[class1] == 'positive'].count()
		n = data[class1][data[class1] == 'negative'].count()
		#print "\t Empty data reached - ", most_common(data)
		val = most_common(parent_data)
		Node.append(MakeTreeNode(val,True,pos=p,neg=n))
		return Node

	#Rule 5
	if (len(data.index) < m):
		p = data[class1][data[class1] == 'positive'].count()
		n = data[class1][data[class1] == 'negative'].count()
		if p == n:
			val = most_common(parent_data)
		else:
			val = most_common(data)
		#print "\t Reached m = ",m," - Made leaf - ",val
		Node.append(MakeTreeNode(val, True,pos=p,neg=n))
		return Node

		#Rule 4.i
	if (len(data['class'].unique()) == 1 and data['class'].unique() =='positive'):
		p = data[class1][data[class1] == 'positive'].count()
		n = data[class1][data[class1] == 'negative'].count()
		#print "\t Made First leaf - positive label"
		Node.append(MakeTreeNode('positive', True,pos=p,neg=n))
		return Node
		#Rule 4.i
	if (len(data['class'].unique()) == 1 and data['class'].unique() =='negative'):

		p = data[class1][data[class1] == 'positive'].count()
		n = data[class1][data[class1] == 'negative'].count()
		#print "\t Made Second leaf - negative label "
		Node.append(MakeTreeNode('negative', True,pos=p,neg=n))
		return Node

	 #other cases remaining
	else:

		best_split_attribute = find_best_split(data, attribute)
		#print "  - best attribute_values  - ",best_split_attribute

		if attribute[best_split_attribute] == 'numeric':
			threshold_value = data_split_numerical_threshold(data,best_split_attribute)
			#print "\t In Numeric Child",'{0:6f}'.format(threshold_value))
			

			lower_subset = data[data[best_split_attribute] <= threshold_value]
			higher_subset = data[data[best_split_attribute] > threshold_value] 

			N1pos = lower_subset[class1][lower_subset[class1] == 'positive'].count()
			N1neg = lower_subset[class1][lower_subset[class1] == 'negative'].count()

			N2pos = higher_subset[class1][higher_subset[class1] == 'positive'].count()
			N2neg = higher_subset[class1][higher_subset[class1] == 'negative'].count()

			Tree_New_Node1 = MakeTreeNode(None,False,best_split_attribute,'<=',str(format(threshold_value, '.6f')),'Numeric',pos =N1pos,neg =N1neg)
			Tree_New_Node2 = MakeTreeNode(None,False,best_split_attribute,'>',str(format(threshold_value, '.6f')),'Numeric',pos =N2pos,neg= N2neg)
			#trimmed_attribute = get_trimmed(attribute,best_split_attribute)
			Tree_New_Node1.node = MakeTree(lower_subset, attribute,m,data)
			Tree_New_Node2.node = MakeTree(higher_subset, attribute,m,data)

			Node.append(Tree_New_Node1)
			Node.append(Tree_New_Node2)
	

		else:
			for value in feature_range[best_split_attribute]:
				#print "\t In Nominal Child - ", value
				
				data_temp = data[data[best_split_attribute] == value]
				trimmed_attribute = get_trimmed(attribute,best_split_attribute)

				p = data_temp[class1][data_temp[class1] == 'positive'].count()
				n = data_temp[class1][data_temp[class1] == 'negative'].count()

				Tree_New_Node = MakeTreeNode(None, False, best_split_attribute,'=',value,'Nominal',pos=p,neg=n)
				Tree_New_Node.node = MakeTree(data_temp,trimmed_attribute,m,data) #passing the trimmed attribute
				Node.append(Tree_New_Node)
	return Node



def display(a,level,flag):
	if a.isleaf == True:
		print ": " + a.label,; sys.stdout.softspace=False;
		return
	if flag:
		print ""
	flag = True
	p = level
	while(p):
		print "|\t",; sys.stdout.softspace=False;
		p -=1
	print a.attribute_name + sp + a.operator + sp+ a.operand + sp + "[" + str(a.neg) + sp + str(a.pos) + "]",; sys.stdout.softspace=False;
	for element in a.node:
		display(element,level+1,flag)

			#print type(item[])
			#

def traverse(instance,root):
	if root.isleaf == True:
		print root.label
		if (root.label == instance[class1]):
			global true_count
			true_count +=1
		return
	if root.type == 'Nominal':
		if (instance[root.attribute_name] == root.operand):
			for element in root.node:
				traverse(instance,element)
	else:
		if root.operator == '<=':

			if (instance[root.attribute_name] <= float(root.operand)):
				for element in root.node:
					traverse(instance, element)
		else:
			if (instance[root.attribute_name] > float(root.operand)):
				for element in root.node:
					traverse(instance,element)


#instance,item,predicted_label
def traverse_with_label_count(instance,root,predicted_label):
	if root.isleaf == True:
		predicted_label.append(root.label)
		return
	if root.type == 'Nominal':
		if (instance[root.attribute_name] == root.operand):
			for element in root.node:
				traverse_with_label_count(instance,element,predicted_label)
	else:
		if root.operator == '<=':

			if (instance[root.attribute_name] <= float(root.operand)):
				for element in root.node:
					traverse_with_label_count(instance, element,predicted_label)
		else:
			if (instance[root.attribute_name] > float(root.operand)):
				for element in root.node:
					traverse_with_label_count(instance,element,predicted_label)

def predict(test_data,root):
	for index,instance in test_data.iterrows():
		print str(index+1)+": Actual: "+instance[class1]+ " Predicted:",
		for item in root:
			traverse(instance,item)

#test_data_set,root,predicted_label,given_label
def predict_with_label(test_data_set,root,predicted_label):
	for index,instance in test_data_set.iterrows():
		for item in root:
			traverse_with_label_count(instance,item,predicted_label)

def node_count(root):
	if root.isleaf == True:
		return 1
	else:
		val = 1
		for item in root.node:
			val +=node_count(item)
		return val


def learning_curve_2(data,test_data_set):

	print "x" 
	training_set_percent = [.05,.1,.2,.5,1]
	samples = 10
	len_training_set_size = len(training_set_percent)

	for percentage in training_set_percent:
		
		
		accuracy_value = []

		for count in range(10):
			given_label = []
			predicted_label =[]
			df_sample = data.sample(frac=percentage)
			root = MakeTree(df_sample,feature_type,4)

			predict_with_label(test_data_set,root,predicted_label)
			predict_count = len(predicted_label)

			for z,element in test_data_set.iterrows():
				given_label.append(element[class1])

			predicted_correct = 0.0

			for p in range(len(predicted_label)):
				if(given_label[p] == predicted_label[p]):
					predicted_correct += 1.0
			accuracy_value.append(predicted_correct/predict_count)
		accuracy_value_sorted = sorted(accuracy_value)

		average_test_set_accuracy = num.mean(accuracy_value_sorted)

		print str(percentage) + "," +str(format(accuracy_value_sorted[0], '.6f')) + "," + str(format(average_test_set_accuracy, '.6f')) + "," +str(format(accuracy_value_sorted[9], '.6f'))


def learning_curve_3(data,test_data_set):

	print "x" + "," + "accuracy"
	training_set_percent = [2,5,10,20]
	len_training_set_size = len(training_set_percent)

	for percentage in training_set_percent:
		given_label = []
		predicted_label =[]
		accuracy_value = 0.0
		root = MakeTree(data,feature_type,percentage)

		predict_with_label(test_data_set,root,predicted_label)
		for z,element in test_data_set.iterrows():
			given_label.append(element[class1])
		predict_count = len(predicted_label)
		predicted_correct = 0.0
		for p in range(len(predicted_label)):
			if(given_label[p] == predicted_label[p]):
				predicted_correct += 1.0
		accuracy_value = predicted_correct/predict_count

		print str(percentage) + "," + str(format(accuracy_value, '.6f'))

def start():
	train_file = "heart_train.arff"
	data,metadata,feature_range,feature_type = parse_input(train_file)
	training_data = pd.DataFrame.from_records(data)	
	m = 2

	root = MakeTree(training_data,feature_type,m)
	
	test_file = "heart_test.arff"
        
	data1, meta1 = arff.loadarff(open(test_file, 'r'))
  	test_data_set = pd.DataFrame.from_records(data1)

  	#learning_curve_2(training_data,test_data_set)
  	#learning_curve_3(training_data,test_data_set)
  	
	flag = False
	#calling the tree
	for item in root:
		display(item,0,flag)
		flag = True
	print ""
	
	print "<Predictions for the Test Set Instances>"
	predict(test_data_set,root)
	print "Number of correctly classified: "+ str(true_count) +" Total number of test instances:" + str(len(test_data_set.index))
	
	#val = 1
	#for item in root:
	#	val += node_count(item)

if __name__ == "__main__":
	
	start()






