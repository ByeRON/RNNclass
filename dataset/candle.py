import os.path
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/candledata.pkl"

file_name  = "kebinGBPJPY_M1.txt"
valid_date = "2013-02-04 00:00:00"

def comvert_dataframe(file_name):
	file_path = dataset_dir + '/' + file_name

	print("Converting " + file_name + " to pandas Dataframe")
	df = pd.read_table(file_path, 
			names=("date","time","open", "high", "low", "close", "volume"),
			dtype = {"date" : "object", "time" : "object"}
		)
	
	df["time"] = pd.to_datetime(df["date"].str.cat(df["time"], sep=' '))
	df.index = df["time"]
	df = df.drop(["date", "volume", "time"], axis=1)

	print("Done")
	return df

def extract_valid(df):
	df = df[valid_date:]
	return df

def init_candle():
	dataset = comvert_dataframe(file_name)

	#extract valid data fpr kebbin
	dataset = extract_valid(dataset)

	print("Creating pickle file ...")
	with open(save_file, 'wb') as f:
		pickle.dump(dataset, f, -1)
	print("Done")
	
def reshape_candle(data, start, end, sticks):
	arr = data.values[start:end].reshape(-1, 4 * sticks)
	return arr

def make_mask(data_size, mask_size):
	return np.random.choice(data_size, mask_size, replace=False)

def resample_candle(data, rate):
	data = data.resample(rate).agg(
		{
			"open" : "first",
			"high" : "max"  ,
			"low"  : "min"  ,
			"close": "last"
		}
	)
	data = data.dropna()
	return data[["open", "high", "low", "close"]]

def make_seqdata(data, price, tau=20):
	x_ = []
	t_ = []

	data = data[price]
	data = data.values
	itr = data.shape[0] - tau
	for i in range(itr):
		x_.append(data[i:i+tau])
		t_.append(data[i+tau])
	x_ = np.asarray(x_)
	t_ = np.asarray(t_)

	x_ = x_.reshape(-1, tau, 1)
	t_ = t_.reshape(-1,1)


	"""
	x_ = np.array([])
	t_ = np.array([])

	data = data[price]
	data = data.values
	itr = data.shape[0] - tau
	for i in range(itr):
		x_ = np.append(x_, data[i:i+tau])
		t_ = np.append(t_, data[i+tau])
	x_ = x_.reshape(-1, tau, 1)
	t_ = t_.reshape(-1,1)
	"""

	return (x_ , t_)

def split_train_test(x_, t_, prob):
	dataset = {}

#	train_mask = make_mask(x_.shape[0], int(x_.shape[0] * prob * 100) // 100)
#	test_mask  = make_mask(x_.shape[0], int(x_.shape[0] * (1 - prob) * 100) // 100)

#	dataset["x_train"] = x_[train_mask]
#	dataset["t_train"] = t_[train_mask]
#	dataset["x_test"]  = x_[test_mask]
#	dataset["t_test"]  = t_[test_mask]

	prob = 1 - prob
	dataset['x_train'], dataset['x_test'], dataset['t_train'], dataset['t_test'] = train_test_split(x_, t_, test_size=prob)

	return dataset

def normalize_candle(dataset, norm):
	dataset["x_train"] = dataset["x_train"] / norm
	dataset["t_train"] = dataset["t_train"] / norm
	dataset["x_test"] = dataset["x_test"] / norm
	dataset["t_test"] = dataset["t_test"] / norm

	return dataset

def load_candle(normalize=True, rate="1H", tau=20, price='close'):

	print("load dataset ... ")
	if not os.path.exists(save_file):
		init_candle()

	with open(save_file, 'rb') as f:
		data = pickle.load(f)
	
	"""
	data = data.resample(rate).agg(
		{
			"open" : "first",
			"high" : "max"  ,
			"low"  : "min"  ,
			"close": "last"
		}
	)
	data = data.dropna()
	data = data[["open", "high", "low", "close"]]
	
	print(data)
	"""
	if rate != "1min":
		data = resample_candle(data, rate)

	"""
	x_ = np.empty((0, 4 * in_sticks))
	t_ = np.empty((0, 4 * out_sticks))

	itr = data.shape[0] - (in_sticks + out_sticks - 1)
	for i in range(itr):
		#start = i
		#end   = start + in_sticks
		#arr = data.values[start:end].reshape(-1, 4 * in_sticks)
		arr = reshape_candle(data, i, i + in_sticks, in_sticks)
		#x_train = np.append(x_train,np.array([tmp[0]]), axis=0)
		x_ = np.concatenate( [x_, np.array([arr[0]]) ], axis=0)

		#start = i + in_sticks
		#end   = start + out_sticks
		#arr = data.values[start:end].reshape(-1, 4 * out_sticks)
		arr = reshape_candle(data, i + in_sticks, i + in_sticks + out_sticks, out_sticks)
		t_ = np.concatenate( [t_, np.array([arr[0]]) ], axis=0)
	"""
	(x_, t_) = make_seqdata(data, price=price, tau=tau)

	"""
	train_mask_size = x_.shape[0] * 8 // 10
	train_mask = np.random.choice(x_.shape[0], train_mask_size, replace=False)
	test_mask_size = x_.shape[0] - train_mask_size
	test_mask = np.random.choice(x_.shape[0], test_mask_size, replace=False)
	"""
	"""
	train_mask = make_mask(x_.shape[0], x_.shape[0] * 8 // 10)
	test_mask  = make_mask(x_.shape[0], x_.shape[0] * 2 // 10)

	dataset = {}

	dataset["x_train"] = x_[train_mask]
	dataset["t_train"] = t_[train_mask]
	dataset["x_test"]  = x_[test_mask]
	dataset["t_test"]  = t_[test_mask]
	"""
	dataset = split_train_test(x_, t_, 0.8)

	if normalize:
		norm = data["close"].max()
		dataset = normalize_candle(dataset, norm)

	print("Done!")
	return (dataset['x_train'], dataset['t_train']), (dataset['x_test'], dataset['t_test']) 

if __name__ == "__main__":
	load_candle()
