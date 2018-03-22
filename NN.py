import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

#User-defined module
from dataset_kebin.candle import load_candle
from Layer import Layer
from Loss import Loss

import matplotlib.pyplot as plt

def main():
	#layer configuration
	input_dim = 1
	output_dim = 1
	hidden_dims = [128]
	actvs = []
	tau = 256

	#for test run without saved checkpoint
	#model = DNN(mode="train", trunc=True, save=False)
	model = DNN(mode="train", trunc=True, save=False)

	#(should not use) for test run and destroy saved checkpoint
	#model = DNN(mode="train", trunc=True, save=True)

	#for resume saved run and save checkpoint
	#model = DNN(mode="train", trunc=False, save=True)

	#for for test run with saved checkpoint
	#model = DNN(mode="train", trunc=False, save=False)

	#initialize params
	model.init_layer_param(input_dim, output_dim, hidden_dims, actvs, tau)
	model.init_loss_param("mse")

	model.define_placeholder()

	(x_train, t_train), (x_test, t_test) = load_candle(normalize=True, rate='15min', price='close', tau=tau)

	#for sin
	#(x_train, t_train), (x_test, t_test) = load_seq_sin(T=200, tau=tau, rate=0.7)
	
	train_size = int(len(x_train) * 0.8)
	validation_size = len(x_train) - train_size

	x_train, x_validation, t_train, t_validation = train_test_split(x_train, t_train, test_size=validation_size)

	print(x_validation.shape)
	print(t_validation.shape)
	if model.modes["train"] is True:
		#epoch
		batch_size = 32
		keep_prob = 1.0
		epochs= 512

		model.fit(x_train, t_train, x_validation, t_validation, keep_prob, epochs, batch_size)
		accuracy = model.evaluate(x_test, t_test)
		print("evaluate : ", accuracy)

		model.plot()

		if model.is_save is True:
			model.save_session()

	if model.modes["eval"] is True:
		accuracy = model.evaluate(x_test, t_test)
		print(accuracy)

	if model.modes['construct'] is True:
		model.make_model()
		print(None)


class DNN:
	def __init__(self, mode, trunc, save):
		self.weights = []
		self.biases = []
		self.input_dim = None
		self.output_dim = None
		self.hidden_dims = None
		self.actvs = None
		self.tau = None
		self.loss = None

		self._x = None
		self._t = None
		self._keep_prob = None

		self.logits = None
		self._log = {
			"accuracy" : [],
			"loss"     : []
			}

		self._sess = tf.Session()
		self.ckpt_dir = "./Save"
		self.ckpt_path = "./Save/model.ckpt"

		self.modes = {
			"train" : False,
			"eval"  : False,
			'construct' : False
		}
		self.set_mode(mode)
		self.trunc = trunc
		self.is_save  = save

	def __del__(self):
		self._sess.close()

	def plot(self):
		fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9,6), squeeze=False)
		axes[0,0].plot(self._log['loss'])
		plt.savefig('./Output/loss.png')

	def set_mode(self, mode):
		self.modes[mode] = True

	def is_trunc(self):
		if self.trunc is True:
			return True
		else:
			return False

	def save_session(self):
		saver = tf.train.Saver()
		saver.save(self._sess, self.ckpt_path)

	def has_ckpt(self):
		ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
		if ckpt:
			return True
		else:
			return False

	def restore_session(self, path):
		saver = tf.train.Saver()
		saver.restore(self._sess, path)

	def delete_session(self):
		self._sess.close()

	def init_layer_param(self, input_dim, output_dim, hidden_dims, actvs, tau):
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.hidden_dims = hidden_dims
		self.actvs = actvs
		self.tau = tau

	def init_loss_param(self, loss):
		self.loss = loss

	def init_weight(self, shape):
		initial = tf.truncated_normal(shape, stddev=0.01)
		return tf.Variable(initial)

	def init_bias(self, shape):
		initial = tf.zeros(shape)
		return tf.Variable(initial)

	def define_placeholder(self):
		#define variable for TensorFlow
		self._x = tf.placeholder(tf.float32, shape=[None, self.tau, self.input_dim])
		self._t = tf.placeholder(tf.float32, shape=[None, self.output_dim])
		self._keep_prob = tf.placeholder(tf.float32)
		self.batch_size = tf.placeholder(tf.int32, shape=[])

	def set_dropout(self, src, keep_prob):
		return tf.nn.dropout(src, keep_prob)

	def infer(self, x, keep_prob):
		hidden_units = self.hidden_dims[0]
		cell = tf.contrib.rnn.BasicRNNCell(hidden_units)# hidden_units = self.hidden_dims[0]
		#cell = tf.contrib.rnn.BasicLSTMCell(hidden_units)
		initial_state = cell.zero_state(self.batch_size, tf.float32)

		state = initial_state
		cell_outputs = []
		with tf.variable_scope("RNN"):
			for t in range(self.tau):
				if t > 0:
					tf.get_variable_scope().reuse_variables()
				(cell_output, state) = cell(x[:,t,:], state)
				cell_outputs.append(cell_output)

		output = cell_outputs[-1]

		V = self.init_weight([hidden_units, self.output_dim])
		c = self.init_bias([self.output_dim])
		y = tf.matmul(output, V) + c
		return y
	
	def train(self, loss):
		optimizer = tf.train.AdamOptimizer(learning_rate = 0.001, beta1=0.9, beta2=0.999)
		train_step = optimizer.minimize(loss)
		return train_step

	def calc_loss(self, y, t):
		#call Loss class and set loss function
		loss = Loss(self.loss).switch_loss()
		return loss.calc(y, t)

	def set_layer(self, src, src_dim, dst_dim, actv_func):
		self.weights.append(self.init_weight([src_dim, dst_dim]))
		self.biases.append(self.init_bias([dst_dim]))

		#call Layer class and set acticate function
		layer = Layer(actv_func).switch_layer()
		return layer.set(src, self.weights[-1], self.biases[-1])

	def calc_accuracy(self, y, t):
		correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(t, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
		return accuracy

	def evaluate(self, x_test, t_test):
		if self.modes["train"] is True:
			#acc_op = self.calc_accuracy(self._logits, self._t)
			acc_op = self.calc_loss(self._logits, self._t)

		if self.modes["eval"] is True:
			y = self.infer(self._x, self._keep_prob)
			loss_op = self.calc_loss(y, self._t)
			train_op = self.train(loss_op)

			acc_op = self.calc_accuracy(y, self._t)

			ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
			if self.should_restore() is True:
				print(ckpt.model_checkpoint_path)
				self.restore_session(ckpt.model_checkpoint_path)
			else:
				init = tf.global_variables_initializer()
				self._sess.run(init)

		accuracy = acc_op.eval(session=self._sess,
			feed_dict = 
			{
				self._x : x_test,
				self._t : t_test,
				self.batch_size : x_test.shape[0]
			}
		)



		return accuracy

	def should_restore(self):
		if self.trunc is False and self.has_ckpt() is True:
			return True
		else:
			return False

	def make_model(self):
		#define model
		y = self.infer(self._x, self._keep_prob)
		#loss = DNN.calc_cross_entropy(y, t)
		loss_op = self.calc_loss(y, self._t)
		train_op = self.train(loss_op)
		return None

	def fit(self, x_train, t_train, x_validation, t_validation, prob, epochs, batch_size):
		#define model
		y = self.infer(self._x, self._keep_prob)
		#loss = DNN.calc_cross_entropy(y, t)
		loss_op = self.calc_loss(y, self._t)
		train_op = self.train(loss_op)

		#for restore
		self._logits = y

		#define calc accuracy
		acc_op = self.calc_accuracy(y, self._t)

		ckpt = tf.train.get_checkpoint_state(self.ckpt_dir)
		if self.should_restore():
			print(ckpt.model_checkpoint_path)
			self.restore_session(ckpt.model_checkpoint_path)
		else:
			init = tf.global_variables_initializer()
			self._sess.run(init)

		batchs = x_train.shape[0] // batch_size

		count = 0
		print(x_validation.shape)
		for epoch in range(epochs):
			for i in range(batchs):
				batch_mask = np.random.choice(x_train.shape[0], batch_size)
				x_batch = x_train[batch_mask]
				t_batch = t_train[batch_mask]

				train_op.run(session=self._sess, 
					feed_dict = 
					{
						self._x : x_batch,
						self._t : t_batch,
						self.batch_size : batch_size
					}
				)
				
			loss = loss_op.eval(session=self._sess,
				feed_dict =
				{
					self._x : x_validation,
					self._t : t_validation,
					self.batch_size : x_validation.shape[0]
				}
			)
			self._log["loss"].append(loss)
			"""
			accuracy = acc_op.eval(session=self._sess,
				feed_dict =
					{
						self._x : x_train,
						self._t : t_train,
						self.batch_size : 123
					}
			)
			self._log["accuracy"].append(accuracy)
			print("epoch : ", epoch, "loss : ", loss, "accuracy : ", accuracy)
			"""
			print("epoch : ", epoch, "loss : ", loss)

			#pick one sequential data (to give batch_size = 1)
			batch_size = 1
			batch_mask = np.random.choice(x_validation.shape[0], batch_size)
			x_batch = x_validation[batch_mask]
			t_batch = t_validation[batch_mask]

			predict = y.eval(session=self._sess,
				feed_dict =
				{
					self._x : x_batch,
					self.batch_size : x_batch.shape[0]
				}
			)

			print('gosa ',t_batch - predict)
			if np.abs(t_batch - predict) < 0.0001:
				count += 1
			print('ratio : ' , count / (epoch+1))
			fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9,6), squeeze=False)
			correct = np.append(x_batch, t_batch)
			predict = np.append(x_batch, predict)
			axes[0,0].set_ylim(np.min(correct),np.max(correct))
			axes[0,0].plot(correct, color='red', linestyle='dashed')
			axes[0,0].plot(predict, linestyle='solid')
			plt.savefig('./Output/' + str(epoch) +'.png')
			plt.close()

		

if __name__ == '__main__':
	main()
