import tensorflow as tf
#Loss class is called by calc_loss from DNN class
class Loss:
	def __init__(self, loss):
		self.loss = loss

	def calc(self, y, t):
		pass

	def switch_loss(self):
		if   self.loss is "mse":
			return MSELoss()

		elif self.loss is "cross_entropy":
			return CrossEntropyLoss()
			
		else:
			return DefaultLoss()

class MSELoss(Loss):
	def __init__(self):
		pass

	def calc(self, y, t):
		return tf.reduce_mean(tf.square(y - t))

class CrossEntropyLoss(Loss):
	def __init__(self):
		pass

	def calc(self, y, t):
		return tf.reduce_mean(-tf.reduce_sum(t * tf.log(y), reduction_indices=[1]))

class DefaultLoss(Loss):
	def __init__(self):
		pass

	def calc(self, y, t):
		return y

if __name__ == '__main__':
	printf('Layer module for TensorFlow Neural Network')