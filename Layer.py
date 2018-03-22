import tensorflow as tf
#Layer class is called by set_layer from DNN class
class Layer:
	def __init__(self, actv):
		self.actv = actv

	def set(self, src, weight, bias):
		pass

	def switch_layer(self):
		if   self.actv is "relu":
			return ReLULayer()

		elif self.actv is "softmax":
			return SoftmaxLayer()

		elif self.actv is "tanh":
			return TanhLayer()

		elif self.actv is "sigmoid":
			return SigmoidLayer()
			
		elif self.actv is "identity":
			return IdentityLayer()

		else:
			return None
		
class ReLULayer(Layer):
	def __init__(self):
		pass

	def set(self, src, weight, bias):
		return tf.nn.relu(tf.matmul(src, weight) + bias)

class SoftmaxLayer(Layer):
	def __init__(self):
		pass

	def set(self, src, weight, bias):
		return tf.nn.softmax(tf.matmul(src, weight) + bias)

class TanhLayer(Layer):
	def __init__(self):
		pass

	def set(self, src, weight, bias):
		return tf.nn.tanh(tf.matmul(src, weight) + bias)

class SigmoidLayer(Layer):
	def __init__(self):
		pass

	def set(self, src, weight, bias):
		return tf.nn.sigmoid(tf.matmul(src, weight) + bias)

class IdentityLayer(Layer):
	def __init__(self):
		pass

	def set(self, src, weight, bias):
		return tf.nn.l2_normalize(tf.matmul(src, weight) + bias)

if __name__ == '__main__':
	printf('Loss module for TensorFlow Neural Network')