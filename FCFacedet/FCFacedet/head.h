#include <caffe/common.hpp>
#include <caffe/layer.hpp>
#include <caffe/layer_factory.hpp>
#include <caffe/layers/input_layer.hpp>
#include <caffe/layers/inner_product_layer.hpp>
#include <caffe/layers/dropout_layer.hpp>
#include <caffe/layers/conv_layer.hpp>
#include <caffe/layers/relu_layer.hpp>
#include <caffe/layers/prelu_layer.hpp>

#include <caffe/layers/pooling_layer.hpp>
#include <caffe/layers/lrn_layer.hpp>
#include <caffe/layers/softmax_layer.hpp>
#include<caffe/layers/split_layer.hpp>

#include "osplatformutil.h"

namespace caffe
{
#if defined I_OS_WIN32
	extern INSTANTIATE_CLASS(InputLayer);
	extern INSTANTIATE_CLASS(InnerProductLayer);
	extern INSTANTIATE_CLASS(DropoutLayer);
	extern INSTANTIATE_CLASS(ConvolutionLayer);
	REGISTER_LAYER_CLASS(Convolution);
	extern INSTANTIATE_CLASS(ReLULayer);
	REGISTER_LAYER_CLASS(ReLU);

	extern INSTANTIATE_CLASS(PoolingLayer);
	REGISTER_LAYER_CLASS(Pooling);
	/*extern INSTANTIATE_CLASS(LRNLayer);
	REGISTER_LAYER_CLASS(LRN);*/
	extern INSTANTIATE_CLASS(SoftmaxLayer);
	REGISTER_LAYER_CLASS(Softmax);

	extern INSTANTIATE_CLASS(PReLULayer);
	//REGISTER_LAYER_CLASS(PReLU);
	extern INSTANTIATE_CLASS(SplitLayer);
#endif
}
