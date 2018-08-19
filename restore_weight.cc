#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include <string>
#include <bitset>
#include <queue>
#include <vector>

using namespace tensorflow;
using namespace std;

/*
    Register Restore operation
*/

REGISTER_OP("RestoreWeight")
  .Input("index: int8")
  .Input("centers: float")
  .Input("shape: int32")
  .Input("freq: int32")
  .Output("output: float")
  .Doc(R"doc(
  Use index and centers to restore the weights in original structure,
  and pass to the conv2d op. Cannot only index and centers because of the
  specific way Eigen choose to do the convolution.
  )doc");

struct huffman_node {
	int id;																				//character
	int freq;																			//frequency of the character
	huffman_node* left;
	huffman_node* right;
	huffman_node()
	{//constructer
		left = right = NULL;
	}
};

/*
    Restore Operation CPU
*/

int32* get_labels(const int32* freq, string huffman_code, int num, int32 samples) {
  
  class compare {//a object funtion to set comparing rule of priority queue
  public:
      bool operator()(const huffman_node* c1, const huffman_node* c2) const {
        return c1->freq > c2->freq;
      }
  };
  huffman_node* root;
  priority_queue<huffman_node*, vector<huffman_node*>, compare> pq;
  auto* node_array = new huffman_node[num];
  for(unsigned int i = 0; i < num; i++) {
    node_array[i].freq = freq[i];
    node_array[i].id = i;
  }
  for(int i = 0; i < num; i++) pq.push(&node_array[i]);
  priority_queue<huffman_node*, vector<huffman_node*>, compare> temp(pq);
  while (temp.size() > 1) {//create the huffman tree with highest frequecy characher being leaf from bottom to top
    root = new huffman_node;
    root->freq = 0;
    root->left = temp.top();
    root->freq += temp.top()->freq;
    temp.pop();
    root->right = temp.top();
    root->freq += temp.top()->freq;
    temp.pop();
    temp.push(root);
  }

  // bool start = true;
  int label_index = 0;
  huffman_node* cur = root;
  int32* labels = new int32[samples];
//  cout<<huffman_code<<endl;
  for (unsigned int i = 0; i < huffman_code.size(); i++) {
    // if (start) tmp = root;
    if (huffman_code.at(i) == '0') {
      cur = cur->left;
//      cout<<"left"<<endl;
    }
    if (huffman_code.at(i) == '1') {
      cur = cur->right;
//      cout<<"right"<<endl;
    }
    if (cur->left == nullptr && cur->right == nullptr) {
//      cout<<"i: "<<i<<"; freq: "<<cur->freq<<"; id: "<<cur->id<<endl;
      labels[label_index] = cur->id;
      label_index++;
      cur = root;
    }
  }
  // cout<<"[Debug] decoding result is correct? "<<(label_index == samples)<<endl;
  return labels;
}




class RestoreWeightOp : public OpKernel {
public:
  explicit RestoreWeightOp(OpKernelConstruction* context) : OpKernel(context) {
  }

  void Compute(OpKernelContext* context) override {

    // get the index tensor(the last byte is the length info)
    const Tensor& index = context->input(0);

    // get the centers tensor
    const Tensor& centers = context->input(1);

    // get the shape tensor
    const Tensor& shape = context->input(2);

    // get the size tensor
    const Tensor& freq = context->input(3);

    //Check that centers are two dimensional
    DCHECK_EQ(index.dims(), 1);
    DCHECK_EQ(centers.dims(), 2);
    DCHECK_EQ(shape.dims(), 1);
    DCHECK_EQ(freq.dims(), 1);
    DCHECK_EQ(centers.dims(), 2);

    auto shape_tensor = shape.tensor<int32, 1>();
    int filter_height = shape_tensor(0);
    int filter_width = shape_tensor(1);
    int input_depth = shape_tensor(2);
    int output_depth = shape_tensor(3);


    // create output shape
    TensorShape output_shape;

    output_shape.AddDim(filter_height);
    output_shape.AddDim(filter_width);
    output_shape.AddDim(input_depth);
    output_shape.AddDim(output_depth);
    // create output tensor
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));

    // get the corresponding Eigen tensors for data access

    // Scope root = Scope::NewRootScope();

    auto centers_tensor = centers.tensor<float, 2>();
    auto output_tensor = output->tensor<float, 4>();
    auto index_tensor = index.tensor<int8, 1>();
    auto freq_tensor = freq.tensor<int32, 1>();

    int32 *freq_array = new int32[centers.dim_size(0)];
    for (int i = 0; i < freq.dim_size(0); i++) freq_array[i] = freq_tensor(i);

    string huffman_code = "";

    for (int i = 0; i < index.dim_size(0); i++) {
      huffman_code.append(bitset<8>(index_tensor(i)).to_string());
    }
    string appendix_string;
    appendix_string = huffman_code.substr(huffman_code.size() - 8, 8);
    int appendix_int = 0;
    for (int i = 0; i < 8; i++) {
      if (appendix_string.at(i) == '1') appendix_int = i;
    }
    huffman_code = huffman_code.substr(0, huffman_code.size() - appendix_int - 8);
    // cout<<"[Debug] appendix_len is:"<<appendix_int<<endl;

    int32* result_labels = get_labels(freq_array, huffman_code, freq.dim_size(0), output_depth*filter_height*input_depth);

    for (int i = 0; i < output_depth*filter_height*input_depth; i++) {
      int32& label = result_labels[i];
      // cout<<"labels is: "<<label<<endl;
      for (int j = i * filter_width; j < i * filter_width + filter_width; j++) {
        output_tensor(i % filter_height, j % filter_width, (i / filter_height)%input_depth, i / (filter_height * input_depth))
                = centers_tensor(label, j % filter_width);
        // cout<<"assign "<<centers_tensor(label, j % filter_width)<<" from "<<"( "<<label<<","<<j%filter_width<<" )"<<" to "<<"( "<<i / (filter_height * input_depth)<<","<<i % filter_height<<","<<j % filter_width<<","<<(i % filter_height * input_depth) / filter_height<<" )"<<endl;
      }
    }
    // cout<<"[Debug] the shape of output is: ["<<output->dim_size(0)<<", "<<output->dim_size(1)<<", "<<output->dim_size(2)<<","<<output->dim_size(3)<<"]"<<endl;
  }
};

REGISTER_KERNEL_BUILDER(Name("RestoreWeight").Device(DEVICE_CPU), RestoreWeightOp);
