/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * Copyright (c) 2018 by Contributors
 * \file trt.cc
 * \brief TensorRT operation registration
 * \author Marek Kolodziej, Clement Fuji Tsang
*/

#if MXNET_USE_TENSORRT

#include "./tensorrt-inl.h"

#include <mxnet/base.h>
#include <nnvm/graph.h>
#include <nnvm/pass_functions.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "../../common/serialization.h"
#include "../../common/utils.h"
#include "../../ndarray/ndarray_function.h"
#include "../../operator/nn/activation-inl.h"
#include "../../operator/nn/batch_norm-inl.h"
#include "../../operator/nn/convolution-inl.h"
#include "../../operator/nn/fully_connected-inl.h"
#include "../../operator/nn/pooling-inl.h"
#include "../../operator/softmax_output-inl.h"

// #include <onnx/checker.h>

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(TRTParam);

OpStatePtr GetPtrMapping(nvinfer1::ICudaEngine* trt_engine,
                         trt::NameToIdx_t input_map,
                         trt::NameToIdx_t output_map) {
  TRTEngineParam param;
  for (int b = 0; b < trt_engine->getNbBindings(); ++b) {
    const std::string& binding_name = trt_engine->getBindingName(b);
    if (trt_engine->bindingIsInput(b)) {
      param.binding_map.emplace_back(input_map[binding_name],
                                     trt::TypeIO::Inputs);
    } else {
      param.binding_map.emplace_back(output_map[binding_name],
                                     trt::TypeIO::Outputs);
    }
  }
  param.trt_executor = trt_engine->createExecutionContext();
  return OpStatePtr::Create<TRTEngineParam>(param);
}

OpStatePtr TRTCreateState(const nnvm::NodeAttrs& attrs, Context ctx,
                          const std::vector<TShape>& ishape,
                          const std::vector<int>& itype) {
  const TRTParam& node_param = nnvm::get<TRTParam>(attrs.parsed);

  ::onnx::ModelProto model_proto;
  bool success = model_proto.ParseFromString(node_param.serialized_onnx_graph);
  if (!success) {
    LOG(FATAL) << "Problems parsing serialized ONNX model.";
  }
  auto graph = model_proto.graph();
  auto first_input_type = graph.input(0).type().tensor_type();
  auto dim_value = first_input_type.shape().dim(0).dim_value();
  uint64_t batch_size = static_cast<uint64_t>(dim_value);

  nvinfer1::ICudaEngine* const trt_engine = ::onnx_to_trt::onnxToTrtCtx(
      node_param.serialized_onnx_graph, batch_size, 1 << 30);

  LOG(INFO) << "TensorRT engine instantiated!!!";

  trt::NameToIdx_t output_map;
  for (auto& el : node_param.output_map) {
    output_map[el.first] = std::get<0>(el.second);
  }
  return GetPtrMapping(trt_engine, node_param.input_map, output_map);
}

void TRTParamParser(nnvm::NodeAttrs* attrs) {
  using namespace mshadow;

  TRTParam param_;

  try {
    param_.Init(attrs->dict);
    common::Deserialize(&param_.input_map, param_.serialized_input_map);
    common::Deserialize(&param_.output_map, param_.serialized_output_map);
    param_.onnx_pb_graph.ParseFromString(param_.serialized_onnx_graph);
  } catch (const dmlc::ParamError& e) {
    std::ostringstream os;
    os << e.what();
    os << ", in operator " << attrs->op->name << "("
       << "name=\"" << attrs->name << "\"";
    for (const auto& k : attrs->dict) {
      os << ", " << k.first << "=\"" << k.second << "\"";
    }
    os << ")";
    throw dmlc::ParamError(os.str());
  }

  attrs->parsed = std::move(param_);
}

template <>
void TRTCompute<cpu>(const OpStatePtr& state, const OpContext& ctx,
                     const std::vector<TBlob>& inputs,
                     const std::vector<OpReqType>& req,
                     const std::vector<TBlob>& outputs) {
  LOG(FATAL) << "TRTCompute not implemented on the CPU";
}

inline bool TRTInferShape(const NodeAttrs& attrs, std::vector<TShape>* in_shape,
                          std::vector<TShape>* out_shape) {
  const auto node_param = nnvm::get<TRTParam>(attrs.parsed);
  for (auto& el : node_param.output_map) {
    (*out_shape)[std::get<0>(el.second)] = std::get<1>(el.second);
  }
  return true;
}

inline bool TRTInferStorageType(const NodeAttrs& attrs, const int dev_mask,
                                DispatchMode* dispatch_mode,
                                std::vector<int>* in_storage_type,
                                std::vector<int>* out_storage_type) {
  return storage_type_assign(out_storage_type, mxnet::kDefaultStorage,
                             dispatch_mode, DispatchMode::kFCompute);
}

inline bool TRTInferType(const NodeAttrs& attrs, std::vector<int>* in_dtype,
                         std::vector<int>* out_dtype) {
  const auto node_param = nnvm::get<TRTParam>(attrs.parsed);
  for (auto& el : node_param.output_map) {
    (*out_dtype)[std::get<0>(el.second)] = std::get<3>(el.second);
  }
  return true;
}

inline std::vector<std::string> TRTListInputNames(const NodeAttrs& attrs) {
  std::vector<std::string> output;
  const auto node_param = nnvm::get<TRTParam>(attrs.parsed);
  output.reserve(node_param.input_map.size());
  for (auto& el : node_param.input_map) {
    output[el.second] = el.first;
  }
  return output;
}

inline std::vector<std::string> TRTListOutputNames(const NodeAttrs& attrs) {
  std::vector<std::string> output;
  const auto node_param = nnvm::get<TRTParam>(attrs.parsed);
  output.reserve(node_param.output_map.size());
  for (auto& el : node_param.output_map) {
    output[std::get<0>(el.second)] = el.first;
  }
  return output;
}

NNVM_REGISTER_OP(_trt_op)
    .describe(R"code(TRT operation (one engine)
)code" ADD_FILELINE)
    .set_num_inputs([](const NodeAttrs& attrs) {
      const auto node_param = nnvm::get<TRTParam>(attrs.parsed);
      return node_param.input_map.size();
    })
    .set_num_outputs([](const NodeAttrs& attrs) {
      const auto node_param = nnvm::get<TRTParam>(attrs.parsed);
      return node_param.output_map.size();
    })
    .set_attr_parser(TRTParamParser)
    .set_attr<nnvm::FInferShape>("FInferShape", TRTInferShape)
    .set_attr<nnvm::FInferType>("FInferType", TRTInferType)
    .set_attr<nnvm::FListInputNames>("FListInputNames", TRTListInputNames)
    .set_attr<nnvm::FListOutputNames>("FListOutputNames", TRTListOutputNames)
    .set_attr<FCreateOpState>("FCreateOpState", TRTCreateState)
    .set_attr<FStatefulCompute>("FStatefulCompute<cpu>", TRTCompute<cpu>)
    .set_attr<FInferStorageType>("FInferStorageType", TRTInferStorageType);

op::TRTParam ConvertNnvmGraphToOnnx(
    const nnvm::Graph& g,
    std::unordered_map<std::string, NDArray>* const shared_buffer) {
    op::TRTParam trt_param;
    op::trt::NameToIdx_t trt_input_map;
    op::trt::InferenceMap_t trt_output_map;

  const nnvm::IndexedGraph& ig = g.indexed_graph();
  const StorageTypeVector& storage_types =
      g.GetAttr<StorageTypeVector>("storage_type");
  const DTypeVector& dtypes = g.GetAttr<DTypeVector>("dtype");
  const ShapeVector& shape_inputs = g.GetAttr<ShapeVector>("shape_inputs");

  for (auto& e : storage_types) {
    if (e != mshadow::kFloat32) {
      LOG(FATAL) << "ONNX converter does not support types other than float32 "
                    "right now.";
    }
  }

  ModelProto model_proto;
  // Need to determine IR versions and features to support
  model_proto.set_ir_version(static_cast<int64>(2));
  GraphProto* graph_proto = model_proto.mutable_graph();
  //  graph_proto->set_name(graph_name);

  std::unordered_map<std::string, TShape> placeholder_shapes =
      GetPlaceholderShapes(shape_inputs, ig);
  std::unordered_map<std::string, uint32_t> output_lookup = GetOutputLookup(ig);
  uint32_t current_input = 0;

  // can't do a foreach over IndexedGraph since it doesn't
  // implement begin(), etc.
  for (uint32_t node_idx = 0; node_idx < ig.num_nodes(); ++node_idx) {
    const IndexedGraph::Node& node = ig[node_idx];
    const nnvm::Node* source = node.source;
    const NodeAttrs& attrs = source->attrs;
    const Op* op = source->op();

    std::string node_name = attrs.name;
    // Here, "variable" actually means anything that's not an op,
    // i.e. a constant (weights) or a placeholder
    if (source->is_variable()) {
      // Is this a placeholder? 
      if (shared_buffer->count(node_name) == 0) {
        // This fixes the problem with a SoftmaxOutput node during inference,
        // but it's hacky.
        // Need to figure out how to properly fix it.
        if (node_name.find("label") != std::string::npos) {
          current_input++;
          continue;
        }
        trt_input_map.emplace(node_name, current_input++);
        ConvertPlaceholder(node_name, placeholder_shapes, graph_proto);
      } else {
        // If it's not a placeholder, then by exclusion it's a constant.
        ConvertConstant(graph_proto, node_name, shared_buffer);
      }  // is_placeholder
    } else {
      // It's an op, rather than a "variable" (constant or placeholder)
      NodeProto* node_proto = graph_proto->add_node();
      node_proto->set_name(node_name);
      if (converter_map.count(op->name) == 0) {
        LOG(FATAL) << "Conversion for node of type " << op->name << " (node "
                   << node_name << ") "
                   << " is not supported yet.";
      }
      // Find function ptr to a converter based on the op name, and
      // invoke the converter. This looks unsafe because find may not
      // succeed, but it does because we're in the operator logic after
      // testing that this node name does not represent a variable.
      converter_map.find(op->name)->second(node_proto, attrs, ig, node.inputs);
      // Add all inputs to the current node (i.e. add graph edges)
      for (const nnvm::IndexedGraph::NodeEntry& entry : node.inputs) {
        std::string in_node_name = ig[entry.node_id].source->attrs.name;
        // As before, we're not adding labels e.g. for SoftmaxOutput,
        // but I wish there was a less hacky way to do it than name matching.
        if (in_node_name.find("label") != std::string::npos) {
          continue;
        }
        node_proto->add_input(in_node_name);
      }
      // The node's output will have the same name as the node name.
      node_proto->add_output(node_name);
      // See if the current node is an output node
      auto out_iter = output_lookup.find(node_name);
      // We found an output
      if (out_iter != output_lookup.end()) {
        ConvertOutput(&trt_output_map, graph_proto, out_iter, node_name, g,
                      storage_types, dtypes);
      }  // output found
    }    // conversion function exists
  }      // loop over i from 0 to num_nodes

  LOG(INFO) << "ONNX graph construction complete.";
  model_proto.SerializeToString(&trt_param.serialized_onnx_graph);
  common::Serialize<op::trt::NameToIdx_t>(trt_input_map,
                                          &trt_param.serialized_input_map);
  common::Serialize<op::trt::InferenceMap_t>(trt_output_map,
                                             &trt_param.serialized_output_map);
  //  onnx::checker::check_model(model_proto);
  return trt_param;
}

void ConvertConvolution(NodeProto* node_proto, const NodeAttrs& attrs,
                        const nnvm::IndexedGraph& ig,
                        const array_view<IndexedGraph::NodeEntry>& inputs) {
  const op::ConvolutionParam& conv_param =
      nnvm::get<op::ConvolutionParam>(attrs.parsed);

  node_proto->set_op_type("Conv");

  const TShape kernel = conv_param.kernel;
  const TShape stride = conv_param.stride;
  const TShape dilate = conv_param.dilate;
  const TShape pad = conv_param.pad;
  // const uint32_t num_filter = conv_param.num_filter;
  const uint32_t num_group = conv_param.num_group;
  // const bool no_bias = conv_param.no_bias;
  const dmlc::optional<int> layout = conv_param.layout;

  // kernel shape
  AttributeProto* const kernel_shape = node_proto->add_attribute();
  kernel_shape->set_name("kernel_shape");
  kernel_shape->set_type(AttributeProto::INTS);

  for (int kval : kernel) {
    kernel_shape->add_ints(static_cast<int64>(kval));
  }

  // pads
  AttributeProto* const pads = node_proto->add_attribute();
  pads->set_name("pads");
  pads->set_type(AttributeProto::INTS);

  for (int kval : pad) {
    pads->add_ints(static_cast<int64>(kval));
    pads->add_ints(static_cast<int64>(kval));
  }

  // dilations
  AttributeProto* const dilations = node_proto->add_attribute();
  dilations->set_name("dilations");
  dilations->set_type(AttributeProto::INTS);
  for (int kval : dilate) {
    dilations->add_ints(static_cast<int64>(kval));
  }

  // strides
  AttributeProto* const strides = node_proto->add_attribute();
  strides->set_name("strides");
  strides->set_type(AttributeProto::INTS);
  for (int kval : stride) {
    strides->add_ints(static_cast<int64>(kval));
  }

  // group
  AttributeProto* const group = node_proto->add_attribute();
  group->set_name("group");
  group->set_type(AttributeProto::INT);
  group->set_i(static_cast<int64>(num_group));
}  // end ConvertConvolution

void ConvertPooling(NodeProto* node_proto, const NodeAttrs& attrs,
                    const nnvm::IndexedGraph& ig,
                    const array_view<IndexedGraph::NodeEntry>& inputs) {
  const op::PoolingParam& pooling_param =
      nnvm::get<op::PoolingParam>(attrs.parsed);

  const TShape kernel = pooling_param.kernel;
  const TShape stride = pooling_param.stride;
  const TShape pad = pooling_param.pad;
  const int pool_type = pooling_param.pool_type;
  // const int pooling_convention = pooling_param.pooling_convention;
  const bool global_pool = pooling_param.global_pool;

  if (global_pool) {
    if (pool_type == 0) {
      node_proto->set_op_type("GlobalMaxPool");
    } else {
      node_proto->set_op_type("GlobalAveragePool");
    }
    return;

  } else {
    // kernel_shape
    AttributeProto* const kernel_shape = node_proto->add_attribute();
    kernel_shape->set_name("kernel_shape");
    kernel_shape->set_type(AttributeProto::INTS);
    for (int kval : kernel) {
      kernel_shape->add_ints(static_cast<int64>(kval));
    }

    // pads
    AttributeProto* const pads = node_proto->add_attribute();
    pads->set_name("pads");
    pads->set_type(AttributeProto::INTS);
    for (int kval : pad) {
      pads->add_ints(static_cast<int64>(kval));
    }

    // strides
    AttributeProto* const strides = node_proto->add_attribute();
    strides->set_name("strides");
    strides->set_type(AttributeProto::INTS);
    for (int kval : stride) {
      strides->add_ints(static_cast<int64>(kval));
    }

    if (pool_type == 0) {
      node_proto->set_op_type("MaxPool");

    } else {
      node_proto->set_op_type("AveragePool");
    }  // average pooling
    // not global pooling
  }
}  // end ConvertPooling

void ConvertActivation(NodeProto* node_proto, const NodeAttrs& attrs,
                       const nnvm::IndexedGraph& ig,
                       const array_view<IndexedGraph::NodeEntry>& inputs) {
  const op::ActivationParam& act_param =
      nnvm::get<op::ActivationParam>(attrs.parsed);
  std::string act_type;
  switch (act_param.act_type) {
    case op::activation::kReLU:
      act_type = "Relu";
      break;
    case op::activation::kSigmoid:
      act_type = "Sigmoid";
      break;
    case op::activation::kTanh:
      act_type = "Tanh";
      break;
    case op::activation::kSoftReLU:
      // act_type = "SoftReLU";
      throw dmlc::Error("SoftReLU is not supported in ONNX");
      break;
    default:
      throw dmlc::Error("Activation of such type doesn't exist");
  }

  node_proto->set_op_type(act_type);
}

void ConvertFullyConnected(NodeProto* node_proto, const NodeAttrs& attrs,
                           const nnvm::IndexedGraph& ig,
                           const array_view<IndexedGraph::NodeEntry>& inputs) {
  // const op::FullyConnectedParam &act_param =
  //    nnvm::get<op::FullyConnectedParam>(attrs.parsed);

  node_proto->set_op_type("Gemm");

  // const int num_hidden = act_param.num_hidden;
  // const bool no_bias = act_param.no_bias;
  // Whether to collapse all but the first axis of the input data tensor.
  // const bool flatten = act_param.flatten;

  AttributeProto* const alpha = node_proto->add_attribute();
  alpha->set_name("alpha");
  alpha->set_type(AttributeProto::FLOAT);
  alpha->set_f(1.0f);

  AttributeProto* const beta = node_proto->add_attribute();
  beta->set_name("beta");
  beta->set_type(AttributeProto::FLOAT);
  beta->set_f(1.0f);

  AttributeProto* const broadcast = node_proto->add_attribute();
  broadcast->set_name("broadcast");
  broadcast->set_type(AttributeProto::INT);
  broadcast->set_i(1);

  AttributeProto* const transA = node_proto->add_attribute();
  transA->set_name("transA");
  transA->set_type(AttributeProto::INT);
  transA->set_i(0);

  AttributeProto* const transB = node_proto->add_attribute();
  transB->set_name("transB");
  transB->set_type(AttributeProto::INT);
  transB->set_i(1);
}

void ConvertSoftmaxOutput(NodeProto* node_proto, const NodeAttrs& attrs,
                          const nnvm::IndexedGraph& ig,
                          const array_view<IndexedGraph::NodeEntry>& inputs) {
  node_proto->set_op_type("Softmax");

  // Setting by default to 1 since MxNet doesn't provide such an attribute
  // for softmax
  // in its node params. This attribute is only relevant when the input is
  // coerced
  // to 2D, and in that case dimension 0 is assumed to be the batch
  // dimension.
  AttributeProto* const axis = node_proto->add_attribute();
  axis->set_name("axis");
  axis->set_type(AttributeProto::INT);
  axis->set_i(1);
}

void ConvertFlatten(NodeProto* node_proto, const NodeAttrs& attrs,
                    const nnvm::IndexedGraph& ig,
                    const array_view<IndexedGraph::NodeEntry>& inputs) {
  node_proto->set_op_type("Flatten");

  // Setting by default to 1 since MxNet doesn't provide such an attribute
  // for Flatten
  // in its node params. This attribute is only relevant when the input is
  // coerced
  // to 2D, and in that case dimension 0 is assumed to be the batch
  // dimension.
  AttributeProto* const axis = node_proto->add_attribute();
  axis->set_name("axis");
  axis->set_type(AttributeProto::INT);
  axis->set_i(1);
}

void ConvertBatchNorm(NodeProto* node_proto, const NodeAttrs& attrs,
                      const nnvm::IndexedGraph& ig,
                      const array_view<IndexedGraph::NodeEntry>& inputs) {
  node_proto->set_op_type("BatchNormalization");
  const op::BatchNormParam& param = nnvm::get<op::BatchNormParam>(attrs.parsed);

  AttributeProto* const epsilon = node_proto->add_attribute();
  epsilon->set_name("epsilon");
  epsilon->set_type(AttributeProto::FLOAT);
  epsilon->set_f(static_cast<float>(param.eps));

  AttributeProto* const is_test = node_proto->add_attribute();
  is_test->set_name("is_test");
  is_test->set_type(AttributeProto::INT);
  is_test->set_i(1);

  AttributeProto* const momentum = node_proto->add_attribute();
  momentum->set_name("momentum");
  momentum->set_type(AttributeProto::FLOAT);
  momentum->set_f(param.momentum);

  AttributeProto* const spatial = node_proto->add_attribute();
  spatial->set_name("spatial");
  spatial->set_type(AttributeProto::INT);
  spatial->set_i(1);

  AttributeProto* const consumed = node_proto->add_attribute();
  consumed->set_name("consumed_inputs");
  consumed->set_type(AttributeProto::INTS);

  for (int i = 0; i < 5; i++) {
    int val = (i < 3) ? 0 : 1;
    consumed->add_ints(static_cast<int64>(val));
  }
}

void ConvertElementwiseAdd(NodeProto* node_proto, const NodeAttrs& attrs,
                           const nnvm::IndexedGraph& ig,
                           const array_view<IndexedGraph::NodeEntry>& inputs) {
  node_proto->set_op_type("Add");
  AttributeProto* const axis = node_proto->add_attribute();
  axis->set_name("axis");
  axis->set_type(AttributeProto::INT);
  axis->set_i(1);

  AttributeProto* const broadcast = node_proto->add_attribute();
  broadcast->set_name("broadcast");
  broadcast->set_type(AttributeProto::INT);
  broadcast->set_i(0);  // 1
}

std::unordered_map<std::string, TShape> GetPlaceholderShapes(
    const ShapeVector& shape_inputs, const nnvm::IndexedGraph& ig) {
  std::unordered_map<std::string, TShape> placeholder_shapes;
  for (uint32_t i = 0; i < shape_inputs.size(); ++i) {
    std::string name = ig[ig.input_nodes()[i]].source->attrs.name;
    TShape shp = shape_inputs[i];
    if (shp.ndim() > 0) {
      placeholder_shapes.emplace(name, shp);
    }
  }
  return placeholder_shapes;
}

std::unordered_map<std::string, uint32_t> GetOutputLookup(
    const nnvm::IndexedGraph& ig) {
  std::unordered_map<std::string, uint32_t> output_lookup;
  const std::vector<nnvm::IndexedGraph::NodeEntry>& graph_outputs =
      ig.outputs();
  for (uint32_t i = 0; i < graph_outputs.size(); ++i) {
    const uint32_t id = graph_outputs[i].node_id;
    const IndexedGraph::Node ig_node = ig[id];
    const nnvm::Node* const source = ig_node.source;
    const std::string name = source->attrs.name;
    output_lookup.emplace(name, i);
  }
  return output_lookup;
}

void ConvertPlaceholder(
    const std::string& node_name,
    const std::unordered_map<std::string, TShape>& placeholder_shapes,
    GraphProto* const graph_proto) {
  auto val_info_proto = graph_proto->add_input();
  auto type_proto = val_info_proto->mutable_type()->mutable_tensor_type();
  auto shape_proto = type_proto->mutable_shape();

  val_info_proto->set_name(node_name);
  // Will support fp16, etc. in the near future
  type_proto->set_elem_type(TensorProto_DataType_FLOAT);
  auto entry_shape = placeholder_shapes.find(node_name)->second;

  for (const auto& elem : entry_shape) {
    TensorShapeProto_Dimension* const tsp_dim = shape_proto->add_dim();
    tsp_dim->set_dim_value(static_cast<int64>(elem));
  }
}

void ConvertConstant(
    GraphProto* const graph_proto, const std::string& node_name,
    std::unordered_map<std::string, NDArray>* const shared_buffer) {
  NodeProto* const node_proto = graph_proto->add_node();
  node_proto->set_name(node_name);
  node_proto->add_output(node_name);
  node_proto->set_op_type("Constant");

  const NDArray nd = shared_buffer->find(node_name)->second;
  const TBlob blob = nd.data();
  const TShape shape = blob.shape_;
  // const int type_flag = blob.type_flag_;
  const int32_t size = shape.Size();

  // Use MSHADOW_TYPE_SWITCH for e.g. fp16 support !!!
  // const int dev_mask = blob.dev_mask();
  // const int dev_id = blob.dev_id();

  std::shared_ptr<float> shared_data_ptr(new float[size]);
  float* const data_ptr = shared_data_ptr.get();
  nd.SyncCopyToCPU(static_cast<void*>(data_ptr), size);

  AttributeProto* const tensor_attr = node_proto->add_attribute();
  tensor_attr->set_name("value");
  tensor_attr->set_type(AttributeProto::TENSOR);

  TensorProto* const tensor_proto = tensor_attr->mutable_t();
  tensor_proto->set_data_type(TensorProto_DataType_FLOAT);
  for (auto& dim : shape) {
    tensor_proto->add_dims(static_cast<int64>(dim));
  }

  for (int blob_idx = 0; blob_idx < size; ++blob_idx) {
    tensor_proto->add_float_data(data_ptr[blob_idx]);
  }
}

void ConvertOutput(
    op::trt::InferenceMap_t* const trt_output_map,
    GraphProto* const graph_proto,
    const std::unordered_map<std::string, uint32_t>::iterator& out_iter,
    const std::string& node_name, const nnvm::Graph& g,
    const StorageTypeVector& storage_types, const DTypeVector& dtypes) {
  const nnvm::IndexedGraph& ig = g.indexed_graph();
  uint32_t out_idx = ig.entry_id(ig.outputs()[out_iter->second]);
  TShape out_shape = g.GetAttr<nnvm::ShapeVector>("shape")[out_idx];
  int storage_type = storage_types[out_idx];
  int dtype = dtypes[out_idx];

  // This should work with fp16 as well
  op::trt::InferenceTuple_t out_tuple{out_iter->second, out_shape, storage_type,
                                      dtype};

  trt_output_map->emplace(node_name, out_tuple);

  auto graph_out = graph_proto->add_output();
  auto tensor_type = graph_out->mutable_type()->mutable_tensor_type();
  auto tensor_shape_proto = tensor_type->mutable_shape();
  graph_out->set_name(node_name);

  // Also support fp16.
  tensor_type->set_elem_type(TensorProto_DataType_FLOAT);

  for (int64_t dim_shp : out_shape) {
    TensorShapeProto_Dimension* const tsp_dim = tensor_shape_proto->add_dim();
    tsp_dim->set_dim_value(static_cast<int64>(dim_shp));
  }
}

}  // namespace op
}  // namespace mxnet

#endif  // MXNET_USE_TENSORRT
