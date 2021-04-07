// This file is part of Instance Stixels:
// https://github.com/tudelft-iv/instance-stixels
//
// Copyright (c) 2020 Thomas Hehn.
//
// Instance Stixels is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// Instance Stixels is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with Instance Stixels. If not, see <http://www.gnu.org/licenses/>.
//
// Originally, it was part of the NVIDIA TensorRT samples and released under
// Apache License, Version 2.0: http://www.apache.org/licenses/LICENSE-2.0
// https://github.com/NVIDIA/TensorRT/blob/release/7.0/samples/opensource/sampleOnnxMNIST/sampleOnnxMNIST.cpp
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.

#include "TRTUtils/buffers.h"
#include "TRTUtils/common.h"
#include "TRTUtils/logging.h"
#include "TRTUtils/parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>

#ifndef TRTOnnxCNN_HPP_
#define TRTOnnxCNN_HPP_

class TRTOnnxCNN
{
    struct InferDeleter
    {
        template <typename T>
        void operator()(T* obj) const
        {
            if (obj)
            {
                obj->destroy();
            }
        }
    };

    template <typename T>
    using InferUniquePtr = std::unique_ptr<T, InferDeleter>;

public:
    TRTOnnxCNN(std::string onnxFileName,
               std::vector<std::string> inputTensorNames,
               std::vector<std::string> outputTensorNames)
        : mOnnxFileName(onnxFileName),
          mInputTensorNames(inputTensorNames),
          mOutputTensorNames(outputTensorNames), mEngine(nullptr) {}

    bool build();
    std::vector<int32_t> infer(std::vector<float> data);
    int32_t* inferOnDevice(std::vector<float> data);

private:
    Logger logger;

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    int mNumber{0};             //!< The number to classify

    // These were originally params from argparse.
    int mBatchSize{1};                  //!< Number of inputs in a batch
    int mDlaCore{-1};                   //!< Specify the DLA core to run network on.
    bool mInt8{false};                  //!< Allow runnning the network in Int8 mode.
    bool mFp16{true};                  //!< Allow running the network in FP16 mode.
    std::vector<std::string> mInputTensorNames;
    std::vector<std::string> mOutputTensorNames;
    std::string mOnnxFileName; //!< Filename of ONNX file of a network

    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
    std::unique_ptr<TRTUtils::BufferManager> mBuffers;

    bool constructNetwork(InferUniquePtr<nvinfer1::IBuilder>& builder,
        InferUniquePtr<nvinfer1::INetworkDefinition>& network, InferUniquePtr<nvinfer1::IBuilderConfig>& config,
        InferUniquePtr<nvonnxparser::IParser>& parser);
    void copyToInputBuffer(float* data, int size);
};

#endif /* TRTOnnxCNN_HPP_ */
