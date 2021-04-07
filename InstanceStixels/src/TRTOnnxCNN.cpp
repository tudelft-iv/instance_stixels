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

#include "TRTOnnxCNN.hpp"
#include <memory>

bool TRTOnnxCNN::build()
{
    LOG_INFO(logger) << "Starting build.\n";
    auto builder = InferUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(logger.getTRTLogger()));
    if (!builder)
    {
        return false;
    }

    LOG_INFO(logger) << "Creating network.\n";
    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = InferUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return false;
    }

    LOG_INFO(logger) << "Setup config.\n";
    auto config = InferUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return false;
    }

    LOG_INFO(logger) << "Parsing network.\n";
    auto parser = InferUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, logger.getTRTLogger()));
    if (!parser)
    {
        return false;
    }

    LOG_INFO(logger) << "Construct network.\n";
    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return false;
    }

    LOG_INFO(logger) << "Build engine.\n";
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        builder->buildEngineWithConfig(*network, *config), TRTOnnxCNN::InferDeleter());
    if (!mEngine)
    {
        return false;
    }

    assert(network->getNbInputs() == 1);
    mInputDims = network->getInput(0)->getDimensions();
    assert(mInputDims.nbDims == 4);

    assert(network->getNbOutputs() == 1);
    mOutputDims = network->getOutput(0)->getDimensions();
    LOG_INFO(logger) << "Output dims: ";
    for(int idx = 0; idx < mOutputDims.nbDims; idx++){
        LOG_INFO(logger) << mOutputDims.d[idx] << "x";
    }
    LOG_INFO(logger) << "\n";
    assert(mOutputDims.nbDims == 4);

    LOG_INFO(logger) << "Create buffer.\n";
    mBuffers =
        std::unique_ptr<TRTUtils::BufferManager>(
                new TRTUtils::BufferManager(mEngine, mBatchSize));
    LOG_INFO(logger) << "Done building.\n";
    return true;
}

bool TRTOnnxCNN::constructNetwork(InferUniquePtr<nvinfer1::IBuilder>& builder,
    InferUniquePtr<nvinfer1::INetworkDefinition>& network, InferUniquePtr<nvinfer1::IBuilderConfig>& config,
    InferUniquePtr<nvonnxparser::IParser>& parser)
{
    //LOG_INFO(logger) << "Constructing from file " << mOnnxFileName << "\n";
    auto parsed =
        parser->parseFromFile(
                mOnnxFileName.c_str(),
                static_cast<int>(logger.getReportableSeverity()));
    if (!parsed)
    {
        return false;
    }

    builder->setMaxBatchSize(mBatchSize);
    constexpr size_t MAX_WORKSPACE_SIZE = 16 * (1 << 20); // 16 MiB
    //constexpr size_t MAX_WORKSPACE_SIZE = 16 * (1 << 30); // 16 GiB
    //// 16_MiB -> ~19.2fps
    //constexpr size_t MAX_WORKSPACE_SIZE = 1ULL << 30;
    //// 1 GB rather slow ~18.5fps
    //constexpr size_t MAX_WORKSPACE_SIZE = 1024ULL*1024ULL*1024ULL*10ULL;
    //// 10 GB rather (18.5fps), building takes really long!
    builder->setMaxWorkspaceSize(MAX_WORKSPACE_SIZE);
    config->setMaxWorkspaceSize(MAX_WORKSPACE_SIZE);

    if (mFp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mInt8)
    {
        config->setFlag(BuilderFlag::kINT8);
        TRTUtils::setAllTensorScales(network.get(), 127.0f, 127.0f);
    }

    TRTUtils::enableDLA(builder.get(), config.get(), mDlaCore);

    return true;
}

int32_t* TRTOnnxCNN::inferOnDevice(std::vector<float> data)
{
    if(mBuffers == nullptr){
        return nullptr;
    }

    auto context = InferUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return nullptr;
    }

    // Read the input data into the managed mBuffers
    assert(mInputTensorNames.size() == 1);
    //LOG_INFO(logger) << "Copy input to buffer.\n";
    copyToInputBuffer(data.data(), data.size());

    //LOG_INFO(logger) << "Copy input to device.\n";
    // Memcpy from host input mBuffers to device input mBuffers
    mBuffers->copyInputToDevice();
    //LOG_INFO(logger) << "executeV2.\n";

    bool status = context->executeV2(mBuffers->getDeviceBindings().data());
    if (!status)
    {
        return nullptr;
    }

    int32_t* d_output =
        static_cast<int32_t*>(mBuffers->getDeviceBuffer(mOutputTensorNames[0]));
    return d_output;
}

std::vector<int32_t> TRTOnnxCNN::infer(std::vector<float> data)
{
    if(mBuffers == nullptr){
        return std::vector<int32_t>{};
    }

    auto context = InferUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return std::vector<int32_t>{};
    }

    // Read the input data into the managed mBuffers
    assert(mInputTensorNames.size() == 1);
    //LOG_INFO(logger) << "Copy input to buffer.\n";
    copyToInputBuffer(data.data(), data.size());

    //LOG_INFO(logger) << "Copy input to device.\n";
    // Memcpy from host input mBuffers to device input mBuffers
    mBuffers->copyInputToDevice();
    //LOG_INFO(logger) << "executeV2.\n";

    bool status = context->executeV2(mBuffers->getDeviceBindings().data());
    if (!status)
    {
        return std::vector<int32_t>{};
    }

    //LOG_INFO(logger) << "Copy output to host.\n";
    // Memcpy from device output mBuffers to host output mBuffers
    mBuffers->copyOutputToHost();

    //LOG_INFO(logger) << "Copy output to result.\n";
    // return result
    int32_t* output = static_cast<int32_t*>(mBuffers->getHostBuffer(mOutputTensorNames[0]));
    //LOG_INFO(logger) << "Compute output_size.\n";
    int output_size =
        std::accumulate(mOutputDims.d, mOutputDims.d + mOutputDims.nbDims,
                        1, std::multiplies<int>());
    //LOG_INFO(logger) << "Output size = " << output_size << "\n";
    std::vector<int32_t> result(output_size);
    std::copy(output, output+output_size, result.begin());
    return result;
}

// TODO: documentation, catch errors
// Optimize: do not copy host memory, but use the same pointer
void TRTOnnxCNN::copyToInputBuffer(
    float* data, int size)
{
    float* hostDataBuffer =
        static_cast<float*>(mBuffers->getHostBuffer(mInputTensorNames[0]));
    std::copy(data, data+size, hostDataBuffer);
}
