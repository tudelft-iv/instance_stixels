// This file is part of Instance Stixels:
// https://github.com/tudelft-iv/instance-stixels
//
// Copyright (c) 2019 Thomas Hehn.
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

#include "Segmentation.h"

Segmentation::Segmentation(const char* fname){
    std::cout << "Opening file " << fname << "\n";
    H5::H5File file(fname, H5F_ACC_RDONLY);
    H5::DataSet dataset = file.openDataSet("nlogprobs");
    assert(dataset.getTypeClass() == H5T_FLOAT);

    H5::DataSpace dataspace = dataset.getSpace();

    int ndims = dataspace.getSimpleExtentNdims();

    std::vector<hsize_t> shape(ndims);
    dataspace.getSimpleExtentDims(shape.data());

    Resize(shape);

    // print shape
    std::cout << "Segmentation shape = (" << shape_[0];
    for(hsize_t i = 1; i < shape_.size(); i++){
        std::cout << ", " << shape_[i];
    }
    std::cout << ")\n";

    dataset.read(data_.data(), H5::PredType::NATIVE_FLOAT);
}

void Segmentation::Resize(std::vector<hsize_t> shape){
    shape_ = shape;

    elements_ = shape_[0];
    for(hsize_t i = 1; i < shape.size(); i++)
        elements_ *= shape[i];

    data_.resize(elements_);
}

