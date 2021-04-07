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

#ifndef SEGMENTATION_H_
#define SEGMENTATION_H_

#include <iostream>
#include <vector>
#include <cassert>
#include <cstdint>
#include "H5Cpp.h"

class H5Segmentation {
    public:
        H5Segmentation() {}
        H5Segmentation(const char* fname);

        void LoadFile(const char* fname);
        std::vector<hsize_t> get_shape() const { return shape_; }
        std::vector<int32_t> data() { return data_; }

    private:
        void Resize(std::vector<hsize_t> shape);

        std::vector<hsize_t> shape_;
        std::vector<int32_t> data_;
        hsize_t elements_;
};

#endif /* SEGMENTATION_H_ */
