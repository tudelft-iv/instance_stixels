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

#include <algorithm>
#include <cmath>

#include "catch.hpp"
#include "testdata.h"

#include "Segmentation.h"

bool test_file(){
    Segmentation segmentation(TESTDATA::h5_filename.c_str());

    //for(int i = 0; i < flat.size(); i++){
    //    std::cout << segmentation[i] << " - " << flat[i] 
    //              << " = " << (segmentation[i] - flat[i])
    //              << " = " << (segmentation[i] == flat[i]) << "\n";
    //}
    //std::cout << "\n";

    // Comparing difference against 1e-7 is fine, since values are in similar range.
    return std::equal(TESTDATA::flat.begin(), TESTDATA::flat.end(), 
                      segmentation.begin(),
                      [] (const float& a, const float& b) 
                         { return std::abs(a - b) < 1e-7; });
}

TEST_CASE("Testing segmentation class.", "[segmentation]"){
    REQUIRE(test_file() == true);
}
