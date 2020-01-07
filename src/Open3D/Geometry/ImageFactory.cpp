// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include <algorithm>
#include <iostream>

#include <Eigen/Dense>
#include <cmath>

#include "Open3D/Camera/PinholeCameraIntrinsic.h"
#include "Open3D/Geometry/Image.h"

namespace open3d {
namespace geometry {

std::shared_ptr<Image> Image::CreateDepthToCameraDistanceMultiplierFloatImage(
        const camera::PinholeCameraIntrinsic &intrinsic) {
    auto fimage = std::make_shared<Image>();
    fimage->Prepare(intrinsic.width_, intrinsic.height_, 1, 4);
    float ffl_inv[2] = {
            1.0f / (float)intrinsic.GetFocalLength().first,
            1.0f / (float)intrinsic.GetFocalLength().second,
    };
    float fpp[2] = {
            (float)intrinsic.GetPrincipalPoint().first,
            (float)intrinsic.GetPrincipalPoint().second,
    };
    std::vector<float> xx(intrinsic.width_);
    std::vector<float> yy(intrinsic.height_);
    for (int j = 0; j < intrinsic.width_; j++) {
        xx[j] = (j - fpp[0]) * ffl_inv[0];
    }
    for (int i = 0; i < intrinsic.height_; i++) {
        yy[i] = (i - fpp[1]) * ffl_inv[1];
    }
    for (int i = 0; i < intrinsic.height_; i++) {
        float *fp =
                (float *)(fimage->data_.data() + i * fimage->BytesPerLine());
        for (int j = 0; j < intrinsic.width_; j++, fp++) {
            *fp = sqrtf(xx[j] * xx[j] + yy[i] * yy[i] + 1.0f);
        }
    }
    return fimage;
}

std::shared_ptr<Image> Image::CreateWeightImage(
        const camera::PinholeCameraIntrinsic &intrinsic) const {

    auto output = std::make_shared<Image>();


    output->Prepare(intrinsic.width_, intrinsic.height_, 1, 4);
    auto focal_length = intrinsic.GetFocalLength();
    auto principal_point = intrinsic.GetPrincipalPoint();

    #ifdef _OPENMP
    #ifdef _WIN32
    #pragma omp parallel for schedule(static)
    #else
    #pragma omp parallel for collapse(2) schedule(static)
    #endif
    #endif
    for (int i = 0; i < output->height_; i++) {
        for (int j = 0; j < output->width_; j++) {
            float *p = output->PointerAt<float>(j, i);
            float *ip = PointerAt<float>(j, i);
            double weight = 0.0;  

            if (*ip > 0) {
                if(i > 0 && j > 0 && i < output->height_-1 && j < output->width_-1){

                    // computing normalized vertex
                    double z = (double)(*ip);
                    double x = (j - principal_point.first) * z / focal_length.first;
                    double y =
                            (i - principal_point.second) * z / focal_length.second;
                    Eigen::Vector3d point = Eigen::Vector3d(x, y, z);
                    Eigen::Vector3d v_norm = point.normalized();

                    //computing normalized normal
                    float *dx1 = PointerAt<float>(j+1, i);
                    float *dx2 = PointerAt<float>(j-1, i);

                    float *dy1 = PointerAt<float>(j, i+1);
                    float *dy2 = PointerAt<float>(j, i-1);

                    double dzdx = (((double)*dx1 - (double)*dx2)/2.0)*1000.0;
                    double dzdy = (((double)*dy1 - (double)*dy2)/2.0)*1000.0;

                    Eigen::Vector3d normal = Eigen::Vector3d(-dzdx, -dzdy, 1.0);
                    Eigen::Vector3d n_norm = normal.normalized();

                    // Eigen::Vector3d captureDir = Eigen::Vector3d(0, 0, 1.0);

                    // double w1 = abs(captureDir.dot(n_norm));
                    // double w2 = abs(captureDir.dot(v_norm));
                    
                    // weight = w1;
                    // weight = w2;
                    double w = abs(n_norm.dot(v_norm));
                    weight = w * w;
                    
                    // // Adding gaussian weight centering at principle point
                    // double w = (double)output->width_/2.0;
                    // double h = (double)output->height_/2.0;
                    // double w_x = ((double)j - w)/w ;
                    // double w_y = ((double)i - h)/h;

                    // double d = sqrt(w_x * w_x + w_y * w_y);  
                    // weight = exp(-((d*d)/(2.0))); // assuming mu = 0 and sigma = 1.0
                }
                else{
                    weight = 1.0f; // if this weight is set to 0, some of the points are missing in final integrated reconstruction. Therefore assigning it 1.0
                }

            }
            *p = (float)weight;
        }
    }
    return output;
}

std::shared_ptr<Image> Image::CreateFloatImage(
        Image::ColorToIntensityConversionType type /* = WEIGHTED*/) const {
    auto fimage = std::make_shared<Image>();
    if (IsEmpty()) {
        return fimage;
    }
    fimage->Prepare(width_, height_, 1, 4);
    for (int i = 0; i < height_ * width_; i++) {
        float *p = (float *)(fimage->data_.data() + i * 4);
        const uint8_t *pi =
                data_.data() + i * num_of_channels_ * bytes_per_channel_;
        if (num_of_channels_ == 1) {
            // grayscale image
            if (bytes_per_channel_ == 1) {
                *p = (float)(*pi) / 255.0f;
            } else if (bytes_per_channel_ == 2) {
                const uint16_t *pi16 = (const uint16_t *)pi;
                *p = (float)(*pi16);
            } else if (bytes_per_channel_ == 4) {
                const float *pf = (const float *)pi;
                *p = *pf;
            }
        } else if (num_of_channels_ == 3) {
            if (bytes_per_channel_ == 1) {
                if (type == Image::ColorToIntensityConversionType::Equal) {
                    *p = ((float)(pi[0]) + (float)(pi[1]) + (float)(pi[2])) /
                         3.0f / 255.0f;
                } else if (type ==
                           Image::ColorToIntensityConversionType::Weighted) {
                    *p = (0.2990f * (float)(pi[0]) + 0.5870f * (float)(pi[1]) +
                          0.1140f * (float)(pi[2])) /
                         255.0f;
                }
            } else if (bytes_per_channel_ == 2) {
                const uint16_t *pi16 = (const uint16_t *)pi;
                if (type == Image::ColorToIntensityConversionType::Equal) {
                    *p = ((float)(pi16[0]) + (float)(pi16[1]) +
                          (float)(pi16[2])) /
                         3.0f;
                } else if (type ==
                           Image::ColorToIntensityConversionType::Weighted) {
                    *p = (0.2990f * (float)(pi16[0]) +
                          0.5870f * (float)(pi16[1]) +
                          0.1140f * (float)(pi16[2]));
                }
            } else if (bytes_per_channel_ == 4) {
                const float *pf = (const float *)pi;
                if (type == Image::ColorToIntensityConversionType::Equal) {
                    *p = (pf[0] + pf[1] + pf[2]) / 3.0f;
                } else if (type ==
                           Image::ColorToIntensityConversionType::Weighted) {
                    *p = (0.2990f * pf[0] + 0.5870f * pf[1] + 0.1140f * pf[2]);
                }
            }
        }
    }
    return fimage;
}

template <typename T>
std::shared_ptr<Image> Image::CreateImageFromFloatImage() const {
    auto output = std::make_shared<Image>();
    if (num_of_channels_ != 1 || bytes_per_channel_ != 4) {
        utility::LogError(
                "[CreateImageFromFloatImage] Unsupported image format.");
    }

    output->Prepare(width_, height_, num_of_channels_, sizeof(T));
    const float *pi = (const float *)data_.data();
    T *p = (T *)output->data_.data();
    for (int i = 0; i < height_ * width_; i++, p++, pi++) {
        if (sizeof(T) == 1) *p = static_cast<T>(*pi * 255.0f);
        if (sizeof(T) == 2) *p = static_cast<T>(*pi);
    }
    return output;
}

template std::shared_ptr<Image> Image::CreateImageFromFloatImage<uint8_t>()
        const;
template std::shared_ptr<Image> Image::CreateImageFromFloatImage<uint16_t>()
        const;

ImagePyramid Image::CreatePyramid(size_t num_of_levels,
                                  bool with_gaussian_filter /*= true*/) const {
    std::vector<std::shared_ptr<Image>> pyramid_image;
    pyramid_image.clear();
    if ((num_of_channels_ != 1) || (bytes_per_channel_ != 4)) {
        utility::LogError("[CreateImagePyramid] Unsupported image format.");
    }

    for (size_t i = 0; i < num_of_levels; i++) {
        if (i == 0) {
            std::shared_ptr<Image> input_copy_ptr = std::make_shared<Image>();
            *input_copy_ptr = *this;
            pyramid_image.push_back(input_copy_ptr);
        } else {
            if (with_gaussian_filter) {
                // https://en.wikipedia.org/wiki/Pyramid_(image_processing)
                auto level_b = pyramid_image[i - 1]->Filter(
                        Image::FilterType::Gaussian3);
                auto level_bd = level_b->Downsample();
                pyramid_image.push_back(level_bd);
            } else {
                auto level_d = pyramid_image[i - 1]->Downsample();
                pyramid_image.push_back(level_d);
            }
        }
    }
    return pyramid_image;
}

}  // namespace geometry
}  // namespace open3d
