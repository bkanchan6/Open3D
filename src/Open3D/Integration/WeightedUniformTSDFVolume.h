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

#pragma once

#include "Open3D/Geometry/VoxelGrid.h"
#include "Open3D/Integration/TSDFVolume.h"

namespace open3d {

namespace geometry {

class WeightedTSDFVoxel : public Voxel {
public:
    WeightedTSDFVoxel() : Voxel() {}
    WeightedTSDFVoxel(const Eigen::Vector3i &grid_index) : Voxel(grid_index) {}
    WeightedTSDFVoxel(const Eigen::Vector3i &grid_index, const Eigen::Vector3d &color)
        : Voxel(grid_index, color) {}
    ~WeightedTSDFVoxel() {}

public:
    float tsdf_ = 0;
    float weight_ = 0;
    float weight_color = 0;
};

}  // namespace geometry

namespace integration {

class WeightedUniformTSDFVolume : public TSDFVolume {
public:
    WeightedUniformTSDFVolume(double length,
                      int resolution,
                      double sdf_trunc,
                      TSDFVolumeColorType color_type,
                      const Eigen::Vector3d &origin = Eigen::Vector3d::Zero());
    ~WeightedUniformTSDFVolume() override;

public:
    void Reset() override;
    void Integrate(const geometry::RGBDImage &image,
                   const camera::PinholeCameraIntrinsic &intrinsic,
                   const Eigen::Matrix4d &extrinsic) override;
    std::shared_ptr<geometry::PointCloud> ExtractPointCloud() override;
    std::shared_ptr<geometry::TriangleMesh> ExtractTriangleMesh() override;

    /// Debug function to extract the voxel data into a VoxelGrid
    std::shared_ptr<geometry::PointCloud> ExtractVoxelPointCloud() const;
    std::shared_ptr<geometry::VoxelGrid> ExtractVoxelGrid() const;

    /// Faster Integrate function that uses depth_to_camera_distance_multiplier
    /// precomputed from camera intrinsic
    // also apply weight factors
    void IntegrateWithDepthToCameraDistanceMultiplierWeight(
            const geometry::RGBDImage &image,
            const camera::PinholeCameraIntrinsic &intrinsic,
            const Eigen::Matrix4d &extrinsic,
            const geometry::Image &depth_to_camera_distance_multiplier,
            const geometry::Image &weight_map);

    inline int IndexOf(int x, int y, int z) const {
        return x * resolution_ * resolution_ + y * resolution_ + z;
    }

    inline int IndexOf(const Eigen::Vector3i &xyz) const {
        return IndexOf(xyz(0), xyz(1), xyz(2));
    }

public:
    std::vector<geometry::WeightedTSDFVoxel> voxels_;
    Eigen::Vector3d origin_;
    double length_;
    int resolution_;
    int voxel_num_;

private:
    Eigen::Vector3d GetNormalAt(const Eigen::Vector3d &p);

    double GetTSDFAt(const Eigen::Vector3d &p);
};

}  // namespace integration
}  // namespace open3d
