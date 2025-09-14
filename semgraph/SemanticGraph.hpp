#pragma once

#include <vector>
#include <Eigen/Core>
#include <yaml-cpp/yaml.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/common/transforms.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <opencv2/opencv.hpp>
#include <tbb/parallel_for.h>
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <chrono>

#include "Coreutils.hpp" 
#include "SemanticCluster.hpp"
#include "FastIcp.hpp"
#include "PlaneIcp.hpp"
#include "Max_clique.hpp"
#include "NDT.h"

namespace SDSGM
{
    class SemanticGraph
    {
    private:
        struct VoxelHash{
            size_t operator()(const Eigen::Vector3i &voxel) const {
                const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
                return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349663 ^ vec[2] * 83492791);
                }
            };
        struct VoxelBlock{
            VoxelBlock(std::initializer_list<Eigen::Vector3d> points, std::initializer_list<int> labels, std::initializer_list<Eigen::Vector3d> volumes) : points(points), lables(labels), volumes(volumes) {}
            std::vector<Eigen::Vector3d> points;
            std::vector<int> lables;
            std::vector<Eigen::Vector3d> volumes;
            inline void AddPoint(const Eigen::Vector3d &point, const int &lable, const Eigen::Vector3d &volume) {
                points.push_back(point);
                lables.push_back(lable);
                volumes.push_back(volume);
            }
        };
        bool show=true;
        bool remap=true;
        bool cluster_view = false;

 
        double deltaA = 2;
        double deltaR = 0.35; 
        double deltaP = 1.2; 

        // graph
        double edge_th = 30;        
        double sub_interval = 3;
        double box_diff_th =  2;    
        float distance_th = 6;
        //outlier node pair filter
        double inlier_th = 0.7;
        float inlier_rate_th = 0.5;
        float dist_th = 0.2;
        
        YAML::Node learning_map;
        std::vector<int> label_map;
        typedef std::tuple<u_char, u_char, u_char> Color;
        std::map<uint32_t, Color> _color_map, _argmax_to_rgb;
        std::shared_ptr<pcl::visualization::CloudViewer> viewer;
    public:
        int frame_count = 0;
        double total_time = 0;
        SemanticGraph(std::string conf_file);
        ~SemanticGraph();

        // load LiDAR scan
        std::pair<std::vector<Eigen::Vector3d>,std::vector<int>> load_cloud(const std::vector<float> & values_cloud, const std::vector<uint32_t> & values_label, int num);
        
        // graph
        SDSGM::Graph build_graph(std::vector<SDSGM::Bbox> cluster_boxes, double edge_dis_th, double subinterval);
        
        //scan desc
        std::vector<float> gen_scan_descriptors(const std::vector<float> & values_cloud, const std::vector<uint32_t> & values_label, int num);


        
        // mathcing node and outlier pruning
        std::tuple<V3d_i,V3d_i>  find_correspondences_withidx(SDSGM::Graph graph1,
                                                                SDSGM::Graph graph2);

        std::tuple<std::vector<Eigen::Vector3d>,std::vector<Eigen::Vector3d>> outlier_pruning(SDSGM::Graph graph1,SDSGM::Graph graph2,
                                                                                                V3d_i match_node1, V3d_i match_node2);

        // ransac
        std::tuple<Eigen::Isometry3d,double>inlier_refine(std::vector<Eigen::Vector3d> match_node1,
                                                            std::vector<Eigen::Vector3d> match_node2,
                                                            Graph &graph1,
                                                            Graph &graph2,
                                                            int& best_inlier_num);
                                                            

        Eigen::Isometry3d solveSVD(std::vector<Eigen::Vector3d> match_node1,
                                    std::vector<Eigen::Vector3d> match_node2);

        // similarity 
        double loop_pairs_similarity(const std::string  cloud_file1, const std::string label_file1,
                                                                         const std::string cloud_file2, const std::string label_file2);
        Eigen::Matrix4d loop_pose(const std::string  cloud_file1, const std::string label_file1,
                                                                         const std::string cloud_file2, const std::string label_file2);
        double get_sim(const std::vector<float>vec1, const std::vector<float>vec2);
        double calculate_sim(Eigen::MatrixXf &desc1, Eigen::MatrixXf &desc2, int yaw);

        std::pair<std::vector<Eigen::Vector3d>, std::vector<int>> pcl2eigen(pcl::PointCloud<pcl::PointXYZL>::Ptr pointcloud);
        std::tuple<double, double, double> rotationMatrixToEuler(const Eigen::Matrix3d& R);
    };
}