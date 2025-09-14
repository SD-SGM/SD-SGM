#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>

#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <random>
#include <cmath>
#include "SemanticGraph.hpp"
#include "NDT.h"

namespace SDSGM
{
    inline double square(double x) { return x * x; }
    struct ResultTuple
    {
        ResultTuple()
        {
            JTJ.setZero();
            JTr.setZero();
        }

        ResultTuple operator+(const ResultTuple &other)
        {
            this->JTJ += other.JTJ;
            this->JTr += other.JTr;
            return *this;
        }

        Eigen::Matrix<double, 6, 6> JTJ;
        Eigen::Matrix<double, 6, 1> JTr;
    };

    SemanticGraph::SemanticGraph(std::string conf_file)
    {
        auto data_cfg = YAML::LoadFile(conf_file);
        show = data_cfg["show"].as<bool>();
        cluster_view = data_cfg["cluster_view"].as<bool>();
        remap = data_cfg["remap"].as<bool>();

        // cluster
        deltaA = data_cfg["deltaA"].as<double>();
        deltaR = data_cfg["deltaR"].as<double>();
        deltaP = data_cfg["deltaP"].as<double>();

        edge_th = data_cfg["edge_th"].as<double>();
        sub_interval = data_cfg["edge_sub_interval"].as<double>();
        box_diff_th = data_cfg["box_diff_th"].as<double>();
        dist_th = data_cfg["dist_th"].as<float>();
        inlier_th = data_cfg["inlier_th"].as<double>();
        distance_th = data_cfg["distance_th"].as<float>();
        inlier_rate_th = data_cfg["inlier_rate_th"].as<float>();
        if (show)
        {
            viewer.reset(new pcl::visualization::CloudViewer("viewer"));
        }
        auto color_map = data_cfg["color_map"];
        learning_map = data_cfg["learning_map"];
        label_map.resize(260);
        for (auto it = learning_map.begin(); it != learning_map.end(); ++it)
        {
            label_map[it->first.as<int>()] = it->second.as<int>();
        }
        YAML::const_iterator it;
        for (it = color_map.begin(); it != color_map.end(); ++it)
        {
            // Get label and key
            int key = it->first.as<int>(); // <- key
            Color color = std::make_tuple(
                static_cast<u_char>(color_map[key][0].as<unsigned int>()),
                static_cast<u_char>(color_map[key][1].as<unsigned int>()),
                static_cast<u_char>(color_map[key][2].as<unsigned int>()));
            _color_map[key] = color;
        }
        auto learning_class = data_cfg["learning_map_inv"];
        for (it = learning_class.begin(); it != learning_class.end(); ++it)
        {
            int key = it->first.as<int>(); // <- key
            _argmax_to_rgb[key] = _color_map[learning_class[key].as<unsigned int>()];
        }
    }

    SemanticGraph::~SemanticGraph()
    {
    }

    SDSGM::Graph SemanticGraph::build_graph(std::vector<SDSGM::Bbox> cluster_boxes, double edge_dis_th, double subinterval)
    {
        SDSGM::Graph frame_graph;
        float sub_interval_value = edge_dis_th / subinterval;
        int N = cluster_boxes.size();
        Eigen::MatrixXf AdjacencyMatrix = Eigen::MatrixXf::Zero(N, N);
        Eigen::MatrixXf NodeEmbeddings_length = Eigen::MatrixXf::Zero(N, 3 * subinterval);
        Eigen::MatrixXf NodeEmbeddings_size = Eigen::MatrixXf::Zero(N, 3 * subinterval);
        Eigen::MatrixXf NodeEmbeddings = Eigen::MatrixXf::Zero(N, 6 * subinterval);
        Eigen::VectorXd obj_num = Eigen::VectorXd::Zero(3 * subinterval);

        for (int i = 0; i < N; i++)
        {
            frame_graph.node_labels.emplace_back(cluster_boxes[i].label);
            if (cluster_boxes[i].label == 1)
                frame_graph.node_stable.emplace_back(0);
            else
                frame_graph.node_stable.emplace_back(1);
            frame_graph.node_centers.emplace_back(cluster_boxes[i].center);
            frame_graph.node_dimensions.emplace_back(cluster_boxes[i].dimension);
            std::vector<std::pair<int, double>> vertex_edges;
            std::vector<double> vertex_size;
            if (cluster_boxes[i].label == 1)
                frame_graph.car_num++;
            if (cluster_boxes[i].label == 2)
                frame_graph.trunk_num++;
            if (cluster_boxes[i].label == 3)
                frame_graph.pole_like_num++;
            frame_graph.global_desc[(cluster_boxes[i].label - 1) * 3 + min(2, (int)cluster_boxes[i].size)]++;
            for (int j = 0; j < N; j++)
            {
                double edge = (cluster_boxes[i].center - cluster_boxes[j].center).norm();
                if (edge < edge_dis_th && edge != 0)
                {
                    vertex_edges.emplace_back(std::make_pair(cluster_boxes[j].label, edge));
                    vertex_size.emplace_back(cluster_boxes[j].size);
                    AdjacencyMatrix(i, j) = 1;
                }
            }
            // build vertes desc
            obj_num.setZero();
            for (size_t m = 0; m < vertex_edges.size(); m++)
            {
                if (vertex_edges[m].first == 1)
                { // x - car
                    NodeEmbeddings_length(i, int(vertex_edges[m].second / sub_interval_value)) += vertex_edges[m].second;
                    NodeEmbeddings_size(i, int(vertex_edges[m].second / sub_interval_value)) += vertex_size[m];
                    obj_num(int(vertex_edges[m].second / sub_interval_value))++;
                }
                else if (vertex_edges[m].first == 2)
                { // x - truck
                    NodeEmbeddings_length(i, subinterval + int(vertex_edges[m].second / sub_interval_value)) += vertex_edges[m].second;
                    NodeEmbeddings_size(i, subinterval + int(vertex_edges[m].second / sub_interval_value)) += vertex_size[m];
                    obj_num(subinterval + int(vertex_edges[m].second / sub_interval_value))++;
                }
                else if (vertex_edges[m].first == 3)
                { // x - pole
                    NodeEmbeddings_length(i, 2 * subinterval + int(vertex_edges[m].second / sub_interval_value)) += vertex_edges[m].second;
                    NodeEmbeddings_size(i, 2 * subinterval + int(vertex_edges[m].second / sub_interval_value)) += vertex_size[m];
                    obj_num(2 * subinterval + int(vertex_edges[m].second / sub_interval_value))++;
                }
            }
            for (int j = 0; j < obj_num.size(); j++)
            {
                if (obj_num[j] == 0)
                    continue;
                NodeEmbeddings_length(i, j) /= obj_num(j);
                NodeEmbeddings_size(i, j) /= obj_num(j);
            }
        }

        if (frame_graph.node_labels.size() == 0)
            return frame_graph;

        NodeEmbeddings.leftCols(3 * subinterval) = NodeEmbeddings_length;
        NodeEmbeddings.rightCols(3 * subinterval) = NodeEmbeddings_size;

        for (int i = 0; i < N; i++)
        {
            Eigen::MatrixXf evec_sort_row = NodeEmbeddings.row(i);
            std::vector<float> node_desf(evec_sort_row.data(), evec_sort_row.data() + evec_sort_row.size());
            frame_graph.node_desc.emplace_back(node_desf);
        }

        return frame_graph;
    }

    // 从文件中读取数据，分别是点云文件和对应的标签文件,点云中的每一个点都有对应的标签
    std::pair<std::vector<Eigen::Vector3d>, std::vector<int>> SemanticGraph::load_cloud(const std::vector<float> &values_cloud, const std::vector<uint32_t> &values_label, int num_points)
    {
        std::vector<Eigen::Vector3d> pc_out(num_points);
        std::vector<int> label_out(num_points);
        for (int i = 0; i < num_points; ++i)
        {
            uint32_t sem_label = label_map[(int)(values_label[i] & 0x0000ffff)];
            // 舍弃所有不需要的节点
            if (sem_label == 0)
                continue;
            pc_out[i] = Eigen::Vector3d(values_cloud[4 * i], values_cloud[4 * i + 1], values_cloud[4 * i + 2]); // 舍弃了第四维的数据
            label_out[i] = (int)sem_label;
        }
        return std::make_pair(pc_out, label_out);
    }

    double SemanticGraph::loop_pairs_similarity(const std::string cloud_file1, const std::string label_file1,
                                                const std::string cloud_file2, const std::string label_file2)
    {
        // process pointcloud data
        std::ifstream in_label1(label_file1, std::ios::binary);
        in_label1.seekg(0, std::ios::end);
        uint32_t num_points1 = in_label1.tellg() / sizeof(uint32_t);
        in_label1.seekg(0, std::ios::beg);
        std::vector<uint32_t> values_label1(num_points1);
        in_label1.read((char *)&values_label1[0], num_points1 * sizeof(uint32_t));
        std::ifstream in_cloud1(cloud_file1, std::ios::binary);
        std::vector<float> values_cloud1(4 * num_points1);
        in_cloud1.read((char *)&values_cloud1[0], 4 * num_points1 * sizeof(float));

        std::ifstream in_label2(label_file2, std::ios::binary);
        in_label2.seekg(0, std::ios::end);
        uint32_t num_points2 = in_label2.tellg() / sizeof(uint32_t);
        in_label2.seekg(0, std::ios::beg);
        std::vector<uint32_t> values_label2(num_points2);
        in_label2.read((char *)&values_label2[0], num_points2 * sizeof(uint32_t));
        std::ifstream in_cloud2(cloud_file2, std::ios::binary);
        std::vector<float> values_cloud2(4 * num_points2);
        in_cloud2.read((char *)&values_cloud2[0], 4 * num_points2 * sizeof(float));

        // global strcutural desc
        NDT ndtmc_manager_1(1);
        auto sem_desc1 = ndtmc_manager_1.readpc(values_cloud1, values_label1, num_points1);
        auto desc1 = ndtmc_manager_1.createNDT();
        NDT ndtmc_manager_2(1);
        auto sem_desc2 = ndtmc_manager_2.readpc(values_cloud2, values_label2, num_points2);
        auto desc2 = ndtmc_manager_2.createNDT();

        std::pair<double, int> result = NDT::distanceBtnNDTScanContext(desc1, desc2);
        std::pair<double, int> result_sem = NDT::calculate_sim(sem_desc1, sem_desc2, result.second);

        // semantic graph
        double simility_score = 0;
        auto cloud1 = load_cloud(values_cloud1, values_label1, num_points1);
        auto cloud2 = load_cloud(values_cloud2, values_label2, num_points2);

        // cluster 
        pcl::PointCloud<pcl::PointXYZL>::Ptr background_points1(new pcl::PointCloud<pcl::PointXYZL>());
        pcl::PointCloud<pcl::PointXYZL>::Ptr background_points2(new pcl::PointCloud<pcl::PointXYZL>());
        auto cluster_box1 = SDSGM::cluster(cloud1.first, cloud1.second, *background_points1, deltaA, deltaR, deltaP, cluster_view);
        auto cluster_box2 = SDSGM::cluster(cloud2.first, cloud2.second, *background_points2, deltaA, deltaR, deltaP, cluster_view);

        // semantic graph
        auto graph1 = build_graph(cluster_box1, edge_th, sub_interval);
        auto graph2 = build_graph(cluster_box2, edge_th, sub_interval);

        auto [match_node_inGraph1, match_node_inGraph2] = find_correspondences_withidx(graph1, graph2);
        auto [refine_match_node_inGraph1, refine_match_node_inGraph2] = outlier_pruning(graph1, graph2, match_node_inGraph1, match_node_inGraph2);
        double distance = 100;
        int best_inlier = 0;
        int yaw = -1;
        if (refine_match_node_inGraph1.size() >= 3)
        {
            auto [trans, score] = inlier_refine(refine_match_node_inGraph1, refine_match_node_inGraph2, graph1, graph2, best_inlier);
            auto eulerangle = rotationMatrixToEuler(trans.rotation());
            yaw = std::get<0>(eulerangle);
            Eigen::Vector3d translation = trans.matrix().block<3, 1>(0, 3);
            distance = translation.norm();
            simility_score = score;
            if (distance > distance_th)
                simility_score = 0;
        }
        else
        {
            simility_score = 0;
            yaw = 10000;
        }
        yaw = yaw < 0 ? yaw + 360 : yaw;
        int node_num = min(graph1.car_num, graph2.car_num) + min(graph1.pole_like_num, graph2.pole_like_num) + min(graph1.trunk_num, graph2.trunk_num);
        double inlier_rate = (node_num == 0) ? 0 : (double)best_inlier / node_num;

        // fusion
        double score = 0;

        std::pair<double, int> ssc_score;
        ssc_score = NDT::calculate_sim(sem_desc1, sem_desc2, yaw);
        if (ssc_score.first == 0 && result_sem.first == 0)
        {
            if (inlier_rate <= inlier_th)
                return 0;
            else
                ssc_score.first = 1;
        }
        int delta_yaw;
        if (simility_score == 0)
        {
            score = result_sem.first * 0.1 / log(node_num < 3 ? 3 : node_num);
        }
        else if (ssc_score.first > result_sem.first)
        {
            score = simility_score;
        }
        else
        {
            delta_yaw = min(abs(result_sem.second - yaw), 360 - abs(result_sem.second - yaw));
            score = simility_score * exp(-delta_yaw / (30 * result.first));
        }
        return score;
    }

    Eigen::Matrix4d SemanticGraph::loop_pose(const std::string cloud_file1, const std::string label_file1, const std::string cloud_file2, const std::string label_file2)
    {
        std::ifstream in_label1(label_file1, std::ios::binary);
        in_label1.seekg(0, std::ios::end);
        uint32_t num_points1 = in_label1.tellg() / sizeof(uint32_t);
        in_label1.seekg(0, std::ios::beg);
        std::vector<uint32_t> values_label1(num_points1);
        in_label1.read((char *)&values_label1[0], num_points1 * sizeof(uint32_t));
        std::ifstream in_cloud1(cloud_file1, std::ios::binary);
        std::vector<float> values_cloud1(4 * num_points1);
        in_cloud1.read((char *)&values_cloud1[0], 4 * num_points1 * sizeof(float));

        std::ifstream in_label2(label_file2, std::ios::binary);
        in_label2.seekg(0, std::ios::end);
        uint32_t num_points2 = in_label2.tellg() / sizeof(uint32_t);
        in_label2.seekg(0, std::ios::beg);
        std::vector<uint32_t> values_label2(num_points2);
        in_label2.read((char *)&values_label2[0], num_points2 * sizeof(uint32_t));
        std::ifstream in_cloud2(cloud_file2, std::ios::binary);
        std::vector<float> values_cloud2(4 * num_points2);
        in_cloud2.read((char *)&values_cloud2[0], 4 * num_points2 * sizeof(float));

        NDT ndtmc_manager_1(1);
        auto sem_desc1 = ndtmc_manager_1.readpc(values_cloud1, values_label1, num_points1);
        auto desc1 = ndtmc_manager_1.createNDT();
        NDT ndtmc_manager_2(1);
        auto sem_desc2 = ndtmc_manager_2.readpc(values_cloud2, values_label2, num_points2);
        auto desc2 = ndtmc_manager_2.createNDT();

        std::pair<double, int> result = NDT::distanceBtnNDTScanContext(desc1, desc2);
        std::pair<double, int> result_sem = NDT::calculate_sim(sem_desc1, sem_desc2, result.second);


        double simility_score = 0;
        auto cloud1 = load_cloud(values_cloud1, values_label1, num_points1);
        auto cloud2 = load_cloud(values_cloud2, values_label2, num_points2);


        pcl::PointCloud<pcl::PointXYZL>::Ptr background_points1(new pcl::PointCloud<pcl::PointXYZL>());
        pcl::PointCloud<pcl::PointXYZL>::Ptr background_points2(new pcl::PointCloud<pcl::PointXYZL>());
        auto cluster_box1 = SDSGM::cluster(cloud1.first, cloud1.second, *background_points1, deltaA, deltaR, deltaP, cluster_view);
        auto cluster_box2 = SDSGM::cluster(cloud2.first, cloud2.second, *background_points2, deltaA, deltaR, deltaP, cluster_view);

        auto graph1 = build_graph(cluster_box1, edge_th, sub_interval);
        auto graph2 = build_graph(cluster_box2, edge_th, sub_interval);

        auto [match_node_inGraph1, match_node_inGraph2] = find_correspondences_withidx(graph1, graph2);
        auto [refine_match_node_inGraph1, refine_match_node_inGraph2] = outlier_pruning(graph1, graph2, match_node_inGraph1, match_node_inGraph2);
        double distance = 100;
        int best_inlier = 0;
        int yaw = -1;
        Eigen::Matrix4d transform;
        transform << 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0;
        if (refine_match_node_inGraph1.size() >= 3)
        {
            auto [trans, score] = inlier_refine(refine_match_node_inGraph1, refine_match_node_inGraph2, graph1, graph2, best_inlier);
            FastIcp instanceIcp(0.25, 0.25, 1, 5);
            transform = instanceIcp.get_trans(cloud1, cloud2, trans.matrix());
            auto eulerangle = rotationMatrixToEuler(trans.rotation());
            yaw = std::get<0>(eulerangle);
            Eigen::Vector3d translation = trans.matrix().block<3, 1>(0, 3);
            distance = translation.norm();
            simility_score = score;
            if (distance > distance_th)
                simility_score = 0;
        }
        else
        {
            simility_score = 0;
            yaw = 10000;
        }
        yaw = yaw < 0 ? yaw + 360 : yaw;
        int node_num = min(graph1.car_num, graph2.car_num) + min(graph1.pole_like_num, graph2.pole_like_num) + min(graph1.trunk_num, graph2.trunk_num);
        double inlier_rate = (node_num == 0) ? 0 : (double)best_inlier / node_num;

        double score = 0;
        std::pair<double, int> ssc_score;
        ssc_score = NDT::calculate_sim(sem_desc1, sem_desc2, yaw);
        if (ssc_score.first == 0 && result_sem.first == 0)
        {
            if (inlier_rate <= inlier_th)
                return transform;
            else
                ssc_score.first = 1;
        }
        int delta_yaw;
        if (simility_score == 0)
        {
            score = result_sem.first * 0.1 / log(node_num < 3 ? 3 : node_num);
        }
        else if (ssc_score.first > result_sem.first)
        {
            score = simility_score;
        }
        else
        {
            delta_yaw = min(abs(result_sem.second - yaw), 360 - abs(result_sem.second - yaw));
            score = simility_score * exp(-delta_yaw / (30 * result.first));
        }

        return transform;
    }

    std::vector<float> SemanticGraph::gen_scan_descriptors(const std::vector<float> &values_cloud, const std::vector<uint32_t> &values_label, int num)
    {

        NDT ndt_manager(1);
        ndt_manager.readpc(values_cloud, values_label, num);
        auto desc = ndt_manager.createNDT();
        std::vector<float> scan_desc;
        scan_desc.resize(40);
        for (int i = 0; i < desc.rows(); i++)
        {
            for (int j = 0; j < desc.cols(); j++)
            {
                if (desc(i, j) == 0)
                    continue;
                else if (desc(i, j) < 0.4)
                    scan_desc[2 * i]++;
                else
                    scan_desc[2 * i + 1]++;
            }
        }
        float sum_of_squares_graph = std::inner_product(scan_desc.begin(), scan_desc.end(), scan_desc.begin(), 0.0f);
        float l2_norm_graph = std::sqrt(sum_of_squares_graph);
        if (l2_norm_graph == 0)
            l2_norm_graph = 0.000001f;
        for (auto it = scan_desc.begin(); it != scan_desc.end(); ++it)
        {
            *it /= l2_norm_graph;
        }
        return scan_desc;
    }

    double SemanticGraph::get_sim(const std::vector<float> vec1, const std::vector<float> vec2)
    {
        assert(vec1.size() == vec2.size());
        int no_zero = 0;
        Eigen::VectorXf dist_vec(vec1.size());
        for (size_t i = 0; i < vec1.size(); i++)
        {
            if (vec1[i] == 0 && vec2[i] == 0)
            {
                dist_vec[i] = 0;
                continue;
            }
            dist_vec[i] = float(std::abs(vec1[i] - vec2[i]) / (vec1[i] + vec2[i]));
            no_zero++;
        }
        if (no_zero == 0)
            return 0;
        double sim = 1 - (dist_vec.sum() / vec1.size());
        return sim;
    }

    std::tuple<V3d_i, V3d_i> SemanticGraph::find_correspondences_withidx(SDSGM::Graph graph1, SDSGM::Graph graph2)
    {
        std::vector<Eigen::Vector3d> query_nodes_center;
        std::vector<Eigen::Vector3d> match_nodes_center;
        std::vector<int> query_nodes_idx;
        std::vector<int> match_nodes_idx;

        // node correspondence
        for (size_t i = 0; i < graph1.node_desc.size(); i++)
        {
            int max_score = 0;
            int id = -1;
            for (size_t j = 0; j < graph2.node_desc.size(); j++)
            {
                if (graph1.node_labels[i] == graph2.node_labels[j])
                { // check node's label
                    double node_des_cost = 1 - get_sim(graph1.node_desc[i], graph2.node_desc[j]);
                    if (node_des_cost > 0.5 || node_des_cost < 0)
                        continue;
                    if (std::abs(graph1.node_dimensions[i].x() - graph2.node_dimensions[j].x()) > box_diff_th ||
                        std::abs(graph1.node_dimensions[i].y() - graph2.node_dimensions[j].y()) > box_diff_th ||
                        std::abs(graph1.node_dimensions[i].z() - graph2.node_dimensions[j].z()) > box_diff_th)
                        continue;
                    query_nodes_center.emplace_back(graph1.node_centers[i]);
                    query_nodes_idx.emplace_back(i);
                    match_nodes_center.emplace_back(graph2.node_centers[j]);
                    match_nodes_idx.emplace_back(j);
                }
            }
        }
        return {std::make_pair(query_nodes_center, query_nodes_idx), std::make_pair(match_nodes_center, match_nodes_idx)};
    }

    std::tuple<std::vector<Eigen::Vector3d>, std::vector<Eigen::Vector3d>> SemanticGraph::outlier_pruning(SDSGM::Graph graph1, SDSGM::Graph graph2, V3d_i match_node1, V3d_i match_node2)
    {

        std::vector<Eigen::Vector3d> inlier_match_node1;
        std::vector<Eigen::Vector3d> inlier_match_node2;
        std::vector<std::vector<int>> match_node_dist(match_node1.first.size(), std::vector<int>(match_node2.first.size(), 0));

        assert(match_node1.first.size() == match_node2.first.size());
        for (int i = 0; i < match_node1.first.size(); i++)
        {
            for (int j = 0; j < match_node1.first.size(); j++)
            {
                double dist1 = (match_node1.first[i] - match_node1.first[j]).norm();
                double dist2 = (match_node2.first[i] - match_node2.first[j]).norm();
                if (std::abs(dist1 - dist2) < dist_th)
                {
                    match_node_dist[i][j] = 1;
                }
                else
                {
                    match_node_dist[i][j] = 0;
                }
            }
        }
        // maxclique
        MaxClique MaxClique(match_node1.first.size(), match_node_dist);
        vector<int> match_index = MaxClique.solve();
        for (int i = 0; i < match_index.size(); i++)
        {
            inlier_match_node1.emplace_back(match_node1.first[match_index[i]]);
            inlier_match_node2.emplace_back(match_node2.first[match_index[i]]);
        }
        return {inlier_match_node1, inlier_match_node2};
    }

    std::tuple<Eigen::Isometry3d, double> SemanticGraph::inlier_refine(std::vector<Eigen::Vector3d> match_node1,
                                                                          std::vector<Eigen::Vector3d> match_node2,
                                                                          Graph &graph1,
                                                                          Graph &graph2,
                                                                          int &best_inlier_num)
    {
        assert(match_node1.size() == match_node2.size());

        float voxel_size = 0.5;
        tsl::robin_map<Eigen::Vector3i, VoxelBlock, VoxelHash> node2_voxel_map;

        for (int m = 0; m < graph2.node_centers.size(); m++)
        {
            Eigen::Vector3d point = graph2.node_centers[m];
            int lable = graph2.node_labels[m];
            Eigen::Vector3d volume = graph2.node_dimensions[m];
            auto voxel = Eigen::Vector3i((point / voxel_size).template cast<int>());
            auto search = node2_voxel_map.find(voxel);
            if (search != node2_voxel_map.end())
            {
                auto &voxel_block = search.value();
                voxel_block.AddPoint(point, lable, volume);
            }
            else
            {
                node2_voxel_map.insert({voxel, VoxelBlock{{point}, {lable}, {volume}}});
            }
        }
        Eigen::Isometry3d trans_matrix = solveSVD(match_node1, match_node2);
        std::vector<Eigen::Vector3d> matched_node_in_graph1, matched_node_in_graph2;
        for (size_t n = 0; n < graph1.node_centers.size(); n++)
        {
            Eigen::Vector3d match_node;
            bool if_inlier = false;
            Eigen::Vector3d trans_match_node1 = trans_matrix * graph1.node_centers[n];
            auto kx = static_cast<int>(trans_match_node1[0] / voxel_size);
            auto ky = static_cast<int>(trans_match_node1[1] / voxel_size);
            auto kz = static_cast<int>(trans_match_node1[2] / voxel_size);
            std::vector<Eigen::Vector3i> voxels;
            voxels.reserve(9);
            for (int i = kx - 1; i <= kx + 1; ++i)
            {
                for (int j = ky - 1; j <= ky + 1; ++j)
                {
                    for (int k = kz - 1; k <= kz + 1; ++k)
                    {
                        voxels.emplace_back(i, j, k);
                    }
                }
            }

            std::vector<std::tuple<Eigen::Vector3d, int, Eigen::Vector3d>> neighboors;

            std::for_each(voxels.cbegin(), voxels.cend(), [&](const auto &voxel){
            bool repeat = false;
            auto search = node2_voxel_map.find(voxel);
            if (search != node2_voxel_map.end()) {
                const auto &points = search->second.points;
                const auto &lables = search->second.lables;
                const auto &volumes = search->second.volumes;
                if (!points.empty()) {
                    for(int point_num=0; point_num < points.size(); point_num++){
                        for_each(matched_node_in_graph2.cbegin(), matched_node_in_graph2.cend(),[&](const auto &matched_node){
                            if(matched_node == points[point_num])
                            repeat = true;        
                        });
                        if(repeat) {repeat = false; continue;}
                        neighboors.emplace_back(std::make_tuple(points[point_num],lables[point_num],volumes[point_num]));
                    }
                }
            } });

            for (int neigh_num = 0; neigh_num < neighboors.size(); neigh_num++)
            {
                if (std::get<1>(neighboors[neigh_num]) == graph1.node_labels[n] && abs(std::get<2>(neighboors[neigh_num])[0] + std::get<2>(neighboors[neigh_num])[1] - graph1.node_dimensions[n][0] - graph1.node_dimensions[n][1]) < 1.5 && abs(std::get<2>(neighboors[neigh_num])[2] - graph1.node_dimensions[n][2]) < 1)
                {
                    match_node = std::get<0>(neighboors[neigh_num]);
                    if_inlier = true;
                }
            }
            if (if_inlier)
            {
                matched_node_in_graph2.emplace_back(match_node);
                matched_node_in_graph1.emplace_back(graph1.node_centers[n]);
            }
        }

        int inlier_num = 0;
        double loss = 0;

        Eigen::Isometry3d refine_trans_matrix = solveSVD(matched_node_in_graph1, matched_node_in_graph2);
        std::vector<Eigen::Vector3d> matched_node_2;

        for (size_t n = 0; n < graph1.node_desc.size(); n++)
        {
            bool if_inlier = false;
            double trans_node_dis = 1e3;
            Eigen::Vector3d match_node;
            Eigen::Vector3d refine_trans_match_node1 = refine_trans_matrix * graph1.node_centers[n];
            auto kx = static_cast<int>(refine_trans_match_node1[0] / voxel_size);
            auto ky = static_cast<int>(refine_trans_match_node1[1] / voxel_size);
            auto kz = static_cast<int>(refine_trans_match_node1[2] / voxel_size);
            std::vector<Eigen::Vector3i> refine_voxels;
            refine_voxels.reserve(9);
            for (int i = kx - 1; i < kx + 1 + 1; ++i)
            {
                for (int j = ky - 1; j < ky + 1 + 1; ++j)
                {
                    for (int k = kz - 1; k < kz + 1 + 1; ++k)
                    {
                        refine_voxels.emplace_back(i, j, k);
                    }
                }
            }

            std::vector<std::tuple<Eigen::Vector3d, int, Eigen::Vector3d>> neighboors;

            std::for_each(refine_voxels.cbegin(), refine_voxels.cend(), [&](const auto &voxel)
                          {
                    bool repeat = false;
                    auto search = node2_voxel_map.find(voxel);
                    if (search != node2_voxel_map.end()) {
                        const auto &points = search->second.points;
                        const auto &lables = search->second.lables;
                        const auto &volumes = search->second.volumes;
                        if (!points.empty()) {
                            for(int point_num=0; point_num < points.size(); point_num++){
                                for_each(matched_node_2.cbegin(), matched_node_2.cend(),[&](const auto &matched_node){
                                    if(matched_node == points[point_num])
                                    repeat = true;        
                                });
                                if(repeat) {repeat = false; continue;}
                                neighboors.emplace_back(std::make_tuple(points[point_num],lables[point_num],volumes[point_num]));
                            }
                        }
                    } });

            for (int neigh_num = 0; neigh_num < neighboors.size(); neigh_num++)
            {
                if (std::get<1>(neighboors[neigh_num]) = graph1.node_labels[n] && abs(std::get<2>(neighboors[neigh_num])[0] + std::get<2>(neighboors[neigh_num])[1] - graph1.node_dimensions[n][0] - graph1.node_dimensions[n][1]) < 1.5 && abs(std::get<2>(neighboors[neigh_num])[2] - graph1.node_dimensions[n][2]) < 1)
                {
                    double trans_node_dis_t = (std::get<0>(neighboors[neigh_num]) - refine_trans_match_node1).norm();
                    if (trans_node_dis_t < trans_node_dis)
                    {
                        trans_node_dis = trans_node_dis_t;
                        match_node = std::get<0>(neighboors[neigh_num]);
                    }
                    if_inlier = true;
                }
            }
            if (if_inlier)
            {
                inlier_num++;
                loss = loss + trans_node_dis;
                matched_node_2.emplace_back(match_node);
            }
            else
            {
                loss += 1;
            }
        }
        double score = exp(-loss / inlier_num);
        best_inlier_num = inlier_num;
        if (inlier_num <= 4 && (double)inlier_num / (graph1.node_desc.size(), graph2.node_desc.size()) < inlier_rate_th)
            score = 0;
        return {refine_trans_matrix, score};
    }

    Eigen::Isometry3d SemanticGraph::solveSVD(std::vector<Eigen::Vector3d> match_node1, std::vector<Eigen::Vector3d> match_node2)
    {

        if (match_node1.empty() || match_node2.empty() || match_node1.size() != match_node2.size())
        {
            std::cout<<"Error! solve SVD: input pointcloud size is not same or empty"<<std::endl;
        }

        int N = match_node1.size();
        Eigen::Vector3d node1_sum{0, 0, 0}, node2_sum{0, 0, 0};
        for (int i = 0; i < N; i++)
        {
            node1_sum += match_node1[i];
            node2_sum += match_node2[i];
        }
        Eigen::Vector3d node1_mean = node1_sum / N;
        Eigen::Vector3d node2_mean = node2_sum / N;

        std::vector<Eigen::Vector3d> node1_list, node2_list;
        for (int i = 0; i < N; i++)
        {
            node1_list.emplace_back(match_node1[i] - node1_mean);
            node2_list.emplace_back(match_node2[i] - node2_mean);
        }

        Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
        for (int i = 0; i < N; i++)
        {
            W += node1_list[i] * node2_list[i].transpose();
        }
        for (size_t i = 0; i < W.size(); i++)
        {
            if (std::isnan(W(i)))
            {
                std::cout << "error: the input points are wrong, can't solve with SVD." << std::endl;
            }
        }

        // svd
        Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::Matrix3d U = svd.matrixU();
        Eigen::Matrix3d V = svd.matrixV();
        Eigen::Matrix3d E = Eigen::Matrix3d::Identity();

        Eigen::Matrix3d R_temp = V * (U.transpose());

        if (R_temp.determinant() < 0)
        {
            E(2, 2) = -1;
        }
        Eigen::Matrix3d R = V * E * (U.transpose());
        Eigen::Vector3d T = node2_mean - R * node1_mean;

        Eigen::Isometry3d trans_matrix = Eigen::Isometry3d::Identity();
        trans_matrix.linear() = R;
        trans_matrix.translation() = T;

        return trans_matrix;
    }

    double SemanticGraph::calculate_sim(Eigen::MatrixXf &desc1, Eigen::MatrixXf &desc2, int yaw_in)
    {

        int sectors = desc1.cols();
        int rings = desc1.rows();
        double score = 0;
        for (int yaw = yaw_in - 6; yaw < yaw_in + 6; yaw++)
        {
            if (yaw < 0)
                yaw += 360;
            if (yaw > 359)
                yaw -= 360;
            double similarity = 0;
            int valid_num = 0;
            for (int p = 0; p < rings; p++)
            {
                for (int q = 0; q < sectors; q++)
                {
                    if (desc1(p, q) == 0 && desc2(p, (q + yaw) % sectors) == 0)
                    {
                        continue;
                    }
                    valid_num++;

                    if (desc1(p, q) == desc2(p, (q + yaw) % sectors))
                    {
                        similarity++;
                    }
                }
                if (score < similarity / valid_num)
                    score = similarity / valid_num;
            }
        }
        return score;
    }

    std::pair<std::vector<Eigen::Vector3d>, std::vector<int>> SemanticGraph::pcl2eigen(pcl::PointCloud<pcl::PointXYZL>::Ptr pointcloud)
    {
        std::vector<Eigen::Vector3d> eigen_pc(pointcloud->size());
        std::vector<int> eigen_pc_label(pointcloud->size());

        tbb::parallel_for(size_t(0), pointcloud->size(), [&](size_t i)
                          {
            eigen_pc[i] =  Eigen::Vector3d((*pointcloud)[i].x, (*pointcloud)[i].y, (*pointcloud)[i].z);
            eigen_pc_label[i] = (*pointcloud)[i].label; });
        return std::make_pair(eigen_pc, eigen_pc_label);
    }

    std::tuple<double, double, double> SemanticGraph::rotationMatrixToEuler(const Eigen::Matrix3d& R) {
        double sy = sqrt(R(0, 0) * R(0, 0) + R(1, 0) * R(1, 0));
        bool singular = sy < 1e-6; 
        double yaw1, pitch1, roll1;
        double yaw2, pitch2, roll2;
        double yaw, pitch, roll;
        if (!singular) {
            yaw1 = atan2(R(1, 0), R(0, 0));
            pitch1 = atan2(-R(2, 0), sy);
            roll1 = atan2(R(2, 1), R(2, 2));

            yaw2 = atan2(-R(1, 0), -R(0, 0));
            pitch2 = atan2(-R(2, 0), -sy);
            roll2 = atan2(-R(2, 1), -R(2, 2));
        } else {
            yaw1 = atan2(-R(1, 2), R(1, 1));
            pitch1 = atan2(-R(2, 0), sy);
            roll1 = 0;

            yaw2 = atan2(R(1, 2), -R(1, 1));
            pitch2 = atan2(-R(2, 0), -sy);
            roll2 = 0;
        }

    
        yaw1 = yaw1 * 180.0 / M_PI;
        pitch1 = pitch1 * 180.0 / M_PI;
        roll1 = roll1 * 180.0 / M_PI;

        yaw2 = yaw2 * 180.0 / M_PI;
        pitch2 = pitch2 * 180.0 / M_PI;
        roll2 = roll2 * 180.0 / M_PI;
        if(abs(pitch1)+abs(roll1) < abs(pitch2)+abs(roll2)){
            yaw = yaw1;
            pitch = pitch1;
            roll = roll1;
        }else{
            yaw = yaw2;
            pitch = pitch2;
            roll = roll2;
        }

        return std::make_tuple(yaw, pitch, roll);
    }
}
