//
// Created by liaolizhou on 23-4-3.
//

#include "NDT.h"

NDT::NDT(float resolution_) : resolution(resolution_)
{
    ndt_pointcloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
};

Eigen::MatrixXd NDT::readpc(const std::vector<float> & values_cloud, const std::vector<uint32_t> & values_label, int num_points)
{   
    // auto start = chrono::high_resolution_clock::now();
    pc_in = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    Eigen::MatrixXd desc(24,360);
    desc.setZero();
    for (uint32_t i = 0; i < num_points; i+=2) {
        // 处理点云数据
        pcl::PointXYZI p;
        p.x = values_cloud[4*i];
        p.y = values_cloud[4*i+1];
        p.z = values_cloud[4*i+2] + 1.73;  // 假设需要对 z 做偏移
        pc_in->points.push_back(p);

        uint32_t sem_label;
        sem_label = values_label[i];
        switch (sem_label){
            case 40: //road
            case 48: //side walk
            case 60: sem_label = 1; break; //lane_marking
            case 50: sem_label = 4; break; //building
            case 51: sem_label = 5; break; //fence
            case 70: sem_label = 3; break; //vegetation
            default: continue;
        }
        int ring_idx, sector_idx;
        ring_idx = (sqrt(p.x * p.x + p.y * p.y) -5) / 1.875;
        sector_idx = xy2theta(p.x, p.y);
        if(ring_idx > 23 || ring_idx <0 || sector_idx > 359 || sector_idx < 0) continue;
        desc(ring_idx, sector_idx) = sem_label>desc(ring_idx, sector_idx) ? sem_label : desc(ring_idx, sector_idx);
    }
    // for(int i=0;i<desc.rows();i++){
    //     for(int j=0;j<desc.cols();j++){
    //         std::cout<<desc(i,j)<<" ";
    //     }
    //     std::cout<<std::endl;
    // }
    return desc;
    // auto end = chrono::high_resolution_clock::now();
    // auto duration = chrono::duration<double, std::milli>(end-start).count();
    //     std::cout<<"read time: "<<duration<<"\n";
}

double NDT::distDirect(const Eigen::MatrixXd &_frame_sc, Eigen::MatrixXd _submap_sc)
{
    // Calculate correlation similarity
    double sc_sim = 0;
    int no_zero_col = 0;
    for(int col_idx = 0; col_idx < _frame_sc.cols(); col_idx++){
        Eigen::VectorXd col_sc1 = _frame_sc.col(col_idx);
        Eigen::VectorXd col_sc2 = _submap_sc.col(col_idx);
        if(col_sc1.norm() == 0 || col_sc2.norm() == 0)
        continue;
        no_zero_col++;
        sc_sim += col_sc1.dot(col_sc2)/(col_sc1.norm()*col_sc2.norm());
    }
   sc_sim = sc_sim / no_zero_col;

    return 1.0 - abs(sc_sim);

} // distDirectSC

Eigen::MatrixXd NDT::circshift(Eigen::MatrixXd _mat, const Eigen::MatrixXd &_mat2)
{
    Eigen::MatrixXd shifted_mat = Eigen::MatrixXd::Zero(_mat.rows(), _mat.cols() + _mat2.cols());
    shifted_mat.block(0, 0, _mat.rows(), _mat.cols()) = _mat;
    shifted_mat.block(0, _mat.cols(), _mat.rows(), _mat2.cols()) = _mat.block(0, 0, _mat.rows(), _mat2.cols());
    return shifted_mat;
} // circshift

Eigen::MatrixXd NDT::circshift(Eigen::MatrixXd _mat)
{
    Eigen::MatrixXd expanded_mat = Eigen::MatrixXd::Zero(_mat.rows(), 2 * _mat.cols() - 1);
    expanded_mat.block(0, 0, _mat.rows(), _mat.cols()) = _mat;
    for (int i = 1; i < _mat.cols(); ++i)
    {
        expanded_mat.block(0, i + _mat.cols() - 1, _mat.rows(), 1) = _mat.block(0, i, _mat.rows(), 1);
    }
    return expanded_mat;
} // circshift

std::pair<double, int> NDT::distanceBtnNDTScanContext(Eigen::MatrixXd &framesc,
                                                           Eigen::MatrixXd &submapsc)
{
    // // Calculate a_mean and b_mean
    double a_mean = framesc.mean();
    double b_mean = submapsc.mean();
    // // Center a_mix and b_mix around their means
    Eigen::MatrixXd a = framesc.array() - a_mean;
    a = (a.array() == -a_mean).select(0, a); // replace -a_mean with 0
    Eigen::MatrixXd b = submapsc.array() - b_mean;
    b = (b.array() == -b_mean).select(0, b); // replace -b_mean with 0
    auto frame_sc = a;
    auto submap_sc = b;
    std::pair<double, int> result = std::make_pair(1000,-1);
    int len_cols = static_cast<int>(submap_sc.cols());
    Eigen::MatrixXd submap_sc_ = circshift(submap_sc);
 
    for (int num_shift = 0; num_shift < len_cols; num_shift+=2)
    {
        Eigen::MatrixXd submap_sc_shifted = submap_sc_.block(0, num_shift, frame_sc.rows(), frame_sc.cols()); 
        double cur_sc_dist = distDirect(frame_sc, submap_sc_shifted);

        if (cur_sc_dist == 0)
        continue;
        if (cur_sc_dist < result.first)
        result = std::make_pair(cur_sc_dist, num_shift*6);
    }   
    result.first = 1 - result.first;
    return result;
} // distanceBtnNDTScanContext

void NDT::transformToNDTForm(float resolution_)
{
    // pcl::PointCloud<pcl::PointXYZI>::Ptr cloudPtr = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>(*pc_in);
    voxel_grid_frame.setLeafSize(resolution_, resolution_, resolution_);
    voxel_grid_frame.setInputCloud(pc_in);
    // Initiate voxel structure.
    voxel_grid_frame.filter(true);
} // transformToNDTForm



Eigen::MatrixXd NDT::getNDTLeaves_new(const std::map<std::size_t, Leaf> &leaves_)
{
    GridCell grid[PC_NUM_RING][PC_NUM_SECTOR];
    histogram.resize(num_intervals);
    for (const auto &leave : leaves_)
    {
        if (leave.second.nr_points < 5 || leave.second.evals_(2) == 0){
            continue;
        }
        // Eigen::Matrix3d cov = leave.second.cov_;
        pcl::PointXYZI point;
        point.x = static_cast<float>(leave.second.mean_(0));
        point.y = static_cast<float>(leave.second.mean_(1));
        point.z = static_cast<float>(leave.second.mean_(2));
        int ring_idx, sector_idx;
        if (!CalculateSectorId(point, ring_idx, sector_idx) || point.z < 0 || point.z > PC_MAX_Z)
        {
            continue;
        }
      
        double value = leave.second.evals_(2) * leave.second.evals_(0) / (leave.second.evals_(1) * leave.second.evals_(1));
        //向量全局描述符生成，用于初步筛选
        double interval_size = 3 / double(num_intervals);
        if (value < 3) {
            int his_idx = static_cast<int>(std::floor(value / interval_size));
            histogram[his_idx]++;
        }
        //矩阵全局描述符生成，用于计算相似度
        if (value > 0 && value < 2.4) {
            int shape_id = static_cast<int>(std::floor(value / 0.3));
            grid[ring_idx - 1][sector_idx - 1].addShape(shape_id+1);
        }
    }


    //直方图模长为1
    float histogram_norm = 0;
    for_each(histogram.cbegin(),histogram.cend(), [&] (auto value) {
        histogram_norm += value * value;
    });
    for(int i=0;i<num_intervals;i++){
        histogram[i] /= sqrt(histogram_norm);
    }

    //     Compute NDT matrix using shape weights
    Eigen::MatrixXd desc(PC_NUM_RING, PC_NUM_SECTOR);
    desc.setZero();
    for (int i = 0; i < PC_NUM_RING; i++)
    {
        for (int j = 0; j < PC_NUM_SECTOR; j++)
        {
            double weight = 0;
            if (grid[i][j].shape_max == -1) 
                continue;
            desc(i, j) += double(grid[i][j].shape_max)/ PC_NUM_Z;
        }
    }
    return desc;
} // getNDTLeaves_new

float NDT::xy2theta(const float &_x, const float &_y)
{
    if ((_x >= 0) & (_y >= 0))
        return static_cast<float>((180 / M_PI) * atan(_y / _x));

    if ((_x < 0) & (_y >= 0))
        return static_cast<float>(180 - ((180 / M_PI) * atan(_y / (-_x))));

    if ((_x < 0) & (_y < 0))
        return static_cast<float>(180 + ((180 / M_PI) * atan(_y / _x)));

    if ((_x >= 0) & (_y < 0))
        return static_cast<float>(360 - ((180 / M_PI) * atan((-_y) / _x)));
    return 0.0;
} // xy2theta

bool NDT::CalculateSectorId(pcl::PointXYZI &pt_in, int &ring_idx, int &sctor_idx)
{
    // xyz to ring, sector
    float azim_range = sqrt(pt_in.x * pt_in.x + pt_in.y * pt_in.y);
    float azim_angle = xy2theta(pt_in.x, pt_in.y);
    // if range is out of roi, pass
    if (azim_range > PC_MAX_RADIUS)
        return false;

    ring_idx = std::max(std::min(PC_NUM_RING, int(ceil((azim_range / PC_MAX_RADIUS) * PC_NUM_RING))), 1);
    sctor_idx = std::max(std::min(PC_NUM_SECTOR, int(ceil((azim_angle / 360.0) * PC_NUM_SECTOR))), 1);
    return true;
} // CalculateSectorId





Eigen::MatrixXd NDT::createNDT()
{
    transformToNDTForm(resolution);    
    return getNDTLeaves_new(voxel_grid_frame.getLeaves());
     
}


 std::pair<double, int> NDT::calculate_sim(Eigen::MatrixXd &desc1, Eigen::MatrixXd &desc2, int yaw){
        auto score = std::make_pair(0.0,0);
        int sectors = desc1.cols();
        int rings = desc1.rows();
        int yaw_ = yaw - 6;
        for(int i = 0; i<=12; i++){
            if(yaw_ < 0 ) yaw_ += 360;
            if(yaw_ > 359) yaw_ -= 360;
            int valid_num = 0;
            double similarity = 0;
            for (int p = 0; p < sectors; p++)
            {
                for (int q = 0; q < rings; q++)
                {
                    if (desc1(q, p) == 0 && desc2(q, (p+yaw_)%sectors) == 0)
                    {
                        continue;
                    }
                    valid_num++;
                    if (desc1(q, p) == desc2(q, (p+yaw_)%sectors))
                    {
                        similarity++;
                    }
                }
            }
            if(similarity/valid_num > score.first && similarity/valid_num > 0.3){
                score.first = similarity/valid_num;
                score.second = yaw_;
            } 
            yaw_ ++; 
        }
        
        return score;
    }
