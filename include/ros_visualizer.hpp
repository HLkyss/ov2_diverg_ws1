/**
*    This file is part of OV²SLAM.
*    
*    Copyright (C) 2020 ONERA
*
*    For more information see <https://github.com/ov2slam/ov2slam>
*
*    OV²SLAM is free software: you can redistribute it and/or modify
*    it under the terms of the GNU General Public License as published by
*    the Free Software Foundation, either version 3 of the License, or
*    (at your option) any later version.
*
*    OV²SLAM is distributed in the hope that it will be useful,
*    but WITHOUT ANY WARRANTY; without even the implied warranty of
*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*    GNU General Public License for more details.
*
*    You should have received a copy of the GNU General Public License
*    along with OV²SLAM.  If not, see <https://www.gnu.org/licenses/>.
*
*    Authors: Maxime Ferrera     <maxime.ferrera at gmail dot com> (ONERA, DTIS - IVA),
*             Alexandre Eudes    <first.last at onera dot fr>      (ONERA, DTIS - IVA),
*             Julien Moras       <first.last at onera dot fr>      (ONERA, DTIS - IVA),
*             Martial Sanfourche <first.last at onera dot fr>      (ONERA, DTIS - IVA)
*/
#pragma once


#include <ros/ros.h>

#include <std_msgs/Header.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Bool.h>

#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <cv_bridge/cv_bridge.h>

#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>

#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/TransformStamped.h>

#include <visualization_msgs/Marker.h>
#include <tf/transform_broadcaster.h>

#include <pcl_ros/point_cloud.h>

#include <sophus/se3.hpp>

#include "camera_visualizer.hpp"

#include <iostream>

class RosVisualizer {
    
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    
    RosVisualizer(ros::NodeHandle &n)
        : cameraposevisual_(1, 0, 0, 1)
    {
        std::cout << "\nROS visualizer is being created...\n";

        pub_image_track_ = n.advertise<sensor_msgs::Image>("image_track", 1000);
        pub_image_track_l_ = n.advertise<sensor_msgs::Image>("image_track_l", 1000);
        pub_image_track_r_ = n.advertise<sensor_msgs::Image>("image_track_r", 1000);

        pub_vo_traj_ = n.advertise<visualization_msgs::Marker>("vo_traj", 1000);
        pub_vo_pose_ = n.advertise<geometry_msgs::PoseStamped>("vo_pose", 1000);

        pub_vo_traj_r_ = n.advertise<visualization_msgs::Marker>("vo_traj_r", 1000);
        pub_vo_pose_r_ = n.advertise<geometry_msgs::PoseStamped>("vo_pose_r", 1000);

        vo_traj_msg_.type = visualization_msgs::Marker::LINE_STRIP;
        vo_traj_msg_.color.a = 1.0;
        vo_traj_msg_.color.r = 0.25;
        vo_traj_msg_.color.g = 1.0;
        vo_traj_msg_.color.b = 0.25;
        vo_traj_msg_.scale.x = 0.02;

        vo_traj_msg_r_.type = visualization_msgs::Marker::LINE_STRIP;
        vo_traj_msg_r_.color.a = 1.0;
        vo_traj_msg_r_.color.r = 1.0;
        vo_traj_msg_r_.color.g = 0.5;
        vo_traj_msg_r_.color.b = 0.0;
        vo_traj_msg_r_.scale.x = 0.02;

        camera_pose_visual_pub_ = n.advertise<visualization_msgs::MarkerArray>("cam_pose_visual", 1000);
        camera_pose_visual_pub_r_ = n.advertise<visualization_msgs::MarkerArray>("cam_pose_visual_r", 1000);

        cameraposevisual_.setScale(0.1);
//        cameraposevisual_.setScale(0.5);//相机可视化放大
        cameraposevisual_.setLineWidth(0.01);

        pub_point_cloud_ = n.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("point_cloud", 1000);
        pub_point_cloud_l_ = n.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("point_cloud_left", 1000);
        pub_point_cloud_r_ = n.advertise<pcl::PointCloud<pcl::PointXYZRGB>>("point_cloud_right", 1000);

        pub_kfs_traj_ = n.advertise<visualization_msgs::Marker>("kfs_traj", 1000);
        pub_kfs_pose_ = n.advertise<visualization_msgs::MarkerArray>("local_kfs_window", 1000);

        pub_final_kfs_traj_ = n.advertise<visualization_msgs::Marker>("final_kfs_traj", 1000);

        kfs_traj_msg_ = vo_traj_msg_;
        kfs_traj_msg_.color.r = 0.25;
        kfs_traj_msg_.color.g = 0.25;
        kfs_traj_msg_.color.b = 1.;

        kfs_traj_msg_r_ = vo_traj_msg_r_;
        kfs_traj_msg_r_.color.r = 0.25;
        kfs_traj_msg_r_.color.g = 0.25;
        kfs_traj_msg_r_.color.b = 1.;

        final_kfs_traj_msg_ = vo_traj_msg_;
        final_kfs_traj_msg_.color.r = 0.75;
        final_kfs_traj_msg_.color.g = 0.25;
        final_kfs_traj_msg_.color.b = 0.25;

        final_kfs_traj_msg_r_ = vo_traj_msg_r_;
        final_kfs_traj_msg_r_.color.r = 0.75;
        final_kfs_traj_msg_r_.color.g = 0.25;
        final_kfs_traj_msg_r_.color.b = 0.25;
    }

    void pubTrackImage(const cv::Mat &imgTrack, const double time)
    {
        if( pub_image_track_.getNumSubscribers() == 0 ) {
            return;
        }

        std_msgs::Header header;
        header.frame_id = "world";
        header.stamp = ros::Time(time);
        sensor_msgs::ImagePtr imgTrackMsg = cv_bridge::CvImage(header, "rgb8", imgTrack).toImageMsg();
        pub_image_track_.publish(imgTrackMsg);
    }

    void pubTrackImage(const cv::Mat &imgTrack, const ros::Publisher& pub_image_track, const double time)
    {
        if( pub_image_track.getNumSubscribers() == 0 ) {
            return;
        }

        std_msgs::Header header;
        header.frame_id = "world";
        header.stamp = ros::Time(time);
        sensor_msgs::ImagePtr imgTrackMsg = cv_bridge::CvImage(header, "rgb8", imgTrack).toImageMsg();
        pub_image_track.publish(imgTrackMsg);
    }

    void pubVO(const Sophus::SE3d &Twc, const double time, const Sophus::SE3d &T_right_left)
    {
        // 1. Publish marker message
        // =========================
        vo_traj_msg_.header.stamp = ros::Time(time);
        vo_traj_msg_.header.frame_id = "world";

        geometry_msgs::Point p;
        const Eigen::Vector3d &twc = Twc.translation();
        p.x = twc.x(); p.y = twc.y(); p.z = twc.z();

        if( p.x > 50. || p.y > 50. || p.z > 50 ) {
            if( vo_traj_msg_.scale.x < 0.1 ) {
                vo_traj_msg_.scale.x *= 20;  
                kfs_traj_msg_.scale.x *= 20; 
                final_kfs_traj_msg_.scale.x *= 20;     
            }
        }

        vo_traj_msg_.points.push_back(p);

        pub_vo_traj_.publish(vo_traj_msg_);

        // 2. Publish Pose Stamped + tf
        // ============================
        geometry_msgs::PoseStamped Twc_msg;
        geometry_msgs::Quaternion q;
        const Eigen::Quaterniond eigen_q(Twc.unit_quaternion());

        Twc_msg.pose.position = p;
        q.x = eigen_q.x(); q.y = eigen_q.y();
        q.z = eigen_q.z(); q.w = eigen_q.w();

        Twc_msg.pose.orientation = q;

        Twc_msg.header = vo_traj_msg_.header;

        pub_vo_pose_.publish(Twc_msg);
//        std::cout<<"pose = "<<p.x<<" "<<p.y<<" "<<p.z<<std::endl;
//        std::cout<<"q = "<<q.x<<" "<<q.y<<" "<<q.z<<" "<<q.w<<std::endl;

        tf::Transform transform;
        transform.setOrigin(tf::Vector3(p.x, p.y, p.z));
        tf::Quaternion qtf(q.x, q.y, q.z, q.w);
        transform.setRotation(qtf);

        static tf::TransformBroadcaster br;
        br.sendTransform(tf::StampedTransform(transform, ros::Time(time), "world", "camera"));

        // 3. Publish camera visual
        // =========================
        cameraposevisual_.reset();
        cameraposevisual_.add_pose(twc, eigen_q);
        cameraposevisual_.setImageBoundaryColor(1, 0, 0);
        cameraposevisual_.setOpticalCenterConnectorColor(1, 0, 0);
        cameraposevisual_.publish_by(camera_pose_visual_pub_, Twc_msg.header);

        //add 右目相机可视化 更换Twc和p
        // 2. 构建当前位姿矩阵 T_wc
        Sophus::SE3d T_wc_r = Twc * T_right_left.inverse();// 试出来应该是这样，但是为什么：基于全局坐标系的旋转变换左乘旋转矩阵，基于自身坐标系的旋转变换右乘旋转矩阵
        //推一下检查一下，假设左右各乘一个点坐标，则： T_wc_r * p = Twc * T_right_left.inverse() * p
        // 右目到世界*右目点 = 左目到世界*右目到左目*右目点，成立，所以没错
        Eigen::Vector3d t_r = T_wc_r.translation();
        Eigen::Quaterniond eigen_q_r(T_wc_r.unit_quaternion());
        geometry_msgs::Point p_r;
        p_r.x = t_r.x();
        p_r.y = t_r.y();
        p_r.z = t_r.z();
        geometry_msgs::Quaternion q_r;
        q_r.x = eigen_q_r.x();
        q_r.y = eigen_q_r.y();
        q_r.z = eigen_q_r.z();
        q_r.w = eigen_q_r.w();
        geometry_msgs::PoseStamped Twc_msg_r;
        Twc_msg_r.pose.position = p_r;       // 更新平移
        Twc_msg_r.pose.orientation = q_r;   // 更新旋转
        Twc_msg_r.header = vo_traj_msg_.header;
        cameraposevisual_.reset();
        cameraposevisual_.add_pose(t_r, eigen_q_r);
        cameraposevisual_.setImageBoundaryColor(1, 0, 0);
        cameraposevisual_.setOpticalCenterConnectorColor(1, 0, 0);
        cameraposevisual_.publish_by(camera_pose_visual_pub_r_, Twc_msg_r.header);

        // if( vo_traj_msg_.points.size() >= 3600 ) {
        //     size_t nbpts = vo_traj_msg_.points.size();
        //     std::vector<geometry_msgs::Point> vtmp;
        //     // if( nbpts / 20 < 3600 ) {
        //         vtmp.reserve(nbpts / 20);
        //         for( size_t i = 0 ; i < nbpts ; i+=20 ) {
        //             vtmp.push_back(vo_traj_msg_.points.at(i));
        //         }
        //     // } 
        //     // else {
        //     //     vtmp.reserve(nbpts / 100);
        //     //     for( size_t i = 0 ; i < nbpts ; i+=100 ) {
        //     //         vtmp.push_back(vo_traj_msg_.points.at(i));
        //     //     }
        //     // }

        //     vo_traj_msg_.points.swap(vtmp);
        // }

        return;
    }

    void pubVO_r(const Sophus::SE3d &Twc, const double time, const Sophus::SE3d &T_right_left)//todo stage2
    {
        // 1. Publish marker message
        // =========================
        vo_traj_msg_r_.header.stamp = ros::Time(time);
        vo_traj_msg_r_.header.frame_id = "world";

        geometry_msgs::Point p;
        const Eigen::Vector3d &twc = Twc.translation();
        p.x = twc.x(); p.y = twc.y(); p.z = twc.z();

        if( p.x > 50. || p.y > 50. || p.z > 50 ) {
            if( vo_traj_msg_r_.scale.x < 0.1 ) {
                vo_traj_msg_r_.scale.x *= 20;
                kfs_traj_msg_r_.scale.x *= 20;
                final_kfs_traj_msg_r_.scale.x *= 20;
            }
        }

        vo_traj_msg_r_.points.push_back(p);

        pub_vo_traj_r_.publish(vo_traj_msg_r_);

        // 2. Publish Pose Stamped + tf
        // ============================
        geometry_msgs::PoseStamped Twc_msg;
        geometry_msgs::Quaternion q;
        const Eigen::Quaterniond eigen_q(Twc.unit_quaternion());

        Twc_msg.pose.position = p;
        q.x = eigen_q.x(); q.y = eigen_q.y();
        q.z = eigen_q.z(); q.w = eigen_q.w();

        Twc_msg.pose.orientation = q;

        Twc_msg.header = vo_traj_msg_r_.header;

        pub_vo_pose_r_.publish(Twc_msg);
//        std::cout<<"pose = "<<p.x<<" "<<p.y<<" "<<p.z<<std::endl;
//        std::cout<<"q = "<<q.x<<" "<<q.y<<" "<<q.z<<" "<<q.w<<std::endl;

        tf::Transform transform;
        transform.setOrigin(tf::Vector3(p.x, p.y, p.z));
        tf::Quaternion qtf(q.x, q.y, q.z, q.w);
        transform.setRotation(qtf);

//        static tf::TransformBroadcaster br;// todo 这个要不要改
//        br.sendTransform(tf::StampedTransform(transform, ros::Time(time), "world", "camera"));
        static tf::TransformBroadcaster br_r;
        br_r.sendTransform(tf::StampedTransform(transform, ros::Time(time), "world", "camera"));

        // 3. Publish camera visual todo 右目可以不可视化相机，暂时不用右目的信息更新相机可视化位置了，只用左目的位姿结果更新rviz中相机可视化位姿。rviz中关键帧可视化也是只用左目结果，右目位姿结果在rviz中只可视化一个轨迹线
        /*// =========================
        cameraposevisual_.reset();
        cameraposevisual_.add_pose(twc, eigen_q);
        cameraposevisual_.setImageBoundaryColor(1, 0, 0);
        cameraposevisual_.setOpticalCenterConnectorColor(1, 0, 0);
        cameraposevisual_.publish_by(camera_pose_visual_pub_, Twc_msg.header);

        //add 右目相机可视化 更换Twc和p
        // 2. 构建当前位姿矩阵 T_wc
        Sophus::SE3d T_wc_r = Twc * T_right_left.inverse();// 试出来应该是这样，但是为什么：基于全局坐标系的旋转变换左乘旋转矩阵，基于自身坐标系的旋转变换右乘旋转矩阵
        //推一下检查一下，假设左右各乘一个点坐标，则： T_wc_r * p = Twc * T_right_left.inverse() * p
        // 右目到世界*右目点 = 左目到世界*右目到左目*右目点，成立，所以没错
        Eigen::Vector3d t_r = T_wc_r.translation();
        Eigen::Quaterniond eigen_q_r(T_wc_r.unit_quaternion());
        geometry_msgs::Point p_r;
        p_r.x = t_r.x();
        p_r.y = t_r.y();
        p_r.z = t_r.z();
        geometry_msgs::Quaternion q_r;
        q_r.x = eigen_q_r.x();
        q_r.y = eigen_q_r.y();
        q_r.z = eigen_q_r.z();
        q_r.w = eigen_q_r.w();
        geometry_msgs::PoseStamped Twc_msg_r;
        Twc_msg_r.pose.position = p_r;       // 更新平移
        Twc_msg_r.pose.orientation = q_r;   // 更新旋转
        Twc_msg_r.header = vo_traj_msg_.header;
        cameraposevisual_.reset();
        cameraposevisual_.add_pose(t_r, eigen_q_r);
        cameraposevisual_.setImageBoundaryColor(1, 0, 0);
        cameraposevisual_.setOpticalCenterConnectorColor(1, 0, 0);
        cameraposevisual_.publish_by(camera_pose_visual_pub_r_, Twc_msg_r.header);
*/

        // if( vo_traj_msg_.points.size() >= 3600 ) {
        //     size_t nbpts = vo_traj_msg_.points.size();
        //     std::vector<geometry_msgs::Point> vtmp;
        //     // if( nbpts / 20 < 3600 ) {
        //         vtmp.reserve(nbpts / 20);
        //         for( size_t i = 0 ; i < nbpts ; i+=20 ) {
        //             vtmp.push_back(vo_traj_msg_.points.at(i));
        //         }
        //     // }
        //     // else {
        //     //     vtmp.reserve(nbpts / 100);
        //     //     for( size_t i = 0 ; i < nbpts ; i+=100 ) {
        //     //         vtmp.push_back(vo_traj_msg_.points.at(i));
        //     //     }
        //     // }

        //     vo_traj_msg_.points.swap(vtmp);
        // }

        return;
    }

    void addVisualKF(const Sophus::SE3d &Twc) 
    {
        const Eigen::Quaterniond eigen_q(Twc.unit_quaternion());
        CameraPoseVisualization kfposevisual(0, 0, 1, 1);
        kfposevisual.add_pose(Twc.translation(), eigen_q);
        kfposevisual.setImageBoundaryColor(0, 0, 1);
        kfposevisual.setOpticalCenterConnectorColor(0, 0, 1);
        vkeyframesposevisual_.push_back(kfposevisual);
    }

    void pubVisualKFs(const double time) 
    {
        if( pub_kfs_pose_.getNumSubscribers() == 0 ) {
            return;
        }

        std_msgs::Header header;
        header.frame_id = "world";
        header.stamp = ros::Time(time);

        visualization_msgs::MarkerArray markerArray_msg;

        int j = 0;

        for( auto &camposeviz : vkeyframesposevisual_ )
        {
            for( auto &marker : camposeviz.m_markers )
            {
                marker.header = header;
                marker.id += j;
                markerArray_msg.markers.push_back(marker);
            }
            j++;
        }

        pub_kfs_pose_.publish(markerArray_msg);

        vkeyframesposevisual_.clear();
    }

    void pubPointCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcloud, const double time) 
    {
        if( pub_point_cloud_.getNumSubscribers() == 0 ) {
            return;
        }

        std_msgs::Header header;
        header.frame_id = "world";
        header.stamp = ros::Time(time);

        pcloud->header = pcl_conversions::toPCL(header);
        pub_point_cloud_.publish(pcloud);
    }

    void pubPointCloud_left(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcloud, const double time)
    {
        if( pub_point_cloud_l_.getNumSubscribers() == 0 ) {
            return;
        }

        std_msgs::Header header;
        header.frame_id = "world";
        header.stamp = ros::Time(time);

        pcloud->header = pcl_conversions::toPCL(header);
        pub_point_cloud_l_.publish(pcloud);
    }
    void pubPointCloud_right(pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcloud, const double time)
    {
        if( pub_point_cloud_r_.getNumSubscribers() == 0 ) {
            return;
        }

        std_msgs::Header header;
        header.frame_id = "world";
        header.stamp = ros::Time(time);

        pcloud->header = pcl_conversions::toPCL(header);
        pub_point_cloud_r_.publish(pcloud);
    }

    void addKFsTraj(const Sophus::SE3d &Twc)
    {
        geometry_msgs::Point p;
        const Eigen::Vector3d twc = Twc.translation();
        p.x = twc.x(); p.y = twc.y(); p.z = twc.z();

        kfs_traj_msg_.points.push_back(p);
    }

    void clearKFsTraj()
    {   
        kfs_traj_msg_.points.clear();
    }

    void pubKFsTraj(const double time)
    {   
        if( pub_kfs_traj_.getNumSubscribers() == 0 ) {
            return;
        }

        kfs_traj_msg_.header.stamp = ros::Time(time);
        kfs_traj_msg_.header.frame_id = "world";

        pub_kfs_traj_.publish(kfs_traj_msg_);
    }

    void pubFinalKFsTraj(const Sophus::SE3d &Twc, const double time)
    {   
        if( pub_final_kfs_traj_.getNumSubscribers() == 0 ) {
            return;
        }

        final_kfs_traj_msg_.header.stamp = ros::Time(time);
        final_kfs_traj_msg_.header.frame_id = "world";

        geometry_msgs::Point p;
        const Eigen::Vector3d twc = Twc.translation();
        p.x = twc.x(); p.y = twc.y(); p.z = twc.z();

        final_kfs_traj_msg_.points.push_back(p);

        pub_final_kfs_traj_.publish(final_kfs_traj_msg_);

        return;
    }

    ros::Publisher pub_image_track_;
    ros::Publisher pub_image_track_l_, pub_image_track_r_;

    ros::Publisher pub_vo_traj_, pub_vo_pose_;
    ros::Publisher pub_vo_traj_r_, pub_vo_pose_r_;
    visualization_msgs::Marker vo_traj_msg_;
    visualization_msgs::Marker vo_traj_msg_r_;

    ros::Publisher camera_pose_visual_pub_, camera_pose_visual_pub_r_;
    CameraPoseVisualization cameraposevisual_;

    ros::Publisher pub_point_cloud_;
    ros::Publisher pub_point_cloud_l_, pub_point_cloud_r_;

    ros::Publisher pub_kfs_pose_;
    std::vector<CameraPoseVisualization> vkeyframesposevisual_;

    ros::Publisher pub_kfs_traj_, pub_final_kfs_traj_;

    visualization_msgs::Marker kfs_traj_msg_, final_kfs_traj_msg_;
    visualization_msgs::Marker kfs_traj_msg_r_, final_kfs_traj_msg_r_;
};