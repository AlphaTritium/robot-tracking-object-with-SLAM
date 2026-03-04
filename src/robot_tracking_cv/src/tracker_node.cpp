#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose2_d.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>

using std::placeholders::_1;

// ANSI color codes
#define COLOR_RESET   "\033[0m"
#define COLOR_GREEN   "\033[32m"
#define COLOR_YELLOW  "\033[33m"
#define COLOR_CYAN    "\033[36m"

namespace hw_vision_ctrl {

class UltimateTracker : public rclcpp::Node
{
public:
  explicit UltimateTracker(const rclcpp::NodeOptions & options = rclcpp::NodeOptions())
  : Node("ultimate_tracker", options)
  {
    // ----- Control parameters -----
    declare_parameter("target_distance", 0.5);
    declare_parameter("dist_tolerance", 0.05);
    declare_parameter("angle_tolerance", 3.0);
    declare_parameter("kp_linear", 0.8);
    declare_parameter("kd_linear", 0.1);
    declare_parameter("kp_angular", 1.2);
    declare_parameter("kd_angular", 0.2);
    declare_parameter("kp_yaw", 0.5);
    declare_parameter("kd_yaw", 0.1);
    declare_parameter("max_linear_vel", 0.5);
    declare_parameter("max_angular_vel", 0.8);

    // ----- Detection parameters -----
    declare_parameter("purple_h_lower", 120);
    declare_parameter("purple_h_upper", 160);
    declare_parameter("purple_s_lower", 50);
    declare_parameter("purple_s_upper", 255);
    declare_parameter("purple_v_lower", 50);
    declare_parameter("purple_v_upper", 255);
    declare_parameter("min_contour_area", 15.0);
    declare_parameter("min_solidity", 0.6);
    declare_parameter("gaussian_kernel", 5);
    declare_parameter("morph_kernel", 3);
    declare_parameter("edge_margin", 20);
    declare_parameter("canny_lower", 50);
    declare_parameter("canny_upper", 150);
    declare_parameter("hough_threshold", 40);
    declare_parameter("min_line_length", 30);
    declare_parameter("max_line_gap", 10);

    // ----- Orbit behaviour -----
    declare_parameter("orbit_turn_vel", 0.3);
    declare_parameter("orbit_back_vel", -0.15);
    declare_parameter("cube_size", 0.4);          // physical side length (m)
    declare_parameter("approach_speed", 0.2);

    // Retrieve parameters
    target_dist_ = get_parameter("target_distance").as_double();
    dist_tol_ = get_parameter("dist_tolerance").as_double();
    angle_tol_deg_ = get_parameter("angle_tolerance").as_double();
    kp_lin_ = get_parameter("kp_linear").as_double();
    kd_lin_ = get_parameter("kd_linear").as_double();
    kp_ang_ = get_parameter("kp_angular").as_double();
    kd_ang_ = get_parameter("kd_angular").as_double();
    kp_yaw_ = get_parameter("kp_yaw").as_double();
    kd_yaw_ = get_parameter("kd_yaw").as_double();
    max_lin_vel_ = get_parameter("max_linear_vel").as_double();
    max_ang_vel_ = get_parameter("max_angular_vel").as_double();

    purple_h_lower_ = get_parameter("purple_h_lower").as_int();
    purple_h_upper_ = get_parameter("purple_h_upper").as_int();
    purple_s_lower_ = get_parameter("purple_s_lower").as_int();
    purple_s_upper_ = get_parameter("purple_s_upper").as_int();
    purple_v_lower_ = get_parameter("purple_v_lower").as_int();
    purple_v_upper_ = get_parameter("purple_v_upper").as_int();
    min_contour_area_ = get_parameter("min_contour_area").as_double();
    min_solidity_ = get_parameter("min_solidity").as_double();
    gaussian_kernel_ = get_parameter("gaussian_kernel").as_int();
    morph_kernel_ = get_parameter("morph_kernel").as_int();
    edge_margin_ = get_parameter("edge_margin").as_int();
    canny_lower_ = get_parameter("canny_lower").as_int();
    canny_upper_ = get_parameter("canny_upper").as_int();
    hough_thresh_ = get_parameter("hough_threshold").as_int();
    min_line_len_ = get_parameter("min_line_length").as_int();
    max_line_gap_ = get_parameter("max_line_gap").as_int();

    orbit_turn_ = get_parameter("orbit_turn_vel").as_double();
    orbit_back_ = get_parameter("orbit_back_vel").as_double();
    cube_size_ = get_parameter("cube_size").as_double();
    approach_speed_ = get_parameter("approach_speed").as_double();

    // PID state
    last_lin_error_ = 0.0;
    last_ang_error_ = 0.0;
    last_time_ = now();

    // Yaw tracking
    prev_target_yaw_ = 0.0;
    prev_yaw_time_ = now();
    target_yaw_valid_ = false;

    // General state
    last_z_ = 0.0;
    last_angle_ = 0.0;
    have_last_pose_ = false;

    RCLCPP_INFO(get_logger(), COLOR_CYAN "[UltimateTracker] Node started." COLOR_RESET);

    // Subscriptions
    img_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/camera/image_raw", 10, std::bind(&UltimateTracker::image_callback, this, _1));
    info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
        "/camera/camera_info", 1, std::bind(&UltimateTracker::info_callback, this, _1));

    // Publishers
    cmd_pub_ = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
    pose_pub_ = create_publisher<geometry_msgs::msg::Pose2D>("/robot_target_pose", 10);
    debug_pub_ = create_publisher<sensor_msgs::msg::Image>("/tracker/debug_view", 10);
    mask_pub_ = create_publisher<sensor_msgs::msg::Image>("/tracker/mask", 10);

    // 3D object points – four corners of the square face (size cube_size_)
    float half = cube_size_ / 2.0f;
    object_points_ = {
      cv::Point3f(-half, -half, 0.0f),  // TL
      cv::Point3f( half, -half, 0.0f),  // TR
      cv::Point3f( half,  half, 0.0f),  // BR
      cv::Point3f(-half,  half, 0.0f)   // BL
    };
  }

private:
  void info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
  {
    if (!has_camera_info_) {
      camera_matrix_ = cv::Mat(3, 3, CV_64F, (void*)msg->k.data()).clone();
      dist_coeffs_ = cv::Mat(1, 5, CV_64F, (void*)msg->d.data()).clone();
      fx_ = camera_matrix_.at<double>(0,0);
      has_camera_info_ = true;
      RCLCPP_INFO(get_logger(), COLOR_GREEN "[SUCCESS] Camera info received." COLOR_RESET);
    }
  }

  // Order four points clockwise from top‑left
  std::vector<cv::Point2f> order_points_clockwise(const std::vector<cv::Point2f>& pts)
  {
    if (pts.size() != 4) return pts;
    cv::Point2f centroid(0,0);
    for (const auto& p : pts) centroid += p;
    centroid *= 1.0f/4.0f;

    std::vector<cv::Point2f> ordered = pts;
    std::sort(ordered.begin(), ordered.end(),
        [centroid](const cv::Point2f& a, const cv::Point2f& b) {
          return atan2(a.y - centroid.y, a.x - centroid.x) <
                 atan2(b.y - centroid.y, b.x - centroid.x);
        });

    int tl_idx = 0;
    double min_sum = ordered[0].x + ordered[0].y;
    for (int i = 1; i < 4; ++i) {
      double s = ordered[i].x + ordered[i].y;
      if (s < min_sum) { min_sum = s; tl_idx = i; }
    }
    std::rotate(ordered.begin(), ordered.begin() + tl_idx, ordered.end());
    return ordered;
  }

  // Project object points to image using last known pose
  std::vector<cv::Point2f> project_object_points(const cv::Mat& rvec, const cv::Mat& tvec)
  {
    std::vector<cv::Point2f> img_pts;
    cv::projectPoints(object_points_, rvec, tvec, camera_matrix_, dist_coeffs_, img_pts);
    return img_pts;
  }

  // Fill missing dots (2-3 visible) using last known pose
  bool fill_missing_from_last_pose(const std::vector<cv::Point2f>& detected,
                                    std::vector<cv::Point2f>& full_set)
  {
    if (!have_last_pose_ || detected.empty()) return false;
    std::vector<cv::Point2f> projected = project_object_points(last_rvec_, last_tvec_);
    const double MATCH_DIST = 50.0;

    std::vector<bool> used(4, false);
    std::vector<cv::Point2f> result(4);

    for (const auto& d : detected) {
      int best = -1;
      double best_dist = MATCH_DIST;
      for (int j = 0; j < 4; ++j) {
        if (used[j]) continue;
        double dist = cv::norm(d - projected[j]);
        if (dist < best_dist) {
          best_dist = dist;
          best = j;
        }
      }
      if (best >= 0) {
        used[best] = true;
        result[best] = d;
      }
    }
    for (int j = 0; j < 4; ++j) {
      if (!used[j]) result[j] = projected[j];
    }
    full_set = result;
    return true;
  }

  void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    if (!has_camera_info_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
          COLOR_YELLOW "[WAITING] Camera info not yet received." COLOR_RESET);
      return;
    }

    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (cv_bridge::Exception& e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }

    cv::Mat frame = cv_ptr->image.clone();
    cv::Mat display = frame.clone();
    cv::Mat binary, edges;

    // ----- Preprocessing -----
    int gauss_size = gaussian_kernel_ | 1;
    cv::GaussianBlur(frame, frame, cv::Size(gauss_size, gauss_size), 0);
    cv::Mat hsv;
    cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);

    // ----- Purple mask -----
    cv::Scalar lower(purple_h_lower_, purple_s_lower_, purple_v_lower_);
    cv::Scalar upper(purple_h_upper_, purple_s_upper_, purple_v_upper_);
    cv::inRange(hsv, lower, upper, binary);

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(morph_kernel_, morph_kernel_));
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
    mask_pub_->publish(*cv_bridge::CvImage(msg->header, "mono8", binary).toImageMsg());

    // ----- Edge detection for Hough lines -----
    cv::Canny(binary, edges, canny_lower_, canny_upper_);
    std::vector<cv::Vec4i> lines;
    cv::HoughLinesP(edges, lines, 1, CV_PI/180, hough_thresh_, min_line_len_, max_line_gap_);
    for (const auto& l : lines) {
      cv::line(display, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(255,0,0), 1);
    }

    // ----- Find purple blobs (dots) -----
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    std::vector<std::pair<double, cv::Point2f>> candidates;
    for (const auto& cnt : contours) {
      double area = cv::contourArea(cnt);
      if (area < min_contour_area_) continue;

      std::vector<cv::Point> hull;
      cv::convexHull(cnt, hull);
      double hull_area = cv::contourArea(hull);
      double solidity = area / std::max(1.0, hull_area);
      if (solidity < min_solidity_) continue;

      cv::Moments m = cv::moments(cnt);
      if (m.m00 != 0) {
        cv::Point2f center(m.m10 / m.m00, m.m01 / m.m00);
        candidates.emplace_back(area, center);
      }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });
    size_t num_detected = std::min(candidates.size(), (size_t)4);
    std::vector<cv::Point2f> detected_pts;
    for (size_t i = 0; i < num_detected; ++i) {
      detected_pts.push_back(candidates[i].second);
      cv::circle(display, candidates[i].second, 4, cv::Scalar(0,255,255), -1);
    }

    // Timing
    rclcpp::Time now = this->now();
    double dt = std::clamp((now - last_time_).seconds(), 0.01, 0.1);
    last_time_ = now;

    geometry_msgs::msg::Twist twist;
    bool pnp_success = false;

    // ================= STATE MACHINE =================
    if (num_detected == 4)
    {
      // ----- STATE: FULL TRACK & ALIGN -----
      std::vector<cv::Point2f> ordered_pts = order_points_clockwise(detected_pts);
      for (int i = 0; i < 4; ++i) {
        cv::line(display, ordered_pts[i], ordered_pts[(i+1)%4], cv::Scalar(0,255,0), 2);
      }

      cv::Mat rvec, tvec;
      if (cv::solvePnP(object_points_, ordered_pts, camera_matrix_, dist_coeffs_, rvec, tvec, false, cv::SOLVEPNP_IPPE) ||
          cv::solvePnP(object_points_, ordered_pts, camera_matrix_, dist_coeffs_, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE))
      {
        double z = tvec.at<double>(2);
        double x = tvec.at<double>(0);
        if (z > 0.1 && !std::isnan(z) && !std::isnan(x))
        {
          pnp_success = true;
          last_rvec_ = rvec.clone();
          last_tvec_ = tvec.clone();
          have_last_pose_ = true;

          double robot_x = z;
          double robot_y = -x;
          double angle_rad = atan2(x, z);
          double angle_deg = angle_rad * 180.0 / CV_PI;
          last_z_ = z;
          last_angle_ = angle_rad;

          // Distance error & avoiding flag
          double dist_error = robot_x - target_dist_;
          bool avoiding = (dist_error < 0);   // your insight: sign flip when avoiding

          // PD for distance
          double linear_cmd = 0.0;
          if (std::abs(dist_error) > dist_tol_) {
            double deriv = (dist_error - last_lin_error_) / dt;
            linear_cmd = kp_lin_ * dist_error + kd_lin_ * deriv;
            last_lin_error_ = dist_error;
          }

          // PD for centering (bearing)
          double angular_cmd = 0.0;
          if (std::abs(angle_deg) > angle_tol_deg_) {
            double deriv = (angle_deg - last_ang_error_) / dt;
            angular_cmd = kp_ang_ * angle_deg + kd_ang_ * deriv;
            last_ang_error_ = angle_deg;
          }

          // APPLY YOUR SIGN FLIP WHEN AVOIDING
          if (avoiding) {
            angular_cmd = -angular_cmd;
          }

          // Face alignment from rotation
          cv::Mat R;
          cv::Rodrigues(rvec, R);
          double obj_zx = R.at<double>(0,2);
          double obj_zz = R.at<double>(2,2);
          double face_yaw_deg = atan2(obj_zx, obj_zz) * 180.0 / CV_PI;

          double face_align = kp_yaw_ * face_yaw_deg;
          if (avoiding) face_align = -face_align;   // also flip face alignment when backing

          // Feedforward from target yaw rate
          double yaw_rate = 0.0;
          if (target_yaw_valid_) {
            double dt_yaw = (now - prev_yaw_time_).seconds();
            if (dt_yaw > 0.0 && dt_yaw < 0.5) {
              yaw_rate = (face_yaw_deg - prev_target_yaw_) / dt_yaw;
            }
          }
          prev_target_yaw_ = face_yaw_deg;
          prev_yaw_time_ = now;
          target_yaw_valid_ = true;
          double ff = -kd_yaw_ * yaw_rate;

          // Combine angular commands
          angular_cmd += face_align + ff;

          // Edge safety (reduce angular speed near border)
          int w = frame.cols, h = frame.rows;
          bool near_edge = false;
          for (const auto& pt : ordered_pts) {
            if (pt.x < edge_margin_ || pt.x > w - edge_margin_ ||
                pt.y < edge_margin_ || pt.y > h - edge_margin_) {
              near_edge = true;
              break;
            }
          }
          if (near_edge) {
            angular_cmd *= 0.5;
            cv::putText(display, "NEAR EDGE", cv::Point(10,90),
                        cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,165,255), 2);
          }

          twist.linear.x = std::clamp(linear_cmd, -max_lin_vel_, max_lin_vel_);
          twist.angular.z = std::clamp(angular_cmd, -max_ang_vel_, max_ang_vel_);

          geometry_msgs::msg::Pose2D pose_msg;
          pose_msg.x = robot_x;
          pose_msg.y = robot_y;
          pose_msg.theta = angle_deg;
          pose_pub_->publish(pose_msg);

          cv::drawFrameAxes(display, camera_matrix_, dist_coeffs_, rvec, tvec, 0.1);
          cv::putText(display, "FULL TRACK", cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,255,0), 2);
        }
      }
    }
    else if (num_detected >= 1 && num_detected <= 3)
    {
      // ----- STATE: APPROACH OR ORBIT -----
      target_yaw_valid_ = false;   // can't estimate yaw reliably

      // Compute centroid of visible dots
      cv::Point2f centroid(0,0);
      for (const auto& pt : detected_pts) centroid += pt;
      centroid *= (1.0f / num_detected);
      cv::circle(display, centroid, 5, cv::Scalar(255,100,0), -1);

      // Estimate distance using maximum pixel spread (if at least 2 dots)
      double estimated_z = last_z_;
      if (num_detected >= 2) {
        double max_px_dist = 0;
        for (size_t i=0; i<num_detected; i++)
          for (size_t j=i+1; j<num_detected; j++)
            max_px_dist = std::max(max_px_dist, cv::norm(detected_pts[i] - detected_pts[j]));
        if (max_px_dist > 1.0) {
          estimated_z = (fx_ * cube_size_) / max_px_dist;
        }
      }

      if (estimated_z > target_dist_ + dist_tol_) {
        // Too far → approach forward while centering the cluster
        double center_err = frame.cols/2.0f - centroid.x;
        twist.linear.x = approach_speed_;
        twist.angular.z = kp_ang_ * (center_err * 0.01);   // simple proportional centering
        cv::putText(display, "APPROACHING", cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,255,0), 2);
      } else {
        // Distance OK or too close → ORBIT to find missing dots
        // Determine orbit direction by comparing vertical edges with dot centroid
        float edge_x_sum = 0.0f;
        int v_edges = 0;
        for (const auto& l : lines) {
          // rough vertical check: dx < dy * 0.5
          if (std::abs(l[0] - l[2]) < std::abs(l[1] - l[3]) * 0.5) {
            edge_x_sum += (l[0] + l[2]) / 2.0f;
            v_edges++;
          }
        }

        double turn_dir = 0.0;
        std::string orbit_msg;
        if (v_edges > 0) {
          float avg_edge_x = edge_x_sum / v_edges;
          if (avg_edge_x > centroid.x + 10.0f) {
            // edge is to the right → missing dots are to the left → orbit left
            turn_dir = 1.0;
            orbit_msg = "ORBIT LEFT (edge right)";
          } else if (avg_edge_x < centroid.x - 10.0f) {
            turn_dir = -1.0;
            orbit_msg = "ORBIT RIGHT (edge left)";
          } else {
            turn_dir = (centroid.x > frame.cols/2.0f) ? 1.0 : -1.0;   // fallback
            orbit_msg = "FALLBACK ORBIT";
          }
        } else {
          // no edges – use screen position
          turn_dir = (centroid.x > frame.cols/2.0f) ? 1.0 : -1.0;
          orbit_msg = "BLIND ORBIT";
        }

        // Execute orbit: back up while turning
        twist.linear.x = orbit_back_;           // negative = reverse
        twist.angular.z = turn_dir * orbit_turn_;
        cv::putText(display, orbit_msg, cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255,100,255), 2);
      }
    }
    else // num_detected == 0
    {
      // ----- STATE: LOST – MANUAL OVERRIDE -----
      twist.linear.x = 0.0;
      twist.angular.z = 0.0;
      last_lin_error_ = 0.0;
      last_ang_error_ = 0.0;
      have_last_pose_ = false;
      target_yaw_valid_ = false;
      cv::putText(display, "TARGET LOST - MANUAL", cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,255), 2);
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 1000, COLOR_YELLOW "[LOST] 0 dots – manual control." COLOR_RESET);
    }

    // Publish commands and debug image
    cmd_pub_->publish(twist);
    debug_pub_->publish(*cv_bridge::CvImage(msg->header, "bgr8", display).toImageMsg());
  }

  // Member variables
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr img_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr info_sub_;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr cmd_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Pose2D>::SharedPtr pose_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr debug_pub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr mask_pub_;

  cv::Mat camera_matrix_, dist_coeffs_;
  bool has_camera_info_ = false;
  double fx_ = 1.0;
  std::vector<cv::Point3f> object_points_;

  // Parameters
  double target_dist_, dist_tol_, angle_tol_deg_;
  double kp_lin_, kd_lin_, kp_ang_, kd_ang_, kp_yaw_, kd_yaw_;
  double max_lin_vel_, max_ang_vel_;

  int purple_h_lower_, purple_h_upper_, purple_s_lower_, purple_s_upper_, purple_v_lower_, purple_v_upper_;
  double min_contour_area_, min_solidity_;
  int gaussian_kernel_, morph_kernel_, edge_margin_;
  int canny_lower_, canny_upper_, hough_thresh_, min_line_len_, max_line_gap_;

  double orbit_turn_, orbit_back_, cube_size_, approach_speed_;

  // State
  double last_lin_error_, last_ang_error_;
  rclcpp::Time last_time_;
  double prev_target_yaw_;
  rclcpp::Time prev_yaw_time_;
  bool target_yaw_valid_;

  double last_z_, last_angle_;
  bool have_last_pose_;
  cv::Mat last_rvec_, last_tvec_;
};

} // namespace hw_vision_ctrl

int main(int argc, char** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<hw_vision_ctrl::UltimateTracker>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}