#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose2_d.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <vector>
#include <cmath>
#include <algorithm>

using std::placeholders::_1;

class TrackerNode : public rclcpp::Node
{
public:
    TrackerNode() : Node("tracker_node")
    {
        // --- Control parameters ---
        this->declare_parameter("target_distance", 0.25);
        // Linear PID gains
        this->declare_parameter("kp_lin", 1.0);
        this->declare_parameter("ki_lin", 0.0);
        this->declare_parameter("kd_lin", 3.0);
        // Angular PID gains
        this->declare_parameter("kp_ang", 1.0);
        this->declare_parameter("ki_ang", 0.0);
        this->declare_parameter("kd_ang", 2.0);
        // Integral wind-up limits
        this->declare_parameter("lin_integral_max", 5.0);
        this->declare_parameter("ang_integral_max", 5.0);
        // Velocity limits
        this->declare_parameter("max_linear_vel", 5.0);
        this->declare_parameter("max_angular_vel", 5.0);

        // --- Detection parameters ---
        this->declare_parameter("use_color_filter", true);
        this->declare_parameter("lower_h", 45);
        this->declare_parameter("lower_s", 100);
        this->declare_parameter("lower_v", 100);
        this->declare_parameter("upper_h", 85);
        this->declare_parameter("upper_s", 255);
        this->declare_parameter("upper_v", 255);
        this->declare_parameter("min_contour_area", 300.0);

        // --- Tracker parameters ---
        this->declare_parameter("tracker_type", "CSRT");
        this->declare_parameter("max_lost_frames", 10);

        // Retrieve parameters
        target_dist_ = this->get_parameter("target_distance").as_double();

        kp_lin_ = this->get_parameter("kp_lin").as_double();
        ki_lin_ = this->get_parameter("ki_lin").as_double();
        kd_lin_ = this->get_parameter("kd_lin").as_double();

        kp_ang_ = this->get_parameter("kp_ang").as_double();
        ki_ang_ = this->get_parameter("ki_ang").as_double();
        kd_ang_ = this->get_parameter("kd_ang").as_double();

        lin_integral_max_ = this->get_parameter("lin_integral_max").as_double();
        ang_integral_max_ = this->get_parameter("ang_integral_max").as_double();

        max_lin_vel_ = this->get_parameter("max_linear_vel").as_double();
        max_ang_vel_ = this->get_parameter("max_angular_vel").as_double();

        use_color_filter_ = this->get_parameter("use_color_filter").as_bool();
        lower_h_ = this->get_parameter("lower_h").as_int();
        lower_s_ = this->get_parameter("lower_s").as_int();
        lower_v_ = this->get_parameter("lower_v").as_int();
        upper_h_ = this->get_parameter("upper_h").as_int();
        upper_s_ = this->get_parameter("upper_s").as_int();
        upper_v_ = this->get_parameter("upper_v").as_int();
        min_contour_area_ = this->get_parameter("min_contour_area").as_double();

        tracker_type_ = this->get_parameter("tracker_type").as_string();
        max_lost_frames_ = this->get_parameter("max_lost_frames").as_int();

        // PID state
        lin_error_sum_ = 0.0;
        last_lin_error_ = 0.0;
        ang_error_sum_ = 0.0;
        last_ang_error_ = 0.0;
        last_time_ = this->now();

        // Other state
        tracking_initialized_ = false;
        have_prev_bbox_ = false;
        lost_counter_ = 0;
        last_z_ = 0.0;
        last_angle_ = 0.0;

        RCLCPP_INFO(this->get_logger(), "Tracker Node Started with PID control.");

        // Subscriptions
        sub_img_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/camera/image_raw", 10, std::bind(&TrackerNode::image_callback, this, _1));
        sub_info_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
            "/camera/camera_info", 10, std::bind(&TrackerNode::info_callback, this, _1));

        // Publishers
        pub_vel_ = this->create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", 10);
        pub_debug_ = this->create_publisher<sensor_msgs::msg::Image>("/tracker/debug_view", 10);
        pub_pose_ = this->create_publisher<geometry_msgs::msg::Pose2D>("/tracker/target_pose", 10);
        pub_mask_ = this->create_publisher<sensor_msgs::msg::Image>("/tracker/mask", 10);

        // 3D object points (order: TL, TR, BR, BL)
        float s = 0.05f;
        object_points_.push_back(cv::Point3f(-s,  s, 0));   // TL
        object_points_.push_back(cv::Point3f( s,  s, 0));   // TR
        object_points_.push_back(cv::Point3f( s, -s, 0));   // BR
        object_points_.push_back(cv::Point3f(-s, -s, 0));   // BL
    }

private:
    void info_callback(const sensor_msgs::msg::CameraInfo::SharedPtr msg)
    {
        if (camera_matrix_.empty()) {
            camera_matrix_ = cv::Mat(3, 3, CV_64F, (void*)msg->k.data()).clone();
            dist_coeffs_ = cv::Mat(1, 5, CV_64F, (void*)msg->d.data()).clone();
            RCLCPP_INFO(this->get_logger(), "Camera info received.");
        }
    }

    void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        if (camera_matrix_.empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "Waiting for camera info...");
            return;
        }

        // Convert ROS image to OpenCV BGR
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception& e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat frame = cv_ptr->image.clone();
        cv::Mat display = frame.clone();
        cv::Mat gray, binary;

        // --- Step 1: Get binary mask (if using colour filter) ---
        if (use_color_filter_) {
            cv::Mat hsv;
            cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
            cv::Scalar lower(lower_h_, lower_s_, lower_v_);
            cv::Scalar upper(upper_h_, upper_s_, upper_v_);
            cv::inRange(hsv, lower, upper, binary);
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
            cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
            cv::morphologyEx(binary, binary, cv::MORPH_CLOSE, kernel);
            pub_mask_->publish(*cv_bridge::CvImage(msg->header, "mono8", binary).toImageMsg());
        } else {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            cv::GaussianBlur(gray, gray, cv::Size(5,5), 0);
            cv::Canny(gray, binary, 50, 150);
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
            cv::dilate(binary, binary, kernel);
        }

        // --- Step 2: Detection (find largest contour) ---
        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        bool detection_success = false;
        cv::Rect bbox;

        if (!contours.empty()) {
            auto largest = std::max_element(contours.begin(), contours.end(),
                [](const std::vector<cv::Point>& a, const std::vector<cv::Point>& b) {
                    return cv::contourArea(a) < cv::contourArea(b);
                });
            double area = cv::contourArea(*largest);
            if (area > min_contour_area_) {
                detection_success = true;
                bbox = cv::boundingRect(*largest);
                cv::rectangle(display, bbox, cv::Scalar(255,0,0), 2);
            }
        }

        // --- Step 3: Tracker management ---
        if (detection_success) {
            if (tracker_type_ == "CSRT")
                tracker_ = cv::TrackerCSRT::create();
            else if (tracker_type_ == "KCF")
                tracker_ = cv::TrackerKCF::create();
            else {
                RCLCPP_WARN(this->get_logger(), "Unknown tracker type, using CSRT");
                tracker_ = cv::TrackerCSRT::create();
            }
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            tracker_->init(gray, bbox);
            tracking_initialized_ = true;
            have_prev_bbox_ = true;
            prev_tracker_bbox_ = bbox;
            lost_counter_ = 0;
            RCLCPP_DEBUG(this->get_logger(), "Tracker re‑initialised");
        }

        // --- Step 4: Get current bounding box from tracker ---
        cv::Rect tracked_bbox;
        bool tracking_ok = false;
        if (tracking_initialized_) {
            cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
            int rows = gray.rows;
            int cols = gray.cols;

            const int MARGIN = 10;
            if (have_prev_bbox_) {
                if (prev_tracker_bbox_.x < MARGIN ||
                    prev_tracker_bbox_.y < MARGIN ||
                    prev_tracker_bbox_.x + prev_tracker_bbox_.width > cols - MARGIN ||
                    prev_tracker_bbox_.y + prev_tracker_bbox_.height > rows - MARGIN) {
                    tracking_initialized_ = false;
                    have_prev_bbox_ = false;
                    RCLCPP_WARN(this->get_logger(), "Tracker near edge – reinit required");
                }
            }

            if (tracking_initialized_) {
                tracking_ok = tracker_->update(gray, tracked_bbox);
                if (tracking_ok) {
                    tracked_bbox.x = std::max(0, std::min(tracked_bbox.x, cols - 1));
                    tracked_bbox.y = std::max(0, std::min(tracked_bbox.y, rows - 1));
                    tracked_bbox.width = std::min(tracked_bbox.width, cols - tracked_bbox.x);
                    tracked_bbox.height = std::min(tracked_bbox.height, rows - tracked_bbox.y);
                    if (tracked_bbox.width <= 0 || tracked_bbox.height <= 0) {
                        tracking_ok = false;
                        tracking_initialized_ = false;
                        have_prev_bbox_ = false;
                        RCLCPP_WARN(this->get_logger(), "Tracker returned empty box after clamping");
                    } else {
                        prev_tracker_bbox_ = tracked_bbox;
                        have_prev_bbox_ = true;
                        cv::rectangle(display, tracked_bbox, cv::Scalar(0,255,255), 2);
                    }
                } else {
                    tracking_initialized_ = false;
                    have_prev_bbox_ = false;
                    RCLCPP_WARN(this->get_logger(), "Tracker lost");
                }
            }
        }

        // --- Step 5: Determine bounding box for PnP ---
        cv::Rect current_bbox;
        bool use_tracker = tracking_initialized_ && tracking_ok;
        if (use_tracker) {
            current_bbox = tracked_bbox;
        } else if (detection_success) {
            current_bbox = bbox;
        } else {
            current_bbox = cv::Rect();
        }

        // --- Step 6: PnP pose estimation ---
        geometry_msgs::msg::Twist twist;
        bool pnp_success = false;

        if (!current_bbox.empty()) {
            int rows = frame.rows;
            int cols = frame.cols;
            current_bbox.x = std::max(0, std::min(current_bbox.x, cols - 1));
            current_bbox.y = std::max(0, std::min(current_bbox.y, rows - 1));
            current_bbox.width = std::min(current_bbox.width, cols - current_bbox.x);
            current_bbox.height = std::min(current_bbox.height, rows - current_bbox.y);

            if (current_bbox.width > 0 && current_bbox.height > 0) {
                std::vector<cv::Point2f> img_pts(4);
                img_pts[0] = cv::Point2f(current_bbox.tl().x, current_bbox.tl().y);
                img_pts[1] = cv::Point2f(current_bbox.br().x, current_bbox.tl().y);
                img_pts[2] = cv::Point2f(current_bbox.br().x, current_bbox.br().y);
                img_pts[3] = cv::Point2f(current_bbox.tl().x, current_bbox.br().y);

                cv::Mat rvec, tvec;
                bool ok = cv::solvePnP(object_points_, img_pts, camera_matrix_, dist_coeffs_,
                                       rvec, tvec, false, cv::SOLVEPNP_IPPE);
                if (!ok) {
                    ok = cv::solvePnP(object_points_, img_pts, camera_matrix_, dist_coeffs_,
                                      rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
                }

                if (ok) {
                    double z = tvec.at<double>(2);
                    double x = tvec.at<double>(0);
                    if (z > 0.1 && !std::isnan(z) && !std::isnan(x)) {
                        pnp_success = true;
                        double angle = atan2(x, z);

                        // Store last valid pose
                        last_z_ = z;
                        last_angle_ = angle;
                        lost_counter_ = 0;

                        // Compute time delta
                        rclcpp::Time now = this->now();
                        double dt = (now - last_time_).seconds();
                        if (dt > 0.1) dt = 0.1;   // safety cap
                        if (dt < 0.0001) dt = 0.01; // avoid division by zero

                        // --- Linear PID (distance) ---
                        double lin_error = z - target_dist_;   // positive means too far
                        lin_error_sum_ += lin_error * dt;
                        lin_error_sum_ = std::clamp(lin_error_sum_, -lin_integral_max_, lin_integral_max_);
                        double lin_deriv = (lin_error - last_lin_error_) / dt;
                        double lin_output = kp_lin_ * lin_error + ki_lin_ * lin_error_sum_ + kd_lin_ * lin_deriv;
                        last_lin_error_ = lin_error;

                        // --- Angular PID (centering) ---
                        double ang_error = -angle;   // positive angle (object left) → positive error → turn left (positive angular.z)
                        ang_error_sum_ += ang_error * dt;
                        ang_error_sum_ = std::clamp(ang_error_sum_, -ang_integral_max_, ang_integral_max_);
                        double ang_deriv = (ang_error - last_ang_error_) / dt;
                        double ang_output = kp_ang_ * ang_error + ki_ang_ * ang_error_sum_ + kd_ang_ * ang_deriv;
                        last_ang_error_ = ang_error;

                        // Apply outputs (positive linear = forward, positive angular = left turn)
                        double linear_cmd = lin_output;
                        double angular_cmd = ang_output;

                        // Clamp
                        linear_cmd = std::clamp(linear_cmd, -max_lin_vel_, max_lin_vel_);
                        angular_cmd = std::clamp(angular_cmd, -max_ang_vel_, max_ang_vel_);

                        twist.linear.x = linear_cmd;
                        twist.angular.z = angular_cmd;

                        // Publish pose
                        geometry_msgs::msg::Pose2D pose_msg;
                        pose_msg.x = z;
                        pose_msg.y = x;
                        pose_msg.theta = angle;
                        pub_pose_->publish(pose_msg);

                        // Draw axes
                        try {
                            cv::drawFrameAxes(display, camera_matrix_, dist_coeffs_, rvec, tvec, 0.1);
                        } catch (...) { }

                        cv::putText(display, "Z: " + std::to_string(z).substr(0,5) + " m",
                                    cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                                    cv::Scalar(0,255,0), 2);
                    }
                }
            }
        }

        // --- Step 7: Temporal filter (if PnP failed) ---
        if (!pnp_success) {
            lost_counter_++;
            if (lost_counter_ <= max_lost_frames_ && last_z_ > 0.1) {
                // Use last known pose with P control only (simple fallback)
                double error_dist = last_z_ - target_dist_;
                // - = follow, + = avoid
                twist.linear.x = -kp_lin_ * error_dist;                     
                twist.angular.z = -kp_ang_ * last_angle_;
                twist.linear.x = std::clamp(twist.linear.x, -max_lin_vel_, max_lin_vel_);
                twist.angular.z = std::clamp(twist.angular.z, -max_ang_vel_, max_ang_vel_);
                cv::putText(display, "LOST (using last pose)", cv::Point(10,60),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,165,255), 2);
            } else {
                twist.linear.x = 0.0;
                twist.angular.z = 0.0;
                cv::putText(display, "TARGET LOST", cv::Point(10,60),
                            cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0,0,255), 2);
            }
        }

        pub_vel_->publish(twist);
        pub_debug_->publish(*cv_bridge::CvImage(msg->header, "bgr8", display).toImageMsg());

        last_time_ = this->now();
    }

    // Member variables
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_img_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr sub_info_;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr pub_vel_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_debug_;
    rclcpp::Publisher<geometry_msgs::msg::Pose2D>::SharedPtr pub_pose_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_mask_;

    cv::Mat camera_matrix_, dist_coeffs_;
    std::vector<cv::Point3f> object_points_;

    // Control parameters
    double target_dist_;
    double kp_lin_, ki_lin_, kd_lin_;
    double kp_ang_, ki_ang_, kd_ang_;
    double lin_integral_max_, ang_integral_max_;
    double max_lin_vel_, max_ang_vel_;

    // PID state
    double lin_error_sum_, last_lin_error_;
    double ang_error_sum_, last_ang_error_;
    rclcpp::Time last_time_;

    // Detection parameters
    bool use_color_filter_;
    int lower_h_, lower_s_, lower_v_;
    int upper_h_, upper_s_, upper_v_;
    double min_contour_area_;

    // Tracker parameters
    std::string tracker_type_;
    int max_lost_frames_;

    // Tracker and state
    cv::Ptr<cv::Tracker> tracker_;
    bool tracking_initialized_;
    bool have_prev_bbox_;
    cv::Rect prev_tracker_bbox_;
    int lost_counter_;
    double last_z_, last_angle_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<TrackerNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}