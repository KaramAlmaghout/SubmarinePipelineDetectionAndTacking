/**
* alpha filter - Kalman filter implementation for a one-state model (position) 
* paper 
*
* @author: Karam Almaghout
* 
*/

#include "alphaFilter.hpp"

alphaFilter::alphaFilter()
{

}

alphaFilter::~alphaFilter()
{
  
}

void alphaFilter::Init(const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& H, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R)
{
  A_ = A;
  B_ = B;
  H_ = H;
  Q_ = Q;
  R_ = R;
    
  sqrt_Q_ = Q_.sqrt();
  sqrt_R_ = R_.sqrt();
        
  int n_states = A_.cols();
  int n_outputs = H_.rows();
  
  // Apply intial states
  x_ =  MatrixXd::Zero(n_states,1);

  // // Inital values:
  P_m_ = MatrixXd::Identity(n_states, n_states);
  x_m_ = MatrixXd::Zero(n_states, 1);
}

void alphaFilter::InitState(const Eigen::MatrixXd& x0)
{
  x_ = x0;
  x_m_ = x0;
}

void alphaFilter::InitStateCovariance(const Eigen::MatrixXd& P0)
{
  P_m_ = P0;
}
  
void alphaFilter::update(const Eigen::MatrixXd& delta_x)
{
  v_ = MatrixXd::Random(A_.rows(),A_.rows());
  w_ = MatrixXd::Random(H_.rows(),A_.rows());
  v_ = sqrt_Q_ * v_;
  w_ = sqrt_R_ * w_;
  x_ = A_ * x_ + B_ * delta_x + v_;
  z_ = H_ * x_ + w_;
  
  // Prior update:
  x_p_ = A_ * x_m_ + B_ * delta_x;
  P_p_ = A_ * P_m_ * A_.transpose() + Q_;
  
  // Measurement update:
  MatrixXd K = P_p_ * H_.transpose() * (H_ * P_p_ * H_.transpose().inverse() + R_);
  x_m_ = x_p_ + K * (z_ - H_ * x_p_);
  P_m_ = P_p_ - K * H_ * P_p_;
  
  // Estimated output is the projection of etimated states to the output function
  z_m_ = H_ * x_m_;
}

void alphaFilter::update(const Eigen::MatrixXd& z, const Eigen::MatrixXd& delta_x)
{
  // Prior update:
  x_p_ = A_ * x_m_ + B_ * delta_x;
  P_p_ = A_ * P_m_ * A_.transpose() + Q_;
  
  // Measurement update:
  MatrixXd K = P_p_ * H_.transpose() * (H_ * P_p_ * H_.transpose() + R_).inverse();
  x_m_ = x_p_ + K * (z - H_ * x_p_);
  P_m_ = P_p_ - K * H_ * P_p_;
  
  // Estimated output is the projection of etimated states to the output function
  z_m_ = H_ * x_m_;
}

Eigen::MatrixXd* alphaFilter::GetCurrentState()
{
    return & x_;
}

Eigen::MatrixXd* alphaFilter::GetCurrentOutput()
{
    return &z_;
}

Eigen::MatrixXd* alphaFilter::GetCurrentEstimatedState()
{
    return &x_m_;
}

Eigen::MatrixXd* alphaFilter::GetCurrentEstimatedOutput()
{
    return &z_m_;
}
