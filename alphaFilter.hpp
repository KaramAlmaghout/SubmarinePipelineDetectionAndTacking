/**
* alpha filter - Kalman filter implementation for a one-state model (position) 
* paper 
*
* @author: Karam Almaghout
* 
*/

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

using namespace Eigen;

#pragma once

class alphaFilter {
public:
  /*!
   * \brief Constructor, nothing happens here.
   */
  alphaFilter();
  /*!
   * \brief Destructor, nothing happens here.
   */
  ~alphaFilter();
  
  /*!
   * @brief Define the system.
   * @param A System matrix
   * @param B Input matrix
   * @param H Output matrix
   * @param Q Process noise covariance
   * @param R Measurement noise covariance
   */
  void Init (const Eigen::MatrixXd& A, const Eigen::MatrixXd& B, const Eigen::MatrixXd& H, const Eigen::MatrixXd& Q, const Eigen::MatrixXd& R);
  
  /*!
   * @brief Initialize the system states.
   * Must be called after InitSystem.
   * If not, called, system states are initialized to zero.
   * @param x0 Inital value for the system state
   */
  void InitState(const Eigen::MatrixXd& x0);
  
  /*!
   * @brief Initialize the state covariance.
   * Must be called after InitSystem.
   * If not called, covariance state is Initialized to an identity matrix.
   * @param P0 Inital value for the state covariance
   */
  void InitStateCovariance(const Eigen::MatrixXd& P0);
  
  /*!
   * @param delta_x The descritized veloity (x(k) - x(k-1))
   */
  void update(const Eigen::MatrixXd& delta_x);
  
  /*!
   * @param z The values of the output from camera detection
   * @param delta_x The descritized veloity (x(k) - x(k-1))
   */
  void update(const Eigen::MatrixXd& z, const Eigen::MatrixXd& delta_x);
 
 /*!
  * @brief Get current simulated true state.
  * @return Current simulated state $x_k$
  */
  Eigen::MatrixXd* GetCurrentState();
  
   /*!
  * @brief Get current simulated true output.
  * This is analogous to the measurements.
  * @return Current simulated output $z_k$
  */
  Eigen::MatrixXd* GetCurrentOutput();
  
 /*!
  * @brief Get current estimated state.
  * @return Current estimated state $\hat{x}_k$
  */
  Eigen::MatrixXd* GetCurrentEstimatedState();
  
   /*!
  * @brief Get current estimated output.
  * This is the filtered measurements, with less noise.
  * @return Current estimated output $\hat{z}_k$
  */
  Eigen::MatrixXd* GetCurrentEstimatedOutput();
  
private:
  
  MatrixXd A_;      ///< System matrix
  MatrixXd B_;      ///< Input matrix
  MatrixXd H_;      ///< Output matrix
  MatrixXd Q_;      ///< Process noise covariance
  MatrixXd R_;      ///< Measurement noise covariance
  MatrixXd v_;   ///< Gaussian process noise
  MatrixXd w_;   ///< Gaussian measurement noise
  
  MatrixXd sqrt_Q_; ///< Process noise stdev
  MatrixXd sqrt_R_; ///< Measurement noise stdev
  
  MatrixXd x_;   ///< State vector
  MatrixXd z_;   ///< Output matrix
 
  MatrixXd x_m_; ///< State vector after measurement update
  MatrixXd x_p_; ///< State vector after a priori update
  
  MatrixXd P_p_;    ///< State covariance after a priori update
  MatrixXd P_m_;    ///< State covariance after measurement update
  
  MatrixXd z_m_; ///< Estimated output
};