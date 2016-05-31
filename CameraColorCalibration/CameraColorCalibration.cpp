/**************************************************
Gauss-Newton�@�ɂ��J�����J���[�L�����u���[�V����
Author: Nishizawa  Date: 2016/05/30

ver. 0.1 ColorChecker��XYZ�ƁC����ɑΉ�����
�J�����摜��RGB�l����͂��ăJ���[�v���t�@�C�����o�͂���
�A���S�Ă�RGB�֘A��BGR��

�g�p���C�u�����FOpenCV 3.1
(���`�㐔�p���C�u�����Ƃ��Ďg�p)
**************************************************/

#include <opencv2\opencv.hpp>
#include <Windows.h>
#include <string>	//	stringstream
#include <sstream>	//	ifstream
#include <fstream>

#pragma region OPENCV3_LIBRARY_LINKER
#ifdef _DEBUG
#define CV_EXT "d.lib"
#else
#define CV_EXT ".lib"
#endif
#define CV_VER  CVAUX_STR(CV_MAJOR_VERSION) CVAUX_STR(CV_MINOR_VERSION) CVAUX_STR(CV_SUBMINOR_VERSION)
#pragma comment(lib, "opencv_world" CV_VER CV_EXT)
#pragma endregion

using namespace cv;
using namespace std;

//	�ŏ����p�����[�^�Ə����l
cv::Vec3d gamma(1.0,1.0,1.0);			//	�K���}�W�� B, G, R
cv::Vec3d Lmax(600.0, 1200.0, 800.0);	//	������x B, G, R
std::vector<cv::Point2d> xy = {			//	����̂�xy�F�x
	cv::Point2d(0.1, 0.2),				//	xB, yB
	cv::Point2d(0.3, 0.5),				//	xG, yG
	cv::Point2d(0.5, 0.3)				//	xR, yR
};


//	�֐��v���g�^�C�v
void GaussNewtonMethod(std::vector<cv::Vec3d> BGRs_, std::vector<cv::Vec3d> XYZs_, cv::Vec3d &gamma_, cv::Vec3d &Lmax_, std::vector<cv::Point2d> &xy_);
cv::Mat JacobianMat(std::vector<cv::Vec3d> BGRs_, std::vector<cv::Vec3d> XYZs_, cv::Vec3d gamma_, cv::Vec3d Lmax_, std::vector<cv::Point2d> xy_);
cv::Vec3d calcError(cv::Vec3d BGR, cv::Vec3d XYZ, cv::Vec3d gamma_, cv::Vec3d Lmax_, std::vector<cv::Point2d> xy_);
cv::Vec3d calcXYZ(cv::Vec3d BGR, cv::Vec3d gamma_, cv::Vec3d Lmax_, std::vector<cv::Point2d> xy_);
cv::Vec3d operator*(cv::Mat m, cv::Vec3d &v);


//	Gauss-Newton�@
void GaussNewtonMethod(std::vector<cv::Vec3d> BGRs_, std::vector<cv::Vec3d> XYZs_, cv::Vec3d &gamma_, cv::Vec3d &Lmax_, std::vector<cv::Point2d> &xy_)
{
	cv::Mat sigma(XYZs_.size() * 3, 1, CV_64FC1);		//	���덷�֐��̗v�f���̃x�N�g�� sigma(params)
	cv::Mat params(12, 1, CV_64FC1);					//	12�̃p�����[�^�x�N�g�� params �̏����l
	for (int i = 0; i < 3; i++) {
		params.at<double>(i*4 + 0) = gamma_[i];
		params.at<double>(i*4 + 1) = Lmax_[i];
		params.at<double>(i*4 + 2) = xy_[i].x;
		params.at<double>(i*4 + 3) = xy_[i].y;
	}
	//	�X�V�l�̕]���֐�error��臒l�ȉ��ɂȂ����甲���o��
	double error = 100.0;
	for (int count = 0; error > 1.0e-5 && count < 1000; count++) {
		//	�X�V��̃p�����[�^
		cv::Vec3d gamma_new(params.at<double>(0 + 0), params.at<double>(0 + 4), params.at<double>(0 + 8));
		cv::Vec3d Lmax_new(params.at<double>(1 + 0), params.at<double>(1 + 4), params.at<double>(1 + 8));
		std::vector<cv::Point2d> xy_new = {
			cv::Point2d(params.at<double>(2 + 0), params.at<double>(3 + 0)),
			cv::Point2d(params.at<double>(2 + 4), params.at<double>(3 + 4)),
			cv::Point2d(params.at<double>(2 + 8), params.at<double>(3 + 8))
		};
		//	���R�r�s�� J(params) �̓��o
		cv::Mat J = JacobianMat(BGRs_, XYZs_, gamma_new, Lmax_new, xy_new);
		//	�덷�x�N�g�� sigma(params) �̓��o
		for (int i = 0; i < XYZs_.size(); i++) {
			cv::Vec3d e3 = calcError(BGRs_[i], XYZs_[i], gamma_new, Lmax_new, xy_new);
			for (int j = 0; j < 3; j++) {
				sigma.at<double>(i * 3 + j) = e3[j];
			}
		}
		//	���� d_params �̌v�Z
		cv::Mat d_params = -(J.t()*J).inv()*J.t()*sigma;
		//cout << J.col(5) << endl;
		//cout << J.t().row(5).t() << endl;
		//cout << d_params << endl;
		//	�����̑傫���̔�r
		error = cv::norm(d_params);
		cout << "error = " << error << endl;
		//	�p�����[�^�̍X�V
		params = params + d_params;
	}
	//	���ʂ��Q�Ɠn��
	gamma_ = cv::Vec3d(params.at<double>(0 + 0), params.at<double>(0 + 4), params.at<double>(0 + 8));
	Lmax_ = cv::Vec3d(params.at<double>(1 + 0), params.at<double>(1 + 4), params.at<double>(1 + 8));
	xy_[0] = cv::Point2d(params.at<double>(2 + 0), params.at<double>(3 + 0));
	xy_[1] = cv::Point2d(params.at<double>(2 + 4), params.at<double>(3 + 4));
	xy_[2] = cv::Point2d(params.at<double>(2 + 8), params.at<double>(3 + 8));
}

//	sigma(params)��params�ɑ΂��郄�R�r�s��J�̓��o
//	XYZ�͑S�ĕʗv�f�Ƃ��Ĉ����CXYZXYZ...�Ɗi�[����Ă���
cv::Mat JacobianMat(std::vector<cv::Vec3d> BGRs_, std::vector<cv::Vec3d> XYZs_, cv::Vec3d gamma_, cv::Vec3d Lmax_, std::vector<cv::Point2d> xy_)
{
	const double dx = 1.0e-5;			//	�p�����[�^�̑���
	std::vector<double> params;			//	�p�����[�^�͑S����12����
	for (int i = 0; i < 3; i++) {		//	B�֌W�S�āCG�֌W�S�āCR�֌W�S�āC�̏��ɕ���
		params.push_back(gamma_[i]);
		params.push_back(Lmax_[i]);
		params.push_back(xy_[i].x);
		params.push_back(xy_[i].y);
	}
	//	���R�r�s��
	cv::Mat Jt(params.size(), XYZs_.size()*3, CV_64FC1);
	for (int i = 0; i < Jt.rows; i++) {			//	�p�����[�^�̕ϐ�
		//	�p�����[�^�̔����ω�
		std::vector<double> params_d;			//	��i�����̃p�����[�^��dx����������������
		for (int k = 0; k < params.size(); k++) {
			if (k == i)
				params_d.push_back(params[k] + dx);
			else
				params_d.push_back(params[k]);
		}
		//	��i�����̔����ω���̃p�����[�^
		cv::Vec3d gamma_d(params_d[0], params_d[0 + 4], params_d[0 + 8]);
		cv::Vec3d Lmax_d(params_d[1], params_d[1 + 4], params_d[1 + 8]);
		std::vector<cv::Point2d> xy_d = {
			cv::Point2d(params_d[2], params_d[3]),
			cv::Point2d(params_d[2 + 4], params_d[3 + 4]),
			cv::Point2d(params_d[2 + 8], params_d[3 + 8])
		};
		for (int j = 0; j < Jt.cols/3; j++) {	//	�F���̕ϐ�
			cv::Vec3d sigma = calcError(BGRs_[j], XYZs_[j], gamma_, Lmax_, xy_);			//	sigma
			cv::Vec3d sigma_d = calcError(BGRs_[j], XYZs_[j], gamma_d, Lmax_d, xy_d);		//	sigma + dsi
			for (int k = 0; k < 3; k++) {		//	XYZ�`�����l���̕ϐ�
				Jt.at<double>(i, j*3+k) = (sigma_d[k] - sigma[k]) / dx;			//	dsi/dxi
			}
		}
	}
	return Jt.t();		//	����J.t()�����߂Ă��̂ł�����x�]�u
}

//	�F�̐���l�Ƒ���l�Ƃ̓��덷�̕����� sigma(params) (X, Y, Z�̏�)
cv::Vec3d calcError(cv::Vec3d BGR, cv::Vec3d XYZ, cv::Vec3d gamma_, cv::Vec3d Lmax_, std::vector<cv::Point2d> xy_)
{
	cv::Vec3d estXYZ = calcXYZ(BGR, gamma_, Lmax_, xy_);
	cv::Vec3d sigma;
	for (int i = 0; i < 3; i++) {
		sigma[i] = sqrt((estXYZ[i] - XYZ[i])*(estXYZ[i] - XYZ[i]));
	}
	return sigma;
}

//	RGB�K���l����͂��ăp�����[�^�����XYZ���o�͂���֐�
cv::Vec3d calcXYZ(cv::Vec3d BGR, cv::Vec3d gamma_, cv::Vec3d Lmax_, std::vector<cv::Point2d> xy_)
{
	//	1. BGR�K���l��BGR����P�x
	cv::Vec3d YBGR;
	for (int i = 0; i < 3; i++) {
		YBGR[i] = Lmax_[i] * pow(BGR[i] / 255.0, gamma_[i]);
	}
	//	2. �F�ʕ�����
	cv::Vec3d XYZ;
	cv::Mat V = (cv::Mat_<double>(3, 3) <<
		xy_[0].x / xy_[0].y, xy_[1].x / xy_[1].y, xy_[2].x / xy_[2].y,
		1.0, 1.0, 1.0,
		(1.0 - xy_[0].x - xy_[0].y) / xy_[0].y, (1.0 - xy_[1].x - xy_[1].y) / xy_[1].y, (1.0 - xy_[2].x - xy_[2].y) / xy_[2].y);
	XYZ = V * YBGR;
	return XYZ;
}

//	Mat * Vec3d �̏�Z�I�y���[�^
cv::Vec3d operator*(cv::Mat m, cv::Vec3d &v)
{
	cv::Mat y = m * (cv::Mat)v;
	return (cv::Vec3d)y;
	//cv::Mat vm = (cv::Mat_<double>(3, 1) << v[0], v[1], v[2]);
	//cv::Mat y = m * vm;
	//return cv::Vec3d(y.at<double>(0, 0), y.at<double>(1, 0), y.at<double>(2, 0));
}

int main(void)
{
	//	���̓f�[�^�i�[�p
	std::vector<cv::Vec3d> BGRs;
	std::vector<cv::Vec3d> XYZs;
	//	�t�@�C���ǂݍ���
	ifstream csvFile("calibdata.csv");
	if (!csvFile) {
		cout << "Error: Input data file not found" << endl;
		return -1;
	}
	string str;
	getline(csvFile, str);		//	1�s�ڂ͓ǂݔ�΂�
	cout << "X\tY\tZ\tB\tG\tR" << endl;
	//	1�s�������o��
	while (getline(csvFile, str)) {
		string token;
		istringstream stream(str);
		vector<double> data;
		//	1�s�̂���������ƃJ���}�𕪊�
		while (getline(stream, token, ',')) {
			double temp = stod(token);		//	���l�ɕϊ�
			data.push_back(temp);
		}
		//	�f�[�^��\��
		cout << data[0] << "\t" << data[1] << "\t" << data[2] << "\t" << data[3] << "\t" << data[4] << "\t" << data[5] << endl;
		XYZs.push_back(Vec3d(data[0], data[1], data[2]));
		BGRs.push_back(Vec3d(data[3], data[4], data[5]));
	}

	cout << "XYZ = " << XYZs[0] << XYZs[1] << XYZs[2] 
		<< "\nEstimated XYZ = " << calcXYZ(BGRs[0], gamma, Lmax, xy) << calcXYZ(BGRs[1], gamma, Lmax, xy) << calcXYZ(BGRs[2], gamma, Lmax, xy) << endl;
	
	GaussNewtonMethod(BGRs, XYZs, gamma, Lmax, xy);

	cout << "\n\n----------------------"
		<< "\n\tResult"
		<< "\n----------------------"
		<< "\ngamma_BGR = " << gamma
		<< "\nLmax_BGR = " << Lmax
		<< "\nxy_B = " << xy[0]
		<< "\nxy_G = " << xy[1]
		<< "\nxy_R = " << xy[2]
		<< endl;

	cout << "XYZ = " << XYZs[0] << XYZs[1] << XYZs[2]
		<< "\nEstimated XYZ = " << calcXYZ(BGRs[0], gamma, Lmax, xy) << calcXYZ(BGRs[1], gamma, Lmax, xy) << calcXYZ(BGRs[2], gamma, Lmax, xy) << endl;

	system("PAUSE");
	return 0;
}
