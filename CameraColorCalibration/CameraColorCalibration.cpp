/**************************************************
Gauss-Newton�@�ɂ��J�����J���[�L�����u���[�V����
Author: Nishizawa  Date: 2016/05/30

ver. 0.1 ColorChecker��XYZ�ƁC����ɑΉ�����
�J�����摜��RGB�l����͂��ăJ���[�v���t�@�C�����o�͂���
�A���S�Ă�RGB�֘A��BGR��
�g�p���C�u�����FOpenCV 3.1
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
cv::Vec3d gamma(2.2,2.2,2.2);			//	�K���}�W�� B, G, R
cv::Vec3d Lmax(200.0, 400.0, 300.0);	//	������x B, G, R
std::vector<cv::Point2d> xy = {				//	����̂�xy�F�x
	cv::Point2d(0.2, 0.2),				//	xB, yB
	cv::Point2d(0.25, 0.5),				//	xG, yG
	cv::Point2d(0.5, 0.3)				//	xR, yR
};


//	�֐��v���g�^�C�v
cv::Mat JacobianMat(std::vector<cv::Vec3d> BGRs_, std::vector<cv::Vec3d> XYZs_, cv::Vec3d gamma_, cv::Vec3d Lmax_, std::vector<cv::Point2d> xy_);
cv::Vec3d calcError(cv::Vec3d BGR, cv::Vec3d XYZ, cv::Vec3d gamma_, cv::Vec3d Lmax_, std::vector<cv::Point2d> xy_);
cv::Vec3d calcXYZ(cv::Vec3d BGR, cv::Vec3d gamma_, cv::Vec3d Lmax_, std::vector<cv::Point2d> xy_);
cv::Vec3d operator*(cv::Mat m, cv::Vec3d &v);

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
	cv::Mat J(params.size(), XYZs_.size()*3, CV_64FC1);
	for (int i = 0; i < J.rows; i++) {			//	�p�����[�^�̕ϐ�
		//	�p�����[�^�̔����ω�
		std::vector<double> params_d;			//	��i�����̃p�����[�^��dx����������������
		for (int k = 0; k < params.size(); k++) {
			if (k == i)
				params_d.push_back(params[k] + dx);
			else
				params_d.push_back(params[k]);
		}
		//	��i�����̔����ω���̃p�����[�^
		cv::Vec3d gamma_d(params_d[0], params_d[0+4], params_d[0+8]);
		cv::Vec3d Lmax_d(params_d[1], params_d[1+4], params_d[1+8]);
		std::vector<cv::Point2d> xy_d = {
			cv::Point2d(params_d[2], params_d[3]),
			cv::Point2d(params_d[2+4], params_d[3+4]),
			cv::Point2d(params_d[2+8], params_d[3+8])
		};
		for (int j = 0; j < J.cols/3; j++) {	//	�F���̕ϐ�
			cv::Vec3d sigma = calcError(BGRs_[i], XYZs_[i], gamma_, Lmax_, xy_);			//	sigma
			cv::Vec3d sigma_d = calcError(BGRs_[i], XYZs_[i], gamma_d, Lmax_d, xy_d);		//	sigma + ds
			for (int k = 0; k < 3; k++) {		//	XYZ�`�����l���̕ϐ�
				J.at<double>(i, j*3+k) = (sigma_d[k] - sigma[k]) / dx;
			}
		}
		params_d.clear();		//	�p�����[�^�x�N�g���̊J��
	}
	return J;
}

//	�F�̐���l�Ƒ���l�Ƃ̓��덷�̕����� sigma(params) (X, Y, Z�̏�)
cv::Vec3d calcError(cv::Vec3d BGR, cv::Vec3d XYZ, cv::Vec3d gamma_, cv::Vec3d Lmax_, std::vector<cv::Point2d> xy_)
{
	cv::Vec3d estXYZ = calcXYZ(BGR, gamma_, Lmax_, xy_);
	cv::Vec3d sigma;
	for (int i = 0; i < 3; i++) {
		sigma[i] = abs(estXYZ[i] - XYZ[i]);
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
	Mat J = JacobianMat(BGRs, XYZs, gamma, Lmax, xy);
	cout << J << endl;

	system("PAUSE");
	return 0;
}
