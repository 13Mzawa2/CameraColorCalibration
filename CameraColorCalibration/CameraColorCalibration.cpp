/**************************************************
Gauss-Newton法によるカメラカラーキャリブレーション
Author: Nishizawa  Date: 2016/05/30

ver. 0.1 ColorCheckerのXYZと，それに対応する
カメラ画像のRGB値を入力してカラープロファイルを出力する
但し全てのRGB関連はBGR順

使用ライブラリ：OpenCV 3.1
(線形代数用ライブラリとして使用)
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

//	最小化パラメータと初期値
cv::Vec3d gamma(1.2,1.2,1.2);			//	ガンマ係数 B, G, R
cv::Vec3d Lmax(200.0, 800.0, 500.0);	//	受光感度 B, G, R
//std::vector<cv::Point2d> xy = {			//	受光体のxy色度
//	cv::Point2d(0.1, 0.2),				//	xB, yB
//	cv::Point2d(0.3, 0.5),				//	xG, yG
//	cv::Point2d(0.5, 0.3)				//	xR, yR
//};
cv::Mat V_BGR2XYZ = (cv::Mat_<double>(3, 3) << 	//	ガンマ補正済みのBGRからXYZへの線形変換行列
	-0.3, 0.2, 2.1,
	0.5, 1.1, 1.2,
	1.5, 1.3, -0.8
	);

//	関数プロトタイプ
cv::Mat calcErrorVector(std::vector<cv::Vec3d> BGRs_, std::vector<cv::Vec3d> XYZs_, cv::Mat params_);
double goldenRatioSearch(std::vector<cv::Vec3d> BGRs_, std::vector < cv::Vec3d> XYZs_, cv::Mat params_, cv::Mat d_params_);
void GaussNewtonMethod(std::vector<cv::Vec3d> BGRs_, std::vector<cv::Vec3d> XYZs_, cv::Vec3d &gamma_, cv::Vec3d &Lmax_, cv::Mat &V_);
cv::Mat JacobianMat(std::vector<cv::Vec3d> BGRs_, std::vector<cv::Vec3d> XYZs_, cv::Vec3d gamma_, cv::Vec3d Lmax_, cv::Mat V_);
cv::Vec3d calcError(cv::Vec3d BGR, cv::Vec3d XYZ, cv::Vec3d gamma_, cv::Vec3d Lmax_, cv::Mat V_);
cv::Vec3d calcXYZ(cv::Vec3d BGR, cv::Vec3d gamma_, cv::Vec3d Lmax_, cv::Mat V_);
cv::Vec3d operator*(cv::Mat m, cv::Vec3d &v);


//	黄金分割探索
//	params_new = params_ + alpha * d_params
//	としてゲイン alpha in [0, 1] を探索する
double goldenRatioSearch(std::vector<cv::Vec3d> BGRs_, std::vector < cv::Vec3d> XYZs_, cv::Mat params_, cv::Mat d_params_)
{
	const double phi = (1.0 + sqrt(5.0)) / 2.0;		//	黄金分割に使用
	double lb = 0.0, ub = 1.0;						//	探索範囲の下界，上界
	double alpha1 = (ub - lb) / (phi + 1.0) + lb;	//	初期内分点1 1:phi
	double alpha2 = (ub - lb) / phi + lb;			//	初期内分点2 phi:1
	cv::Mat params_new1 = params_ + alpha1 * d_params_;		//	内分されたパラメータベクトル
	cv::Mat params_new2 = params_ + alpha2 * d_params_;
	//	内分後の誤差関数1
	cv::Mat sigma1 = calcErrorVector(BGRs_, XYZs_, params_new1);
	//	内分後の誤差関数2
	cv::Mat sigma2 = calcErrorVector(BGRs_, XYZs_, params_new2);
	for (int i = 0; i < 10; i++) {
		if (norm(sigma1) < norm(sigma2)) {
			//	上界をalpha2に更新
			ub = alpha2;
			//	alpha2にalpha1を上書き
			alpha2 = alpha1; 
			params_new2 = params_ + alpha2 * d_params_;
			sigma2 = sigma1.clone();
			//	alpha1を[alpha2, ub]の間の黄金分割点に更新
			alpha1 = (ub - lb) / (phi + 1.0) + lb;
			params_new1 = params_ + alpha1 * d_params_;
			sigma1 = calcErrorVector(BGRs_, XYZs_, params_new1);
		}
		else {
			//	下界をalpha1に更新
			lb = alpha1;
			//	alpha1をalpha2に上書き
			alpha1 = alpha2;
			params_new1 = params_ + alpha1 * d_params_;
			sigma1 = sigma2.clone();
			//	alpha2を[lb, alpha1]の間の黄金分割点に更新
			alpha2 = (ub - lb) / phi + lb;
			params_new2 = params_ + alpha2 * d_params_;
			sigma2 = calcErrorVector(BGRs_, XYZs_, params_new2);
		}
	}
	return (lb + ub) / 2.0;
}
//	誤差ベクトルをパラメータベクトルから直接求める
cv::Mat calcErrorVector(std::vector<cv::Vec3d> BGRs_, std::vector<cv::Vec3d> XYZs_, cv::Mat params_)
{
	//	パラメータベクトルを各要素に変換
	cv::Vec3d gamma_(params_.at<double>(0), params_.at<double>(2), params_.at<double>(4));
	cv::Vec3d Lmax_(params_.at<double>(1), params_.at<double>(3), params_.at<double>(5));
	cv::Mat V_(3, 3, CV_64FC1);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			V_.at<double>(i, j) = params_.at<double>(6 + i * 3 + j);
		}
	}
	cv::Mat sigma(XYZs_.size() * 3, 1, CV_64FC1);
	//	誤差ベクトル sigma(params) の導出
	for (int i = 0; i < XYZs_.size(); i++) {
		cv::Vec3d e3 = calcError(BGRs_[i], XYZs_[i], gamma_, Lmax_, V_);
		for (int j = 0; j < 3; j++) {
			sigma.at<double>(i * 3 + j) = e3[j];
		}
	}
	return sigma;
}

//	Gauss-Newton法
void GaussNewtonMethod(std::vector<cv::Vec3d> BGRs_, std::vector<cv::Vec3d> XYZs_, cv::Vec3d &gamma_, cv::Vec3d &Lmax_, cv::Mat &V_)
{
	cv::Mat sigma(XYZs_.size() * 3, 1, CV_64FC1);		//	二乗誤差関数の要素毎のベクトル sigma(params)
	cv::Mat params(15, 1, CV_64FC1);					//	15個のパラメータベクトル params の初期値
	for (int i = 0; i < 3; i++) {
		params.at<double>(i*2 + 0) = gamma_[i];
		params.at<double>(i*2 + 1) = Lmax_[i];
	}
	for (int i = 0; i < 3; i++) {		//	V_BGR2XYZについては最後に並べる
		for (int j = 0; j < 3; j++) {
			params.at<double>(6 + i * 3 + j) = V_.at<double>(i, j);
		}
	}

	//	更新値の評価関数errorが閾値以下になったら抜け出す
	double error = 100.0;
	for (int count = 0; error > 1.0e-6 && count < 10000; count++) {
		//	更新後のパラメータ
		cv::Vec3d gamma_new(params.at<double>(0), params.at<double>(2), params.at<double>(4));
		cv::Vec3d Lmax_new(params.at<double>(1), params.at<double>(3), params.at<double>(5));
		cv::Mat V_new(3, 3, CV_64FC1);
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 3; j++) {
				V_new.at<double>(i, j) = params.at<double>(6 + i * 3 + j);
			}
		}
		//	ヤコビ行列 J(params) の導出
		cv::Mat J = JacobianMat(BGRs_, XYZs_, gamma_new, Lmax_new, V_new);
		//	誤差ベクトル sigma(params) の導出
		for (int i = 0; i < XYZs_.size(); i++) {
			cv::Vec3d e3 = calcError(BGRs_[i], XYZs_[i], gamma_new, Lmax_new, V_new);
			for (int j = 0; j < 3; j++) {
				sigma.at<double>(i * 3 + j) = e3[j];
			}
		}
		//	増分 d_params の計算
		cv::Mat d_params = -(J.t()*J).inv()*J.t()*sigma;
		//cout << J.col(5) << endl;
		//cout << J.t().row(5).t() << endl;
		//cout << d_params << endl;
		//	パラメータの更新
		double alpha = goldenRatioSearch(BGRs_, XYZs_, params, d_params);
		params = params + alpha*d_params;
		//	増分の大きさの比較
		error = cv::norm(d_params);
		cout << "count\t" << count << ":\terror = " << error << "\tgain = " << alpha << endl;
	}

	//	結果を参照渡し
	gamma_ = cv::Vec3d(params.at<double>(0 + 0), params.at<double>(0 + 4), params.at<double>(0 + 8));
	Lmax_ = cv::Vec3d(params.at<double>(1 + 0), params.at<double>(1 + 4), params.at<double>(1 + 8));
	cv::Mat VV(3, 3, CV_64FC1);
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 3; j++) {
			VV.at<double>(i, j) = params.at<double>(6 + i * 3 + j);
		}
	}
	V_ = VV.clone();
} 

//	sigma(params)のparamsに対するヤコビ行列Jの導出
//	XYZは全て別要素として扱い，XYZXYZ...と格納されている
cv::Mat JacobianMat(std::vector<cv::Vec3d> BGRs_, std::vector<cv::Vec3d> XYZs_, cv::Vec3d gamma_, cv::Vec3d Lmax_, cv::Mat V_)
{
	const double dx = 1.0e-6;			//	パラメータの増分
	std::vector<double> params;			//	パラメータは全部で12次元
	for (int i = 0; i < 3; i++) {		//	gamma, LmaxについてはB関係全て，G関係全て，R関係全て，の順に並ぶ
		params.push_back(gamma_[i]);
		params.push_back(Lmax_[i]);
	}
	for (int i = 0; i < 3; i++) {		//	V_BGR2XYZについては最後に並べる
		for (int j = 0; j < 3; j++) {
			params.push_back(V_.at<double>(i, j));
		}
	}
	//	ヤコビ行列
	cv::Mat Jt(params.size(), XYZs_.size()*3, CV_64FC1);
	for (int i = 0; i < Jt.rows; i++) {			//	パラメータの変数
		//	パラメータの微小変化
		std::vector<double> params_d;			//	第i成分のパラメータをdxだけ増加したもの
		for (int k = 0; k < params.size(); k++) {
			if (k == i)
				params_d.push_back(params[k] + dx);
			else
				params_d.push_back(params[k]);
		}
		//	第i成分の微小変化後のパラメータ
		cv::Vec3d gamma_d(params_d[0], params_d[2], params_d[4]);
		cv::Vec3d Lmax_d(params_d[1], params_d[3], params_d[5]);
		cv::Mat V_d = (cv::Mat_<double>(3, 3)
			<< params_d[6], params_d[7], params_d[8],
			params_d[9], params_d[10], params_d[11],
			params_d[12], params_d[13], params_d[14]);
		for (int j = 0; j < Jt.cols/3; j++) {	//	色数の変数
			cv::Vec3d sigma = calcError(BGRs_[j], XYZs_[j], gamma_, Lmax_, V_);			//	sigma
			cv::Vec3d sigma_d = calcError(BGRs_[j], XYZs_[j], gamma_d, Lmax_d, V_d);		//	sigma + dsi
			for (int k = 0; k < 3; k++) {		//	XYZチャンネルの変数
				Jt.at<double>(i, j*3+k) = (sigma_d[k] - sigma[k]) / dx;			//	dsi/dxi
			}
		}
	}
	return Jt.t();		//	↑はJ.t()を求めてたのでもう一度転置
}

//	色の推定値と測定値との二乗誤差の平方根 sigma(params) (X, Y, Zの順)
cv::Vec3d calcError(cv::Vec3d BGR, cv::Vec3d XYZ, cv::Vec3d gamma_, cv::Vec3d Lmax_, cv::Mat V_)
{
	cv::Vec3d estXYZ = calcXYZ(BGR, gamma_, Lmax_, V_);
	cv::Vec3d sigma;
	for (int i = 0; i < 3; i++) {
		sigma[i] = sqrt((estXYZ[i] - XYZ[i])*(estXYZ[i] - XYZ[i]));
	}
	return sigma;
}

//	RGB階調値を入力してパラメータを基にXYZを出力する関数
cv::Vec3d calcXYZ(cv::Vec3d BGR, cv::Vec3d gamma_, cv::Vec3d Lmax_, cv::Mat V_)
{
	//	1. BGR階調値→BGR受光輝度
	cv::Vec3d YBGR;
	for (int i = 0; i < 3; i++) {
		YBGR[i] = Lmax_[i] * pow(BGR[i] / 255.0, gamma_[i]);
	}
	//	2. BGR→XYZ変換
	cv::Vec3d XYZ;
	XYZ = V_ * YBGR;
	return XYZ;
}

//	Mat * Vec3d の乗算オペレータ
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
	//	入力データ格納用
	std::vector<cv::Vec3d> BGRs;
	std::vector<cv::Vec3d> XYZs;
	//	ファイル読み込み
	ifstream csvFile("calibdata.csv");
	if (!csvFile) {
		cout << "Error: Input data file not found" << endl;
		return -1;
	}
	string str;
	getline(csvFile, str);		//	1行目は読み飛ばす
	cout << "X\tY\tZ\tB\tG\tR" << endl;
	//	1行だけ取り出す
	while (getline(csvFile, str)) {
		string token;
		istringstream stream(str);
		vector<double> data;
		//	1行のうち文字列とカンマを分割
		while (getline(stream, token, ',')) {
			double temp = stod(token);		//	数値に変換
			data.push_back(temp);
		}
		//	データを表示
		cout << data[0] << "\t" << data[1] << "\t" << data[2] << "\t" << data[3] << "\t" << data[4] << "\t" << data[5] << endl;
		XYZs.push_back(Vec3d(data[0], data[1], data[2]));
		BGRs.push_back(Vec3d(data[3], data[4], data[5]));
	}
	cout << "\n--------------------------------------\n" << endl;
	cout << "\nXYZ = \n" << XYZs[0] << "\n" << XYZs[1] << "\n" << XYZs[2]
		<< "\nEstimated XYZ = \n" << calcXYZ(BGRs[0], gamma, Lmax, V_BGR2XYZ)
		<< "\n" << calcXYZ(BGRs[1], gamma, Lmax, V_BGR2XYZ)
		<< "\n" << calcXYZ(BGRs[2], gamma, Lmax, V_BGR2XYZ) << endl;
	system("PAUSE");
	
	GaussNewtonMethod(BGRs, XYZs, gamma, Lmax, V_BGR2XYZ);

	cout << "\n\n----------------------"
		<< "\n\tResult"
		<< "\n----------------------"
		<< "\ngamma_BGR = " << gamma
		<< "\nLmax_BGR = " << Lmax
		<< "\nV_BGR2XYZ = \n" << V_BGR2XYZ
		<< endl;

	cout << "\nXYZ = \n" << XYZs[0] << "\n" << XYZs[1] << "\n" << XYZs[2]
		<< "\nEstimated XYZ = \n" << calcXYZ(BGRs[0], gamma, Lmax, V_BGR2XYZ)
		<< "\n" << calcXYZ(BGRs[1], gamma, Lmax, V_BGR2XYZ)
		<< "\n" << calcXYZ(BGRs[2], gamma, Lmax, V_BGR2XYZ) << endl;
	system("PAUSE");
	return 0;
}
