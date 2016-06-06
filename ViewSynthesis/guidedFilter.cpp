#include "viewsynth.h"

void guidedFilter(const Mat& src, const Mat& joint, Mat& dest, const int radius, const float eps)
{
	const int BORDER_TYPE = cv::BORDER_REPLICATE;
	Size ksize(2 * radius + 1, 2 * radius + 1);
	Size imsize = src.size();
	const float e = eps;

	Mat sf; src.convertTo(sf, CV_32F, 1.0 / 255);
	Mat jf; joint.convertTo(jf, CV_32F, 1.0 / 255);

	Mat mJoint(imsize, CV_32F);//mean_I
	Mat mSrc(imsize, CV_32F);//mean_p

	boxFilter(jf, mJoint, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//mJoint*K/////////////////////////
	boxFilter(sf, mSrc, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//mSrc*K

	Mat x1(imsize, CV_32F), x2(imsize, CV_32F), x3(imsize, CV_32F);

	multiply(jf, sf, x1);//x1*1
	boxFilter(x1, x3, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//corrI
	multiply(mJoint, mSrc, x1);//;x1*K*K
	x3 -= x1;//x1 div k ->x3*k
	multiply(jf, jf, x1);////////////////////////////////////
	boxFilter(x1, x2, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//x2*K
	multiply(mJoint, mJoint, x1);//x1*K*K

	sf = Mat(x2 - x1) + e;
	divide(x3, sf, x3);//x3->a
	multiply(x3, mJoint, x1);
	x1 -= mSrc;//x1->b
	boxFilter(x3, x2, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//x2*k
	boxFilter(x1, x3, CV_32F, ksize, Point(-1, -1), true, BORDER_TYPE);//x3*k
	multiply(x2, jf, x1);//x1*K
	Mat temp = x1 - x3;//
	temp.convertTo(dest, src.type(), 255);
}