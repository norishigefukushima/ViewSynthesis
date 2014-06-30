#include "stdafx.h"

double YPSNR(const Mat& src1, const Mat& src2)
{
	Mat g1,g2;
	cvtColor(src1,g1,COLOR_BGR2GRAY);
	cvtColor(src2,g2,COLOR_BGR2GRAY);
	return PSNR(g1,g2);
}

double calcBadPixel(const Mat& src, const Mat& ref, int threshold)
{
	Mat g1,g2;
	if(src.channels()==3)
	{
		cvtColor(src,g1,CV_BGR2GRAY);
		cvtColor(ref,g2,CV_BGR2GRAY);
	}
	else
	{
		g1=src;
		g2=ref;
	}
	Mat temp;
	absdiff(g1,g2,temp);
	Mat mask;
	compare(temp,threshold,mask,CMP_GE);
	return 100.0*countNonZero(mask)/src.size().area();
}

void warpShiftSubpix(InputArray src_, OutputArray dest_, double shiftx, double shifty, const int inter_method)
{
	Mat src = src_.getMat();
	dest_.create( src.size(), src.type() );
	Mat dest = dest_.getMat();

	Mat aff = Mat::zeros(2,3,CV_64F);

	aff.at<double>(0,0)=1.0;
	aff.at<double>(0,1)=0.0;
	aff.at<double>(0,2)=shiftx;

	aff.at<double>(1,0)=0.0;
	aff.at<double>(1,1)=1.0;
	aff.at<double>(1,2)=shifty;

	warpAffine(src,dest,aff,src.size(), inter_method,0);
}

void warpShift_(Mat& src, Mat& dst, int shiftx, int shifty)
{
	Mat dest =Mat::zeros(src.size(),src.type());
	
	int width = src.cols;
	int height = src.rows;
	if(shiftx>=0 &&shifty>=0)
	{
		Mat d = dest(Rect( shiftx, shifty, width-shiftx, height-shifty ));
		Mat(src(Rect(0, 0, width-shiftx, height-shifty))).copyTo(d);
	}
	else if(shiftx>=0 &&shifty<0)
	{
		Mat d = dest(Rect( shiftx, 0, width-shiftx, height+shifty));
		Mat(src(Rect( 0, -shifty, width-shiftx, height+shifty))).copyTo(d);
	}
	else if(shiftx<0 &&shifty<0)
	{
		Mat d = dest(Rect( 0, 0, width+shiftx, height+shifty));
		Mat(src(Rect( -shiftx, -shifty, width+shiftx, height+shifty))).copyTo(d);
	}
	else if(shiftx<0 &&shifty>=0)
	{
		Mat d = dest(Rect( 0, shifty, width+shiftx, height-shifty));
		Mat(src(Rect( -shiftx, 0, width+shiftx, height-shifty))).copyTo(d);
	}
	dest.copyTo(dst);
}


void warpShift_(Mat& src, Mat& dest, int shiftx, int shifty, int borderType)
{
	if(dest.empty())dest=Mat::zeros(src.size(),src.type());
	else dest.setTo(0);

	int width = src.cols;
	int height = src.rows;
	if(shiftx>=0 &&shifty>=0)
	{
		Mat im; copyMakeBorder(src,im,shifty,0,shiftx,0, borderType);
		Mat(im(Rect(0, 0, width, height))).copyTo(dest);
	}
	else if(shiftx>=0 &&shifty<0)
	{
		Mat im; copyMakeBorder(src,im,0,-shifty,shiftx,0, borderType);
		Mat(im(Rect(0, -shifty, width, height))).copyTo(dest);
	}
	else if(shiftx<0 &&shifty<0)
	{
		Mat im; copyMakeBorder(src,im,0,-shifty,0, -shiftx, borderType);
		Mat(im(Rect(-shiftx, -shifty, width, height))).copyTo(dest);
	}
	else if(shiftx<0 &&shifty>=0)
	{
		Mat im; copyMakeBorder(src,im,shifty, 0, 0, -shiftx, borderType);
		Mat(im(Rect(-shiftx, 0, width, height))).copyTo(dest);
	}
}


void warpShift(InputArray src_, OutputArray dest_, int shiftx, int shifty, int borderType)
{
	Mat src = src_.getMat();
	dest_.create( src.size(), src.type() );
	Mat dest = dest_.getMat();

	if(borderType<0)
		warpShift_(src,dest,shiftx,shifty);
	else
		warpShift_(src,dest,shiftx,shifty,borderType);
}

void alphaBlend(const Mat& src1, const Mat& src2, double alpha, Mat& dest)
{
	int T;
	Mat s1,s2;
	if(src1.channels()<=src2.channels())T=src2.type();
	else T=src1.type();
	if(dest.empty())dest=Mat::zeros(src1.size(),T);
	if(src1.channels()==src2.channels())
	{
		s1=src1;
		s2=src2;
	}
	else if(src2.channels()==3)
	{
		cvtColor(src1,s1,CV_GRAY2BGR);
		s2=src2;
	}
	else
	{
		cvtColor(src2,s2,CV_GRAY2BGR);
		s1=src1;
	}
	cv::addWeighted(s1,alpha,s2,1.0-alpha,0.0,dest);
}

void alphaBlendSSE_8u(const Mat& src1, const Mat& src2, const Mat& alpha, Mat& dest)
{
	if(dest.empty())dest.create(src1.size(),CV_8U);

	const int imsize = (src1.size().area()/16);
	uchar* s1 = (uchar*)src1.data;
	uchar* s2 = (uchar*)src2.data;
	uchar* a = (uchar*)alpha.data;
	uchar* d = dest.data;

	const __m128i zero = _mm_setzero_si128();
	const __m128i amax = _mm_set1_epi8(char(255));
	int i=0;
	if(s1==d)
	{
		for(;i<imsize;++i)
		{
			__m128i ms1h = _mm_load_si128((__m128i*)(s1));
			__m128i ms2h = _mm_load_si128((__m128i*)(s2));
			__m128i mah = _mm_load_si128((__m128i*)(a));
			__m128i imah = _mm_sub_epi8(amax,mah);

			__m128i ms1l = _mm_unpacklo_epi8(ms1h, zero);
			ms1h = _mm_unpackhi_epi8(ms1h, zero);

			__m128i ms2l = _mm_unpacklo_epi8(ms2h, zero);
			ms2h = _mm_unpackhi_epi8(ms2h, zero);

			__m128i mal = _mm_unpacklo_epi8(mah, zero);
			mah = _mm_unpackhi_epi8(mah, zero);

			__m128i imal = _mm_unpacklo_epi8(imah, zero);
			imah = _mm_unpackhi_epi8(imah, zero);

			ms1l = _mm_mullo_epi16(ms1l,mal);
			ms2l = _mm_mullo_epi16(ms2l,imal);
			ms1l = _mm_add_epi16(ms1l,ms2l);
			//ms1l = _mm_srli_epi16(ms1l,8);
			ms1l = _mm_srai_epi16(ms1l,8);

			ms1h = _mm_mullo_epi16(ms1h,mah);
			ms2h = _mm_mullo_epi16(ms2h,imah);
			ms1h = _mm_add_epi16(ms1h,ms2h);
			//ms1h = _mm_srli_epi16(ms1h,8);
			ms1h = _mm_srai_epi16(ms1h,8);

			_mm_stream_si128((__m128i*)s1,_mm_packs_epi16(ms1l,ms1h));

			s1+=16;
			s2+=16;
			a+=16;
		}
	}
	else
	{
		for(;i<imsize;++i)
		{
			__m128i ms1h = _mm_load_si128((__m128i*)(s1));
			__m128i ms2h = _mm_load_si128((__m128i*)(s2));
			__m128i mah = _mm_load_si128((__m128i*)(a));
			__m128i imah = _mm_sub_epi8(amax,mah);

			__m128i ms1l = _mm_unpacklo_epi8(ms1h, zero);
			ms1h = _mm_unpackhi_epi8(ms1h, zero);

			__m128i ms2l = _mm_unpacklo_epi8(ms2h, zero);
			ms2h = _mm_unpackhi_epi8(ms2h, zero);

			__m128i mal = _mm_unpacklo_epi8(mah, zero);
			mah = _mm_unpackhi_epi8(mah, zero);

			__m128i imal = _mm_unpacklo_epi8(imah, zero);
			imah = _mm_unpackhi_epi8(imah, zero);

			ms1l = _mm_mullo_epi16(ms1l,mal);
			ms2l = _mm_mullo_epi16(ms2l,imal);
			ms1l = _mm_add_epi16(ms1l,ms2l);
			//ms1l = _mm_srli_epi16(ms1l,8);
			ms1l = _mm_srai_epi16(ms1l,8);

			ms1h = _mm_mullo_epi16(ms1h,mah);
			ms2h = _mm_mullo_epi16(ms2h,imah);
			ms1h = _mm_add_epi16(ms1h,ms2h);
			//ms1h = _mm_srli_epi16(ms1h,8);
			ms1h = _mm_srai_epi16(ms1h,8);

			_mm_store_si128((__m128i*)d,_mm_packs_epi16(ms1l,ms1h));

			s1+=16;
			s2+=16;
			a+=16;
			d+=16;
		}
	}

	{
		uchar* s1 = (uchar*)src1.data;
		uchar* s2 = (uchar*)src2.data;
		uchar* a = (uchar*)alpha.data;
		uchar* d = dest.data;
		for(int n=i*16;n<src1.size().area();n++)
		{
			d[n] = (a[n]*s1[n] + (255-a[n])*s2[n])>>8;
		}
	}
}

void alphaBlend(const Mat& src1, const Mat& src2, const Mat& alpha, Mat& dest)
{
	int T;
	Mat s1,s2;
	if(src1.channels()<=src2.channels())T=src2.type();
	else T=src1.type();
	if(dest.empty()) dest=Mat::zeros(src1.size(),T);
	if(dest.type()!=T)dest=Mat::zeros(src1.size(),T);
	if(src1.channels()==src2.channels())
	{
		s1=src1;
		s2=src2;
	}
	else if(src2.channels()==3)
	{
		cvtColor(src1,s1,CV_GRAY2BGR);
		s2=src2;
	}
	else
	{
		cvtColor(src2,s2,CV_GRAY2BGR);
		s1=src1;
	}
	Mat a;
	if(alpha.depth()==CV_8U && s1.channels()==1)
	{
		//alpha.convertTo(a,CV_32F,1.0/255.0);
		alphaBlendSSE_8u(src1,src2,alpha,dest);
		return ;
	}
	else if(alpha.depth()==CV_8U)
	{
		alpha.convertTo(a,CV_32F,1.0/255);
	}
	else 
	{
		alpha.copyTo(a);
	}

	if(dest.channels()==3)
	{
		vector<Mat> ss1(3),ss2(3);
		vector<Mat> ss1f(3),ss2f(3);
		split(s1,ss1);
		split(s2,ss2);	
		for(int c=0;c<3;c++)
		{
			ss1[c].convertTo(ss1f[c],CV_32F);
			ss2[c].convertTo(ss2f[c],CV_32F);
		}
		{
			float* s1r = ss1f[0].ptr<float>(0);
			float* s2r = ss2f[0].ptr<float>(0);

			float* s1g = ss1f[1].ptr<float>(0);
			float* s2g = ss2f[1].ptr<float>(0);

			float* s1b = ss1f[2].ptr<float>(0);
			float* s2b = ss2f[2].ptr<float>(0);


			float* al = a.ptr<float>(0);
			const int size = src1.size().area()/4;
			const int sizeRem = src1.size().area()-size*4;

			const __m128 ones = _mm_set1_ps(1.0f);

			for(int i=size;i--;)
			{
				const __m128 msa = _mm_load_ps(al);
				const __m128 imsa = _mm_sub_ps(ones,msa);
				__m128 ms1 = _mm_load_ps(s1r);
				__m128 ms2 = _mm_load_ps(s2r);
				ms1 = _mm_mul_ps(ms1,msa);
				ms2 = _mm_mul_ps(ms2,imsa);
				ms1 = _mm_add_ps(ms1,ms2);
				_mm_store_ps(s1r,ms1);//store ss1f

				ms1 = _mm_load_ps(s1g);
				ms2 = _mm_load_ps(s2g);
				ms1 = _mm_mul_ps(ms1,msa);
				ms2 = _mm_mul_ps(ms2,imsa);
				ms1 = _mm_add_ps(ms1,ms2);
				_mm_store_ps(s1g,ms1);//store ss1f

				ms1 = _mm_load_ps(s1b);
				ms2 = _mm_load_ps(s2b);
				ms1 = _mm_mul_ps(ms1,msa);
				ms2 = _mm_mul_ps(ms2,imsa);
				ms1 = _mm_add_ps(ms1,ms2);
				_mm_store_ps(s1b,ms1);//store ss1f

				al+=4,s1r+=4,s2r+=4,s1g+=4,s2g+=4,s1b+=4,s2b+=4;
			}
			for(int i=0;i<sizeRem;i++)
			{
				*s1r= *al * *s1r +(1.f-*al) * *s2r;
				*s1g= *al * *s1g +(1.f-*al) * *s2g;
				*s1b= *al * *s1b +(1.f-*al) * *s2b;

				al++,s1r++,s2r++,s1g++,s2g++,s1b++,s2b++;
			}
			for(int c=0;c<3;c++)
			{
				ss1f[c].convertTo(ss1[c],CV_8U);
			}
			merge(ss1,dest);
		}
	}
	else if(dest.channels()==1)
	{
		Mat ss1f,ss2f;
		s1.convertTo(ss1f,CV_32F);
		s2.convertTo(ss2f,CV_32F);
		{
			float* s1r = ss1f.ptr<float>(0);
			float* s2r = ss2f.ptr<float>(0);
			float* al = a.ptr<float>(0);
			const int size = src1.size().area()/4;
			const int nn = src1.size().area() - size*4;
			const __m128 ones = _mm_set1_ps(1.0f);
			for(int i=size;i--;)
			{
				const __m128 msa = _mm_load_ps(al);
				const __m128 imsa = _mm_sub_ps(ones,msa);
				__m128 ms1 = _mm_load_ps(s1r);
				__m128 ms2 = _mm_load_ps(s2r);
				ms1 = _mm_mul_ps(ms1,msa);
				ms2 = _mm_mul_ps(ms2,imsa);
				ms1 = _mm_add_ps(ms1,ms2);
				_mm_store_ps(s1r,ms1);//store ss1f

				al+=4,s1r+=4,s2r+=4;
			}
			for(int i=nn;i--;)
			{
				*s1r = *al * *s1r + (1.0f-*al)* *s2r;
				al++,s1r++,s2r++;
			}
			if(src1.depth()==CV_32F)
				ss1f.copyTo(dest);
			else
				ss1f.convertTo(dest,src1.depth());
		}
	}
}