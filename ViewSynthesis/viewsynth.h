#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
#define CV_LIB_PREFIX comment(lib, "opencv_"

#define CV_LIB_VERSION CVAUX_STR(CV_MAJOR_VERSION)\
    CVAUX_STR(CV_MINOR_VERSION)\
    CVAUX_STR(CV_SUBMINOR_VERSION)

#ifdef _DEBUG
#define CV_LIB_SUFFIX CV_LIB_VERSION "d.lib")
#else
#define CV_LIB_SUFFIX CV_LIB_VERSION ".lib")
#endif

#define CV_LIBRARY(lib_name) CV_LIB_PREFIX CVAUX_STR(lib_name) CV_LIB_SUFFIX


#pragma CV_LIBRARY(core)
#pragma CV_LIBRARY(highgui)
#pragma CV_LIBRARY(imgcodecs)
#pragma CV_LIBRARY(imgproc)
#pragma CV_LIBRARY(calib3d)
#pragma CV_LIBRARY(photo)

void maxFilter(const Mat& src, Mat& dest, Size kernelSize);
void maxFilter(const Mat& src, Mat& dest, int radius);
void minFilter(const Mat& src, Mat& dest, Size kernelSize);
void minFilter(const Mat& src, Mat& dest, int radius);

void warpShift(InputArray src, OutputArray dest, int shiftx, int shifty=0, int borderType=-1);
void warpShiftSubpix(InputArray  src, OutputArray dest, double shiftx, double shifty=0, const int inter_method = cv::INTER_LANCZOS4);

void alphaBlend(const Mat& src1, const Mat& src2, const Mat& alpha, Mat& dest);
void alphaBlend(const Mat& src1, const Mat& src2, double alpha, Mat& dest);

double YPSNR(const Mat& src1, const Mat& src2);
double calcBadPixel(const Mat& src, const Mat& ref, int threshold);

void fillOcclusion(Mat& src, int invalidvalue=0);// for disparity map

enum
{
	FILTER_DEFAULT = 0,
	FILTER_CIRCLE,
	FILTER_RECTANGLE,
	FILTER_SEPARABLE,
	FILTER_SLOWEST,// for just comparison.
};

void GaussianFilter(const Mat src, Mat& dest, int r, float sigma, int method=FILTER_SLOWEST, Mat& mask=Mat());
void weightedGaussianFilter(Mat& src, Mat& weight, Mat& dest,Size ksize, float sigma, int border_type = BORDER_REPLICATE);

class StereoViewSynthesis
{

private:
	void depthfilter(Mat& depth, Mat& depth2,Mat& mask2,int viewstep,double disp_amp);
	template <class T>
	void analyzeSynthesizedViewDetail_(Mat& srcL,Mat& srcR, Mat& dispL,Mat& dispR, double alpha, int invalidvalue, double disp_amp,Mat& srcsynth, Mat& ref);
	template <class T>
	void viewsynth (const Mat& srcL, const Mat& srcR, const Mat& dispL, const Mat& dispR, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp, int disptype);
	template <class T>
	void makeMask_(Mat& srcL,Mat& srcR, Mat& dispL,Mat& dispR, double alpha, int invalidvalue, double disp_amp);
	template <class T>
	void viewsynthSingle(Mat& src,Mat& disp, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp, int disptype);

public:
	//warping parameters
	enum 
	{
		WAPR_IMG_INV= 0,//Mori et al.
		WAPR_IMG_FWD_SUB_INV, //Zenger et al.
	};
	int warpMethod;

	int warpInterpolationMethod;//Nearest, Linear or Cubic
	bool warpSputtering;
	int large_jump;

	//warped depth filtering parameters
	enum 
	{
		DEPTH_FILTER_SPECKLE = 0,
		DEPTH_FILTER_MEDIAN,
		DEPTH_FILTER_MEDIAN_ERODE,
		DEPTH_FILTER_CRACK,
		DEPTH_FILTER_MEDIAN_BILATERAL,
		DEPTH_FILTER_NONE
	};
	int depthfiltermode;
	int warpedMedianKernel;

	int warpedSpeckesWindow;
	int warpedSpeckesRange;

	int bilateral_r;
	float bilateral_sigma_space;
	float bilateral_sigma_color;

	//blending parameter

	int blendMethod;
	double blend_z_thresh;

	//post filtering parameters
	enum 
	{
		POST_GAUSSIAN_FILL=0,
		POST_FILL,
		POST_NONE
	};
	int postFilterMethod;

	enum 
	{
		FILL_OCCLUSION_LINE = 0,
		FILL_OCCLUSION_REFLECT = 1,
		FILL_OCCLUSION_STRETCH = -1,
		FILL_OCCLUSION_HV=2,
		FILL_OCCLUSION_INPAINT_NS=3, // OpenCV Navier-Stokes algorithm
		FILL_OCCLUSION_INPAINT_TELEA=4, // OpenCV A. Telea algorithm
	};
	int inpaintMethod;

	double inpaintr;//parameter for opencv inpaint 
	int canny_t1;
	int canny_t2;

	Size occBlurSize;

	Size boundaryKernelSize;
	double boundarySigma;
	double boundaryGaussianRatio;

	//preset
	enum 
	{
		PRESET_FASTEST = 0,
		PRESET_SLOWEST,
	};

	StereoViewSynthesis();
	StereoViewSynthesis(int preset);
	void init(int preset);

	void operator()(Mat& src,Mat& disp, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp);
	void operator()(const Mat& srcL, const Mat& srcR, const Mat& dispL, const Mat& dispR, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp);

	Mat diskMask;
	Mat allMask;//all mask
	Mat boundaryMask;//disparity boundary
	Mat nonOcclusionMask;
	Mat occlusionMask;//half and full occlusion
	Mat fullOcclusionMask;//full occlusion
	Mat nonFullOcclusionMask; //bar of full occlusion
	Mat halfOcclusionMask;//left and right half ooclusion

	void viewsynthSingleAlphaMap(Mat& src,Mat& disp, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp, int disptype);
	void alphaSynth(Mat& srcL,Mat& srcR, Mat& dispL,Mat& dispR, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp);
	void noFilter(Mat& srcL,Mat& srcR, Mat& dispL,Mat& dispR, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp);
	void analyzeSynthesizedViewDetail(Mat& srcL,Mat& srcR, Mat& dispL,Mat& dispR, double alpha, int invalidvalue, double disp_amp,Mat& srcsynth, Mat& ref);
	void analyzeSynthesizedView(Mat& srcsynth, Mat& ref);
	void makeMask(Mat& srcL,Mat& srcR, Mat& dispL,Mat& dispR, double alpha, int invalidvalue, double disp_amp);
	void makeMask(Mat& srcL,Mat& srcR, Mat& dispL,Mat& dispR, double alpha, int invalidvalue, double disp_amp, Mat& srcsynth, Mat& ref);

	void check(Mat& srcL,Mat& srcR, Mat& dispL,Mat& dispR, Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp, Mat& ref);
	void check(Mat& src,Mat& disp,Mat& dest, Mat& destdisp, double alpha, int invalidvalue, double disp_amp, Mat& ref);
	void preview(Mat& srcL,Mat& srcR, Mat& dispL,Mat& dispR,int invalidvalue, double disp_amp);
	void preview(Mat& src, Mat& disp,int invalidvalue, double disp_amp);
};
