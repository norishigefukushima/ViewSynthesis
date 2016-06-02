#include "viewsynth.h"

void fillOcclusionHV(Mat& src, int invalid=0)
{
	Mat dst = src.clone();
	fillOcclusion(dst,invalid);
	Mat dst2;
	transpose(src,dst2);
	fillOcclusion(dst2,invalid);
	transpose(dst2,src);
	min(src,dst,src);
}


void guiViewSynthesis()
{
	vector<string> sequence(50);
	int s_index=0;
	sequence[s_index++]="teddyH";
	sequence[s_index++]="conesH";
	sequence[s_index++]="Aloe";
	sequence[s_index++]="Art";
	sequence[s_index++]="Baby1";
	sequence[s_index++]="Baby2";
	sequence[s_index++]="Baby3";
	sequence[s_index++]="Books";
	sequence[s_index++]="Bowling1";
	sequence[s_index++]="Bowling2";
	sequence[s_index++]="Cloth1";
	sequence[s_index++]="Cloth2";
	sequence[s_index++]="Cloth3";
	sequence[s_index++]="Cloth4";
	sequence[s_index++]="Dolls";
	sequence[s_index++]="Flowerpots";
	sequence[s_index++]="Lampshade1";
	sequence[s_index++]="Lampshade2";
	sequence[s_index++]="Midd1";
	sequence[s_index++]="Midd2";
	sequence[s_index++]="Reindeer";
	sequence[s_index++]="Laundry";
	sequence[s_index++]="Moebius";
	sequence[s_index++]="Monopoly";
	sequence[s_index++]="Plastic";
	sequence[s_index++]="Rocks1";
	sequence[s_index++]="Rocks2";
	sequence[s_index++]="Wood1";
	sequence[s_index++]="Wood2";
	int s_index_max=s_index;
	s_index=2;

	
	string wname = "view";
	namedWindow(wname);

	int alpha = 0; createTrackbar("alpha", wname, &alpha,100);
	createTrackbar("index", wname, &s_index,s_index_max-1);
	int dilation_rad = 1; createTrackbar("dilation r", wname, &dilation_rad,10);
	int blur_rad = 1; createTrackbar("blur r", wname, &blur_rad,10);
	int isOcc = 2; createTrackbar("is Occ", wname, &isOcc,2);
	int isOccD = 1; createTrackbar("is OccD", wname, &isOccD,3);
	int zth = 32; createTrackbar("z thresh", wname, &zth,100);
	int grate = 100; createTrackbar("grate", wname, &grate,100);
	int ljump = 0; createTrackbar("ljump", wname, &ljump,1000);
	int ncore = 1;createTrackbar("ncore", wname, &ncore,32);
	int blend = 0; createTrackbar("blend", wname, &blend,1);
	int inter = 2; createTrackbar("inter", wname, &inter,2);
	int alphaPos = 500; createTrackbar("apos", wname, &alphaPos,1000);
	int color = 300; createTrackbar("color", wname, &color,2000);
	int space = 300; createTrackbar("space", wname, &space,2000);
	int iter = 0; createTrackbar("iter", wname, &iter,30);

	int key = 0;
	int dispAmp = 2;
	while(key!='q')
	{
		Mat dest,destdisp, max_disp_l, max_disp_r;
		Mat show;
		StereoViewSynthesis svs(StereoViewSynthesis::PRESET_SLOWEST);
		
		//cout<<"img/stereo/"+sequence[s_index]<<endl;
		Mat matdiL = imread("img/stereo/"+sequence[s_index]+"/disp1.png",0);
		Mat matdiR = imread("img/stereo/"+sequence[s_index]+"/disp5.png",0);
		Mat matimL= imread("img/stereo/"+sequence[s_index]+"/view1.png");
		Mat matimR= imread("img/stereo/"+sequence[s_index]+"/view5.png");

		Mat ref= imread("img/stereo/"+sequence[s_index]+"/view3.png");

		fillOcclusion(matdiL);
		if((matdiL.size().area()-countNonZero(matdiL)!=0))fillOcclusionHV(matdiL);
		fillOcclusion(matdiR);
		if((matdiL.size().area()-countNonZero(matdiR)!=0))fillOcclusionHV(matdiR);
		
		
		alphaBlend(matimL,matdiL,alpha/100.0,show);
		imshow("disp",show);

		svs.blendMethod=blend;
		svs.large_jump = ljump;

		svs.boundaryGaussianRatio = (double)grate/100.0;
		svs.blend_z_thresh=zth;

		svs.occBlurSize = Size(2*blur_rad+1,2*blur_rad+1);
		svs.inpaintMethod = StereoViewSynthesis::FILL_OCCLUSION_HV;
		
		if(isOcc==0) svs.postFilterMethod=StereoViewSynthesis::POST_NONE;
		if(isOcc==1) svs.postFilterMethod=StereoViewSynthesis::POST_FILL;
		if(isOcc==2) svs.postFilterMethod=StereoViewSynthesis::POST_GAUSSIAN_FILL;
		if(isOccD==0)svs.inpaintMethod=StereoViewSynthesis::FILL_OCCLUSION_LINE;
		if(isOccD==1)svs.inpaintMethod=StereoViewSynthesis::FILL_OCCLUSION_HV;
		if(isOccD==2)svs.inpaintMethod=StereoViewSynthesis::FILL_OCCLUSION_INPAINT_NS;
		if(isOccD==3)svs.inpaintMethod=StereoViewSynthesis::FILL_OCCLUSION_INPAINT_TELEA;
		//
		svs.warpInterpolationMethod = inter;
		{	
			maxFilter(matdiL, max_disp_l,dilation_rad);
			maxFilter(matdiR, max_disp_r,dilation_rad);
			svs(matimL,matimR,max_disp_l,max_disp_r, dest, destdisp,alphaPos*0.001,0,dispAmp);
		}
		
		alphaBlend(ref,dest,alpha/100.0,dest);

		imshow(wname, dest);
		key = waitKey(1);
		if(key=='f') alpha = (alpha == 0) ? 100 : 0;
		cout<<YPSNR(dest,ref)<<endl;
	}
}

int main(int argc, char* argv[])
{
	guiViewSynthesis();
	return 0;
}