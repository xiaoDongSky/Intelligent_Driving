#include <stdio.h>
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <time.h>
using namespace std;
using namespace cv;



/*********************************/
/*     车道线检测的参数，可删    */
#define BIN_MIN_THRESHOLD 120
#define BIN_MAX_THRESHOLD 255
#define CANNY_MIN_THRESHOLD 50
#define CANNY_MAX_THRESHOLD 60
#define	HOUGH_TRESHOLD 50		// line approval vote threshold
#define	HOUGH_MIN_LINE_LENGTH 10	// remove lines shorter than this treshold
#define	HOUGH_MAX_LINE_GAP 150   // join lines to one with smaller than this gaps
#define LINE_REJECT_DEGREES 20
#define MIDDLE_THRESH 0.25
#define VEHICLE_UP_RATIO 0.4
#define VEHICLE_BOTTOM_RATIO 0.9
#define RHO 1
#define THETA CV_PI/180.0



/*********************************/
/*       车尾检测的估计参数      */
#define MinDistance 2.5
#define CamHeight 1.2
#define VehicleWidth 1.9



/*********************************/
/*    视频大小，可改成320*240    */
#define VIDEO_WIDTH 640
#define VIDEO_HEIGHT 480



/*********************************/
/*    1:加辅助线   0:无辅助线    */
#define TESTMODE 1



//检测函数需要用到的链表结构
struct ListNode{
	CvRect rect;
	int times;
	ListNode *next;
};

//检测函数需要用到的变量集合
struct VehicleVariable{
	ListNode *StoreHead;
	ListNode *StoreTail;
	ListNode *Valid;
	int count;		//hogdetect没有检测到任何区域时候该叠加器+1
	bool check;
};

//确定检测窗口ROI需要用到的变量集合
struct ROIVariable{
	CvSize video_size;
	bool initialized;
	int vanishingpointx;
	int vanishingpointy;
	int fixedvanishingpointx;
	int fixedvanishingpointy;
	int vanishingpointrange;
	int upline;
	int downline;
	int count1;
	int count2;
	int count3;
	CvRect ROIRect;
	float ROIRatio;
	float ROIRatioStep;
	int ROIRatioChange;
	int ROIRatioHold;
};

//测距需要用到的变量集合
struct DistanceVariable
{
	int Distance; 
	char DistanceDisplay[20];
};

struct RecordDistance
{
	int Distance1; 
	int Distance2;
	int Distance3;
	int DistanceShow;
};

//检测函数需要用到的函数
bool isValid(CvRect src,CvRect dst,CvRect rect, int pty)
{
	int diff1,diff2;
	diff1=abs(src.x-dst.x);
	diff2=abs(src.y-dst.y);
	if(diff1<(src.height*0.5) && (diff2<src.width*0.5) && dst.x-rect.x>rect.width*0.05 && dst.x-rect.x<rect.width*0.8 && dst.x+dst.width-rect.x>rect.width*0.2 && dst.x+dst.width-rect.x<rect.width*0.95 && pty-dst.y>0 && dst.y+dst.height-pty>0 && (dst.y+dst.height-pty)/(pty-dst.y)<10){
		return true;
	}else{
		return false;
	}
}

//检测函数需要用到的函数
void GetValid(CvRect tempRect,ListNode *&StoreHead,ListNode *&StoreTail,ListNode *Valid,CvRect rect,int pty)
{
	ListNode *ptr=StoreHead;
	ListNode *tmp;
	bool add=true;

	while(ptr!=NULL){
		if(isValid(ptr->rect,tempRect,rect,pty)){
			ptr->times++;
			add=false;
			if(ptr->times==3){
				Valid->rect=tempRect;
				Valid->times=0;
				Valid->next=Valid;
				break;
			}
		}
		ptr=ptr->next;
	}

	if(add){
		StoreTail->next=new ListNode;
		StoreTail=StoreTail->next;
		StoreTail->next=NULL;
		StoreTail->rect=tempRect;
		StoreTail->times=0;
	}

	if(Valid->next!=NULL){
		while(StoreHead!=NULL){
			tmp=StoreHead;
			StoreHead=StoreHead->next;
			delete tmp;
		}
	}
}


//函数目的：检测车辆，返回tempRect
//如未检测到目标，函数返回cvRect(1,1,1,1)
//SvmVariable为变量集合，具体见main函数
CvRect SvmDetector(IplImage* img, IplImage* img2,CvRect rect ,VehicleVariable* Vehicle,cv::HOGDescriptor hog,int pty)
{
	vector<cv::Rect> found;
	cvSetImageROI(img,rect);
	cvResize(img,img2);
	cvResetImageROI(img);
	CvRect tempRect=cvRect(1,1,1,1);
	hog.detectMultiScale(img2, found, 0, cv::Size(8,8), cv::Size(0,0), 1.1, 5);
	cvSetImageROI(img,cvRect(0,0,img2->width,img2->height));
	if(TESTMODE==1)cvCopy(img2,img);
	cvResetImageROI(img);
	if (found.size() > 0)
	{
		Vehicle->check=false;
		for (int i=0; i<found.size(); i++)
		{
			tempRect = cvRect(found[i].x, found[i].y, found[i].width, found[i].height);
			tempRect.x=tempRect.x*rect.width/img2->width+rect.x;
			tempRect.y=tempRect.y*rect.height/img2->height+rect.y;
			tempRect.width=tempRect.width*rect.width/img2->width;
			tempRect.height=tempRect.height*rect.height/img2->height;
			if(Vehicle->Valid->next==NULL)
			{
				if(Vehicle->StoreHead==NULL)
				{
					Vehicle->StoreHead=new ListNode;
					Vehicle->StoreHead->next=NULL;
					Vehicle->StoreHead->rect=tempRect;
					Vehicle->StoreHead->times=0;
					Vehicle->StoreTail=Vehicle->StoreHead;
				}else
				{
					GetValid(tempRect,Vehicle->StoreHead,Vehicle->StoreTail,Vehicle->Valid,rect,pty);
				}
			}else
			{
				if(isValid(Vehicle->Valid->rect,tempRect,rect,pty)){
					Vehicle->Valid->rect=tempRect;
					Vehicle->check=true;
					break;
				}
			}
		}
		tempRect=cvRect(1,1,1,1);
		if(!Vehicle->check){
			Vehicle->count++;
		}else{
			Vehicle->count=0;
		}

	}else{
		Vehicle->count++;
	}

	if(Vehicle->count>10){			//连续10帧都没有检测到有效区域
		Vehicle->count=0;
		Vehicle->Valid->next=NULL;
	}

	if(Vehicle->Valid->next!=NULL){
		tempRect = Vehicle->Valid->rect;
	}

	return tempRect;
}


//函数目的：检测车道线（可删）
//最终结果通过CvPoint leftlanepoints[2]和CvPoint rightlanepoints[2]两个坐标数组传递出来
//两个数组分别代表的含义如下：
//(leftlanepoints[0].x，leftlanepoints[0].y)为左边车道线的左下角坐标
//(leftlanepoints[1].x，leftlanepoints[1].y)为左边车道线的右上角坐标
//(rightlanepoints[0].x，rightlanepoints[0].y)为右边车道线的左上角坐标
//(rightlanepoints[1].x，rightlanepoints[1].y)为右边车道线的右下角坐标
void FindLanes(int nFrmNum,IplImage* img, IplImage* pFrameGray, IplImage* pFrameGrayEdge, IplImage* pFramePrevious,int &LaneDetect, float &LaneRatio,CvMemStorage* houghStorage,CvPoint leftlanepoints[2],CvPoint rightlanepoints[2],CvPoint previousleftlanepoints[2],CvPoint previousrightlanepoints[2])
{
	
	cvCvtColor(img,pFrameGray,CV_BGR2GRAY);
	cvSmooth(pFrameGray,pFrameGray);
	cvCanny(pFrameGray,pFrameGrayEdge,CANNY_MIN_THRESHOLD,CANNY_MAX_THRESHOLD);
	
	if(LaneDetect<9)
	{
		if(LaneDetect==0) {cvZero(pFramePrevious);cvCopy(pFrameGrayEdge,pFramePrevious);}
		cvOr(pFrameGrayEdge,pFramePrevious,pFramePrevious);
		cvThreshold(pFramePrevious,pFramePrevious,250,255,THRESH_BINARY);
		LaneDetect++;
	}
	else
	{
		cvSetImageROI(pFramePrevious,cvRect(0,(int)img->height*(1-LaneRatio),img->width-20,(int)img->height*LaneRatio));
		if(TESTMODE==2) imshow("pFramePrevious",Mat(pFramePrevious));
		LaneDetect=0;
		CvSeq* lines = cvHoughLines2(pFramePrevious, houghStorage, CV_HOUGH_PROBABILISTIC, RHO, THETA, HOUGH_TRESHOLD, HOUGH_MIN_LINE_LENGTH, HOUGH_MAX_LINE_GAP);
		int leftmidx,rightmidx;
		leftmidx=0;
		rightmidx=img->width-1;
		leftlanepoints[0].x=rightlanepoints[0].x=leftlanepoints[1].x=rightlanepoints[1].x=-1000;
		for(int i = 0; i < lines->total; i++ )
		{
			CvPoint* line = (CvPoint*)cvGetSeqElem(lines,i);
			int midx=(line[1].x+line[0].x)*0.5;
			int dx = line[1].x - line[0].x;
			int dy = line[1].y - line[0].y;
			float angle = atan2f(dy, dx) * 180.0/CV_PI;
			if (fabs(angle) <= LINE_REJECT_DEGREES) continue;
			dx==0?1:dx;
			if(midx<img->width/2*(1.0-MIDDLE_THRESH) && midx>leftmidx && angle<0 && angle>LINE_REJECT_DEGREES-90)
			{
				leftmidx=midx;
				leftlanepoints[0].x=line[0].x+(pFramePrevious->height*LaneRatio-line[0].y)*dx/(float)dy;
				leftlanepoints[0].y=pFramePrevious->height*LaneRatio-1;
				leftlanepoints[1].x=line[1].x+(-line[1].y)*dx/(float)dy;
				leftlanepoints[1].y=0;
			}
			else if(midx>img->width/2*(1.0+MIDDLE_THRESH) && midx<rightmidx && angle>0 && angle<90-LINE_REJECT_DEGREES)
			{
				rightmidx=midx;
				rightlanepoints[0].x=line[0].x+(-line[0].y)*dx/(float)dy;
				rightlanepoints[0].y=0;
				rightlanepoints[1].x=line[1].x+(pFramePrevious->height*LaneRatio-line[1].y)*dx/(float)dy;
				rightlanepoints[1].y=pFramePrevious->height*LaneRatio-1;
			}
		}
		cvResetImageROI(pFramePrevious);
		if(leftlanepoints[0].x!=-1000 && rightlanepoints[0].x!=-1000)
		{
			if(((float)rightlanepoints[0].x-(float)leftlanepoints[1].x)<0.35*((float)rightlanepoints[1].x-(float)leftlanepoints[0].x)) 
			{
				LaneRatio-=0.01;
			}
			else if(((float)rightlanepoints[0].x-(float)leftlanepoints[1].x)>0.45*((float)rightlanepoints[1].x-(float)leftlanepoints[0].x)) 
			{
				LaneRatio+=0.01;
			}
		}

		if(leftlanepoints[0].x==-1000)
		{
			leftlanepoints[0]=previousleftlanepoints[0];
			leftlanepoints[1]=previousleftlanepoints[1];
		}
		if(rightlanepoints[0].x==-1000)
		{
			rightlanepoints[0]=previousrightlanepoints[0];
			rightlanepoints[1]=previousrightlanepoints[1];
		}
		if(leftlanepoints[0].x!=-1000)
		{

			if(previousleftlanepoints[0].x!=-1000)
			{
				leftlanepoints[0].x=previousleftlanepoints[0].x+0.4*(leftlanepoints[0].x-previousleftlanepoints[0].x);
				leftlanepoints[1].x=previousleftlanepoints[1].x+0.4*(leftlanepoints[1].x-previousleftlanepoints[1].x);
			}
			previousleftlanepoints[0]=leftlanepoints[0];
			previousleftlanepoints[1]=leftlanepoints[1];
		}
		if(rightlanepoints[0].x!=-1000)
		{

			if(previousrightlanepoints[0].x!=-1000)
			{
				rightlanepoints[0].x=previousrightlanepoints[0].x+0.4*(rightlanepoints[0].x-previousrightlanepoints[0].x);
				rightlanepoints[1].x=previousrightlanepoints[1].x+0.4*(rightlanepoints[1].x-previousrightlanepoints[1].x);
			}
			previousrightlanepoints[0]=rightlanepoints[0];
			previousrightlanepoints[1]=rightlanepoints[1];
		}
	}
}

//该函数在LocateROI函数中所用到：目的是计算vanishingpoint的纵坐标
//刚开始时，如果已经检测到了车道线信息，就计算消隐点的纵坐标（相当于地平线），一旦确定了该纵坐标，之后很长时间之内不再更新，目的是防止颠簸和抖动
int findvanishingline(ROIVariable* ROI)
{
	if(ROI->downline-ROI->upline<=2) {ROI->downline=ROI->upline; return ROI->upline;}
	if(!ROI->initialized && ROI->vanishingpointx!=-1000 && ROI->vanishingpointy!=-1000) {ROI->initialized=true;ROI->upline=ROI->vanishingpointy-ROI->vanishingpointrange;ROI->downline=ROI->vanishingpointy+ROI->vanishingpointrange; return ROI->vanishingpointy;}
	if(ROI->vanishingpointy>=ROI->upline-(ROI->downline-ROI->upline)/4 && ROI->vanishingpointy<=(ROI->upline+ROI->downline)/2) {ROI->count1++;}
	else if(ROI->vanishingpointy>(ROI->upline+ROI->downline)/2 && ROI->vanishingpointy<=ROI->downline+(ROI->downline-ROI->upline)/4) {ROI->count2++;}
	else {ROI->count3++;}
	if(ROI->count1>=3) {ROI->downline=(ROI->upline+ROI->downline)/2; ROI->count1=ROI->count2=0;}
	else if(ROI->count2>=3) {ROI->upline=(ROI->upline+ROI->downline)/2; ROI->count1=ROI->count2=0;}
	else if(ROI->count3>=10) {ROI->upline=ROI->upline-(ROI->downline-ROI->upline)/2;ROI->downline=ROI->downline+(ROI->downline-ROI->upline)/2;ROI->count1=ROI->count2=ROI->count3=0;}
	return (ROI->upline+ROI->downline)/2;
}


//函数目的：确认所需扫描的检测窗口ROI的大小和位置，每帧都会根据检测到的车辆框来修正ROI的大小和位置
//最终检测窗口的信息通过ROI->ROIRect传递
void LocateROI(ROIVariable* ROI, CvPoint leftlanepoints[2], CvPoint rightlanepoints[2], IplImage* img, int nFrmNum )
{
	/*cout<<"leftlanepoints[0] ("<<leftlanepoints[0].x<<","<<leftlanepoints[0].y<<")"<<endl;
	cout<<"leftlanepoints[1] ("<<leftlanepoints[1].x<<","<<leftlanepoints[1].y<<")"<<endl;
	cout<<"rightlanepoints[0] ("<<rightlanepoints[0].x<<","<<rightlanepoints[0].y<<")"<<endl;
	cout<<"rightlanepoints[1] ("<<rightlanepoints[1].x<<","<<rightlanepoints[1].y<<")"<<endl;*/
	if(leftlanepoints[0].x!=-1000 && rightlanepoints[0].x!=-1000)
	{
		ROI->vanishingpointx=(leftlanepoints[0].x*(rightlanepoints[1].x-rightlanepoints[0].x)-rightlanepoints[1].x*(leftlanepoints[0].x-leftlanepoints[1].x))/(float)((rightlanepoints[1].x-rightlanepoints[0].x)-(leftlanepoints[0].x-leftlanepoints[1].x));
		ROI->vanishingpointy=img->height-1-(leftlanepoints[0].y-leftlanepoints[1].y)*(rightlanepoints[1].x-leftlanepoints[0].x)/(float)((rightlanepoints[1].x-leftlanepoints[0].x)-abs(rightlanepoints[0].x-leftlanepoints[1].x));
	}
	if(nFrmNum>1 && nFrmNum%10000<=1)
	{
		ROI->vanishingpointrange=8;
		ROI->upline=ROI->fixedvanishingpointy-ROI->vanishingpointrange;
		ROI->downline=ROI->fixedvanishingpointy+ROI->vanishingpointrange;
		ROI->count1=ROI->count2=ROI->count3=0;
	}
	if(nFrmNum%10==0) ROI->fixedvanishingpointy=findvanishingline(ROI);
	if(leftlanepoints[0].x==-1000 || rightlanepoints[0].x==-1000 || ROI->fixedvanishingpointx<0 || ROI->fixedvanishingpointx>ROI->video_size.width || ROI->fixedvanishingpointy<0 || ROI->fixedvanishingpointy>ROI->video_size.height)
	{
		ROI->fixedvanishingpointx=ROI->video_size.width*0.5;
		ROI->fixedvanishingpointy=ROI->video_size.height*0.5;
		ROI->vanishingpointrange=32;
		ROI->count1=ROI->count2=ROI->count3=0;
		ROI->initialized=false;
	}

	if(ROI->ROIRatioChange==1)
	{
		if(ROI->ROIRatioHold==1 || ROI->ROIRatio>=0.5) {
			ROI->ROIRatioHold=0;
			if(ROI->ROIRatio<1.0) {ROI->ROIRatio+=ROI->ROIRatioStep;} else{ROI->ROIRatioChange=-1;}
		}else{
			ROI->ROIRatioHold=1;
		}
	}else if(ROI->ROIRatioChange==-1)
	{
		if(ROI->ROIRatioHold==1 || ROI->ROIRatio>=0.5) {
			ROI->ROIRatioHold=0;
			if(ROI->ROIRatio>0.11) {ROI->ROIRatio-=ROI->ROIRatioStep;} else{ ROI->ROIRatioChange=1;}
		}else{
			ROI->ROIRatioHold=1;
		}
	}

	ROI->ROIRect.x=ROI->fixedvanishingpointx-ROI->video_size.width*0.2*ROI->ROIRatio;
	ROI->ROIRect.y=ROI->fixedvanishingpointy-ROI->video_size.height*0.2*ROI->ROIRatio;
	ROI->ROIRect.width=ROI->video_size.width*0.4*ROI->ROIRatio;
	ROI->ROIRect.height=ROI->video_size.width*0.4*ROI->ROIRatio;
}


//函数目的：利用已经检测到的车辆框的信息，调整了ROIRatio参数，即重新校正后续帧检测窗口ROI的大小和位置，提升后续帧检测的准确度和速度
void RelocateROI(CvRect tempRect, ROIVariable* ROI,IplImage* img)
{
	if(tempRect.height!=1 && tempRect.width!=1)
	{
		ROI->ROIRatioChange=ROI->ROIRatioHold=0;
		if(tempRect.x+tempRect.width*0.5>=ROI->video_size.width*0.4 && tempRect.x+tempRect.width*0.5<=ROI->video_size.width*0.6) ROI->fixedvanishingpointx=tempRect.x+tempRect.width*0.5;
		if((tempRect.x-ROI->ROIRect.x<=ROI->ROIRect.width*0.2 && tempRect.x-ROI->ROIRect.x+tempRect.width>=ROI->ROIRect.width*0.8) && ROI->ROIRatio<1) ROI->ROIRatio+=ROI->ROIRatioStep;
		else if((tempRect.x-ROI->ROIRect.x>=ROI->ROIRect.width*0.25 && tempRect.x-ROI->ROIRect.x+tempRect.width<=ROI->ROIRect.width*0.75) && ROI->ROIRatio>0.11) ROI->ROIRatio-=ROI->ROIRatioStep;
		cvRectangle(img, cvPoint(tempRect.x,tempRect.y),cvPoint(tempRect.x+tempRect.width-1,tempRect.y+tempRect.height-1),CV_RGB(255,0,0), 2);
	}
	else
	{
		ROI->fixedvanishingpointx=ROI->vanishingpointx;
		if(ROI->ROIRatioChange==0)
		{
			ROI->ROIRatio=0.1;
			ROI->ROIRatioChange=1;
		}
	}
}


//函数目的：利用车辆框的宽度和一些估计参数测距
//最终结果显示在车辆框的正上方
void CalculateDistance(CvRect tempRect, ROIVariable* ROI, DistanceVariable* Dist)
{
	if(tempRect.x!=1 && tempRect.y!=1)
	{
		int TempDistance=ROI->video_size.height*0.5*MinDistance*VehicleWidth/(tempRect.width*CamHeight)-3;
		if(TempDistance>=60)
		{
			Dist->Distance=TempDistance/10*10;
		}
		else if(TempDistance>=30)
		{
			Dist->Distance=TempDistance/5*5;
		}
		else if(abs(Dist->Distance-TempDistance)>=10 || TempDistance<=10)
		{
			Dist->Distance=TempDistance;
		}
		else if(abs(Dist->Distance-TempDistance)>=2)
		{
			Dist->Distance<TempDistance?Dist->Distance++:Dist->Distance--;
		}
		//sprintf(Dist->DistanceDisplay,"%dm",Dist->Distance);
		//putText(Mat(img),Dist->DistanceDisplay,cvPoint(tempRect.x+(int)(0.5*tempRect.width)-20,tempRect.y-10),FONT_HERSHEY_SIMPLEX ,0.7,CV_RGB(255,0,0),2);
	}
}

//近距离测距稳定
void DistanceStable(DistanceVariable* Dist, RecordDistance* Rec,IplImage* img,CvRect tempRect)
{
	if(Dist->Distance<8){
		Rec->Distance1=Rec->Distance2;
		Rec->Distance2=Rec->Distance3;
		Rec->Distance3=Dist->Distance;
		if(Rec->Distance1==Rec->Distance2 && Rec->Distance2==Rec->Distance3){
			Rec->DistanceShow=Dist->Distance;
		}
		sprintf(Dist->DistanceDisplay,"%dm",Rec->DistanceShow);
		putText(Mat(img),Dist->DistanceDisplay,cvPoint(tempRect.x+(int)(0.5*tempRect.width)-20,tempRect.y-10),FONT_HERSHEY_SIMPLEX ,0.7,CV_RGB(255,0,0),2);
	}else{
		sprintf(Dist->DistanceDisplay,"%dm",Dist->Distance);
		putText(Mat(img),Dist->DistanceDisplay,cvPoint(tempRect.x+(int)(0.5*tempRect.width)-20,tempRect.y-10),FONT_HERSHEY_SIMPLEX ,0.7,CV_RGB(255,0,0),2);
	}
}

//函数目的：画车道线，画辅助线，写上标注等
//可通过 #define TESTMODE 1 改成0来禁用
void Display(CvPoint leftlanepoints[2], CvPoint rightlanepoints[2], IplImage* img, ROIVariable* ROI, float LaneRatio)
{
	if(leftlanepoints[0].x!=-1000) cvLine(img,cvPoint(leftlanepoints[0].x,leftlanepoints[0].y+(1-LaneRatio)*img->height-1),cvPoint(leftlanepoints[1].x,leftlanepoints[1].y+(1-LaneRatio)*img->height-1),CV_RGB(255,0,0),3);
	if(rightlanepoints[0].x!=-1000) cvLine(img,cvPoint(rightlanepoints[0].x,rightlanepoints[0].y+(1-LaneRatio)*img->height-1),cvPoint(rightlanepoints[1].x,rightlanepoints[1].y+(1-LaneRatio)*img->height-1),CV_RGB(255,0,0),3);

	if(TESTMODE==1)
	{
		cvCircle(img,cvPoint(ROI->vanishingpointx,ROI->vanishingpointy),2,CV_RGB(255,0,0),2);
		cvCircle(img,cvPoint(ROI->fixedvanishingpointx,ROI->fixedvanishingpointy),2,CV_RGB(0,255,0),2);
	}

	if(ROI->upline==ROI->downline)
	{
		if(TESTMODE==1)
		{
			putText(Mat(img),"VP Confirmed",cvPoint(5,ROI->video_size.height-8),FONT_HERSHEY_SIMPLEX ,0.5,CV_RGB(0,255,0));
			cvLine(img,cvPoint(ROI->fixedvanishingpointx,0),cvPoint(ROI->fixedvanishingpointx,ROI->video_size.height-1),CV_RGB(0,255,0),1);
			cvLine(img,cvPoint(0,ROI->upline),cvPoint(ROI->video_size.width-1,ROI->upline),CV_RGB(0,255,0),1);
		}
	}
	else
	{
		if(TESTMODE==1)
		{
			putText(Mat(img),"Calculating VP..",cvPoint(5,ROI->video_size.height-8),FONT_HERSHEY_SIMPLEX ,0.5,CV_RGB(255,0,0));
		}
	}
	if(TESTMODE==1)cvRectangle(img,cvPoint(ROI->ROIRect.x,ROI->ROIRect.y),cvPoint(ROI->ROIRect.x+ROI->ROIRect.width-1,ROI->ROIRect.y+ROI->ROIRect.height-1),CV_RGB(0,255,0),1,8);
}



 void ShadowFind(CvRect &rect, IplImage* img, int &PrevHeight)
 {
	 IplImage* grey=cvCreateImage(cvSize(img->width,img->height),IPL_DEPTH_8U,1); 
	 //char widthdisplay[20];
	 uchar* data=(uchar *)grey->imageData;  
     int step = grey->widthStep/sizeof(uchar); 
	 int GreyValue=255;
	 int y=rect.y+rect.height;
	 int x=rect.x+rect.width/2;
	 cvCvtColor(img,grey,CV_BGR2GRAY);
	 for(int i=0;i<rect.height/3;i++){
		if(data[(rect.y+rect.height-i)*step+x]<GreyValue){
			GreyValue=data[(rect.y+rect.height-i)*step+x];
			y=rect.y+rect.height-i;
		}
		if(data[(rect.y+rect.height+i)*step+x]<GreyValue){
			GreyValue=data[(rect.y+rect.height+i)*step+x];
			y=rect.y+rect.height+i;
		}
	 }
	 if(PrevHeight==1000){
		 PrevHeight=y-rect.y;
		 rect.height=y-rect.y;
	 }else{
		 if(abs(PrevHeight-(y-rect.y))>20){
			 rect.height=y-rect.y;
			 PrevHeight=rect.height;
		 }else{
			 rect.height=PrevHeight;
		 }
	 }
	 //sprintf(widthdisplay,"%dm",rect.width);
	 //putText(Mat(img),widthdisplay,cvPoint(30,30),FONT_HERSHEY_SIMPLEX ,0.7,CV_RGB(255,0,0),2);
	 cvReleaseImage(&grey);
 }



 CvRect FindContours(IplImage* src)
{
	CvMemStorage* storage = cvCreateMemStorage(0);
	//IplImage* dst = cvCreateImage( cvGetSize(src), 8, 3);
	//cvCvtColor(src,dst,CV_GRAY2BGR);
	CvScalar color = CV_RGB( 255, 0, 0);
	CvSeq* contours=0;
	int flag=0;
	CvRect aRect;
	
	 //建立一个空序列存储每个四边形的四个顶点
   // CvSeq* squares = cvCreateSeq( 0, sizeof(CvSeq), sizeof(CvPoint), storage );

	//cvFindContours( src, storage, &contours, sizeof(CvContour),CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );//外界边界h_next 和 孔用v_next连接
	cvFindContours( src, storage, &contours, sizeof(CvContour), CV_RETR_CCOMP, CV_CHAIN_APPROX_SIMPLE );

	 for( ; contours != 0; contours = contours->h_next)
            {
				//使用边界框的方式
				aRect = cvBoundingRect( contours, 1 );
				int tmparea=aRect.height*aRect.height;
				if (((double)aRect.width/(double)aRect.height>2)
				&& ((double)aRect.width/(double)aRect.height<6)&& tmparea>=100&&tmparea<=2500)
			{
				//cvRectangle(dst,cvPoint(aRect.x,aRect.y),cvPoint(aRect.x+aRect.width ,aRect.y+aRect.height),color,2);
				flag=1;
				break;
				//cvDrawContours( dst, contours, color, color, -1, 1, 8 );
			}
		}

	//cvReleaseImage(&dst);
	cvReleaseMemStorage(&storage);

	if(flag==1){
		return aRect;
	}else{
		return cvRect(1,1,1,1);
	}
}


 void LinPlate(CvRect rect, IplImage* imginput)
 {
	 cvSetImageROI(imginput,cvRect(rect.x,rect.y+rect.height/3,rect.width,rect.height/2));
	 IplImage* img = cvCreateImage(cvSize(rect.width,rect.height/2),8,1);;
	 //cvResize(imginput,img);
	 cvCvtColor(imginput,img,CV_BGR2GRAY); 
	 cvSmooth(img,img,CV_MEDIAN);
	 CvRect aRect;

	 IplImage* imgS=cvCreateImage(cvGetSize(img),IPL_DEPTH_16S,1);
	 IplImage* imgTh=cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);
	 IplImage* temp=cvCreateImage(cvGetSize(img),IPL_DEPTH_8U,1);

	 cvSobel(img,imgS,2,0,3);
     cvNormalize(imgS,imgTh,255,0,CV_MINMAX);

	 cvThreshold( imgTh, imgTh, 100, 255, CV_THRESH_BINARY );

	 for (int k=0; k<img->height; k++)
		for(int j=0; j<img->width; j++)
			{
				imgTh->imageData[k*img->widthStep+j] = 255 - imgTh->imageData[k*img->widthStep+j];
			}

	IplConvKernel* K=cvCreateStructuringElementEx(3,1,0,0,CV_SHAPE_RECT);
	IplConvKernel* K1=cvCreateStructuringElementEx(3,3,0,0,CV_SHAPE_RECT);
		
	cvMorphologyEx(imgTh,imgTh,temp,K,CV_MOP_CLOSE,10);
	cvMorphologyEx(imgTh,imgTh,temp,K1,CV_MOP_OPEN,1);

	

	aRect=FindContours(imgTh);
	if(aRect.x!=1){
		cvRectangle(imginput,cvPoint(aRect.x,aRect.y),cvPoint(aRect.x+aRect.width ,aRect.y+aRect.height),CV_RGB( 255, 0, 0),2);
	}
	cvResetImageROI(imginput);
	cvReleaseImage(&img);
	cvReleaseImage(&imgTh);
	cvReleaseImage(&imgS);
	cvReleaseImage(&temp);
	cvReleaseStructuringElement(&K);
	cvReleaseStructuringElement(&K1);
 }

//主函数
void main()
{
	//确定ROI所需用到的变量集
	ROIVariable* ROI=new ROIVariable;
	ROI->video_size.height = VIDEO_HEIGHT;
	ROI->video_size.width = VIDEO_WIDTH;
	ROI->initialized=false;
	ROI->fixedvanishingpointx=ROI->video_size.width*0.5;
	ROI->fixedvanishingpointy=ROI->video_size.height*0.5;
	ROI->vanishingpointx=ROI->vanishingpointy=-1000;
	ROI->vanishingpointrange=32;
	ROI->upline=ROI->video_size.height*0.5-ROI->vanishingpointrange;
	ROI->downline=ROI->video_size.height*0.5+ROI->vanishingpointrange;
	ROI->count1=ROI->count2=ROI->count3=0;
	ROI->ROIRatio=0.1;
	ROI->ROIRatioStep=0.05;
	ROI->ROIRatioChange=1;
	ROI->ROIRatioHold=0;

	//检测函数中需要用到的变量集
	VehicleVariable* Vehicle=new VehicleVariable;
	Vehicle->StoreHead=NULL;
	Vehicle->StoreTail=NULL;
	Vehicle->Valid=new ListNode;
	Vehicle->Valid->next=NULL;
	Vehicle->count=0;
	CvRect tempRect;
	int PrevHeight=1000;

	//测距需要用到的变量
	DistanceVariable* Dist=new DistanceVariable;
	RecordDistance* Rec=new RecordDistance;
	Dist->Distance=0; 

	//检测车道线需要用到的变量
	IplImage* pFramePrevious=0, *pFrameGray=0, *pFrameGrayEdge=0;
	CvMemStorage* houghStorage = cvCreateMemStorage(0);
	CvPoint leftlanepoints[2], rightlanepoints[2],previousleftlanepoints[2],previousrightlanepoints[2];
	leftlanepoints[0].x=rightlanepoints[0].x=leftlanepoints[1].x=rightlanepoints[1].x=-1000;
	previousleftlanepoints[0].x=previousrightlanepoints[0].x=previousleftlanepoints[1].x=previousrightlanepoints[1].x=-1000;
	int LaneDetect=0;
	float LaneRatio=0.33;
	float VehicleUpRatio=VEHICLE_UP_RATIO;
	float VehicleBottomRatio=VEHICLE_BOTTOM_RATIO;


	//读入SVM分类器
	vector<float> x;
	ifstream fileIn("SVMDetector.txt", ios::in);
	float val = 0.0f;
	while(!fileIn.eof())
	{
		fileIn>>val;
		x.push_back(val);
	}
	fileIn.close();
	cv::HOGDescriptor hog(cv::Size(48,48), cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), 9);
	hog.setSVMDetector(x);



	//CvCapture* cap = cvCaptureFromCAM(-1);
	CvCapture* cap = cvCreateFileCapture("day2.mp4");
	//CvCapture* cap = cvCreateFileCapture("D:/Cheng/研究生/1学期/项目——车尾检测/视频/day2.mp4");
	//cvSetCaptureProperty(cap,CV_CAP_PROP_POS_FRAMES,24300);

	int nFrmNum=0;	//第几帧
	IplImage* oriimg=NULL;		//oriimg为输入的原始图像
	IplImage* img = cvCreateImage(cvSize(ROI->video_size.width,ROI->video_size.height),8,3);	//img为预设大小的图像
	oriimg=cvQueryFrame(cap);
	cvResize(oriimg,img);	//输入图像转换成某个预设尺寸
	IplImage* ResizedImg=cvCreateImage(cvSize(120,120),img->depth,img->nChannels);		//120*120固定大小的待扫描窗口，分类器每帧都扫描这个窗口

	while(oriimg=cvQueryFrame(cap)){
		cvResize(oriimg,img);
		nFrmNum++;
		/*cout<<"nFrmNum = "<<nFrmNum<<endl;*/
		if(nFrmNum == 1)	//第一帧初始化
		{
			pFrameGray = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 1);
			pFrameGrayEdge = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 1);
			pFramePrevious = cvCreateImage(cvSize(img->width, img->height), IPL_DEPTH_8U, 1);
		}

		//因为后面那个检测车道线的函数耗时较长，可以用几个点的坐标来暂时替代
		/*if(VIDEO_WIDTH==640)
		{
			leftlanepoints[0]=cvPoint(83,479);
			leftlanepoints[1]=cvPoint(230,322);
			rightlanepoints[0]=cvPoint(362,322);
			rightlanepoints[1]=cvPoint(579,479);
		}
		else if(VIDEO_WIDTH==320)
		{
			leftlanepoints[0]=cvPoint(41,239);
			leftlanepoints[1]=cvPoint(115,161);
			rightlanepoints[0]=cvPoint(181,161);
			rightlanepoints[1]=cvPoint(289,239);
		}*/
		

		FindLanes(nFrmNum, img, pFrameGray, pFrameGrayEdge, pFramePrevious, LaneDetect, LaneRatio, houghStorage, leftlanepoints, rightlanepoints, previousleftlanepoints, previousrightlanepoints);	//车道线检测，通过leftlanepoints和rightlanepoints传递检测结果

		LocateROI(ROI, leftlanepoints, rightlanepoints, img, nFrmNum);	//通过车道线计算消隐点，从而初步确定ROI位置和大小（绿色框）

		tempRect=SvmDetector(img, ResizedImg, ROI->ROIRect, Vehicle, hog, ROI->fixedvanishingpointy);	//检测车辆（红色框）

		RelocateROI(tempRect, ROI, img);	//通过车辆框校正ROI（绿色框）

		CalculateDistance(tempRect, ROI, Dist);	//测距，直接标注在图像上

		DistanceStable(Dist,Rec,img,tempRect);

		Display(leftlanepoints, rightlanepoints, img, ROI, LaneRatio);	//画辅助线，可通过 #define TESTMODE 1 改成0来禁用
		
		imshow("img",Mat(img));		//显示最终窗口



		char c=cvWaitKey(1);
		if(c==27){break;}	//Esc键退出
		else if(c==32){ c=cvWaitKey(0); if(c==32) {continue;} }		//空格键暂停，再按一次继续
		else if(c=='q'){		//q键跳过处理过程，相当于快进，e键继续处理
			while(1){
				oriimg=cvQueryFrame(cap);
				cvResize(oriimg,img);
				cvShowImage("img",img);
				if(cvWaitKey(5)=='w') break;
			}
		}



	}
	cvReleaseCapture(&cap);
	cvReleaseImage(&img);
	cvReleaseImage(&ResizedImg);
	cvReleaseImage(&pFramePrevious);
	cvReleaseImage(&pFrameGray);
	cvReleaseImage(&pFrameGrayEdge);
	cvReleaseMemStorage(&houghStorage);
	cvDestroyWindow("img");
	return;
}