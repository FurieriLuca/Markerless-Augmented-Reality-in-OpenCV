//Augmented Layer of a Book. Code written in 2015 for the course "Computer Vision and Image Processing" at the University of Bologna. 



#ifdef _WIN32
#include "stdafx.h"


#ifdef __APPLE__
#include "cv.h"
#include "cxcore.h"
#include "highgui.h"
#endif

#include "RansacEstimator.h"

#define CT_LKT		0x01

#define CT_HARRIS	0x02

#define MAXPOINTS	1000
#define FQUALITYLEVEL	0.05f
#define FMINDISTANCE	1.0f



void fnMatchFeatures(IplImage *pRefImage,IplImage *pTrgImage,CvPoint2D32f *pRefCorners,CvPoint2D32f* pTrgCorners,int *pCorners,CvSize sWinSize,int iPyrLevels,int iFlag);
int fnExtractFeatures(IplImage *pImage,CvPoint2D32f *vCorners,int *pCorners,float fQualityLevel, float fMinDistance,CvRect sROI,int iCornerType);

#ifdef __APPLE__
int main(int argc, char* argv[])
#endif

#ifdef WIN32
int _tmain(int argc, _TCHAR* argv[])
#endif
{
	RansacEstimator *pEstimator=new RansacEstimator(FULLPROJECTIVE);
	int iCorners=MAXPOINTS, MINiCorners=MAXPOINTS, referenceiCorners;
	CvPoint2D32f *vReferenceCorners=(CvPoint2D32f*)malloc(sizeof(CvPoint2D32f)*iCorners);
	CvPoint2D32f *vSrcCorners=(CvPoint2D32f*)malloc(sizeof(CvPoint2D32f)*iCorners);
	CvPoint2D32f *vDstCorners=(CvPoint2D32f*)malloc(sizeof(CvPoint2D32f)*iCorners);

	int frameCount = 0;
	int w, ws, h, i, j, k, wsl, wsb, wslb, diff, npixelF2R = 0, npixelF2F = 0, diffpixel;;
	int res;
	int tmp, tmp1, tmp2, tmp3;
	double dtmp;

	char buf[100];
	char sempreF;
	char ridotto;

// Mode of operation
	printf("Press 'F' for Frame-to-Frame, 'R' for Frame-to-Reference, any oher key for novel mixed algorithm\n");
	fgets(buf, 99,stdin);
	sempreF=buf[0];
	if (sempreF != 'F' && sempreF != 'R')
		sempreF = 'M';

	printf("Press 'R' for reduced mask, any oher key for exended mask\n");
	fgets(buf, 99,stdin);
	ridotto=buf[0];
	if (ridotto != 'R')
		ridotto = 'E';

	IplImage *layerMask;
	if (ridotto == 'R')
		layerMask=cvLoadImage("Stuff/AugmentedLayerMaskRidotto.PNG");
	else
		layerMask=cvLoadImage("Stuff/AugmentedLayerMask.PNG");



	IplImage *layer=cvLoadImage("Stuff/AugmentedLayer.PNG");
	IplImage *reference= cvLoadImage("Stuff/ReferenceFrame.png");
	CvCapture *cap = cvCaptureFromAVI("Stuff/Multiple View.avi");

	if(layer==NULL){
		printf("Layer not found\n");
		return -1;
	}
	if(layerMask==NULL){
		printf("layerMask not found\n");
		return -1;
	}
	if(reference==NULL){
		printf("reference not found\n");
		return -1;
	}
	if(cap==NULL){
		printf("Video not found\n");
		return -1;
	}

	double fps = cvGetCaptureProperty(cap, CV_CAP_PROP_FPS);
	CvSize sImageSize=cvGetSize(reference);
	sprintf(buf,"Stuff/Out%c%c.avi", sempreF, ridotto);
	CvVideoWriter *VideoOut=cvCreateVideoWriter(buf, CV_FOURCC('M','J','P','G'), fps, sImageSize, 1 );

	IplImage *layerWarp= cvCreateImage(cvGetSize(layer), 8, layer->nChannels);
	IplImage *LablayerWarp= cvCreateImage(cvGetSize(layer), 8, layer->nChannels);

	IplImage *layerMaskWarp= cvCreateImage(cvGetSize(layerMask), 8, layerMask->nChannels);
	IplImage *differenceMask= cvCloneImage(layerMask);
	IplImage *differenceMaskWarp= cvCreateImage(cvGetSize(layerMask), 8, layerMask->nChannels);

	IplImage *objectMask= cvCreateImage(cvGetSize(reference), 8, 1);
	IplImage *objectMaskWarp= cvCreateImage(cvGetSize(reference), 8, 1);
	IplImage *frameMask = cvCreateImage(cvGetSize(reference), 8, 1);
	wsb = frameMask->widthStep;

	IplImage *thisFrame = cvCloneImage(reference);
	IplImage *finalFrame = cvCreateImage(cvGetSize(reference), 8, reference->nChannels);
	IplImage *prevFrame= cvCreateImage(cvGetSize(reference), 8, reference->nChannels);
	IplImage *Labframe = cvCreateImage(cvGetSize(reference), 8, reference->nChannels);
	IplImage *LabframeCopy = cvCreateImage(cvGetSize(reference), 8, reference->nChannels);
	IplImage *referenceWarp = cvCreateImage(cvGetSize(reference), 8, reference->nChannels);
	IplImage *LabreferenceWarp = cvCreateImage(cvGetSize(reference), 8, reference->nChannels);

	IplImage *frameVideo;

	IplImage *imgdif= cvCreateImage(cvGetSize(reference), 8, 1);

	CvMat* pHomography=cvCreateMat(3,3,CV_32FC1);
	CvMat* diffHomography=cvCreateMat(3,3,CV_32FC1);
	CvMat* prevMixedHomography=cvCreateMat(3,3,CV_32FC1);
	CvMat* pF2RHomography=cvCreateMat(3,3,CV_32FC1);
	CvMat* pF2FHomography=cvCreateMat(3,3,CV_32FC1);
	cvSetIdentity(prevMixedHomography);


	cvNamedWindow("Image",CV_WINDOW_AUTOSIZE);

	wsl = layer->widthStep;
	w = reference->width;
	ws = reference->widthStep;
	h = reference->height;

	//Expansion of the augmented layer wih the conent of the reference, so as to avoid black artifacts at the edges.


	for(j=0; j<h; j++){//row scanning
		for(i=0; i<w*3; i+=3) {
			if (((unsigned char*)(layerMask->imageData))[j*wsl + i] == 0) {
				((unsigned char*)(layer->imageData))[j*wsl + i] = ((unsigned char*)(reference->imageData))[j*ws + i];
				((unsigned char*)(layer->imageData))[j*wsl + i+1] = ((unsigned char*)(reference->imageData))[j*ws + i+1];
				((unsigned char*)(layer->imageData))[j*wsl + i+2] = ((unsigned char*)(reference->imageData))[j*ws + i+2];
			}
		}
	}
	// conversion to Lablayer and Reference.
	IplImage *Lablayer=cvCloneImage(layer);
	cvCvtColor(layer, Lablayer, CV_RGB2Lab);
	

	IplImage *Labreference=cvCloneImage(reference);
	cvCvtColor(reference, Labreference, CV_RGB2Lab);

	// Construction of an objectMask: using the B channel of reference
	for(j=0; j<h; j++){//row scanning
		for(i=0; i<w; i++) {
			((unsigned char*)(objectMask->imageData))[j*wsb + i]
				= ((unsigned char*)(reference->imageData))[j*ws + i*3];
		}
	}
	cvThreshold(objectMask, objectMask, 0.0, 255, CV_THRESH_OTSU);

	// construction of the differential mask: contains "Azzari" and the additional picture 
	for(j=0; j<h; j++){//row scanning
		for(i=0; i<w*3; i+=3) {
			if (((unsigned char*)(layerMask->imageData))[j*wsl + i] == 255) {
				tmp1=((unsigned char*)(reference->imageData))[j*ws + i] -((unsigned char*)(layer->imageData))[j*wsl + i];
				if (tmp1 < 0) tmp1 = 0-tmp1;
				
				tmp2=((unsigned char*)(reference->imageData))[j*ws + i +1] -((unsigned char*)(layer->imageData))[j*wsl + i +1];
				if (tmp2 < 0) tmp2 = 0-tmp2;
				
				tmp3=((unsigned char*)(reference->imageData))[j*ws + i +2] -((unsigned char*)(layer->imageData))[j*wsl + i +2];
				if (tmp3 < 0) tmp3 = 0-tmp3;
	
				if (tmp1 < 30 && tmp2 < 30 && tmp3 < 30) {
					((unsigned char*)(differenceMask->imageData))[j*wsl + i] = 0;
					((unsigned char*)(differenceMask->imageData))[j*wsl + i +1] = 0;
					((unsigned char*)(differenceMask->imageData))[j*wsl + i +2] = 0;
				}
			}
		}
	}

	res=1;
	while (res!=0) {


		cvCopy(thisFrame, prevFrame);
		res = cvGrabFrame( cap );

		if (res!=0) {
			frameVideo = cvRetrieveFrame( cap );
			if (frameVideo->origin == 1)
				cvFlip(frameVideo,thisFrame);
			else
				cvCopy(frameVideo,thisFrame);
		} else
			continue;

// F2R
if (sempreF != 'F') {
		// Extraction of feature points in the Reference
		iCorners=MAXPOINTS;
		fnExtractFeatures(reference,vReferenceCorners,&iCorners,FQUALITYLEVEL,FMINDISTANCE,cvRect(0,0,sImageSize.width,sImageSize.height),CT_LKT);
		referenceiCorners = iCorners;

		// Matching of feature points. reference -> frame
		iCorners = referenceiCorners;
		fnMatchFeatures(reference,thisFrame,vReferenceCorners,vDstCorners,&iCorners,cvSize(15,15),3,0);
		if (iCorners < MINiCorners) MINiCorners = iCorners;

		// Robust estimation of the transformation through RANSAC algorithm
		pEstimator->m_fRansacModelEstimation(vDstCorners,vReferenceCorners,iCorners,2);
		//Reshaping of parameters vector computed through RANSAC into a 3X3 matrix
		for(k=0,i=0;i<3;i++)
			for(j=0;j<3;j++,k++)
				cvmSet(pHomography,i,j,cvmGet(pEstimator->m_pFinalTransformationMat,k,0));

		cvCopy(pHomography,pF2RHomography);

		// Comparison between warped objectMask and binarized frame. 
		cvWarpPerspective(objectMask,objectMaskWarp,pHomography,CV_WARP_FILL_OUTLIERS);

		// Take B channel of reference frame
		for(j=0; j<h; j++){//row scanning
			for(i=0; i<w; i++) {
				((unsigned char*)(frameMask->imageData))[j*wsb + i] = ((unsigned char*)(thisFrame->imageData))[j*ws + i*3];
			}
		}
		cvThreshold(frameMask, frameMask, 0.0, 255, CV_THRESH_OTSU);

		npixelF2R = 0;
		for(j=0; j<h; j++)//row scanning
			for(i=0; i<w; i++)
				if (((unsigned char*)(frameMask->imageData))[j*wsb + i] == ((unsigned char*)(objectMaskWarp->imageData))[j*wsb + i]) //3IMM {
					npixelF2R++;
}

// F2F
if (sempreF != 'R') {
		//Extraction of feature points of previous frame
		iCorners=MAXPOINTS;
		fnExtractFeatures(prevFrame,vSrcCorners,&iCorners,FQUALITYLEVEL,FMINDISTANCE,cvRect(0,0,sImageSize.width,sImageSize.height),CT_LKT);
		// Matching of feature points. Previous frame -> Present frame
		fnMatchFeatures(prevFrame,thisFrame,vSrcCorners,vDstCorners,&iCorners,cvSize(15,15),3,0);
		if (iCorners < MINiCorners) MINiCorners = iCorners;

		//Robust transformation through RANSAC. 
		pEstimator->m_fRansacModelEstimation(vDstCorners,vSrcCorners,iCorners,2);
		//Reshaping of parameters vector computed through RANSAC into a 3X3 matrix
		for(k=0,i=0;i<3;i++)
			for(j=0;j<3;j++,k++)
				cvmSet(diffHomography,i,j,cvmGet(pEstimator->m_pFinalTransformationMat,k,0));

		cvMatMul(diffHomography,prevMixedHomography,pHomography);
		
		// Comparison between warped objectMask and binarized frame. 
		cvWarpPerspective(objectMask,objectMaskWarp,pHomography,CV_WARP_FILL_OUTLIERS);

		npixelF2F = 0;
		for(j=0; j<h; j++)//row scanning
			for(i=0; i<w; i++)
				if (((unsigned char*)(frameMask->imageData))[j*wsb + i] == ((unsigned char*)(objectMaskWarp->imageData))[j*wsb + i]) //3IMM {
					npixelF2F++;
}

// ***** CHOOSE BEST HOMOGRAPHY

// Omography chosen depending on operating mode
diffpixel = npixelF2R - npixelF2F;
if (sempreF == 'F') diffpixel=-1;
if (sempreF == 'R') diffpixel=1;

		if ( diffpixel > 0) {
				printf("frame %d:\tutilizzo REFERENCE\t(F %d\tR %d\tF-R %d)\n", frameCount, npixelF2F, npixelF2R, npixelF2F - npixelF2R);
				cvCopy(pF2RHomography,pHomography);
		} else {
				printf("frame %d:\tutilizzo FRAME\t\t(F %d\tR %d\tF-R %d)\n", frameCount, npixelF2F, npixelF2R, npixelF2F - npixelF2R);
		}
		cvCopy(pHomography,prevMixedHomography);



		// Application of the homography on: Labreference, Lablayer, layerMask
		// reference e mask con interpolazione lineare (mask no, perche' deve rimanere bineizzata)
		cvWarpPerspective(Labreference,LabreferenceWarp,pHomography,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS);
		cvWarpPerspective(Lablayer,LablayerWarp,pHomography,CV_INTER_LINEAR+CV_WARP_FILL_OUTLIERS);

		cvWarpPerspective(layerMask,layerMaskWarp,pHomography,CV_WARP_FILL_OUTLIERS);
		cvWarpPerspective(differenceMask,differenceMaskWarp,pHomography,CV_WARP_FILL_OUTLIERS);

		cvCvtColor(thisFrame, Labframe, CV_RGB2Lab);
		cvCopy(Labframe,LabframeCopy);

		//Substitution of transparent pixels in the mask homography with the corresponding pixels.
		//For every pixel, compute the variation for all 3 Lab channels between initial reference frame and present frame. Apply the same variation
		//on the layer pixels.



		// Computation of the color: for every mask pixel, channel A = frame*layer/reference
		// Azzari and Logo part: channel B and C = frame
		// All other pixels: channel B and C = frame*layer/reference

		for(j=0; j<h; j++){//row scanning
			for(i=0; i<w*3; i+=3) {
				if (((unsigned char*)(layerMaskWarp->imageData))[j*wsl + i] == 255) {
					tmp=((unsigned char*)(Labframe->imageData))[j*ws + i]
					+((unsigned char*)(LablayerWarp->imageData))[j*wsl + i]
					-((unsigned char*)(LabreferenceWarp->imageData))[j*ws + i];
					
					((unsigned char*)(LabframeCopy->imageData))[j*ws + i] = (tmp <= 255) ? tmp : 255;
					
					tmp=((unsigned char*)(Labframe->imageData))[j*ws + i+1]
					+((unsigned char*)(LablayerWarp->imageData))[j*wsl + i+1]
					-((unsigned char*)(LabreferenceWarp->imageData))[j*ws + i+1];
					
					if (((unsigned char*)(differenceMaskWarp->imageData))[j*wsl + i+1] == 255)
					  tmp=((unsigned char*)(LablayerWarp->imageData))[j*wsl + i+1];
					((unsigned char*)(LabframeCopy->imageData))[j*ws + i+1] = (tmp <= 255) ? tmp : 255;
					
					tmp=((unsigned char*)(Labframe->imageData))[j*ws + i+2]
					+((unsigned char*)(LablayerWarp->imageData))[j*wsl + i+2]
					-((unsigned char*)(LabreferenceWarp->imageData))[j*ws + i+2];
					
					if (((unsigned char*)(differenceMaskWarp->imageData))[j*wsl + i+2] == 255)
						tmp=((unsigned char*)(LablayerWarp->imageData))[j*wsl + i+2];
					((unsigned char*)(LabframeCopy->imageData))[j*ws + i+2] = (tmp <= 255) ? tmp : 255;
				}
			}
		}
		
		// Conversion in RGB
		cvCvtColor(LabframeCopy, finalFrame, CV_Lab2RGB);


		cvShowImage("Image",finalFrame);


		cvWriteFrame(VideoOut, finalFrame);

		frameCount++;

		cvWaitKey(1);

	}
	cvReleaseVideoWriter(&VideoOut);

	return 0;
}

int fnExtractFeatures(IplImage *pImage,CvPoint2D32f *vCorners,int *pCorners,float fQualityLevel, float fMinDistance,CvRect sROI,int iCornerType)
{
	if ((*pCorners)<=0) return 0;

	CvSize sImageSize=cvGetSize(pImage);

	IplImage* pGrayImage=cvCreateImage(sImageSize,IPL_DEPTH_8U,1);
	cvCvtColor(pImage,pGrayImage,CV_RGB2GRAY);
	IplImage* pEigImage=cvCreateImage(sImageSize,IPL_DEPTH_32F,1);
	IplImage* pTempImage=cvCreateImage(sImageSize,IPL_DEPTH_32F,1);
 	cvSetImageROI( pGrayImage, sROI );
	cvSetImageROI( pEigImage, sROI );
	cvSetImageROI( pTempImage, sROI );
 	//OpenCV Beta 5
	switch (iCornerType)
	{
		case CT_LKT:
			cvGoodFeaturesToTrack(pGrayImage,pEigImage,pTempImage,vCorners, pCorners,fQualityLevel,fMinDistance);
 			cvFindCornerSubPix(pGrayImage, vCorners,*pCorners, cvSize(5,5), cvSize(-1,-1),cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,5,1));
			break;
		case CT_HARRIS:
			cvGoodFeaturesToTrack(pGrayImage,pEigImage,pTempImage,vCorners, pCorners,fQualityLevel,fMinDistance,NULL,3,1);
			cvFindCornerSubPix(pGrayImage, vCorners,*pCorners, cvSize(5,5), cvSize(-1,-1),cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS,5,1));
			break;
	}
 	// E' necessario aggiungere alle coordinate delle feature le coordinate iniziali del ROI perchè la funzione non lo fa automaticamente!!
	for (int iIndex=0;iIndex<*pCorners;iIndex++)
	{
		vCorners[iIndex].x+=(float)sROI.x;
		vCorners[iIndex].y+=(float)sROI.y;
	}
 	cvReleaseImage(&pGrayImage);
	cvReleaseImage(&pEigImage);
	cvReleaseImage(&pTempImage);

	return *pCorners;
}

void fnMatchFeatures(IplImage *pRefImage,IplImage *pTrgImage,CvPoint2D32f *pRefCorners,CvPoint2D32f* pTrgCorners,
		int *pCorners,CvSize sWinSize,int iPyrLevels,int iFlag)
{
	if (!(*pCorners))
		return;

	CvSize sImageSize;
	CvSize sPyrSize;

	sImageSize=cvGetSize(pRefImage);
	IplImage* pGrayRefImage=cvCreateImage(sImageSize,IPL_DEPTH_8U,1);
	cvCvtColor(pRefImage,pGrayRefImage,CV_RGB2GRAY);
	sPyrSize.width=sImageSize.width+8;
	sPyrSize.height=sImageSize.height/3;
	IplImage *pPyrA=cvCreateImage(sPyrSize,IPL_DEPTH_8U ,1);

	sImageSize=cvGetSize(pTrgImage);
	IplImage* pGrayTrgImage=cvCreateImage(sImageSize,IPL_DEPTH_8U,1);
	cvCvtColor(pTrgImage,pGrayTrgImage,CV_RGB2GRAY);
	sPyrSize.width=sImageSize.width+8;
	sPyrSize.height=sImageSize.height/3;
	IplImage *pPyrB=cvCreateImage(sPyrSize,IPL_DEPTH_8U ,1);

	//Le immagini reference e target devono essere della stessa dimensione
	CvTermCriteria sTermCriteria=cvTermCriteria(CV_TERMCRIT_ITER,100,20);
	cvCalcOpticalFlowPyrLK( pGrayRefImage, pGrayTrgImage, NULL, NULL, pRefCorners,pTrgCorners,*pCorners, sWinSize,iPyrLevels,NULL,NULL,sTermCriteria ,iFlag);

	//CV_LKFLOW_PYR_A_READY , pyramid for the first frame is precalculated before the call;
	//CV_LKFLOW_PYR_B_READY , pyramid for the second frame is precalculated before the call;
	//CV_LKFLOW_INITIAL_GUESSES , array B contains initial coordinates of features before the function call.

	//Eliminate those features that get mapped outside of the image rectangle.
	int iValidFeat=0;
	for (int iFeat=0;iFeat<*pCorners;iFeat++) {
		if (pTrgCorners[iFeat].x>=0 && pTrgCorners[iFeat].x<pRefImage->width && pTrgCorners[iFeat].y>=0 && pTrgCorners[iFeat].y<pRefImage->height) {
			pTrgCorners[iValidFeat].x=pTrgCorners[iFeat].x;
			pTrgCorners[iValidFeat].y=pTrgCorners[iFeat].y;

			pRefCorners[iValidFeat].x=pRefCorners[iFeat].x;
			pRefCorners[iValidFeat].y=pRefCorners[iFeat].y;
			iValidFeat++;
		}

	}

	*pCorners=iValidFeat;

	cvReleaseImage(&pGrayRefImage);
	cvReleaseImage(&pGrayTrgImage);

	cvReleaseImage(&pPyrA);
	cvReleaseImage(&pPyrB);
}

