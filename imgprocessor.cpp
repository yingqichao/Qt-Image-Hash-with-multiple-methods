//第一种指纹：原始 hash 法。
//输入图像，如果是 RGB 图像则转换为灰度图像
//缩放，将图像缩放为 8x8 大小的图像
//求均值，计算此 8x8 图像所有像素的均值
//量化，比较 8x8 图像中每一个像素像素与均值大小关系，如果大于均值，则指纹上该位对应的值为1；否则，为0。
//组合，将这64bit的0和1组成这幅图像的指纹。

#include "imgprocessor.h"
#include <QFileDialog>
#include <QFile>
#include <QTextStream>
#include <QPainter>
#include <QColorDialog>
#include <QColor>
#include <QTextList>
#include <QSplitter>
#include <QTime>
#include <QApplication>
#include <QPushButton>
#include <QHBoxLayout>
#include <QVBoxLayout>
#include <QLabel>
#include <QMessageBox>
#include <QSignalMapper>
#include "inputdlg.h"
#include "settingdlg.h"
#include <fstream>
#include <map>
#include <bitset>
#include <cstring>
#include <QScrollArea>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>



//#define MAX_IMAGE_SIZE 26
#define PI 3.1415926
#define hashLength 6400

//#define fingerSize = 8;
using namespace std;
using namespace cv;


struct MARK_MATCH {
//    MARK_MATCH(int a,string b)
//        {
//            this->mark=a;
//            this->address=b;
//        };
    int mark = 0;
    int totalDistance = 0;
    double ssim = 0;
    string address;
};

map<string,string> obm;

int cmp(const void *a, const void *b) {
    int cmpA = ((MARK_MATCH *)a)->mark - ((MARK_MATCH *)b)->mark;
    int cmpB = ((MARK_MATCH *)a)->totalDistance - ((MARK_MATCH *)b)->totalDistance;
    return (cmpA==0)?cmpB:-cmpA;
}

int cmp_Simple(const void *a, const void *b) {
    return ((MARK_MATCH *)a)->mark - ((MARK_MATCH *)b)->mark;
}

int cmp_ssim(const void *a, const void *b) {
    return ((MARK_MATCH *)b)->ssim - ((MARK_MATCH *)a)->ssim;
}

void sleep(unsigned int msec){
    QTime reachTime = QTime:: currentTime().addMSecs(msec);
    while(QTime::currentTime()<reachTime){
        QCoreApplication::processEvents(QEventLoop::AllEvents,100);
    }
}

string bitTohex(bitset<hashLength> &target){
    string str;
    for(int i = 0; i < hashLength; i=i+4){
        int sum = 0;
        string s;
        sum += target[i] + (target[i+1]<<1) + (target[i+2]<<2) + (target[i+3]<<3);
        stringstream ss;
        ss << hex <<sum;    
        ss >> s;
        str += s;
    }
    return str;
}

string bitTohex(bitset<64> &target){
    string str;
    for(int i = 0; i < 64; i=i+4){
        int sum = 0;
        string s;
        sum += target[i] + (target[i+1]<<1) + (target[i+2]<<2) + (target[i+3]<<3);
        stringstream ss;
        ss << hex <<sum;
        ss >> s;
        str += s;
    }
    return str;
}

string bitTohex(bitset<3240> &target){
    string str;
    for(int i = 0; i < 3240; i=i+4){
        int sum = 0;
        string s;
        sum += target[i] + (target[i+1]<<1) + (target[i+2]<<2) + (target[i+3]<<3);
        stringstream ss;
        ss << hex <<sum;
        ss >> s;
        str += s;
    }
    return str;
}

string ImagePHash(string input, cv::Mat image, bool wr)
{
    bitset<64> hash = 0; // 用于保存hash值
    cv::Mat imageGray; // 转换后的灰度图像
    cv::Mat imageFinger; // 缩放后的8x8的指纹图像
    int fingerSize = 8; // 指纹图像的大小
    int dctSize = 32; // dct变换的尺寸大小

    if (3 == image.channels()) // rgb -> gray
    {
        cv::cvtColor(image, imageGray, CV_RGB2GRAY);
    }
    else
    {
        imageGray = image.clone();
    }

    cv::resize(imageGray, imageFinger, cv::Size(dctSize, dctSize)); // 图像缩放
    imageFinger.convertTo(imageFinger, CV_32F); // 转换为浮点型
    cv::dct(imageFinger, imageFinger); // 对缩放后的图像进行dct变换
    imageFinger = imageFinger(cv::Rect(0, 0, fingerSize, fingerSize)); // 取低频区域

    /* 对dct变换后的系数取对数 */
    for (int i = 0; i < fingerSize; i++)
    {
        float* data = imageFinger.ptr<float>(i);
        for (int j = 0; j < fingerSize; j++)
        {
            data[j] = logf(abs(data[j]));
        }
    }

    cv::Scalar imageMean = cv::mean(imageFinger); // 求均值

    /* 计算图像哈希指纹，小于等于均值为0，大于为1 */
    for (int i = 0; i < fingerSize; i++)
    {
        float* data = imageFinger.ptr<float>(i);
        for (int j = 0; j < fingerSize; j++)
        {
            if (data[j] > imageMean[0])
            {
//                hash = (hash << 1) + 1;
                hash[fingerSize*i+j] = 1;
            }
            else
            {
//                hash = hash << 1;
                hash[fingerSize*i+j] = 0;
            }
        }
    }

    string ImagePHash = bitTohex(hash);

    if(wr)  {
        obm.insert(make_pair(input, ImagePHash));
    }

    return ImagePHash;
}

Mat ImgProcessor::matchers(string input, Mat img_1,bool wr){
    //由于hash值与图像是一一对应存储的，因此不需要obm
    std::vector<KeyPoint> keypoints_1;

    //创建两张图像的描述子，类型是Mat类型
    Mat descriptors_1;

    //创建一个ORB类型指针orb，ORB类是继承自Feature2D类
    //class CV_EXPORTS_W ORB : public Feature2D
    //这里看一下create()源码：参数较多，不介绍。
    //creat()方法所有参数都有默认值，返回static　Ptr<ORB>类型。
    /*
    CV_WRAP static Ptr<ORB> create(int nfeatures=500,
                                   float scaleFactor=1.2f,
                                   int nlevels=8,
                                   int edgeThreshold=31,
                                   int firstLevel=0,
                                   int WTA_K=2,
                                   int scoreType=ORB::HARRIS_SCORE,
                                   int patchSize=31,
                                   int fastThreshold=20);
    */
    //所以这里的语句就是创建一个Ptr<ORB>类型的orb，用于接收ORB类中create()函数的返回值
    Ptr<ORB> orb = ORB::create();


    //第一步：检测Oriented FAST角点位置.
    //detect是Feature2D中的方法，orb是子类指针，可以调用
    //看一下detect()方法的原型参数：需要检测的图像，关键点数组，第三个参数为默认值
    /*
    CV_WRAP virtual void detect( InputArray image,
                                 CV_OUT std::vector<KeyPoint>& keypoints,
                                 InputArray mask=noArray() );
    */
    orb->detect(img_1, keypoints_1);


    //第二步：根据角点位置计算BRIEF描述子
    //compute是Feature2D中的方法，orb是子类指针，可以调用
    //看一下compute()原型参数：图像，图像的关键点数组，Mat类型的描述子
    /*
    CV_WRAP virtual void compute( InputArray image,
                                  CV_OUT CV_IN_OUT std::vector<KeyPoint>& keypoints,
                                  OutputArray descriptors );
    */
    orb->compute(img_1, keypoints_1, descriptors_1);

    Mat mostImpactfulPoints = descriptors_1.rowRange(0,100).clone();
    if(wr){
        FileStorage fs(input+".xml", FileStorage::WRITE);
        fs<<"mostImpactfulPoints"<<mostImpactfulPoints;
        fs.release();
    }
    return mostImpactfulPoints;

}

Scalar getMSSIM(Mat  inputimage1, Mat inputimage2)
{
    Mat i1 = inputimage1;
    Mat i2 = inputimage2;
    const double C1 = 6.5025, C2 = 58.5225;
    int d = CV_32F;
    Mat I1, I2;
    i1.convertTo(I1, d);
    i2.convertTo(I2, d);
    Mat I2_2 = I2.mul(I2);
    Mat I1_2 = I1.mul(I1);
    Mat I1_I2 = I1.mul(I2);
    Mat mu1, mu2;
    GaussianBlur(I1, mu1, Size(11, 11), 1.5);
    GaussianBlur(I2, mu2, Size(11, 11), 1.5);
    Mat mu1_2 = mu1.mul(mu1);
    Mat mu2_2 = mu2.mul(mu2);
    Mat mu1_mu2 = mu1.mul(mu2);
    Mat sigma1_2, sigma2_2, sigma12;
    GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
    sigma1_2 -= mu1_2;
    GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
    sigma2_2 -= mu2_2;
    GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
    sigma12 -= mu1_mu2;
    Mat t1, t2, t3;
    t1 = 2 * mu1_mu2 + C1;
    t2 = 2 * sigma12 + C2;
    t3 = t1.mul(t2);
    t1 = mu1_2 + mu2_2 + C1;
    t2 = sigma1_2 + sigma2_2 + C2;
    t1 = t1.mul(t2);
    Mat ssim_map;
    divide(t3, t1, ssim_map);
    Scalar mssim = mean(ssim_map);
    return mssim;
}

string ImgProcessor::histogram(string input, Mat matSrc2,bool wr){
        cv::resize(matSrc2, matSrc2, cv::Size(300,300), 0, 0, cv::INTER_CUBIC);
        Mat image;
        cv::cvtColor(matSrc2, image, CV_BGR2HSV);//CV_BGR2HSV
        bitset<3240> bt;
        int pointer = 0;
        for(int i=0;i<9;i++){
            Rect rect((i%3)*100,(i/3)*100,100,100);
            vector<Mat> mv(3);

            cv::split(image(rect),mv);

            //创建直方图
            int arr_size_h = 255;                 //定义一个变量用于表示直方图行宽
            float hranges_arr_h[] = { 0, 180 };       //图像方块范围数组
            float *phranges_arr_h = hranges_arr_h;      //cvCreateHist参数是一个二级指针，所以要用指针指向数组然后传参
            CvHistogram *hist_h = cvCreateHist(1, &arr_size_h, CV_HIST_ARRAY, &phranges_arr_h, 1);    //创建一个一维的直方图，行宽为255，多维密集数组，方块范围为0-180，bin均化

            MatND hist;
            int bins = 255;
            int hist_size[] = {bins};
            float range[] = { 0, 180 };
            const float* ranges[] = { range};
            int channels[] = {0};
            calcHist(&mv[0], 1, channels, Mat(), // do not use mask
                     hist, 1, hist_size, ranges,
                     true, // the histogram is uniform
                     false);

                float p[36];

                for(int j=0;j<180;j++){
                    if(j%5==0)  p[j/5] = 0;
                    p[j/5] += hist.at<float>(j)/10000.0;
                }
                for(int j=0;j<36;j++){
                    float tmp = p[j];
                    for(int k=0;k<10;k++){
                        tmp *= 2;
                        if(tmp>=1){
                            bt[pointer] = 1;tmp -= 1;
                        }else{
                            bt[pointer] = 0;
                        }
                        pointer++;
                    }
                }
        }

        string ImageHash = bitTohex(bt);

        if(wr)  {
            obm.insert(make_pair(input, ImageHash));
        }

        return ImageHash;
}


string ImgProcessor::otsu(string input, Mat matSrc2,bool wr){

    Mat gray;
    cv::resize(matSrc2, matSrc2, cv::Size(80, 80), 0, 0, cv::INTER_CUBIC);
    cv::cvtColor(matSrc2, gray, CV_BGR2GRAY);
    Mat dst;
    cv::threshold(gray, dst, 0, 255, CV_THRESH_OTSU);
//    imshow("dst", dst);
//    cv::waitKey();
    bitset<hashLength> phash;int k = 0;
    int iAvg2 = 0, i = 0, j = 0, tmp = 0, tmp1 = 0, iDiffNum = 0;


    for (i = 0; i < 80; i++)
    {
        uchar* data2 = dst.ptr<uchar>(i);
        tmp = i * 80;
        for (j = 0; j < 80; j++)
        {
            phash[k] = data2[j]==0?0:1 ;
            k++;
        }
    }

    string ImageHash = bitTohex(phash);

    if(wr)  {
        obm.insert(make_pair(input, ImageHash));
    }

    return ImageHash;
}

string ImgProcessor::mark(string input, Mat matSrc2,bool wr) {
    bitset<hashLength> phash;int arr2[6400];
    int iAvg2 = 0, i = 0, j = 0, tmp = 0, tmp1 = 0, iDiffNum = 0;

    cv::Mat matDst1, matDst2;
    cv::resize(matSrc2, matDst2, cv::Size(80, 80), 0, 0, cv::INTER_CUBIC);
    cv::cvtColor(matDst2, matDst2, COLOR_BGR2GRAY);

    for (i = 0; i < 80; i++)

    {

        uchar* data2 = matDst2.ptr<uchar>(i);
        tmp = i * 80;
        for (j = 0; j < 80; j++)
        {
            tmp1 = tmp + j;

            arr2[tmp1] = data2[j] ;

            iAvg2 += arr2[tmp1];
        }
    }


    iAvg2 /= 6400;

    for (i = 0; i < 6400; i++)
    {

        phash[i] = (arr2[i] >= iAvg2) ? 1 : 0;
    }

    string ImageHash = bitTohex(phash);

    if(wr)  {
        obm.insert(make_pair(input, ImageHash));
    }

    return ImageHash;
}

map<string,string> read(string foldpath){
    map<string,string> dict;
    fstream     f(foldpath+ "/" + "hashcode.txt ");

    string  line;
        while(getline(f,line))
        {
            string key = line;
            getline(f,line);
            dict.insert(make_pair(key,line));
        }
    f.close();
    return dict;
}

void saveFile(string foldpath){
    map<string,string>::iterator iter = obm.begin();

    ofstream   ofresult( foldpath+ "/" + "hashcode.txt ");
    if (!ofresult.is_open()) {
       cout << "File is open fail!" << endl;
    }
    while (iter != obm.end()) {
       ofresult <<iter->first<< endl;
       ofresult <<iter->second<< endl;
       iter++;
    }
    ofresult.close();
}

bitset<hashLength> stringToHex(string hashCode){
    int sum  =0;bitset<hashLength> target;
    for(int i=0;i<hashLength/4;i++){
        char high = hashCode[i];
        if(high>='0' && high<='9')
                sum = high-'0';
        else if(high>='a' && high<='f')
                sum = high - 'a' + 10;
        else if(high>='A' && high<='F')
                sum = high - 'A' + 10;
        target[i*4] = (sum/8==0)?0:1;
        sum -=   target[i*4]*8;
        target[i*4+1] = (sum/4==0)?0:1;
        sum -=   target[i*4+1]*4;
        target[i*4+2] = (sum/2==0)?0:1;
        sum -=   target[i*4+2]*2;
        target[i*4+3] = (sum==0)?0:1;
    }

    return target;
}

bitset<3240> stringToHex_3240(string hashCode){
    int sum  =0;bitset<3240> target;
    for(int i=0;i<3240/4;i++){
        char high = hashCode[i];
        if(high>='0' && high<='9')
                sum = high-'0';
        else if(high>='a' && high<='f')
                sum = high - 'a' + 10;
        else if(high>='A' && high<='F')
                sum = high - 'A' + 10;
        target[i*4] = (sum/8==0)?0:1;
        sum -=   target[i*4]*8;
        target[i*4+1] = (sum/4==0)?0:1;
        sum -=   target[i*4+1]*4;
        target[i*4+2] = (sum/2==0)?0:1;
        sum -=   target[i*4+2]*2;
        target[i*4+3] = (sum==0)?0:1;
    }

    return target;
}

bitset<64> stringToHex_64(string hashCode){
    int sum  =0;bitset<64> target;
    for(int i=0;i<64/4;i++){
        char high = hashCode[i];
        if(high>='0' && high<='9')
                sum = high-'0';
        else if(high>='a' && high<='f')
                sum = high - 'a' + 10;
        else if(high>='A' && high<='F')
                sum = high - 'A' + 10;
        target[i*4] = (sum/8==0)?0:1;
        sum -=   target[i*4]*8;
        target[i*4+1] = (sum/4==0)?0:1;
        sum -=   target[i*4+1]*4;
        target[i*4+2] = (sum/2==0)?0:1;
        sum -=   target[i*4+2]*2;
        target[i*4+3] = (sum==0)?0:1;
    }

    return target;
}


void ImgProcessor::hashTest(){
    QTextCodec *code = QTextCodec::codecForName("GB2312");
    string name = code->fromUnicode(filename).data();
    QString newFileName = QString::fromStdString(name);
    string name1 = code->fromUnicode(foldname).data();
    QString newFoldName = QString::fromStdString(name1);
//    if(!changed){
//        QMessageBox::information(this,
//                                 tr("Tips"),
//                                 codec->toUnicode("已经是最新结果！"));
//        return;
//    }
    if(newFileName.compare(QString::fromStdString("* Not Selected *")) == 0){
        QMessageBox::information(this,
                                 tr("Error"),
                                 codec->toUnicode("请先选择需要检测的图像！"));
        return;
    }
    else if(newFoldName.compare(QString::fromStdString("* Not Selected *")) == 0){
        QMessageBox::information(this,
                                 tr("Error"),
                                 codec->toUnicode("请先选择需要检测的文件夹！"));
        return;
    }

    QScrollArea *s = new QScrollArea(0);
    QWidget *w = new QWidget(s);
    QGridLayout *pLayout = new QGridLayout(w);
    QGridLayout *bottomLayout = new QGridLayout(0);
    QWidget *dockWidget = new QWidget(0);
    if(method==1 || method==2){
        QLabel *picName = new QLabel;
        picName->setText(codec->toUnicode("图像名"));
        picName->setFont(QFont("Timers" , 10, QFont::Bold));
        QLabel *hash = new QLabel;
        hash->setText(codec->toUnicode("哈希偏差"));
        hash->setFont(QFont("Timers" , 10, QFont::Bold));
        pLayout->addWidget(picName, 0, 0, 1, 1);
        pLayout->addWidget(hash, 0, 1, 1, 1);

        changed = false;
        string targetHash;
        if(method==1)   targetHash = mark(name,imread(name),false);
        else    targetHash = otsu(name,imread(name),false);
        map<string,string> dict = read(newFoldName.toStdString());
        int len = dict.size();
        MARK_MATCH *mark = new MARK_MATCH[len];
        bitset<hashLength> targetBitSet = stringToHex(targetHash);
        map<string,string>::iterator ite = dict.begin();
        for(int i=0;i<dict.size();i++){
            bitset<hashLength> cmpBitSet = stringToHex(ite->second);
            int iDiffNum = 0;

            for (int j = 0; j < hashLength; j++)
                if (targetBitSet[j] != cmpBitSet[j])
                    ++iDiffNum;

            mark[i].mark = iDiffNum;
            mark[i].address = ite->first;
            ite++;
        }
        qsort(mark, dict.size(), sizeof(MARK_MATCH), cmp_Simple);

        matches.clear();
        ClickedLabel *match;
        for(int i=0;i<9;i++){
                match = new ClickedLabel;
                match->setText("..");
                matches.push_back(match);
            }

        QSignalMapper *signalMapper = new QSignalMapper(this);
        int len1 = dict.size();
        QLabel *pn = new QLabel[len1];QLabel *hs = new QLabel[len1];
        for(int i=0;i<dict.size();i++){
            QString fname = QString::fromStdString(mark[i].address);
            if(i<9 && mark[i].mark<=threshold){
                QPixmap *img=new QPixmap;
                img->load(fname);
                *img = img->scaled(120, 120);
                list<ClickedLabel*>::iterator iter=matches.begin();
                for(int ix=0; ix<i; ++ix) {
                    ++iter;
                }
                (*iter)->setPixmap(*img);
                (*iter)->setRoute(mark[i].address);

                bottomLayout->addWidget(*iter, i/3, i%3, 1, 1);
                connect(*iter, SIGNAL(clicked()), signalMapper, SLOT(map()));
                signalMapper->setMapping(*iter, i);
            }
            pn[i].setText(QString::fromStdString("..."+mark[i].address).right(30));
            pn[i].setFont(QFont("Timers" , 8));
            hs[i].setText(QString::number(mark[i].mark));
            hs[i].setFont(QFont("Timers" , 8));
            pLayout->addWidget(&pn[i], i+1, 0, 1, 1);
            pLayout->addWidget(&hs[i], i+1, 1, 1, 1);

        }
        connect(signalMapper, SIGNAL(mapped(int)), this, SLOT(onclick(int)));

    }else if(method==0){

        QLabel *picName = new QLabel;
        picName->setText(codec->toUnicode("图像名"));
        picName->setFont(QFont("Timers" , 10, QFont::Bold));
        QLabel *hash = new QLabel;
        hash->setText(codec->toUnicode("匹配特征点"));
        hash->setFont(QFont("Timers" , 10, QFont::Bold));
        QLabel *diff = new QLabel;
        diff->setText(codec->toUnicode("平均汉明距离"));
        diff->setFont(QFont("Timers" , 10, QFont::Bold));
        pLayout->addWidget(picName, 0, 0, 1, 1);
        pLayout->addWidget(hash, 0, 1, 1, 1);
        pLayout->addWidget(diff, 0, 2, 1, 1);

        changed = false;
        Mat descriptionA = matchers(name,imread(name),false);
        //遍历所有的xml
        cv::String pattern = name1 + "/*.xml";

        vector<cv::String> fn;
        glob(pattern, fn, false);
        int len = fn.size();
        MARK_MATCH *mark = new MARK_MATCH[len];

        for (int ind = 0;ind < fn.size();ind++) {
            FileStorage fs(fn[ind], FileStorage::READ);
            Mat descriptionB;
            fs["mostImpactfulPoints"] >> descriptionB;

            BFMatcher matcher (NORM_HAMMING);

            vector<DMatch> Dmatch;

            matcher.match(descriptionA, descriptionB, Dmatch);

            int count = 0;int totalD = 0;
            for (int j = 0; j < descriptionA.rows; ++j)
            {
                if (Dmatch[j].distance <= 30){//max(3*min_dist, 30.0)
                    count++;
                    totalD += Dmatch[j].distance;
                }
            }
            mark[ind].mark = count;
            mark[ind].totalDistance = totalD;
            mark[ind].address = fn[ind].substr(0,fn[ind].length()-4);
    //        mark++;
    //        fs.release();

        }

        qsort(mark, len, sizeof(MARK_MATCH), cmp);

        matches.clear();
        ClickedLabel *match;
        for(int i=0;i<9;i++){
                match = new ClickedLabel;
                matches.push_back(match);
            }

        QSignalMapper *signalMapper = new QSignalMapper(this);

        QLabel *pn = new QLabel[len];QLabel *hs = new QLabel[len];QLabel *df = new QLabel[len];
        for(int i=0;i<len;i++){
//            QString fname = QString::fromStdString(mark[i].address);
            QString fname = QString::fromLocal8Bit(mark[i].address.data());
            if(i<9 && mark[i].mark>=threshold){
                QPixmap *img=new QPixmap;
                img->load(fname);
                *img = img->scaled(120, 120);
                list<ClickedLabel*>::iterator iter=matches.begin();
                for(int ix=0; ix<i; ++ix) {
                    ++iter;
                }
                (*iter)->setPixmap(*img);
                (*iter)->setRoute(mark[i].address);

                bottomLayout->addWidget(*iter, i/3, i%3, 1, 1);
                connect(*iter, SIGNAL(clicked()), signalMapper, SLOT(map()));
                signalMapper->setMapping(*iter, i);
            }
    //        pn[i] = new QLabel;hs[i] = new QLabel;
            pn[i].setText(fname.right(30));
            pn[i].setFont(QFont("Timers" , 8));
            hs[i].setText(QString::number(mark[i].mark));
            hs[i].setFont(QFont("Timers" , 8));
            df[i].setText(QString::number(mark[i].totalDistance));
            df[i].setFont(QFont("Timers" , 8));
            pLayout->addWidget(&pn[i], i+1, 0, 1, 1);
            pLayout->addWidget(&hs[i], i+1, 1, 1, 1);
            pLayout->addWidget(&df[i], i+1, 2, 1, 1);

        }
        connect(signalMapper, SIGNAL(mapped(int)), this, SLOT(onclick(int)));

    }else if(method==3){

        QLabel *picName = new QLabel;
        picName->setText(codec->toUnicode("图像名"));
        picName->setFont(QFont("Timers" , 10, QFont::Bold));
        QLabel *hash = new QLabel;
        hash->setText(codec->toUnicode("SSIM"));
        hash->setFont(QFont("Timers" , 10, QFont::Bold));

        pLayout->addWidget(picName, 0, 0, 1, 1);
        pLayout->addWidget(hash, 0, 1, 1, 1);

        changed = false;
        Mat A = imread(name);
        cv::resize(A, A, cv::Size(500,500));

        string format[3]={"/*.jpg","/*.png","/*.tif"};
        cv::String pattern1 = name1 + format[0];
        vector<cv::String> fn1;glob(pattern1, fn1, false);
        pattern1 = name1 + format[1];
        vector<cv::String> fn2;glob(pattern1, fn2, false);
        pattern1 = name1 + format[2];
        vector<cv::String> fn3;glob(pattern1, fn3, false);
        int len = fn1.size()+fn2.size()+fn3.size();
        MARK_MATCH *mark = new MARK_MATCH[len];

        int index = 0;
        for (int ind = 0;ind < fn1.size();ind++) {
            Mat image = imread(fn1[ind]);
            cv::resize(image, image, cv::Size(500,500));
            Scalar SSIM = getMSSIM(A,image);
            mark[index].ssim = (SSIM.val[2]+SSIM.val[1]+SSIM.val[0])/3*100;
            mark[index].address = fn1[ind];
            index++;
        }
        for (int ind = 0;ind < fn2.size();ind++) {
            Mat image = imread(fn2[ind]);
            Scalar SSIM = getMSSIM(A,image);
            mark[index].ssim = (SSIM.val[2]+SSIM.val[1]+SSIM.val[0])/3*100;
            mark[index].address = fn2[ind];
            index++;
        }
        for (int ind = 0;ind < fn3.size();ind++) {
            Mat image = imread(fn3[ind]);
            Scalar SSIM = getMSSIM(A,image);
            mark[index].ssim = (SSIM.val[2]+SSIM.val[1]+SSIM.val[0])/3*100;
            mark[index].address = fn3[ind];
            index++;
        }

        qsort(mark, len, sizeof(MARK_MATCH), cmp_ssim);

        matches.clear();
        ClickedLabel *match;
        for(int i=0;i<9;i++){
                match = new ClickedLabel;
                matches.push_back(match);
            }

        QSignalMapper *signalMapper = new QSignalMapper(this);

        QLabel *pn = new QLabel[len];QLabel *hs = new QLabel[len];QLabel *df = new QLabel[len];
        for(int i=0;i<len;i++){
//            QString fname = QString::fromStdString(mark[i].address);
            QString fname = QString::fromLocal8Bit(mark[i].address.data());
            if(i<9 && mark[i].ssim>=threshold){
                QPixmap *img=new QPixmap;
                img->load(fname);
                *img = img->scaled(120, 120);
                list<ClickedLabel*>::iterator iter=matches.begin();
                for(int ix=0; ix<i; ++ix) {
                    ++iter;
                }
                (*iter)->setPixmap(*img);
                (*iter)->setRoute(mark[i].address);

                bottomLayout->addWidget(*iter, i/3, i%3, 1, 1);
                connect(*iter, SIGNAL(clicked()), signalMapper, SLOT(map()));
                signalMapper->setMapping(*iter, i);
            }
    //        pn[i] = new QLabel;hs[i] = new QLabel;
            pn[i].setText(fname.right(30));
            pn[i].setFont(QFont("Timers" , 8));
            hs[i].setText(QString::number(mark[i].ssim));
            hs[i].setFont(QFont("Timers" , 8));
            pLayout->addWidget(&pn[i], i+1, 0, 1, 1);
            pLayout->addWidget(&hs[i], i+1, 1, 1, 1);
        }
        connect(signalMapper, SIGNAL(mapped(int)), this, SLOT(onclick(int)));

    }else if(method==4){
        QLabel *picName = new QLabel;
        picName->setText(codec->toUnicode("图像名"));
        picName->setFont(QFont("Timers" , 10, QFont::Bold));
        QLabel *hash = new QLabel;
        hash->setText(codec->toUnicode("Distance"));
        hash->setFont(QFont("Timers" , 10, QFont::Bold));
        pLayout->addWidget(picName, 0, 0, 1, 1);
        pLayout->addWidget(hash, 0, 1, 1, 1);

        changed = false;
        string targetHash;
        targetHash = histogram(name,imread(name),false);
        map<string,string> dict = read(newFoldName.toStdString());
        int len = dict.size();
        MARK_MATCH *mark = new MARK_MATCH[len];
        bitset<3240> targetBitSet = stringToHex_3240(targetHash);
        map<string,string>::iterator ite = dict.begin();
        for(int i=0;i<dict.size();i++){
            bitset<3240> cmpBitSet = stringToHex_3240(ite->second);
            int similar = 0;int k=0;double distance = 0;
            for(int block=0;block<9;block++){
                double blockDiff = 0;
                for(int coeff=0;coeff<36;coeff++){
                    double coeffA = 0;double coeffB = 0;
                    for(int ind=0;ind<10;ind++){
                      coeffA += (targetBitSet[k]==1)?pow(2,-ind-1):0;
                      coeffB += (cmpBitSet[k]==1)?pow(2,-ind-1):0;
                      k++;
                    }
                    blockDiff = pow(coeffA-coeffB,2);
                }
                distance += blockDiff;
            }


            mark[i].mark = (int)(distance*10000*10000);
            mark[i].address = ite->first;
            ite++;
        }
        qsort(mark, dict.size(), sizeof(MARK_MATCH), cmp_Simple);

        matches.clear();
        ClickedLabel *match;
        for(int i=0;i<9;i++){
                match = new ClickedLabel;
                match->setText("..");
                matches.push_back(match);
            }

        QSignalMapper *signalMapper = new QSignalMapper(this);
        int len1 = dict.size();
        QLabel *pn = new QLabel[len1];QLabel *hs = new QLabel[len1];
        for(int i=0;i<dict.size();i++){
            QString fname = QString::fromStdString(mark[i].address);
            if(i<9){
                QPixmap *img=new QPixmap;
                img->load(fname);
                *img = img->scaled(120, 120);
                list<ClickedLabel*>::iterator iter=matches.begin();
                for(int ix=0; ix<i; ++ix) {
                    ++iter;
                }
                (*iter)->setPixmap(*img);
                (*iter)->setRoute(mark[i].address);

                bottomLayout->addWidget(*iter, i/3, i%3, 1, 1);
                connect(*iter, SIGNAL(clicked()), signalMapper, SLOT(map()));
                signalMapper->setMapping(*iter, i);
            }
            pn[i].setText(QString::fromStdString("..."+mark[i].address).right(30));
            pn[i].setFont(QFont("Timers" , 8));
            hs[i].setText(QString::number(mark[i].mark));
            hs[i].setFont(QFont("Timers" , 8));
            pLayout->addWidget(&pn[i], i+1, 0, 1, 1);
            pLayout->addWidget(&hs[i], i+1, 1, 1, 1);

        }
        connect(signalMapper, SIGNAL(mapped(int)), this, SLOT(onclick(int)));

    }

    bottomLayout->setHorizontalSpacing(5);
    bottomLayout->setVerticalSpacing(5);
    bottomLayout->setContentsMargins(0,0,0,0);

    dockWidget->setLayout(bottomLayout);

    dock->setWidget(dockWidget);

    w->setLayout(pLayout);
    s->setWidget(w);
    code_dock->setWidget(s);

    return;

}

void ImgProcessor::generateHashCode(){
    if(method==3){
        QMessageBox::information(0,codec->toUnicode("提示"),"This method does not require calculating ImageHash!");
        return;
    }
    obm.clear();
    string format[3]={"/*.jpg","/*.png","/*.tif"};
    for(int i=0;i<3;i++){
        QProgressDialog *progressDlg=new QProgressDialog(this);
        progressDlg->setWindowModality(Qt::WindowModal);
        progressDlg->setMinimumDuration(0);
        progressDlg->setAttribute(Qt::WA_DeleteOnClose, true);
        progressDlg->setWindowTitle(codec->toUnicode("生成哈希"));
        progressDlg->setMinimumSize(480,120);

        QTextCodec *code = QTextCodec::codecForName("GB2312");
        string name = code->fromUnicode(foldname).data();
        cv::String pattern1 = name + format[i];

        vector<cv::String> fn1;
        glob(pattern1, fn1, false);
//        vector<Mat> images1 = read_images_in_folder(fn1);
        progressDlg->setRange(0,fn1.size());

        for (int ind = 0;ind < fn1.size();ind++) {
            try{
                Mat image = imread(fn1[ind]);
                progressDlg->setLabelText("Generating From Image "+QString::number(ind)+" / "+QString::number(fn1.size()));
                if(method==1)
                        mark(fn1[ind],image,true);
                else if(method==0)
                        matchers(fn1[ind],image,true);
                else if(method==2)
                        otsu(fn1[ind],image,true);
                else if(method==4)
                        histogram(fn1[ind],image,true);
            }
            catch(cv::Exception& e){
                QString errorMsg("There is an error in the process!\n\n");
                errorMsg = errorMsg.append(QString::fromStdString(e.msg));
                errorMsg = errorMsg.append("\nDon't worry, you can either press OK to skip this error or send a screenshot of this Message to the author for help.\n");
                QMessageBox::information(0,"Error Report",errorMsg);
            }
            progressDlg->setValue(ind);
        }
        progressDlg->close();
    }

    if(method==1 || method==2 || method==4)
        saveFile(foldname.toStdString());

    QMessageBox::information(0,codec->toUnicode("提示"),"Finished Calculating Image Hash!");
    return;
}


ImgProcessor::ImgProcessor(QWidget *parent)
    : QMainWindow(parent)
{
    codec = QTextCodec::codecForName("GBK");
    QSplitter *splitterMain =new QSplitter(Qt::Vertical,this);
    QPalette pal(splitterMain->palette());
    setDockNestingEnabled(true);

    pal.setColor(QPalette::Background, Qt::gray); 
    splitterMain->setAutoFillBackground(true);
    splitterMain->setPalette(pal);
 
    setCentralWidget(splitterMain);
    setMinimumSize(1200,720);
    sleep(2000);
    imageLabel =new QLabel(splitterMain);
    imageLabel->setAlignment(Qt::AlignCenter);
    imageLabel->setText(codec->toUnicode("使用指南：\n1.选择图像集数据库与待测图像.\n2.按F5键为数据库生成文件夹下所有图像的哈希值.\n3.按F6键得到待测图像与数据库中相似度最高的9张图片.\n\n帮助：\n1.不同的哈希算法得到的图像相似度可能不同.\n2.软件版本：v1.0 Latest Release 2018/9/17."));
    QPalette p;
    p.setColor(QPalette::WindowText,Qt::darkGray);
    imageLabel->setFont(QFont("Timers" , 10));
    imageLabel->setPalette(p);

//    QSplitter *splitterRight =new QSplitter(Qt::Vertical,splitterMain);
//    splitterRight->setOpaqueResize(false);						//(e)

//    mywid1 = new QWidget(splitterRight);
//    QGridLayout *pLayout = new QGridLayout(mywid1);

//    pName = new QLabel;
//    pName->setText(codec->toUnicode("数据库路径"));
//    pName->setFont(QFont("Timers" , 10, QFont::Bold));
//    fName = new QLabel;
//    fName->setText(codec->toUnicode("图像路径 ："));
//    fName->setFont(QFont("Timers" , 10, QFont::Bold));
//    mtd = new QLabel;
//    mtd->setText(codec->toUnicode("使用方法 ："));
//    mtd->setFont(QFont("Timers" , 10, QFont::Bold));
//    pLayout->addWidget(pName, 0, 0, 1, 1);
//    pLayout->addWidget(fName, 1, 0, 1, 1);
//    pLayout->addWidget(mtd, 2, 0, 1, 1);

//    splitterMain->setStretchFactor(10,1);
//    splitterRight->setWindowTitle(codec->toUnicode(("工作路径")));


    setWindowTitle(tr("Image Hash v1.0"));
//    showWidget =new ShowWidget(this);
//    setCentralWidget(showWidget);


    createActions();
    createMenus();
    createToolBars();
    if(img.load("tianti1.jpeg"))
        {
            img = img.scaled(512,512);
//            imageLabel->setPixmap(img);
        }

    //dataset dock
//    dataset_dock=new QDockWidget(codec->toUnicode("数据库"),this);
//    QWidget *datasetWid = new QWidget();
//    QVBoxLayout *vb = new QVBoxLayout(dataset_dock);
//    QLabel *datasetDefault = new QLabel;
//    datasetDefault->setText(codec->toUnicode("选择数据库(CTRL+O)"));
//    datasetDefault->setAlignment(Qt::AlignCenter);
//    QPalette pe;
//    pe.setColor(QPalette::WindowText,Qt::gray);
//    datasetDefault->setFont(QFont("Timers" , 10));
//    datasetDefault->setPalette(pe);
//    vb->addWidget(datasetDefault);
//    datasetWid->setLayout(vb);
//    dataset_dock->setMinimumWidth(300);
//    dataset_dock->setMaximumHeight(480);
//    dataset_dock->setWidget(datasetWid);
//    dataset_dock->setFeatures(QDockWidget::AllDockWidgetFeatures);
//    addDockWidget(Qt::LeftDockWidgetArea,dataset_dock);
//    m_docks.append(dataset_dock);
    //hostImage dock
    hostImage_dock=new QDockWidget(codec->toUnicode("待测图像"),this);
    QWidget *hostImageWid = new QWidget();
    QVBoxLayout *vb1 = new QVBoxLayout(hostImage_dock);
    QLabel *hostImageDefault = new QLabel;
    hostImageDefault->setText(codec->toUnicode("选择待测图像(CTRL+N)"));
    hostImageDefault->setAlignment(Qt::AlignCenter);
    QPalette pe2;
    pe2.setColor(QPalette::WindowText,Qt::gray);
    hostImageDefault->setFont(QFont("Timers" , 10));
    hostImageDefault->setPalette(pe2);
    vb1->addWidget(hostImageDefault);
    hostImageWid->setLayout(vb1);
    hostImage_dock->setWidget(hostImageWid);
    hostImage_dock->setMaximumHeight(480);
    hostImage_dock->setMinimumWidth(300);
    hostImage_dock->setFeatures(QDockWidget::AllDockWidgetFeatures);
    addDockWidget(Qt::LeftDockWidgetArea,hostImage_dock);

    //result dock
    dock=new QDockWidget(codec->toUnicode("匹配结果"),this);
    dock->setMinimumHeight(120);
    dock->setFeatures(QDockWidget::AllDockWidgetFeatures);
    QWidget *dockWidget = new QWidget();
    QVBoxLayout *dw = new QVBoxLayout(dock);
    QLabel *resultDefault = new QLabel;
    resultDefault->setText(codec->toUnicode("匹配图像显示区域"));
    resultDefault->setAlignment(Qt::AlignCenter);
    QPalette pe1;
    pe1.setColor(QPalette::WindowText,Qt::gray);
    resultDefault->setFont(QFont("Timers" , 10));
    resultDefault->setPalette(pe1);
    dw->addWidget(resultDefault);
    dockWidget->setLayout(dw);
    dock->setWidget(dockWidget);
//    addDockWidget(Qt::BottomDockWidgetArea,dock);
    m_docks.append(dock);

    splitDockWidget(hostImage_dock,dock,Qt::Vertical);
    m_docks.append(hostImage_dock);

    //code dock
    code_dock=new QDockWidget(codec->toUnicode("比较结果"),this);
    code_dock->setMinimumWidth(250);
    addDockWidget(Qt::RightDockWidgetArea,code_dock);
    m_docks.append(code_dock);

    //bottom dock
    bottom_dock=new QDockWidget(codec->toUnicode("工作路径"),this);
    bottom_dock->setMinimumHeight(120);
    bottom_dock->setFeatures(QDockWidget::AllDockWidgetFeatures);
    QWidget *mywid1 = new QWidget(bottom_dock);
    QGridLayout *pLayout = new QGridLayout(mywid1);
    QLabel *pName0 = new QLabel;
    pName0->setText(codec->toUnicode("数据库路径:"));
    pName0->setFont(QFont("Timers" , 10, QFont::Bold));
    QLabel *fName0 = new QLabel;
    fName0->setText(codec->toUnicode("图像路径 ："));
    fName0->setFont(QFont("Timers" , 10, QFont::Bold));
    QLabel *mtd0 = new QLabel;
    mtd0->setText(codec->toUnicode("使用方法 ："));
    mtd0->setFont(QFont("Timers" , 10, QFont::Bold));
    pLayout->addWidget(pName0, 0, 0, 1, 1);
    pLayout->addWidget(fName0, 1, 0, 1, 1);
    pLayout->addWidget(mtd0, 2, 0, 1, 1);

    pName = new QLabel;
    pName->setText(codec->toUnicode("** Not Selected **"));
    pName->setFont(QFont("Timers" , 10));
    fName = new QLabel;
    fName->setText(codec->toUnicode("** Not Selected **"));
    fName->setFont(QFont("Timers" , 10));
    te2 =new QComboBox();
    te2->setFixedSize(250, 30);
    te2->addItem(tr("Perceptual Hash"));
    te2->addItem(tr("Simplified Hash"));
    te2->addItem(tr("Otsu's Method"));
    te2->addItem(tr("Structural Similarity"));
    te2->addItem(tr("Local HSV Histogram"));
    connect(te2,SIGNAL(currentIndexChanged(int)),this,SLOT(onMethodChange(int)));
    pLayout->addWidget(pName, 0, 1, 1, 3);
    pLayout->addWidget(fName, 1, 1, 1, 3);
    pLayout->addWidget(te2, 2, 1, 1, 3);
    mywid1->setLayout(pLayout);
    bottom_dock->setWidget(mywid1);
    addDockWidget(Qt::BottomDockWidgetArea,bottom_dock);
    m_docks.append(bottom_dock);

    //bottom_right
    bottom_dock_right=new QDockWidget(codec->toUnicode("参数设定"),this);
    bottom_dock_right->setMinimumHeight(120);
    bottom_dock_right->setFeatures(QDockWidget::AllDockWidgetFeatures);
    QWidget *mywid2 = new QWidget(bottom_dock_right);
    QGridLayout *pLayout1 = new QGridLayout(mywid2);
    QLabel *pName1 = new QLabel;
    pName1->setText(codec->toUnicode("门限:"));
    pName1->setFont(QFont("Timers" , 10, QFont::Bold));

    pLayout1->addWidget(pName1, 0, 0, 1, 1);

    te1 =new QTextEdit();
    te1->setFixedSize(120, 50);
    te1->setText(tr(""));
    connect(te1,SIGNAL(textChanged()),this,SLOT(onThresholdChange()));
    pLayout1->addWidget(te1, 0, 1, 1, 3);
    mywid2->setLayout(pLayout1);
    bottom_dock_right->setWidget(mywid2);
    splitDockWidget(bottom_dock,bottom_dock_right,Qt::Horizontal);
    m_docks.append(bottom_dock_right);

}

void ImgProcessor::onMethodChange(int index){
    method = index;
    changed = true;
}

void ImgProcessor::onThresholdChange(){
    changed = true;
    threshold = te1->toPlainText().toInt();
}


void  ImgProcessor::removeAllDock()
{
    for(int i=0;i<9;++i)
    {
        removeDockWidget(m_docks[i]);
    }
}

void  ImgProcessor::showDock()
{
        for(int i=0;i<m_docks.size();++i)
        {
            m_docks[i]->show();
        }
}


void ImgProcessor::createActions()
{

    openFileAction =new QAction(QIcon(":/new/icon/open.png"),codec->toUnicode("选择数据库"),this);//(a)
    openFileAction->setShortcut(tr("Ctrl+O"));                    //(b)
    openFileAction->setStatusTip(codec->toUnicode("打开一个文件夹"));               //(c)
    connect(openFileAction,SIGNAL(triggered()),this,SLOT(ShowOpenFile()));
 
    NewFileAction =new QAction(QIcon(":/new/icon/new.png"),codec->toUnicode("选择图像"),this);
    NewFileAction->setShortcut(tr("Ctrl+N"));
    NewFileAction->setStatusTip(codec->toUnicode("打开一个文件"));
    connect(NewFileAction,SIGNAL(triggered()),this,SLOT(ShowNewFile()));
 
    exitAction =new QAction(codec->toUnicode("退出"),this);
    exitAction->setShortcut(tr("Ctrl+Q"));
    exitAction->setStatusTip(codec->toUnicode("退出程序"));
    connect(exitAction,SIGNAL(triggered()),this,SLOT(close()));


    rotate90Action =new QAction(QIcon(":/new/icon/rotate90.png"),codec->toUnicode("旋转90°"),this);
    rotate90Action->setStatusTip(tr("ClockWise"));
    connect(rotate90Action,SIGNAL(triggered()),this,SLOT(ShowRotate90()));

    rotate180Action =new QAction(QIcon(":/new/icon/rotate180.png"),codec->toUnicode("旋转180°"), this);
    rotate180Action->setStatusTip(codec->toUnicode("将一幅图旋转180°"));
    connect(rotate180Action,SIGNAL(triggered()),this,SLOT(ShowRotate180()));

    rotate270Action =new QAction(QIcon(":/new/icon/rotate270.png"),codec->toUnicode("旋转270°"), this);
    rotate270Action->setStatusTip(tr("AntiClockwise"));
    connect(rotate270Action,SIGNAL(triggered()),this,SLOT(ShowRotate270()));
 
    mirrorVerticalAction =new QAction(QIcon(":/new/icon/undo.png"),codec->toUnicode("纵向镜像"),this);
    mirrorVerticalAction->setStatusTip(codec->toUnicode("对一幅图做纵向镜像"));
    connect(mirrorVerticalAction,SIGNAL(triggered()),this,SLOT(ShowMirrorVertical()));

    mirrorHorizontalAction =new QAction(QIcon(":/new/icon/redo.png"), codec->toUnicode("横向镜像"),this);
    mirrorHorizontalAction->setStatusTip(codec->toUnicode("对一幅图做横向镜像"));
    connect(mirrorHorizontalAction,SIGNAL(triggered()),this,SLOT(ShowMirrorHorizontal()));
 
   settingModification =new QAction(codec->toUnicode("设置"),this);
   settingModification->setStatusTip(codec->toUnicode("修改设定"));
   settingModification->setShortcut(tr("Ctrl+S"));
    connect(settingModification,SIGNAL(triggered()),this,SLOT(ShowSettingDialog()));

   showImageSpace =new QAction(QIcon(":/new/icon/new.png"),"File:..."+filename.right(20),this);
   settingModification->setStatusTip(codec->toUnicode("图片路径"));
   showWorkSpace =new QAction(QIcon(":/new/icon/open.png"),"File:..."+foldname.right(20),this);
   settingModification->setStatusTip(codec->toUnicode("图片路径"));
 
   runFolder = new QAction(QIcon(":/new/icon/justify.png"),codec->toUnicode("生成哈希"),this);
   runFolder->setShortcut(tr("F5"));
   runFolder->setStatusTip(codec->toUnicode("运行程序"));
   connect(runFolder,SIGNAL(triggered()),this,SLOT(generateHashCode()));
   runHash = new QAction(QIcon(":/new/icon/justify.png"),codec->toUnicode("比较"),this);
   runHash->setShortcut(tr("F6"));
   runHash->setStatusTip(codec->toUnicode("运行程序"));
   connect(runHash,SIGNAL(triggered()),this,SLOT(hashTest()));

   window = new QAction(codec->toUnicode("恢复所有窗口"),this);
   window->setShortcut(tr("Ctrl+Q"));

}

void ImgProcessor::createMenus()
{

    fileMenu =menuBar()->addMenu(codec->toUnicode("文件"));
    fileMenu->addAction(openFileAction);				
    fileMenu->addAction(NewFileAction);

    fileMenu->addSeparator();
    fileMenu->addAction(exitAction);

    rotateMenu =menuBar()->addMenu(codec->toUnicode("编辑"));
    rotateMenu->addAction(rotate90Action);
    rotateMenu->addAction(rotate180Action);
    rotateMenu->addAction(rotate270Action);
    rotateMenu->addAction(mirrorVerticalAction);
    rotateMenu->addAction(mirrorHorizontalAction);

    SettingMenu =menuBar()->addMenu(codec->toUnicode("设置"));
    SettingMenu->addAction(settingModification);
 
    WorkSpaceMenu =menuBar()->addMenu(codec->toUnicode("工作路径"));
    WorkSpaceMenu->addAction(showWorkSpace);
    WorkSpaceMenu->addAction(showImageSpace);

    RunMenu =menuBar()->addMenu(codec->toUnicode("运行"));
    RunMenu->addAction(runFolder);
    RunMenu->addAction(runHash);

    WindowMenu =menuBar()->addMenu(codec->toUnicode("窗口"));
    WindowMenu->addAction(window);
}

void ImgProcessor::createToolBars()
{

    fileTool =new QToolBar(codec->toUnicode("文件"));
    QPalette pe2;
    pe2.setColor(QPalette::WindowText,Qt::black);

    fileTool->setPalette(pe2);
    addToolBar(Qt::LeftToolBarArea,fileTool);
    fileTool->addAction(openFileAction);		
    fileTool->addAction(NewFileAction);


    rotateTool =new QToolBar(codec->toUnicode("旋转"));//Qt::LeftToolBarArea,rotateTool
    addToolBar(Qt::LeftToolBarArea,rotateTool);
    rotateTool->addAction(rotate90Action);
    rotateTool->addAction(rotate180Action);
    rotateTool->addAction(rotate270Action);
    rotateTool->addAction(mirrorVerticalAction);
    rotateTool->addAction(mirrorHorizontalAction);

}

void ImgProcessor::ShowNewFile()
{
    changed = true;
    filename = QFileDialog::getOpenFileName(this, tr("Select image"), ".", tr("Image Files(*.jpg *.png *.tif)"));
    img.load(filename);
    img = img.scaled(512,512,Qt::KeepAspectRatio);
    imageLabel->setPixmap(img);

    ClickedLabel *nf = new ClickedLabel;
    QPixmap *tmp = new QPixmap;
    tmp->load(filename);
    *tmp = tmp->scaled(90, 90);
    nf->setPixmap(*tmp);
    nf->setRoute(filename.toStdString());
    test_img.push_back(nf);
    showImageSpace->setText("File:..."+filename.right(20));
    fName->setText(filename);


    QScrollArea *s = new QScrollArea(0);
    QWidget *w = new QWidget(s);
    QGridLayout *pLayout = new QGridLayout(w);
    QSignalMapper *signalMapper = new QSignalMapper(this);
    list<ClickedLabel*>::iterator iter=test_img.begin();
    for(int i=0;i<test_img.size();i++){
 
            pLayout->addWidget(*iter, i/3, i%3, 1, 1);

            connect(*iter, SIGNAL(clicked()), signalMapper, SLOT(map()));
            signalMapper->setMapping(*iter, i);
            iter++;
    }

    w->setLayout(pLayout);
    s->setWidget(w);
    hostImage_dock->setWidget(s);

    connect(signalMapper, SIGNAL(mapped(int)), this, SLOT(onclickFromTest(int)));

    return;

}

void ImgProcessor::onclick(int i)
{

    list<ClickedLabel*>::iterator iter=matches.begin();
    for(int ix=0; ix<i; ++ix){
      ++iter;
    }

    if((*iter)->route!="None."){
        QPixmap *tmp=new QPixmap;
        tmp->load(QString::fromStdString((*iter)->route));
        *tmp = tmp->scaled(512,512);
        imageLabel->setPixmap(*tmp);
    }

    return;
}

void ImgProcessor::onclickFromDB(int i)
{

    changed = true;
    list<ClickedLabel*>::iterator iter=database_img.begin();
    for(int ix=0; ix<i; ++ix){
      ++iter;
    }

    if((*iter)->route!="None."){
        QPixmap *tmp=new QPixmap;
        tmp->load(QString::fromStdString((*iter)->route));
        *tmp = tmp->scaled(512, 512,Qt::KeepAspectRatio);
        imageLabel->setPixmap(*tmp);
    }


    return;
}

void ImgProcessor::onclickFromTest(int i)
{
    changed = true;
    list<ClickedLabel*>::iterator iter=test_img.begin();
    for(int ix=0; ix<i; ++ix){
      ++iter;
    }

    if((*iter)->route!="None."){
        QPixmap *tmp=new QPixmap;
        tmp->load(QString::fromStdString((*iter)->route));
        *tmp = tmp->scaled(512, 512,Qt::KeepAspectRatio);
        imageLabel->setPixmap(*tmp);
    }

    filename = QString::fromStdString((*iter)->route);


    return;
}


void ImgProcessor::ShowOpenFile()
{
    changed = true;database_img.clear();
//    QScrollArea *s = new QScrollArea(0);

//    QWidget *w = new QWidget(s);

    foldname = QFileDialog::getExistingDirectory(this, tr("Open Directory"), "/home", QFileDialog::ShowDirsOnly);
    showWorkSpace->setText("Fold:..."+foldname.right(20));
    pName->setText(foldname);
//    cv::String pattern = foldname.toStdString() + "/*.tif";
//    vector<cv::String> fn;
//    glob(pattern, fn, false);

//    QGridLayout *pLayout = new QGridLayout(w);
//    ClickedLabel *allImages;
//    QSignalMapper *signalMapper = new QSignalMapper(this);
//    QPixmap *tmp=new QPixmap;
//    for(int i=0;i<fn.size();i++){
   
//            allImages = new ClickedLabel;
//            tmp->load(QString::fromStdString(fn[i].c_str()));
//            *tmp = tmp->scaled(90, 90);
//            allImages->setPixmap(*tmp);
//            allImages->setRoute(fn[i].c_str());
//            database_img.push_back(allImages);
//            pLayout->addWidget(allImages, i/3, i%3, 1, 1);

//            connect(allImages, SIGNAL(clicked()), signalMapper, SLOT(map()));
//            signalMapper->setMapping(allImages, i);

//    }

//    w->setLayout(pLayout);
//    s->setWidget(w);
//    dataset_dock->setWidget(s);

//    connect(signalMapper, SIGNAL(mapped(int)), this, SLOT(onclickFromDB(int)));

}

void ImgProcessor::loadFile(QString filename)
{
//    printf("file name:%s\n",filename.data());
//    QFile file(filename);
//    if(file.open(QIODevice::ReadOnly|QIODevice::Text))
//    {
//        QTextStream textStream(&file);
//        while(!textStream.atEnd())
//        {
//            showWidget->text->append(textStream.readLine());
//            printf("read line\n");
//        }
//        printf("end\n");
//    }
}


void ImgProcessor::ShowRotate90()
{
//    changed = true;
//    if(img.isNull())
//        return;
//    QMatrix matrix;
//    matrix.rotate(90);
//    img = img.transformed(matrix);
//    imageLabel->setPixmap(QPixmap::fromImage(img));
}

void ImgProcessor::ShowRotate180()
{
//    changed = true;
//    if(img.isNull())
//        return;
//    QMatrix matrix;
//    matrix.rotate(180);
//    img = img.transformed(matrix);
//    imageLabel->setPixmap(QPixmap::fromImage(img));
}

void ImgProcessor::ShowRotate270()
{
//    changed = true;
//    if(img.isNull())
//        return;
//    QMatrix matrix;
//    matrix.rotate(270);
//    img = img.transformed(matrix);
//    imageLabel->setPixmap(QPixmap::fromImage(img));
}

void ImgProcessor::ShowMirrorVertical()
{
//    changed = true;
//    if(img.isNull())
//        return;
//    img=img.mirrored(false,true);
//    imageLabel->setPixmap(QPixmap::fromImage(img));
}


void ImgProcessor::ShowMirrorHorizontal()
{
//    changed = true;
//    if(img.isNull())
//        return;
//    img=img.mirrored(true,false);
//    imageLabel->setPixmap(QPixmap::fromImage(img));
}

void ImgProcessor::ShowSettingDialog()
{
    changed = true;
    SettingDlg *settingDlg =new SettingDlg(this);
    settingDlg->show();
}


ImgProcessor::~ImgProcessor()
{

}
