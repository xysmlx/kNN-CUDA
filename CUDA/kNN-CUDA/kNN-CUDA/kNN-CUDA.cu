// #pragma comment(linker, "/STACK:102400000,102400000")
#include <cstdio>
#include <iostream>
#include <cstring>
#include <string>
#include <cmath>
#include <set>
#include <list>
#include <map>
#include <iterator>
#include <cstdlib>
#include <vector>
#include <queue>
#include <ctime>
#include <stack>
#include <algorithm>
#include <functional>
#include <ctime>
#include <fstream>
using namespace std;
typedef long long ll;
const double INF = 1e30;
typedef pair<int, double> prid;

class KNN
{
public:
	static const int maxCol = 10;
	static const int maxRow = 46010;

	static const int testNum = 300; // Data number for test

public:
	struct Node
	{
		double data[maxCol];
		string label;
	};

	struct CMP
	{
		bool operator ()(const prid &A, const prid &B) const
		{
			return A.second < B.second;
		}
	};

public:
	void init(int _k, int _row, int _col, string path); // init
	void input(); // Input data
	void ZScoreNorm(); // Z-Score Norm
	void MaxMinNorm(); // Max min Norm
	double dis(const Node &A, const Node &B); // Calculate distance between A and B
	void CalDis();
	string MaxFreqLabel(); // Calculate freq label
	void knn(); // Run knn

	void CUDAInit(); // Init CUDA

	void debug(); // For debug

public:
	ifstream fin;
	string filepath;

	int therK;
	int row;
	int col;

	Node dataSet[maxRow];
	Node dataTest;

	vector<prid> vec;
	map<string, int> mp;

	// Data for CUDA
	double trainmtx[maxRow][maxCol];
	int trainRow;
	double *GTrainData;
	size_t pitch_d;
	size_t pitch_h;
};

void KNN::init(int _k, int _row, int _col, string path)
{
	therK = _k;
	row = _row;
	col = _col;
	filepath = path;

	cout << therK << " " << row << " " << col << endl;
}

void KNN::input()
{
	fin.open(filepath);
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
			fin >> dataSet[i].data[j];
		fin >> dataSet[i].label;
	}
	fin.close();
	// debug();

	MaxMinNorm();
	// ZScoreNorm();
}

void KNN::ZScoreNorm()
{
	for (int j = 0; j < col; j++)
	{
		double avg = 0;
		for (int i = 0; i < row; i++)
			avg += dataSet[i].data[j];
		avg /= (double)row;

		double sig = 0;
		for (int i = 0; i < row; i++)
			sig += (dataSet[i].data[j] - avg) * (dataSet[i].data[j] - avg);
		sig /= (double)row;
		sig = sqrt(sig);

		for (int i = 0; i < row; i++)
			dataSet[i].data[j] = (dataSet[i].data[j] - avg) / sig;
	}
}

void KNN::MaxMinNorm()
{
	for (int j = 0; j < col; j++)
	{
		double maxx = max(dataSet[0].data[j], dataSet[1].data[j]);
		double minx = min(dataSet[0].data[j], dataSet[1].data[j]);
		for (int i = 0; i < row; i++)
		{
			//if (dataSet[i].data[j] > maxx) maxx = dataSet[i].data[j];
			//else if (dataSet[i].data[j] < minx) minx = dataSet[i].data[j];
			maxx = max(maxx, dataSet[i].data[j]);
			minx = min(minx, dataSet[i].data[j]);
		}
		for (int i = 0; i < row; i++)
			dataSet[i].data[j] = (double)(dataSet[i].data[j] - minx) / (double)(maxx - minx);
	}
}

double KNN::dis(const Node &A, const Node &B)
{
	double ret = 0;
	for (int i = 0; i < col; i++)
		ret += (A.data[i] - B.data[i]) * (A.data[i] - B.data[i]);
	return sqrt(ret);
}

__global__ void DisKernel(double *traindata, double *testdata, double *dis, int pitch, int N, int D)
{
	int tid = blockIdx.x;
	if (tid < N)
	{
		double temp = 0;
		double sum = 0;
		for (int i = 0; i < D; i++)
		{
			temp = *((double*)((char*)traindata + tid * pitch) + i) - testdata[i];
			sum += temp * temp;
		}
		dis[tid] = sum;
	}
}

void KNN::CalDis()
{
	vec.clear();

	double *GTestData;
	double *GDis;
	double *distance;

	distance = new double[trainRow];

	cudaMalloc(&GTestData, col * sizeof(double));
	cudaMalloc(&GDis, trainRow * sizeof(double));

	cudaMemset(GTestData, 0, col * sizeof(double));
	cudaMemset(GDis, 0, trainRow * sizeof(double));

	cudaMemcpy(GTestData, dataTest.data, col * sizeof(double), cudaMemcpyHostToDevice);

	DisKernel << <trainRow, 1 >> >(GTrainData, GTestData, GDis, pitch_d, trainRow, col);

	cudaMemcpy(distance, GDis, trainRow * sizeof(double), cudaMemcpyDeviceToHost);

	for (int i = testNum; i < row; i++)
		vec.push_back(make_pair(i, distance[i - testNum]));

}

string KNN::MaxFreqLabel()
{
	CalDis();
	sort(vec.begin(), vec.end(), CMP());

	mp.clear();
	for (int i = 0; i < therK; i++)
		mp[dataSet[vec[i].first].label]++;

	string ret;
	int cnt = 0;
	for (auto ite = mp.begin(); ite != mp.end(); ite++)
	{
		if (ite->second > cnt)
		{
			cnt = ite->second;
			ret = ite->first;
		}
	}

	return ret;
}

void KNN::knn()
{
	cout << "Test data num: " << testNum << endl;

	int cnt = 0;
	for (int i = 0; i < testNum; i++)
	{
		dataTest = dataSet[i];
		if (MaxFreqLabel() == dataTest.label)
			cnt++;
	}

	//cout << cnt << " " << testNum << endl;

	cout << "Accuracy Rate: " << (double)cnt / (double)testNum << endl;
}

void KNN::CUDAInit()
{
	trainRow = row - testNum;
	for (int i = testNum; i < row; i++)
	{
		for (int j = 0; j < col; j++)
			trainmtx[i - testNum][j] = dataSet[i].data[j];
	}

	pitch_h = col*sizeof(double);

	cudaMallocPitch(&GTrainData, &pitch_d, col * sizeof(double), trainRow);
	cudaMemset(GTrainData, 0, trainRow * col * sizeof(double));
	cudaMemcpy2D(GTrainData, pitch_d, trainmtx, pitch_h, col * sizeof(double), trainRow, cudaMemcpyHostToDevice);
}

void KNN::debug()
{
	ofstream fout;
	fout.open("debug.txt");
	for (int i = 0; i < row; i++)
	{
		for (int j = 0; j < col; j++)
			fout << dataSet[i].data[j] << "\t";
		fout << dataSet[i].label << endl;
	}
}

KNN knn;

void init()
{
	knn.init(7, 3000, 8, "allTypeC.txt");
}
void input()
{
	knn.input();
}
void debug()
{
	//
}
void solve()
{
	clock_t st, ed;
	st = clock();
	knn.knn();
	ed = clock();
	cout << "Time: " << ed - st << endl;
}
void output()
{
	//
}
int main()
{
	init();
	input();
	solve();
	output();

	return 0;
}