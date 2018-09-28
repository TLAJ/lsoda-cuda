/*
 *  Edit by Kaoru O.
 */

//#define EMULATION_MODE
//#define use_export	// uncomment this if project is built using a compiler that
// supports the C++ keyword "export".  If not using such a 
// compiler, be sure not to add cuLsoda.cc to the target, or
// you will get redefinition errors.

#include <stdio.h>
#include <math.h>
#include "cuLsoda_kernel.cu"

#define Fex_and_Jex_definition
const int NEQ = 3; //濃度変化させる物質の種類数
const int NPR = 3; //パラメータ数
const int NIN = 100; //入力物質の濃度の測定時間数

// Linear interpolation
__constant__ double d_input_1d[NIN]; //CPUスレッドセーフでない
__constant__ double d_input_time[NIN];
__constant__ int d_num_input;

__device__ double inputAtTime(double time) {
	int i_itime = 0; //並列演算で呼び出されるので、毎回異なるGPUローカル変数を初期化する

	while (i_itime < d_num_input && d_input_time[i_itime] < time)
		i_itime++;

	if (i_itime < 1)
		return d_input_1d[0];
	if (i_itime > d_num_input-1)
		return d_input_1d[d_num_input-1];

	return d_input_1d[i_itime-1]+(d_input_1d[i_itime]-d_input_1d[i_itime-1])*
			(time-d_input_time[i_itime-1])/(d_input_time[i_itime]-d_input_time[i_itime-1]);
}

struct myFex {
	__device__ void operator()(int *neq, double *t, double *y, double *ydot, double *param)
	{ //並列演算で呼び出されるので、毎回異なるGPUローカル変数を初期化する
		double flux[2];
		//double input = inputAtTime(*t);
		
		//param: k1, k2, k3
		
		flux[0] = param[0]*y[0]*y[1] - param[1]*y[2];
		flux[1] = param[2]*y[2];
		
		ydot[0] = -flux[0];
		ydot[1] = -flux[0];
		ydot[2] = flux[0]-flux[1];
	}
};

struct myJex
{
	__device__ void operator()(int *neq, double *t, double *y, int ml, int mu, double *pd, int nrowpd/*, void *otherData*/)
	{
		return;
	}
};

__global__ void test_kernel(){
	int me = threadIdx.x + blockIdx.x * blockDim.x;
	
	if (me==0) {
		printf("num_input: %d\n",d_num_input);
	}
}

template <typename T> class CudaMem {
protected:
	size_t current_size;
	virtual cudaError_t memFree() = 0;
	virtual cudaError_t memAlloc(size_t req_size) = 0;

public:
	T *ptr;

	CudaMem() {
		current_size = 0;
	}
	virtual ~CudaMem() {} //基底クラスのデストラクタを仮想化しないと、継承先のデストラクタが呼ばれない

	cudaError_t cudaAlloc(size_t req_size) {
		if (current_size == 0) { //初めてメモリ確保
			current_size = req_size;
			return memAlloc(req_size);
		}
		else if (current_size < req_size) { //要求メモリサイズより小さい時
			memFree(); //現在のメモリを破棄
			current_size = req_size;
			return memAlloc(req_size); //新たなメモリ確保
		}
		else {
			return cudaSuccess; //何もせず終了
		}
	}
};

template <typename T> class CudaDeviceMem : public CudaMem<T> {
protected:
	cudaError_t memFree() {
		return cudaFree(this->ptr);
	}
	cudaError_t memAlloc(size_t req_size) {
		return cudaMalloc(&this->ptr, req_size);
	}

public:
	~CudaDeviceMem() {
		if (this->current_size > 0)
			memFree();
	}
};

template <typename T> class CudaHostMem : public CudaMem<T> {
protected: //メモリ関係の関数をオーバーライド
	cudaError_t memFree() {
		return cudaFreeHost(this->ptr);
	}
	cudaError_t memAlloc(size_t req_size) {
		return cudaHostAlloc(&this->ptr, req_size, cudaHostAllocDefault);
	}
public:
	~CudaHostMem() {
		if (this->current_size > 0)
			memFree();
	}
};

CudaHostMem<double> h_input_1d, h_input_time;

CudaHostMem<double>   h_y, h_rwork, h_param_1d, h_time_1d, h_result;
CudaHostMem<int>      h_iwork;
CudaHostMem<struct cuLsodaCommonBlock> h_common;
CudaDeviceMem<double> d_y, d_rwork, d_param_1d, d_time_1d, d_result;
CudaDeviceMem<int>    d_iwork;
CudaDeviceMem<struct cuLsodaCommonBlock> d_common;

extern "C"
{ //
int modelSim(double *init_1d, double *param_1d, double *time_1d, int num_time,
		double *input_1d, double *input_time, int num_input,
		double **result_2d, int num_parallel, int num_group)
{
	//fprintf(stderr, "num_parallel: %d\n", num_parallel);
	int itime,idx;
	num_parallel = num_parallel*num_group;
	/* Global variables */
	h_input_1d.cudaAlloc(sizeof(double)*num_input);
	h_input_time.cudaAlloc(sizeof(double)*num_input);
	
	for (itime=0; itime < num_input; itime++) {
		h_input_1d.ptr[itime] = input_1d[itime];
		h_input_time.ptr[itime] = input_time[itime];
	}
	
	cudaMemcpyToSymbolAsync(d_input_1d, h_input_1d.ptr, sizeof(double)*num_input, 0,cudaMemcpyHostToDevice,0);
	cudaMemcpyToSymbolAsync(d_input_time, h_input_time.ptr, sizeof(double)*num_input, 0,cudaMemcpyHostToDevice,0);
	cudaMemcpyToSymbolAsync(d_num_input, &num_input, sizeof(int), 0,cudaMemcpyHostToDevice,0);
	
    /* Local variables (CPU & GPU) */
	int nrw = 22+NEQ*max(16,NEQ+9);
	int niw = 20+NEQ;
	 //非同期に読み出すのでピンメモリ確保
	h_y.cudaAlloc(sizeof(double)*NEQ*num_parallel);
    h_iwork.cudaAlloc(sizeof(int)*niw*num_parallel);
    h_rwork.cudaAlloc(sizeof(double)*nrw*num_parallel);
    h_common.cudaAlloc(sizeof(struct cuLsodaCommonBlock)*num_parallel);
	h_param_1d.cudaAlloc(sizeof(double)*NPR*num_parallel);
	h_time_1d.cudaAlloc(sizeof(double)*num_time);
	h_result.cudaAlloc(sizeof(double)*num_time*NEQ*num_parallel);
	//デバイスメモリ
	d_y.cudaAlloc(sizeof(double)*NEQ*num_parallel);
	d_iwork.cudaAlloc(sizeof(int)*niw*num_parallel);
	d_rwork.cudaAlloc(sizeof(double)*nrw*num_parallel);
	d_common.cudaAlloc(sizeof(struct cuLsodaCommonBlock)*num_parallel);
	d_param_1d.cudaAlloc(sizeof(double)*NPR*num_parallel);
	d_time_1d.cudaAlloc(sizeof(double)*num_time);
	d_result.cudaAlloc(sizeof(double)*num_time*NEQ*num_parallel);
	/* End Local Block */
	
	/* Assignment of initial values to locals */
	for (idx=0; idx < NEQ*num_parallel; idx++)
		h_y.ptr[idx] = init_1d[idx];
	
	for (idx=0; idx < NPR*num_parallel; idx++)
		h_param_1d.ptr[idx] = param_1d[idx];
	
	for (itime=0; itime < num_time; itime++)
		h_time_1d.ptr[itime] = time_1d[itime];
	
	for (idx = 0; idx < num_parallel; idx++) {
		h_iwork.ptr[niw*idx+4] = h_iwork.ptr[niw*idx+6] = h_iwork.ptr[niw*idx+7] = h_iwork.ptr[niw*idx+8] = 0; //0ならデフォルト
		h_iwork.ptr[niw*idx+5] = 1e+5; //Maximum step number
		h_rwork.ptr[nrw*idx+4] = h_rwork.ptr[nrw*idx+5] = h_rwork.ptr[nrw*idx+6] = (double)0.; //デフォルト
		cuLsodaCommonBlockInit(&h_common.ptr[idx]);
	}
	
	/* Method instantiations for Derivative and Jacobian functions to send to template */
	myFex fex;
	myJex jex;
	
	
	/* 仮想：Grid -> Block -> Thread
	 * 物理：GPU  -> SM    -> Core (CUDA core)
	 * GPUによるが、最大1024 threads/block
	 * 
	 * スレッド数を減らすとブロック数が増える
	 * 物理SMが多いGPUなら、ブロック数を増やしてやれば負荷が分散され速く終わる
	 * SMに割り当てるブロック数にも上限がある
	 * 
	 * スレッド数を増やすと、SMの限られたレジスタ・shared memoryなどを食いつぶす恐れあり
	 */
	int threadsPerBlock = 256; //32の倍数が良い
	int blocksPerGrid = (num_parallel + threadsPerBlock -1)/threadsPerBlock;
	cudaError_t error;
	
	//全ステップを並列計算
	cudaMemcpyAsync(d_time_1d.ptr, h_time_1d.ptr, sizeof(double)*num_time, cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(d_y.ptr, h_y.ptr, sizeof(double)*NEQ*num_parallel, cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(d_rwork.ptr, h_rwork.ptr, sizeof(double)*nrw*num_parallel, cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(d_iwork.ptr, h_iwork.ptr, sizeof(int)*niw*num_parallel, cudaMemcpyHostToDevice, 0);
 	cudaMemcpyAsync(d_common.ptr, h_common.ptr, sizeof(struct cuLsodaCommonBlock)*num_parallel, cudaMemcpyHostToDevice, 0);
	cudaMemcpyAsync(d_param_1d.ptr, h_param_1d.ptr, sizeof(double)*NPR*num_parallel, cudaMemcpyHostToDevice, 0);
	
	cuLsoda<<<blocksPerGrid,threadsPerBlock,
			threadsPerBlock*sizeof(double) + //*t
			4*sizeof(int) + //*jt, *neq, *liw, *lrw
			NEQ*sizeof(double) + //*atol
			2*sizeof(int) + sizeof(double) + //*itol, *iopt, *rtol
			threadsPerBlock*sizeof(double) + //*tout
			(1+threadsPerBlock)*sizeof(int) //*itask, *istate
			>>>(fex, NEQ, d_y.ptr, d_rwork.ptr,	nrw, d_iwork.ptr, niw, jex, d_common.ptr,
					d_param_1d.ptr, NPR, d_result.ptr, d_time_1d.ptr, num_time, num_parallel);
		
	cudaMemcpyAsync(h_result.ptr,d_result.ptr,sizeof(double)*num_time*NEQ*num_parallel, cudaMemcpyDeviceToHost, 0);

    cudaDeviceSynchronize(); //終了まで待機
    error = cudaGetLastError();
	
	if (error != cudaSuccess) {
		printf("Cuda error: %s\n", cudaGetErrorString(error));
		return 1;
		/* invalid arguments: shared memoryの上限に引っかかってもアウト*/
	}
	//fprintf(stderr, "After kernel\n");
    // 返却用ヒープにすべてコピー
    for (itime=0; itime<num_time; itime++)
    	for (idx=0; idx<NEQ*num_parallel; idx++)
    		result_2d[itime][idx] = h_result.ptr[itime*NEQ*num_parallel + idx];
	
	//cudaDeviceReset(); //soなどランタイム関数のとき、今回の実行関連を要初期化？
	
	return 0;
}
}
