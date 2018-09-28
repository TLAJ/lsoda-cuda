//#define EMULATION_MODE
//#define use_export	// uncomment this if project is built using a compiler that
// supports the C++ keyword "export".  If not using such a 
// compiler, be sure not to add cuLsoda.cc to the target, or
// you will get redefinition errors.
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuLsoda_kernel.cpp"

#define Fex_and_Jex_definition
const int NEQ = 3; //濃度変化させる物質の種類数
const int NPR = 3; //パラメータ数

struct myFex
{
	double *input_1d; //複数スレッドで初期化される可能性があるので、インスタンス内部で参照
	double *input_time;
	int num_input;
	int i_itime; //一つのスレッドからしかアクセスされないので、関数外でOK
	//ある時刻の入力物質の濃度を線形補間で求める
	double inputAtTime(double time) {
		i_itime = 0;

		while (i_itime < num_input && input_time[i_itime] < time)
			i_itime++;

		if (i_itime < 1)
			return input_1d[0];
		if (i_itime > num_input-1)
			return input_1d[num_input-1];

		return input_1d[i_itime-1]+(input_1d[i_itime]-input_1d[i_itime-1])*(time-input_time[i_itime-1])/(input_time[i_itime]-input_time[i_itime-1]);
	}

	double input,flux[2];
	double k1, k2, k3;

	void operator()(int *neq, double *t, double *y, double *ydot, double *param)
	{
		k1 = param[0];
		k2 = param[1];
		k3 = param[2];

		//input = inputAtTime(*t);

		flux[0] = k1*y[0]*y[1] - k2*y[2];
		flux[1] = k3*y[2];

		ydot[0] = -flux[0];
		ydot[1] = -flux[0];
		ydot[2] = flux[0]-flux[1];
	}
};

struct myJex
{
	void operator()(int *neq, double *t, double *y, int ml, int mu, double *pd, int nrowpd/*, void *otherData*/)
	{
		return;
	}
};


template <typename T> class HostMem {
public:
	T *ptr;
	size_t current_size;

	HostMem() {
		current_size = 0;
	}
	~HostMem() {
		if (this->current_size > 0)
			free(this->ptr);
	}

	void memAlloc(size_t req_size) {
		if (this->current_size == 0) { //初めてメモリ確保
			this->current_size = req_size;
			this->ptr = (T *)malloc(req_size);
		}
		else if (this->current_size < req_size) { //要求メモリサイズより小さい時
			free(this->ptr); //現在のメモリを破棄
			this->current_size = req_size;
			this->ptr = (T *)malloc(req_size); //新たなメモリ確保
		}
	}
};

HostMem<double> h_t, h_y, h_atol, h_rtol, h_tout, h_rwork;
HostMem<int> h_jt, h_neq, h_liw, h_lrw, h_itol, h_iopt, h_itask, h_iwork, h_istate;
HostMem<struct cuLsodaCommonBlock> h_common;

extern "C"
{
#ifdef _MSC_VER
__declspec(dllexport)
#endif
int modelSim(double *init_1d, double *param_1d, double *time_1d, int num_time,
		double *input_1d, double *input_time_1d, int num_input_1d, double **result_2d)
{
    /* Local variables */
	int nrw = 22+NEQ*max(16,NEQ+9);
	int niw = 20+NEQ;
	int idx;
	h_t.memAlloc(sizeof(double));
	h_y.memAlloc(sizeof(double)*NEQ);
	h_jt.memAlloc(sizeof(int));
	h_neq.memAlloc(sizeof(int));
	h_liw.memAlloc(sizeof(int));
	h_lrw.memAlloc(sizeof(int));
	h_atol.memAlloc(sizeof(double)*NEQ);
	h_itol.memAlloc(sizeof(int));
	h_iopt.memAlloc(sizeof(int));
	h_rtol.memAlloc(sizeof(double));
	h_tout.memAlloc(sizeof(double));
	h_itask.memAlloc(sizeof(int));
	h_iwork.memAlloc(sizeof(int)*niw);
	h_rwork.memAlloc(sizeof(double)*nrw);
	h_istate.memAlloc(sizeof(int));
	h_common.memAlloc(sizeof(struct cuLsodaCommonBlock));
	/* End Local Block */
	
	/* Assignment of initial values to locals */
	for (idx=0; idx<NEQ; idx++) {
		h_y.ptr[idx] = init_1d[idx];
		//h_atol[idx] = (double)1.e-12;
		h_atol.ptr[idx] = (double)1.e-6;
	}
	
	*h_neq.ptr = NEQ;
	*h_t.ptr = time_1d[0];
	*h_tout.ptr = time_1d[1];
	*h_liw.ptr = niw;
	*h_lrw.ptr = nrw;
	*h_itol.ptr = 2; //1
	//*h_rtol = (double)1.e-4;
	*h_rtol.ptr = (double)1.e-6;
	*h_itask.ptr = 1;
	*h_istate.ptr = 1;
	*h_iopt.ptr = 1;
	h_iwork.ptr[4] = h_iwork.ptr[6] = h_iwork.ptr[7] = h_iwork.ptr[8] = 0; // default
	h_iwork.ptr[5] = 1e+5; //Maximum step number
	h_rwork.ptr[4] = h_rwork.ptr[5] = h_rwork.ptr[6] = (double)0.; // default
	*h_jt.ptr = 2;
	cuLsodaCommonBlockInit(h_common.ptr);

	/* Method instantiations for Derivative and Jacobian functions to send to template */
	myFex fex;
	myJex jex;

	/* Struct variables */
	fex.input_1d = input_1d;
	fex.input_time = input_time_1d;
	fex.num_input = num_input_1d;


	cuLsoda(fex, h_neq.ptr, h_y.ptr, h_t.ptr, h_tout.ptr, h_itol.ptr, h_rtol.ptr, h_atol.ptr,
			h_itask.ptr, h_istate.ptr, h_iopt.ptr, h_rwork.ptr, h_lrw.ptr, h_iwork.ptr, h_liw.ptr,
			jex, h_jt.ptr, h_common.ptr, param_1d, result_2d, time_1d, num_time);
	
	return 0;
}
}
