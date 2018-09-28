/*
 *  cuLsoda_kernel.cu
 *  cuLsoda
 *
 */
 #ifndef _CULSODA_CU_H_
 #define _CULSODA_CU_H_
 
 #include "cuLsoda.cu.h"
 
 template<typename Fex, typename Jex>
__global__ void cuLsoda(Fex fex, int NEQ, double *y, double *rwork,
		int nrw, int *iwork, int niw, Jex jac, struct cuLsodaCommonBlock *common,
		double *param_1d, int NPR, double *result, double *time_1d, int num_time, int probSize)
{
	int block_size = blockDim.x;
	int me_thread = threadIdx.x;
	int me = threadIdx.x + blockIdx.x * blockDim.x;
	int iout, idx;
	/* Shared variables */
	extern __shared__ double array[]; //ブロック内shared memory
	double *t     = array; //double*block_size
	int *jt       = (int*)&t[block_size]; //int
	int *neq      = &jt[1]; // int
	int *liw      = &neq[1]; // int
	int *lrw      = &liw[1]; //int
	double *atol  = (double*)&lrw[1]; //double*NEQ
	int *itol     = (int*)&atol[NEQ]; //int
	int *iopt     = &itol[1]; //int
	double *rtol  = (double*)&iopt[1]; //double
	double *tout  = &rtol[1]; //double*block_size
	int *itask    = (int*)&tout[block_size]; //int
	int *istate   = &itask[1]; //int*block_size
	
	/* Assignment of initial values to shareds */
	if (me_thread==0) {
		*neq = NEQ;
		*liw = niw;
		*lrw = nrw;
		*itol = 2;
		//*rtol = (double)1.e-4;
		*rtol = (double)1.e-6;
		*itask = 1;
		*iopt = 1; //iwork, rworkのオプションを有効にする
		*jt = 2;
		
		for (idx=0; idx<NEQ; idx++) {
			//atol[idx] = (double)1.e-12;
			atol[idx] = (double)1.e-6;
		}
	}

	__syncthreads(); //共通シェアードの準備ができるまで待機
	
	if (me < probSize) {
		//グローバルからコピー、スレッドごとに異なるシェアードを参照するので同期の必要はない
		t[me_thread] = time_1d[0];
		tout[me_thread] = time_1d[1];
		istate[me_thread] = 1;
		/* Numeric simulation */
		for (idx=0; idx<NEQ; idx++)
			result[NEQ*me+idx] = y[NEQ*me+idx]; //初期値コピー
		
		for (iout = 1; iout < (num_time-1); ++iout) {			
			dlsoda_(fex, neq, &y[NEQ*me], &t[me_thread], &tout[me_thread], itol,
					rtol, atol, itask, &istate[me_thread], iopt, &rwork[nrw*me],
					lrw, &iwork[niw*me], liw, jac, jt, &common[me], &param_1d[NPR*me]);
			
			for (idx=0; idx<NEQ; idx++)
				result[iout*NEQ*probSize+NEQ*me+idx] = y[NEQ*me+idx];
			
			if (istate[me_thread] < 0) {
				return; //強制終了
			}
			
			tout[me_thread] = time_1d[iout+1];
		}
		//最後のt->toutまでをシミュレーション
		dlsoda_(fex, neq, &y[NEQ*me], &t[me_thread], &tout[me_thread], itol,
				rtol, atol, itask, &istate[me_thread], iopt, &rwork[nrw*me],
				lrw, &iwork[niw*me], liw, jac, jt, &common[me], &param_1d[NPR*me]);
			
		for (idx=0; idx<NEQ; idx++)
			result[(num_time-1)*NEQ*probSize+NEQ*me+idx] = y[NEQ*me+idx];
	}
}


#endif

