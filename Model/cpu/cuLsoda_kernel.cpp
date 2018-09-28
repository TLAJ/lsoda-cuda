/*
 *  cuLsoda_kernel.cu
 *  cuLsoda
 *
 */
 #ifndef _CULSODA_CU_H_
 #define _CULSODA_CU_H_
 
 #include "cuLsoda.cu.h"
 
template<typename Fex, typename Jex>
void cuLsodaFlux(Fex fex, int *neq, int *nfx, double *y, double *t, double *tout, int *itol,
		double *rtol, double *atol, int *itask, int *istate, int *iopt, double *rwork,
		int *lrw, int *iwork, int *liw, Jex jac, int *jt, struct cuLsodaCommonBlock *common,
		double *param, double **result, double **flux, double *time_1d, int ntime)
{
	int iout, idx;

	fex(neq,t,y,result[0],param); //ダミーで常微分計算>fluxが計算できる

	for (idx=0; idx<*nfx; idx++)
		flux[0][idx] = fex.flux[idx];
	for (idx=0; idx<*neq; idx++)
		result[0][idx] = y[idx];
	//printf("time: %f, input: %f, A: %f\n", *t, inputAtTime(*t), y[0]);
	//calc each step
	for (iout = 1; iout < (ntime-1); ++iout)
	{
		dlsoda_(fex, neq, y, t, tout, itol,
					rtol, atol, itask, istate, iopt, rwork,
					lrw, iwork, liw, jac, jt, common, param);

		fex(neq,t,y,result[iout],param); //ダミーで常微分計算>fluxが計算できる

		for (idx=0; idx<*nfx; idx++)
			flux[iout][idx] = fex.flux[idx];
		for (idx=0; idx<*neq; idx++)
			result[iout][idx] = y[idx];

		//printf("time: %f, input: %f, A: %f\n", *t, inputAtTime(*t), y[0]);

		if (*istate < 0) {
			//printf("istate: %d\n", *istate);
			return; //強制終了
		}

		*tout = time_1d[iout+1];
	}
	//final: t->tout
	dlsoda_(fex, neq, y, t, tout, itol,
			rtol, atol, itask, istate, iopt, rwork,
			lrw, iwork, liw, jac, jt, common, param);

	fex(neq,t,y,result[ntime-1],param); //ダミーで常微分計算>fluxが計算できる

	for (idx=0; idx<*nfx; idx++)
		flux[ntime-1][idx] = fex.flux[idx];
	for (idx=0; idx<*neq; idx++)
		result[ntime-1][idx] = y[idx];
}

template<typename Fex, typename Jex>
void cuLsoda(Fex fex, int *neq, double *y, double *t, double *tout, int *itol,
		double *rtol, double *atol, int *itask, int *istate, int *iopt, double *rwork,
		int *lrw, int *iwork, int *liw, Jex jac, int *jt, struct cuLsodaCommonBlock *common,
		double *param, double **result, double *time_1d, int ntime)
{
	int iout, idx;

	for (idx=0; idx<*neq; idx++)
		result[0][idx] = y[idx];
	//printf("time: %f, input: %f, A: %f\n", *t, inputAtTime(*t), y[0]);
	//calc each step
	for (iout = 1; iout < (ntime-1); ++iout)
	{
		dlsoda_(fex, neq, y, t, tout, itol,
					rtol, atol, itask, istate, iopt, rwork,
					lrw, iwork, liw, jac, jt, common, param);

		for (idx=0; idx<*neq; idx++)
			result[iout][idx] = y[idx];

		//printf("time: %f, input: %f, A: %f\n", *t, inputAtTime(*t), y[0]);

		if (*istate < 0) {
			//printf("istate: %d\n", *istate);
			return; //強制終了
		}

		*tout = time_1d[iout+1];
	}
	//final: t->tout
	dlsoda_(fex, neq, y, t, tout, itol,
			rtol, atol, itask, istate, iopt, rwork,
			lrw, iwork, liw, jac, jt, common, param);

	for (idx=0; idx<*neq; idx++)
		result[ntime-1][idx] = y[idx];
}


#endif

