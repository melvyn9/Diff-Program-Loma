#include <math.h>

typedef struct {
	float val;
	float dval;
} _dfloat;
float benchmark3(float x, float y, float z);
void d_benchmark3(float x, float* _dx_Lx1FPc, float y, float* _dy_2bPjNx, float z, float* _dz_mzWAaf, float _dreturn_k5CHXE);
_dfloat make__dfloat(float val, float dval);
void d_user_func(float x, float* _dx_Lx1FPc, float y, float* _dy_2bPjNx, float z, float* _dz_mzWAaf, float _dreturn_k5CHXE);
float benchmark3(float x, float y, float z) {
	return (((x) * (y)) + (sinf((y) + (z)))) + (logf((x) + (z)));
}
void d_benchmark3(float x, float* _dx_Lx1FPc, float y, float* _dy_2bPjNx, float z, float* _dz_mzWAaf, float _dreturn_k5CHXE) {
	float _t_float_6NmcPy[2];
	for (int _i = 0; _i < 2;_i++) {
		_t_float_6NmcPy[_i] = 0;
	}
	int _stack_ptr_float_6NmcPy = (int)(0);
	float _call_t_0_keRCuf;
	_call_t_0_keRCuf = 0;
	float _d_call_t_0_keRCuf_fXAlyq;
	_d_call_t_0_keRCuf_fXAlyq = 0;
	float _call_t_1_HSQD3s;
	_call_t_1_HSQD3s = 0;
	float _d_call_t_1_HSQD3s_3YplNK;
	_d_call_t_1_HSQD3s_3YplNK = 0;
	(_t_float_6NmcPy)[_stack_ptr_float_6NmcPy] = _call_t_0_keRCuf;
	_stack_ptr_float_6NmcPy = (_stack_ptr_float_6NmcPy) + ((int)(1));
	_call_t_0_keRCuf = (y) + (z);
	(_t_float_6NmcPy)[_stack_ptr_float_6NmcPy] = _call_t_1_HSQD3s;
	_stack_ptr_float_6NmcPy = (_stack_ptr_float_6NmcPy) + ((int)(1));
	_call_t_1_HSQD3s = (x) + (z);
	float _adj_0;
	_adj_0 = 0;
	float _adj_1;
	_adj_1 = 0;
	float _adj_2;
	_adj_2 = 0;
	float _adj_3;
	_adj_3 = 0;
	(*_dx_Lx1FPc) += (y) * (_dreturn_k5CHXE);
	(*_dy_2bPjNx) += (x) * (_dreturn_k5CHXE);
	_d_call_t_0_keRCuf_fXAlyq += (cosf(_call_t_0_keRCuf)) * (_dreturn_k5CHXE);
	_d_call_t_1_HSQD3s_3YplNK += (_dreturn_k5CHXE) / (_call_t_1_HSQD3s);
	_stack_ptr_float_6NmcPy = (_stack_ptr_float_6NmcPy) - ((int)(1));
	_call_t_1_HSQD3s = (_t_float_6NmcPy)[_stack_ptr_float_6NmcPy];
	_adj_0 = _d_call_t_1_HSQD3s_3YplNK;
	_adj_1 = _d_call_t_1_HSQD3s_3YplNK;
	_d_call_t_1_HSQD3s_3YplNK = (float)(0.0);
	(*_dx_Lx1FPc) += _adj_0;
	(*_dz_mzWAaf) += _adj_1;
	_stack_ptr_float_6NmcPy = (_stack_ptr_float_6NmcPy) - ((int)(1));
	_call_t_0_keRCuf = (_t_float_6NmcPy)[_stack_ptr_float_6NmcPy];
	_adj_2 = _d_call_t_0_keRCuf_fXAlyq;
	_adj_3 = _d_call_t_0_keRCuf_fXAlyq;
	_d_call_t_0_keRCuf_fXAlyq = (float)(0.0);
	(*_dy_2bPjNx) += _adj_2;
	(*_dz_mzWAaf) += _adj_3;
}
_dfloat make__dfloat(float val, float dval) {
	_dfloat ret;
	ret.val = 0;
	ret.dval = 0;
	(ret).val = val;
	(ret).dval = dval;
	return ret;
}
void d_user_func(float x, float* _dx_Lx1FPc, float y, float* _dy_2bPjNx, float z, float* _dz_mzWAaf, float _dreturn_k5CHXE) {
	float _t_float_6NmcPy[2];
	for (int _i = 0; _i < 2;_i++) {
		_t_float_6NmcPy[_i] = 0;
	}
	int _stack_ptr_float_6NmcPy = (int)(0);
	float _call_t_0_keRCuf;
	_call_t_0_keRCuf = 0;
	float _d_call_t_0_keRCuf_fXAlyq;
	_d_call_t_0_keRCuf_fXAlyq = 0;
	float _call_t_1_HSQD3s;
	_call_t_1_HSQD3s = 0;
	float _d_call_t_1_HSQD3s_3YplNK;
	_d_call_t_1_HSQD3s_3YplNK = 0;
	(_t_float_6NmcPy)[_stack_ptr_float_6NmcPy] = _call_t_0_keRCuf;
	_stack_ptr_float_6NmcPy = (_stack_ptr_float_6NmcPy) + ((int)(1));
	_call_t_0_keRCuf = (y) + (z);
	(_t_float_6NmcPy)[_stack_ptr_float_6NmcPy] = _call_t_1_HSQD3s;
	_stack_ptr_float_6NmcPy = (_stack_ptr_float_6NmcPy) + ((int)(1));
	_call_t_1_HSQD3s = (x) + (z);
	float _adj_0;
	_adj_0 = 0;
	float _adj_1;
	_adj_1 = 0;
	float _adj_2;
	_adj_2 = 0;
	float _adj_3;
	_adj_3 = 0;
	(*_dx_Lx1FPc) += (y) * (_dreturn_k5CHXE);
	(*_dy_2bPjNx) += (x) * (_dreturn_k5CHXE);
	_d_call_t_0_keRCuf_fXAlyq += (cosf(_call_t_0_keRCuf)) * (_dreturn_k5CHXE);
	_d_call_t_1_HSQD3s_3YplNK += (_dreturn_k5CHXE) / (_call_t_1_HSQD3s);
	_stack_ptr_float_6NmcPy = (_stack_ptr_float_6NmcPy) - ((int)(1));
	_call_t_1_HSQD3s = (_t_float_6NmcPy)[_stack_ptr_float_6NmcPy];
	_adj_0 = _d_call_t_1_HSQD3s_3YplNK;
	_adj_1 = _d_call_t_1_HSQD3s_3YplNK;
	_d_call_t_1_HSQD3s_3YplNK = (float)(0.0);
	(*_dx_Lx1FPc) += _adj_0;
	(*_dz_mzWAaf) += _adj_1;
	_stack_ptr_float_6NmcPy = (_stack_ptr_float_6NmcPy) - ((int)(1));
	_call_t_0_keRCuf = (_t_float_6NmcPy)[_stack_ptr_float_6NmcPy];
	_adj_2 = _d_call_t_0_keRCuf_fXAlyq;
	_adj_3 = _d_call_t_0_keRCuf_fXAlyq;
	_d_call_t_0_keRCuf_fXAlyq = (float)(0.0);
	(*_dy_2bPjNx) += _adj_2;
	(*_dz_mzWAaf) += _adj_3;
}
