#include <math.h>

typedef struct {
	float val;
	float dval;
} _dfloat;
float benchmark(float x);
void d_benchmark(float x, float* _dx_79KDgR, float _dreturn_RMMeAS);
_dfloat make__dfloat(float val, float dval);
void d_user_func(float x, float* _dx_79KDgR, float _dreturn_RMMeAS);
float benchmark(float x) {
	return ((x) * (sinf(x))) + (logf(x));
}
void d_benchmark(float x, float* _dx_79KDgR, float _dreturn_RMMeAS) {
	(*_dx_79KDgR) += (sinf(x)) * (_dreturn_RMMeAS);
	(*_dx_79KDgR) += (cosf(x)) * ((x) * (_dreturn_RMMeAS));
	(*_dx_79KDgR) += (_dreturn_RMMeAS) / (x);
}
_dfloat make__dfloat(float val, float dval) {
	_dfloat ret;
	ret.val = 0;
	ret.dval = 0;
	(ret).val = val;
	(ret).dval = dval;
	return ret;
}
void d_user_func(float x, float* _dx_79KDgR, float _dreturn_RMMeAS) {
	(*_dx_79KDgR) += (sinf(x)) * (_dreturn_RMMeAS);
	(*_dx_79KDgR) += (cosf(x)) * ((x) * (_dreturn_RMMeAS));
	(*_dx_79KDgR) += (_dreturn_RMMeAS) / (x);
}
