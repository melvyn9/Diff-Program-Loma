#include <math.h>

typedef struct {
	float val;
	float dval;
} _dfloat;
float mysin(float x);
void d_mysin(float x, float* _dx_AwjBZP, float _dreturn_QOV3zA);
_dfloat make__dfloat(float val, float dval);
void d_user_func(float x, float* _dx_AwjBZP, float _dreturn_QOV3zA);
float mysin(float x) {
	return sinf(x);
}
void d_mysin(float x, float* _dx_AwjBZP, float _dreturn_QOV3zA) {
	(*_dx_AwjBZP) += (cosf(x)) * (_dreturn_QOV3zA);
}
_dfloat make__dfloat(float val, float dval) {
	_dfloat ret;
	ret.val = 0;
	ret.dval = 0;
	(ret).val = val;
	(ret).dval = dval;
	return ret;
}
void d_user_func(float x, float* _dx_AwjBZP, float _dreturn_QOV3zA) {
	(*_dx_AwjBZP) += (cosf(x)) * (_dreturn_QOV3zA);
}

