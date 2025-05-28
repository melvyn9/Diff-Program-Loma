#include <math.h>

typedef struct {
	float val;
	float dval;
} _dfloat;
float square(float x);
_dfloat d_square(_dfloat x);
_dfloat make__dfloat(float val, float dval);
float square(float x) {
	return (x) * (x);
}
_dfloat d_square(_dfloat x) {
	return make__dfloat(((x).val) * ((x).val),(((x).dval) * ((x).val)) + (((x).val) * ((x).dval)));
}
_dfloat make__dfloat(float val, float dval) {
	_dfloat ret;
	ret.val = 0;
	ret.dval = 0;
	(ret).val = val;
	(ret).dval = dval;
	return ret;
}
