#include <math.h>

typedef struct {
	float val;
	float dval;
} _dfloat;
float cube(float x);
_dfloat d_cube(_dfloat x);
_dfloat make__dfloat(float val, float dval);
float cube(float x) {
	return ((x) * (x)) * (x);
}
_dfloat d_cube(_dfloat x) {
	return make__dfloat((((x).val) * ((x).val)) * ((x).val),(((((x).dval) * ((x).val)) + (((x).val) * ((x).dval))) * ((x).val)) + ((((x).val) * ((x).val)) * ((x).dval)));
}
_dfloat make__dfloat(float val, float dval) {
	_dfloat ret;
	ret.val = 0;
	ret.dval = 0;
	(ret).val = val;
	(ret).dval = dval;
	return ret;
}
