#include <math.h>
#include <complex>
#include <iostream>

#define make_ZOmega(a,b,c,d) ((ZOmega){.x=a, .y=b, .z=c, .w=d})

typedef int ITYPE;

std::complex<double> omega(1/sqrt(2), 1/sqrt(2));
std::complex<double> omega2 = omega*omega;
std::complex<double> omega3 = omega*omega*omega;

////////////////////////////////////////////////////////////////////////////////
// ωによる4次体の整数環 Z[ω]　     ω=exp(iπ/4)
// CUDAのBuilt-inのベクトル型を使う　
////////////////////////////////////////////////////////////////////////////////
typedef int4 ZOmega;

////////////////////////////////////////////////////////////////////////////////
// std::complex<double>型に変換
////////////////////////////////////////////////////////////////////////////////
std::complex<double> convert(ZOmega a)
{
    return (double)a.x*omega3 + (double)a.y*omega2 + (double)a.z*omega + (double)a.w;
}

////////////////////////////////////////////////////////////////////////////////
// a*iを計算        (iは虚数単位)
////////////////////////////////////////////////////////////////////////////////
ZOmega multiple_i(ZOmega a)
{
    return make_ZOmega(-a.z, -a.w, a.x, a.y);
}

////////////////////////////////////////////////////////////////////////////////
// a*ωを計算        (ω=exp(iπ/4))
////////////////////////////////////////////////////////////////////////////////
ZOmega multiple_omega(ZOmega a)
{
    return make_ZOmega(a.y, a.z, a.w, -a.x);
}

////////////////////////////////////////////////////////////////////////////////
// 各演算子をオーバーライド
////////////////////////////////////////////////////////////////////////////////
inline ZOmega operator-(ZOmega &a)
{
    return make_ZOmega(-a.x, -a.y, -a.z, -a.w);
}

inline ZOmega operator+(ZOmega a, ZOmega b)
{
    return make_ZOmega(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

inline void operator+=(ZOmega &a, ZOmega b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

inline ZOmega operator-(ZOmega a, ZOmega b)
{
    return make_ZOmega(a.x - b.x, a.y - b.y, a.z - b.y, a.w - b.w);
}

inline void operator-=(ZOmega &a, ZOmega b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
    a.w -= b.w;
}

inline ZOmega operator*(ZOmega a, ZOmega b)
{
    return make_ZOmega(a.x*b.w + a.y*b.z + a.z*b.y + a.w*b.x,
                         -a.x*b.x + a.y*b.w + a.z*b.z + a.w*b.y,
                         -a.x*b.y - a.y*b.x + a.z*b.w + a.w*b.z,
                         -a.x*b.z - a.y*b.y - a.z*b.x + a.w*b.w);
}

inline void operator*=(ZOmega &a, ZOmega b)
{
    ITYPE tmp_x = a.x;
    ITYPE tmp_y = a.y;
    ITYPE tmp_z = a.z;

    a.x = tmp_x*b.w + tmp_y*b.z + tmp_z*b.y + a.w*b.x;
    a.y =-tmp_x*b.x + tmp_y*b.w + tmp_z*b.z + a.w*b.y;
    a.z =-tmp_x*b.y - tmp_y*b.x + tmp_z*b.w + a.w*b.z;
    a.w =-tmp_x*b.z - tmp_y*b.y - tmp_z*b.x + a.w*b.w;
}

std::ostream& operator<<(std::ostream& os, ZOmega &a)
{
    os << a.x << "ω^3 + " << a.y << "ω^2 + " << a.z << "ω + " << a.w;
    return os;
}