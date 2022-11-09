#pragma once

#include <iostream>
#include <math.h>
#include <complex>
#include <map>

typedef int INT;

#define Root2 std::sqrt(2)
//#define Omega std::complex<long double>(1/Root2, 1/Root2)

class ZOmega;

// √2の多項式整数環(Z[√2])クラス
class ZRoot2
{
    INT a;     // a+b√2
    INT b;
public:
    ZRoot2():a(0),b(0){}
    ZRoot2(INT a, INT b):a(a),b(b){}
    inline INT operator[](int n) const
    {
        if(n==0) return a;      // x[0]でx.aを取得
        if(n==1) return b;
        else
        {
            std::cout << "index=" << n << "は参照できません" << std::endl;
            std::exit(1);
        }
    }
    friend std::ostream& operator<<(std::ostream& os, const ZRoot2& x);
    long double get_real(){return a + b*Root2;}    //実数表現を返す
    ZOmega to_ZOmega();
};

INT norm(ZRoot2 x){
    INT a = x[0];    INT b = x[1];
    return std::abs(a*a - 2*b*b);
}

ZRoot2 dot(ZRoot2 x){
    return ZRoot2(x[0], -x[1]);
}

inline ZRoot2 operator+(const ZRoot2& x, const ZRoot2& y){
    return ZRoot2(x[0] + y[0], x[1] + y[1]);
}

inline ZRoot2 operator-(const ZRoot2& x, const ZRoot2& y){
    return ZRoot2(x[0] - y[0], x[1] - y[1]);
}

inline ZRoot2 operator*(const ZRoot2& x, const ZRoot2& y){
    return ZRoot2(x[0]*y[0] + 2*x[1]*y[1], x[0]*y[1] + x[1]*y[0]);
}

inline ZRoot2 operator/(const ZRoot2& x, const ZRoot2& y){
    ZRoot2 numerator = x * dot(y);                     // 有理化後の分子
    INT denominator = y[0]*y[0] - 2*y[1]*y[1];   // 有理化後の分母
    INT a = numerator[0] / denominator;
    INT b = numerator[1] / denominator;
    return ZRoot2(a, b);
}

inline ZRoot2 operator%(const ZRoot2& x, const ZRoot2& y){
    ZRoot2 q = x / y;
    return x - q*y;
}


inline bool operator==(const ZRoot2& x, const ZRoot2& y){
    return ( (x[0] == y[0]) && (x[1] == y[1]) );
}

inline bool operator!=(const ZRoot2& x, const ZRoot2& y){
    return ( (x[0] != y[0]) || (x[1] != y[1]) );
}

std::ostream& operator<<(std::ostream& os, const ZRoot2& x)
{
    os << x.a << " + " << x.b << "√2";
    return os;
}






// ωの多項式整数環(Z[ω])クラス    ω = e^(iπ/4)
class ZOmega
{
    INT a;     // aω^3 + bω^2 + cω + d
    INT b;
    INT c;
    INT d;
public:
    ZOmega():a(0),b(0),c(0),d(0){}
    ZOmega(INT a, INT b, INT c, INT d):a(a),b(b),c(c),d(d){}
    INT get_a() {return a;}
    INT get_b() {return b;}
    INT get_c() {return c;}
    INT get_d() {return d;}
    inline INT operator[](int n) const
    {
        if(n==0) return a;      // x[0]でx.aを取得
        if(n==1) return b;
        if(n==2) return c;
        if(n==3) return d;
        else
        {
            std::cout << "index=" << n << "は参照できません" << std::endl;
            std::exit(1);
        }
    }
    friend std::ostream& operator<<(std::ostream& os, const ZOmega& x);
    ZOmega& operator=(const ZOmega& x);
    ZOmega& operator+=(const ZOmega& x);
    ZOmega& operator-=(const ZOmega& x);
    ZOmega& operator*=(const ZOmega& x);
    ZOmega& operator/=(const ZOmega& x);
    std::complex<long double> get_complex(){
        std::complex<long double> Omega(1/Root2, 1/Root2);
        //return a*Omega*Omega*Omega + b*Omega*Omega + c*Omega + d;
        return (long double)a*Omega*Omega*Omega + (long double)b*Omega*Omega + (long double)c*Omega + (long double)d;
    }    //複素数表現を返す
    void multiple_omega();
    void conj();
};

void ZOmega::multiple_omega(){
    INT tmp_a = a;
    a = b;
    b = c;
    c = d;
    d = -tmp_a;
}

void ZOmega::conj(){
    INT tmp_a = a;
    a = -c;
    b = -b;
    c = -tmp_a;
}

ZOmega ZRoot2::to_ZOmega(){return ZOmega(-b, 0, b, a);}

ZOmega dagger(ZOmega x)
{
    return ZOmega(-x[2], -x[1], -x[0], x[3]);
}

ZOmega dot(ZOmega x)
{
    return ZOmega(-x[0], x[1], -x[2], x[3]);
}

ZRoot2 get_uu_dagger(ZOmega u)
{
    return ZRoot2(u[0]*u[0] + u[1]*u[1] + u[2]*u[2] + u[3]*u[3], u[3]*(u[2] - u[0]) + u[1]*(u[2] + u[0]));
} 

INT norm(ZOmega u)
{
    return norm(get_uu_dagger(u));
}

ZOmega& ZOmega::operator=(const ZOmega& x){
    a = x.a;
    b = x.b;
    c = x.c;
    d = x.d;
    return *this;
}

ZOmega& ZOmega::operator+=(const ZOmega& x){
    a += x.a;
    b += x.b;
    c += x.c;
    d += x.d;
    return *this;
}

ZOmega& ZOmega::operator-=(const ZOmega& x){
    a -= x.a;
    b -= x.b;
    c -= x.c;
    d -= x.d;
    return *this;
}

ZOmega& ZOmega::operator*=(const ZOmega& x){
    INT tmp_a = a;
    INT tmp_b = b;
    INT tmp_c = c;
    a = tmp_a*x.d + tmp_b*x.c + tmp_c*x.b + d*x.a;
    b =-tmp_a*x.a + tmp_b*x.d + tmp_c*x.c + d*x.b;
    c =-tmp_a*x.b - tmp_b*x.a + tmp_c*x.d + d*x.c;
    d =-tmp_a*x.c - tmp_b*x.b - tmp_c*x.a + d*x.d;
    return *this;
}

inline ZOmega operator+(const ZOmega& x, const ZOmega& y){
    ZOmega z = x;
    z += y;
    return z;
}

inline ZOmega operator-(const ZOmega& x, const ZOmega& y){
    ZOmega z = x;
    z -= y;
    return z;
}

inline ZOmega operator*(const ZOmega& x, const ZOmega& y){
    ZOmega z = x;
    z *= y;
    return z;
}

inline ZOmega operator/(const ZOmega& x, const ZOmega& y){
    ZOmega y_dagger = dagger(y);
    ZRoot2 yy_dagger = get_uu_dagger(y);
    ZOmega numerator = x * y_dagger * dot(yy_dagger.to_ZOmega());
    INT denominator = norm(y);
    INT a = numerator[0] / denominator;
    INT b = numerator[1] / denominator;
    INT c = numerator[2] / denominator;
    INT d = numerator[3] / denominator;
    return ZOmega(a,b,c,d);
}

inline ZOmega operator%(const ZOmega& x, const ZOmega& y){
    ZOmega q = x / y;
    return x - q*y;
}

std::ostream& operator<<(std::ostream& os, const ZOmega& x)
{
    os << x.a << "ω^3 + " << x.b << "ω^2 + " << x.c << "ω + " << x.d;
    return os;
}

