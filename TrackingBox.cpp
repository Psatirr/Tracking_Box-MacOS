#include "TrackingBox.h"

//Mode Face (OpenCV) hanya dikompilasi di Windows. Di platform lain (misalnya Mac) otomatis nonaktif dan
//pilihan "Face" tidak men-track apa pun. Untuk memaksa: tambahkan -DTRACKINGBOX_FACE=1 saat compile.
#ifndef TRACKINGBOX_FACE
#ifdef _WIN32
#define TRACKINGBOX_FACE 1
#else
#define TRACKINGBOX_FACE 0
#endif
#endif

#include <math.h>
#include <string.h>
#include <stdio.h>
#include <thread>
#include <condition_variable>
#include <atomic>
#include <algorithm>
using namespace ffglex;

#ifdef _WIN32
#include <windows.h>
#pragma comment( lib, "gdi32.lib" )//Dipakai untuk menggambar teks ke bitmap.
#elif defined( __APPLE__ )
//Mac menggambar teks ke bitmap memakai CoreText (pengganti GDI).
#include <CoreFoundation/CoreFoundation.h>
#include <CoreGraphics/CoreGraphics.h>
#include <CoreText/CoreText.h>
#endif

#if TRACKINGBOX_FACE
//OpenCV (hanya dipakai mode Face). Di-include setelah windows.h supaya makro min/max tidak mengganggu.
#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>
#endif

static CFFGLPluginInfo PluginInfo(
	PluginFactory< TrackingBox >,                         // Create method
	"TRBX",                                               // Plugin unique ID of maximum length 4.
	"Tracking Box",                                       // Plugin name
	2,                                                    // API major version number
	1,                                                    // API minor version number
	2,                                                    // Plugin major version number
	0,                                                    // Plugin minor version number
	FF_EFFECT,                                            // Plugin type
	"Blob and face tracking with shapes, FX, HUD and lines", // Plugin description
	"Tracking Box"                                        // About
);

//Ukuran grid analisis maksimum. Ukuran sebenarnya ditentukan parameter Quality.
static const int GRID_MAX_W = 128;
static const int GRID_MAX_H = 72;

//Ukuran bitmap teks label (dipakai sebagai tekstur).
static const int TEXT_TEX_W   = 2048;
static const int TEXT_TEX_H   = 128;
static const int TEXT_FONT_PX = 96;

//Atlas angka untuk HUD: "0123456789:-." dalam satu baris.
static const int ATLAS_TEX_W = 1024;
static const int ATLAS_TEX_H = 128;
static const wchar_t ATLAS_CHARS[] = L"0123456789:-.";

enum TrackMode
{
	MODE_BRIGHT = 0,
	MODE_COLOR  = 1,
	MODE_MOTION = 2,
	MODE_FACE   = 3
};

struct QualityCfg
{
	int gw;       //Lebar grid analisis
	int gh;       //Tinggi grid analisis
	int samples;  //Sampel per sisi sel (samples x samples per sel)
	int interval; //Analisis tiap N frame
	bool chain;   //true = garis hanya menyambung berurutan (lebih ringan)
};
static const QualityCfg QUALITY[ 3 ] = {
	{ 32, 18, 4, 3, true },  //Low: hemat CPU/GPU
	{ 64, 36, 6, 1, false }, //Balanced
	{ 128, 72, 8, 1, false } //Full performance
};

//Pengaturan mode Face per Quality. Gambar dikecilkan ke w x h (abu-abu) lalu dicek tiap "interval" frame di thread terpisah.
struct FaceCfg
{
	int w;
	int h;
	int interval;   //Kirim gambar baru ke detektor tiap N frame (kalau detektor sedang bebas).
	int threads;    //Thread OpenCV.
	double scale;   //Langkah skala Haar. Lebih besar = lebih cepat, lebih mudah terlewat.
};
static const FaceCfg FACE_QUALITY[ 3 ] = {
	{ 240, 135, 6, 1, 1.20 }, //Low: hemat CPU, wajah harus cukup besar di layar
	{ 320, 180, 3, 2, 1.15 }, //Balanced
	{ 480, 270, 2, 4, 1.10 }  //Full: wajah kecil pun terdeteksi
};
static const int FACE_W_MAX = 480;
static const int FACE_H_MAX = 270;
static const int FACE_HOLD  = 5;//Track wajah ditahan sebanyak ini analisis berturut-turut walau wajah sesaat tidak terdeteksi.

static inline float Lerp( float a, float b, float t )
{
	return a + ( b - a ) * t;
}
static inline float Clamp01( float v )
{
	return v < 0.0f ? 0.0f : ( v > 1.0f ? 1.0f : v );
}
static inline int ClampInt( int v, int lo, int hi )
{
	return v < lo ? lo : ( v > hi ? hi : v );
}

//Setengah sisi shape terkecil dan terbesar (satuan: tinggi gambar) dari parameter Min/Max Shape Size.
static void HalfRange( const float* v, float& minHalf, float& maxHalf )
{
	minHalf = Lerp( 0.01f, 0.25f, Clamp01( v[ PT_MIN_SIZE ] ) );
	maxHalf = Lerp( 0.03f, 0.50f, Clamp01( v[ PT_MAX_SIZE ] ) );
	if( maxHalf < minHalf )
		maxHalf = minHalf;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Shader
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static const char _vertexShaderCode[] = R"(#version 410 core
uniform vec2 MaxUV;

layout( location = 0 ) in vec4 vPosition;
layout( location = 1 ) in vec2 vUV;

out vec2 uv;  //Koordinat untuk membaca tekstur input.
out vec2 nuv; //Koordinat 0..1 di seluruh gambar (untuk menggambar overlay).

void main()
{
	gl_Position = vPosition;
	nuv = vUV;
	uv = vUV * MaxUV;
}
)";

//Pass 1: tiap pixel output = satu sel grid. Isinya skor 0..1:
//  Mode Brightness/Motion: tingkat terang (rata-rata dicampur nilai tertinggi dari sampel di sel itu).
//  Mode Color: seberapa mirip warna sel dengan warna target (tidak terpengaruh terang/gelap).
static const char _analysisFragmentShaderCode[] = R"(#version 410 core
uniform sampler2D InputTexture;
uniform vec2  MaxUV;
uniform vec2  GridSize;
uniform int   Mode;
uniform int   Samples;
uniform vec3  TargetColor;
uniform float Tol;

out vec4 fragColor;

void main()
{
	vec2 cell   = floor( gl_FragCoord.xy );
	vec2 origin = cell / GridSize;
	vec2 size   = vec2( 1.0 ) / GridSize;

	float tmax = max( max( TargetColor.r, TargetColor.g ), TargetColor.b );
	vec3  nt   = TargetColor / max( tmax, 1e-3 );

	float sum = 0.0;
	float mx  = 0.0;
	for( int j = 0; j < 8; ++j )
	{
		if( j >= Samples )
			break;
		for( int i = 0; i < 8; ++i )
		{
			if( i >= Samples )
				break;
			vec2 n = origin + ( vec2( float( i ), float( j ) ) + 0.5 ) / float( Samples ) * size;
			//Input berisi warna premultiplied, jadi area transparan otomatis dianggap gelap.
			vec3 c = texture( InputTexture, n * MaxUV ).rgb;
			float l;
			if( Mode == 1 )
			{
				float mc = max( max( c.r, c.g ), c.b );
				vec3 nc  = c / max( mc, 1e-3 );
				l        = ( 1.0 - clamp( length( nc - nt ) / max( Tol, 1e-3 ), 0.0, 1.0 ) ) * clamp( mc * 3.0, 0.0, 1.0 );
			}
			else
			{
				l = dot( c, vec3( 0.299, 0.587, 0.114 ) );
			}
			sum += l;
			mx = max( mx, l );
		}
	}
	float avg = sum / float( Samples * Samples );
	//Mode 2 = rata-rata murni (dipakai detektor wajah). Mode lain = campuran rata-rata dan nilai tertinggi.
	float l = ( Mode == 2 ) ? avg : mix( avg, mx, 0.5 );
	fragColor = vec4( l, l, l, 1.0 );
}
)";

//Pass 2 (bagian 1): deklarasi, fungsi bantu, bentuk dasar, dan FX.
static const char _mainShaderPart1[] = R"(#version 410 core
uniform sampler2D InputTexture;
uniform sampler2D TextTexture;
uniform sampler2D AtlasTexture;

uniform vec2  MaxUV;
uniform float Aspect;      //lebar / tinggi gambar
uniform float PixelSize;   //1 / tinggi gambar dalam pixel
uniform float Time;
uniform int   TrackCount;
uniform vec4  TrackA[8];   //x, y (0..1), setengah ukuran (satuan: tinggi gambar), alpha
uniform vec4  TrackB[8];   //x dan y sebagai angka 0..999 untuk HUD, indeks, seed

uniform float ShapeOn;
uniform int   ShapeStyle;
uniform int   FxStyle;
uniform float ShapeThick;
uniform vec3  TrackerColor;

uniform float HudOn;
uniform int   HudPos;
uniform float HudH;        //tinggi satu baris teks HUD
uniform float TextAspect;  //0 = tidak ada teks label
uniform vec2  TextUV;
uniform float AtlasAspect; //0 = atlas angka tidak ada
uniform float AtlasCharW;
uniform float AtlasUVy;

uniform float LineOn;
uniform int   LineWay;     //0 lurus, 1 lengkung
uniform float LineDashed;
uniform float CurveAmt;
uniform float LineThick;
uniform float DashLen;
uniform vec3  LineColor;
uniform float LineChain;

uniform float VideoBg;

in vec2 uv;
in vec2 nuv;

out vec4 fragColor;

float Hash11( float x )
{
	return fract( sin( x * 127.1 ) * 43758.5453 );
}
float Lum( vec3 c )
{
	return dot( c, vec3( 0.299, 0.587, 0.114 ) );
}
//Baca gambar input di posisi p (ruang yang sudah dikoreksi aspek).
vec4 Tex( vec2 p )
{
	return textureLod( InputTexture, clamp( p / vec2( Aspect, 1.0 ), vec2( 0.0 ), vec2( 1.0 ) ) * MaxUV, 0.0 );
}
float Cov( float sd, float px )
{
	return 1.0 - smoothstep( -0.5 * px, 0.5 * px, sd );
}
float Ring( float sd, float t, float px )
{
	return 1.0 - smoothstep( 0.5 * t - 0.5 * px, 0.5 * t + 0.5 * px, abs( sd + 0.5 * t ) );
}
float SegDist( vec2 p, vec2 a, vec2 b )
{
	vec2 pa = p - a;
	vec2 ba = b - a;
	float h = clamp( dot( pa, ba ) / max( dot( ba, ba ), 1e-8 ), 0.0, 1.0 );
	return length( pa - ba * h );
}
float SegCov( vec2 p, vec2 a, vec2 b, float t, float px )
{
	return 1.0 - smoothstep( 0.5 * t - 0.5 * px, 0.5 * t + 0.5 * px, SegDist( p, a, b ) );
}

float sdBox( vec2 p, float s )
{
	vec2 d = abs( p ) - vec2( s );
	return length( max( d, 0.0 ) ) + min( max( d.x, d.y ), 0.0 );
}
float sdTri( vec2 p, float r )
{
	const float k = 1.7320508;
	p.x = abs( p.x ) - r;
	p.y = p.y + r / k;
	if( p.x + k * p.y > 0.0 )
		p = vec2( p.x - k * p.y, -k * p.x - p.y ) / 2.0;
	p.x -= clamp( p.x, -2.0 * r, 0.0 );
	return -length( p ) * sign( p.y );
}
float sdHex( vec2 p, float r )
{
	const vec3 k = vec3( -0.866025404, 0.5, 0.577350269 );
	p = abs( p );
	p -= 2.0 * min( dot( k.xy, p ), 0.0 ) * k.xy;
	p -= vec2( clamp( p.x, -k.z * r, k.z * r ), r );
	return length( p ) * sign( p.y );
}
float sdStar5( vec2 p, float r, float rf )
{
	const vec2 k1 = vec2( 0.809016994375, -0.587785252292 );
	const vec2 k2 = vec2( -0.809016994375, -0.587785252292 );
	p.x = abs( p.x );
	p -= 2.0 * max( dot( k1, p ), 0.0 ) * k1;
	p -= 2.0 * max( dot( k2, p ), 0.0 ) * k2;
	p.x = abs( p.x );
	p.y -= r;
	vec2 ba = rf * vec2( -k1.y, k1.x ) - vec2( 0.0, 1.0 );
	float h = clamp( dot( p, ba ) / dot( ba, ba ), 0.0, r );
	return length( p - ba * h ) * sign( p.y * ba.x - p.x * ba.y );
}

vec3 Heat( float x )
{
	x = clamp( x, 0.0, 1.0 );
	return clamp( vec3( 1.5 - abs( 4.0 * x - 3.0 ), 1.5 - abs( 4.0 * x - 2.0 ), 1.5 - abs( 4.0 * x - 1.0 ) ), 0.0, 1.0 );
}
float Bayer4( vec2 fc )
{
	int m[ 16 ] = int[ 16 ]( 0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5 );
	int x       = int( mod( fc.x, 4.0 ) );
	int y       = int( mod( fc.y, 4.0 ) );
	return ( float( m[ y * 4 + x ] ) + 0.5 ) / 16.0;
}

//FX yang diterapkan pada isi shape. b = pixel input asli (premultiplied).
vec4 ApplyFx( int fx, vec2 P, vec4 b )
{
	if( fx == 1 )
		return vec4( vec3( b.a ) - b.rgb, b.a );//Invert
	if( fx == 2 )
		return vec4( floor( b.rgb * 4.0 + 0.5 ) / 4.0, b.a );//Posterize
	if( fx == 3 )
		return vec4( vec3( Lum( b.rgb ) ), b.a );//Monochrome
	if( fx == 4 )
		return vec4( Heat( Lum( b.rgb ) ) * b.a, b.a );//Thermal
	if( fx == 5 )
		return vec4( vec3( step( Bayer4( gl_FragCoord.xy ), Lum( b.rgb ) ) ) * b.a, b.a );//Dither
	if( fx == 6 )
	{
		float blk = 14.0 * PixelSize;//Pixelate
		return Tex( ( floor( P / blk ) + 0.5 ) * blk );
	}
	if( fx == 7 )//Glitch
	{
		float row = floor( P.y / 0.018 );
		float tt  = floor( Time * 10.0 );
		float h   = Hash11( row + tt * 13.7 );
		float dir = ( Hash11( row + tt ) > 0.5 ) ? 1.0 : -1.0;
		float sh  = ( h > 0.82 ) ? ( h - 0.82 ) * 0.9 * dir : 0.0;
		float sp  = 0.004 + step( 0.82, h ) * 0.01;
		vec2 o    = vec2( sh, 0.0 );
		return vec4( Tex( P + o + vec2( sp, 0.0 ) ).r, Tex( P + o ).g, Tex( P + o - vec2( sp, 0.0 ) ).b, b.a );
	}
	if( fx == 8 )//X-Ray
	{
		float l = Lum( b.rgb );
		float e = abs( Lum( Tex( P + vec2( PixelSize * 1.5, 0.0 ) ).rgb ) - l ) + abs( Lum( Tex( P + vec2( 0.0, PixelSize * 1.5 ) ).rgb ) - l );
		vec3 c  = vec3( 0.15, 0.75, 1.0 ) * ( 1.0 - l ) * 0.9 + vec3( e * 5.0 );
		return vec4( c * b.a, b.a );
	}
	return b;//Normal
}
)";

//Pass 2 (bagian 2): bentuk tracker, garis penghubung, dan HUD.
static const char _mainShaderPart2[] = R"(
//Mengembalikan (isi, garis tepi). Isi = area tempat FX diterapkan.
vec2 Shape( vec2 p, float s, float t, float px, int style )
{
	vec2 q  = abs( p );
	float sd = 0.0;
	float ol = 0.0;

	if( style == 0 )//Square
	{
		sd = sdBox( p, s );
		ol = Ring( sd, t, px );
	}
	else if( style == 1 )//Circle
	{
		sd = length( p ) - s;
		ol = Ring( sd, t, px );
	}
	else if( style == 2 )//Triangle
	{
		sd = sdTri( p, s * 0.9 );
		ol = Ring( sd, t, px );
	}
	else if( style == 3 )//Win2000: kotak dengan title bar
	{
		sd         = sdBox( p, s );
		ol         = Ring( sd, t, px );
		float bar  = Cov( sd, px ) * step( 0.62 * s, p.y );
		float cut  = Cov( sdBox( p - vec2( 0.84 * s, 0.81 * s ), 0.07 * s ), px );
		ol         = max( ol, bar * ( 1.0 - cut ) );
	}
	else if( style == 4 )//Scope HUD: lingkaran + tanda silang + tick
	{
		sd = length( p ) - 0.85 * s;
		ol = Ring( sd, t, px );
		ol = max( ol, SegCov( q, vec2( 0.7 * s, 0.0 ), vec2( 1.05 * s, 0.0 ), t, px ) );
		ol = max( ol, SegCov( q.yx, vec2( 0.7 * s, 0.0 ), vec2( 1.05 * s, 0.0 ), t, px ) );
		ol = max( ol, SegCov( q, vec2( 0.0 ), vec2( 0.12 * s, 0.0 ), t, px ) );
		ol = max( ol, SegCov( q.yx, vec2( 0.0 ), vec2( 0.12 * s, 0.0 ), t, px ) );
	}
	else if( style == 5 )//L-Frame: hanya sudut-sudut kotak
	{
		sd = sdBox( p, s );
		ol = Ring( sd, t, px ) * smoothstep( 0.6 * s - px, 0.6 * s, min( q.x, q.y ) );
	}
	else if( style == 6 )//Grid Box: kotak dengan garis 3x3
	{
		sd        = sdBox( p, s );
		ol        = Ring( sd, t, px );
		float gx  = 1.0 - smoothstep( 0.25 * t - 0.5 * px, 0.25 * t + 0.5 * px, abs( q.x - s / 3.0 ) );
		float gy  = 1.0 - smoothstep( 0.25 * t - 0.5 * px, 0.25 * t + 0.5 * px, abs( q.y - s / 3.0 ) );
		ol        = max( ol, max( gx, gy ) * Cov( sd, px ) * 0.8 );
	}
	else if( style == 7 )//Particle: inti kecil dikelilingi titik-titik
	{
		sd         = length( p ) - 0.3 * s;
		ol         = Ring( sd, t, px );
		float a    = atan( p.y, p.x );
		float sec  = 6.2831853 / 8.0;
		float aa   = mod( a + 0.5 * sec, sec ) - 0.5 * sec;
		vec2 dq    = length( p ) * vec2( cos( aa ), sin( aa ) ) - vec2( s, 0.0 );
		float dotd = length( dq ) - max( 1.2 * t, 0.05 * s );
		ol         = max( ol, Cov( dotd, px ) );
	}
	else if( style == 8 )//Prism: segi enam dengan garis kubus
	{
		float r = 0.95 * s;
		sd      = sdHex( p, r );
		ol      = Ring( sd, t, px );
		float R = 1.1547 * r;
		ol      = max( ol, SegCov( p, vec2( 0.0 ), vec2( R, 0.0 ), t, px ) );
		ol      = max( ol, SegCov( p, vec2( 0.0 ), vec2( -0.5 * R, 0.866 * R ), t, px ) );
		ol      = max( ol, SegCov( p, vec2( 0.0 ), vec2( -0.5 * R, -0.866 * R ), t, px ) );
	}
	else//Star
	{
		sd = sdStar5( p, s * 0.85, 0.45 );
		ol = Ring( sd, t, px );
	}

	return vec2( Cov( sd, px ), ol );
}

//Garis dari titik A ke B (lurus atau lengkung, solid atau putus-putus). pid dipakai untuk membalik arah lengkungan.
float LineCov( vec2 P, vec2 A, vec2 B, float t, float px, float pid )
{
	vec2 ab   = B - A;
	float len = max( length( ab ), 1e-4 );
	vec2 nrm  = vec2( -ab.y, ab.x ) / len;
	float sgn = ( mod( pid, 2.0 ) < 0.5 ) ? 1.0 : -1.0;
	float bul = ( LineWay == 1 ) ? len * CurveAmt * 0.35 * sgn : 0.0;
	vec2 M    = 0.5 * ( A + B ) + nrm * bul * 2.0;

	//Buang cepat: pixel di luar kotak pembatas garis.
	vec2 lo = min( min( A, B ), M ) - vec2( t );
	vec2 hi = max( max( A, B ), M ) + vec2( t );
	if( P.x < lo.x || P.y < lo.y || P.x > hi.x || P.y > hi.y )
		return 0.0;

	float d = 1e9;
	float u = 0.0;
	if( LineWay == 0 )
	{
		vec2 pa = P - A;
		float h = clamp( dot( pa, ab ) / ( len * len ), 0.0, 1.0 );
		d       = length( pa - ab * h );
		u       = h * len;
	}
	else
	{
		vec2 prev = A;
		for( int k = 1; k <= 8; ++k )
		{
			float tt  = float( k ) / 8.0;
			vec2 cur  = ( 1.0 - tt ) * ( 1.0 - tt ) * A + 2.0 * ( 1.0 - tt ) * tt * M + tt * tt * B;
			vec2 sa   = cur - prev;
			float h   = clamp( dot( P - prev, sa ) / max( dot( sa, sa ), 1e-8 ), 0.0, 1.0 );
			float dd  = length( P - prev - sa * h );
			if( dd < d )
			{
				d = dd;
				u = ( float( k - 1 ) + h ) / 8.0 * len * 1.12;
			}
			prev = cur;
		}
	}

	float cov = 1.0 - smoothstep( 0.5 * t - 0.5 * px, 0.5 * t + 0.5 * px, d );
	if( LineDashed > 0.5 )
		cov *= step( mod( u, 2.0 * DashLen ), DashLen );
	return cov;
}

//Teks HUD di sekitar shape: baris bawah = koordinat "XXX:YYY", baris atas = teks label dari parameter Text.
float HudText( vec2 P, vec2 C, float s, vec4 B )
{
	float hs     = HudH;
	float hasAt  = ( AtlasAspect > 0.0 ) ? 1.0 : 0.0;
	float hasLab = ( TextAspect > 0.0 ) ? 1.0 : 0.0;
	float cellW  = hs * max( AtlasAspect, 0.0001 );
	float coordW = hasAt * 7.0 * cellW;
	float labW   = hasLab * hs * TextAspect;
	float w      = max( max( coordW, labW ), hs );
	float rows   = max( hasAt + hasLab, 1.0 );
	float h      = rows * hs;
	float gap    = 0.012;

	vec2 o;
	if( HudPos == 0 )//Right
	{
		o = vec2( C.x + s + gap, C.y - 0.5 * h );
		if( o.x + w > Aspect )
			o.x = C.x - s - gap - w;
	}
	else if( HudPos == 1 )//Left
	{
		o = vec2( C.x - s - gap - w, C.y - 0.5 * h );
		if( o.x < 0.0 )
			o.x = C.x + s + gap;
	}
	else if( HudPos == 2 )//Bottom
	{
		o = vec2( C.x - 0.5 * w, C.y - s - gap - h );
		if( o.y < 0.0 )
			o.y = C.y + s + gap;
	}
	else//Top
	{
		o = vec2( C.x - 0.5 * w, C.y + s + gap );
		if( o.y + h > 1.0 )
			o.y = C.y - s - gap - h;
	}
	o.x = clamp( o.x, 0.0, max( Aspect - w, 0.0 ) );
	o.y = clamp( o.y, 0.0, max( 1.0 - h, 0.0 ) );

	vec2 r = P - o;

	//Baris koordinat
	float ci = floor( r.x / cellW );
	float fx = clamp( fract( r.x / cellW ), 0.03, 0.97 );
	float fy = clamp( r.y / hs, 0.0, 1.0 );
	int c    = int( clamp( ci, 0.0, 6.0 ) );
	int xi   = int( B.x );
	int yi   = int( B.y );
	int code = 10;
	if( c == 0 )
		code = ( xi / 100 ) % 10;
	else if( c == 1 )
		code = ( xi / 10 ) % 10;
	else if( c == 2 )
		code = xi % 10;
	else if( c == 4 )
		code = ( yi / 100 ) % 10;
	else if( c == 5 )
		code = ( yi / 10 ) % 10;
	else if( c == 6 )
		code = yi % 10;
	vec2 auv   = vec2( ( float( code ) + fx ) * AtlasCharW, ( 1.0 - fy ) * AtlasUVy );
	float dA   = texture( AtlasTexture, clamp( auv, vec2( 0.0 ), vec2( 1.0 ) ) ).r;
	float inCo = hasAt * step( 0.0, r.y ) * step( r.y, hs ) * step( 0.0, r.x ) * step( r.x, coordW );

	//Baris label
	float ly   = ( r.y - hasAt * hs ) / hs;
	vec2 tuv   = vec2( r.x / max( labW, 1e-5 ) * TextUV.x, ( 1.0 - ly ) * TextUV.y );
	float tA   = texture( TextTexture, clamp( tuv, vec2( 0.0 ), TextUV ) ).r;
	float inLb = hasLab * step( 0.0, r.x ) * step( r.x, labW ) * step( 0.0, ly ) * step( ly, 1.0 );

	return max( dA * inCo, tA * inLb );
}
)";

//Pass 2 (bagian 3): program utama.
static const char _mainShaderPart3[] = R"(
vec4 Over( vec4 dst, vec3 c, float a )
{
	return vec4( c * a, a ) + dst * ( 1.0 - a );
}

void main()
{
	vec4 base = texture( InputTexture, uv );
	vec2 P    = nuv * vec2( Aspect, 1.0 );
	float px  = PixelSize;
	float t   = max( ShapeThick, 1.5 * px );

	//VIDEO BG mati: latar transparan, hanya isi shape (dengan FX) yang tampil.
	vec4 res = ( VideoBg > 0.5 ) ? base : vec4( 0.0 );

	float outA = 0.0;
	float hudA = 0.0;
	for( int i = 0; i < 8; ++i )
	{
		if( i >= TrackCount )
			break;
		vec4 A = TrackA[ i ];
		if( A.w < 0.01 )
			continue;

		vec2 C = A.xy * vec2( Aspect, 1.0 );
		vec2 p = P - C;
		float s = A.z;

		if( ShapeOn > 0.5 && abs( p.x ) < s * 1.3 + t && abs( p.y ) < s * 1.3 + t )
		{
			vec2 sh = Shape( p, s, t, px, ShapeStyle );
			if( sh.x > 0.001 )
				res = mix( res, ApplyFx( FxStyle, P, base ), sh.x * A.w );
			outA = max( outA, sh.y * A.w );
		}

		if( HudOn > 0.5 )
			hudA = max( hudA, HudText( P, C, s, TrackB[ i ] ) * A.w );
	}

	//Garis penghubung antar track: semua ke semua (atau berurutan saja kalau mode hemat).
	float lineA = 0.0;
	if( LineOn > 0.5 )
	{
		float lt = max( LineThick, 1.5 * px );
		for( int i = 0; i < 8; ++i )
		{
			if( i >= TrackCount )
				break;
			vec4 a = TrackA[ i ];
			if( a.w < 0.05 )
				continue;
			for( int j = i + 1; j < 8; ++j )
			{
				if( j >= TrackCount )
					break;
				if( LineChain > 0.5 && j != i + 1 )
					break;
				vec4 b = TrackA[ j ];
				if( b.w < 0.05 )
					continue;
				lineA = max( lineA, LineCov( P, a.xy * vec2( Aspect, 1.0 ), b.xy * vec2( Aspect, 1.0 ), lt, px, float( i * 8 + j ) ) * min( a.w, b.w ) );
			}
		}
	}

	res = Over( res, LineColor, lineA );
	res = Over( res, TrackerColor, outA );
	res = Over( res, TrackerColor, hudA );
	fragColor = res;
}
)";

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Detektor wajah (OpenCV, Haar cascade)
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if TRACKINGBOX_FACE
//FACE_DETECTOR_BEGIN

#ifdef _WIN32
typedef std::wstring PathStr;
#else
typedef std::string PathStr;
#endif

static const char* const FACE_CASCADE_FILE = "haarcascade_frontalface_default.xml";
static const char* const FACE_LOG_FILE     = "TrackingBox_face_log.txt";
static const int FACE_MIN_PX               = 24;//Wajah terkecil yang dicari (dalam pixel gambar analisis).

static PathStr MakePath( const PathStr& dir, const char* name )
{
	PathStr p = dir;
	for( const char* c = name; *c != '\0'; ++c )
		p.push_back( (PathStr::value_type)*c );
	return p;
}

//Folder tempat file plugin (.dll) berada, diakhiri tanda garis miring.
static PathStr PluginDirectory()
{
#ifdef _WIN32
	HMODULE module = NULL;
	if( !GetModuleHandleExW( GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, (LPCWSTR)&PluginDirectory, &module ) )
		return PathStr();
	wchar_t buffer[ 1024 ];
	DWORD n = GetModuleFileNameW( module, buffer, 1024 );
	if( n == 0 || n >= 1024 )
		return PathStr();
	std::wstring path( buffer, n );
	size_t slash = path.find_last_of( L"\\/" );
	if( slash == std::wstring::npos )
		return PathStr();
	return path.substr( 0, slash + 1 );
#else
	return PathStr( "./" );
#endif
}

//Subfolder tempat DLL OpenCV dan file model disimpan.
static PathStr ModelSubDir()
{
#ifdef _WIN32
	return PathStr( L"TrackingBox.Cv\\" );
#else
	return PathStr( "TrackingBox.Cv/" );
#endif
}

static bool ReadWholeFile( const PathStr& path, std::string& out )
{
#ifdef _WIN32
	FILE* f = _wfopen( path.c_str(), L"rb" );
#else
	FILE* f = fopen( path.c_str(), "rb" );
#endif
	if( f == NULL )
		return false;
	out.clear();
	char chunk[ 65536 ];
	size_t n;
	while( ( n = fread( chunk, 1, sizeof( chunk ), f ) ) > 0 )
		out.append( chunk, n );
	fclose( f );
	return !out.empty();
}

template< class S >
static std::string AsciiOnly( const S& s )
{
	std::string r;
	for( size_t i = 0; i < s.size(); ++i )
		r.push_back( (unsigned int)s[ i ] < 128u ? (char)s[ i ] : '?' );
	return r;
}

//Menulis catatan status singkat di sebelah plugin, supaya mudah dicek kalau mode Face tidak jalan.
static void WriteFaceLog( const PathStr& dir, const std::string& message )
{
	PathStr path = MakePath( dir, FACE_LOG_FILE );
#ifdef _WIN32
	FILE* f = _wfopen( path.c_str(), L"w" );
#else
	FILE* f = fopen( path.c_str(), "w" );
#endif
	if( f == NULL )
		return;
	fputs( message.c_str(), f );
	fputs( "\n", f );
	fclose( f );
}

//Detektor wajah yang berjalan di thread sendiri, supaya render Resolume tidak pernah menunggu OpenCV.
//Alur: thread render memberi gambar abu-abu kecil lewat Submit(); thread detektor mengerjakannya; hasil terbaru
//diambil thread render lewat Fetch(). Kalau detektor masih sibuk, gambar baru dilewati (bukan ditumpuk).
class FaceDetector
{
public:
	enum Status
	{
		ST_LOADING = 0,
		ST_READY   = 1,
		ST_FAILED  = 2
	};
	struct Settings
	{
		int minNeighbors;  //Makin besar = makin ketat (sedikit salah deteksi).
		double scaleFactor;//Makin kecil = makin teliti tapi makin berat.
		int threads;       //Jumlah thread OpenCV.
	};

	explicit FaceDetector( const PathStr& pluginDir ) :
		dir( pluginDir ),
		status( ST_LOADING ),
		busy( false ),
		stopRequested( false ),
		hasJob( false ),
		jobW( 0 ),
		jobH( 0 ),
		resultSeq( 0 )
	{
		jobSettings.minNeighbors = 3;
		jobSettings.scaleFactor  = 1.15;
		jobSettings.threads      = 1;
		worker                   = std::thread( &FaceDetector::Run, this );
	}
	~FaceDetector()
	{
		{
			std::lock_guard< std::mutex > lock( mutex );
			stopRequested = true;
		}
		cond.notify_all();
		if( worker.joinable() )
			worker.join();
	}
	FaceDetector( const FaceDetector& ) = delete;
	FaceDetector& operator=( const FaceDetector& ) = delete;

	Status GetStatus() const
	{
		return (Status)status.load();
	}
	bool CanSubmit() const
	{
		return status.load() == ST_READY && !busy.load();
	}

	//gray = w*h byte, baris pertama = BAGIAN BAWAH gambar (urutan asli dari OpenGL).
	bool Submit( const unsigned char* gray, int w, int h, const Settings& settings )
	{
		if( status.load() != ST_READY || w <= 0 || h <= 0 )
			return false;
		std::unique_lock< std::mutex > lock( mutex, std::try_to_lock );
		if( !lock.owns_lock() || hasJob || busy.load() )
			return false;
		jobBuffer.assign( gray, gray + (size_t)w * (size_t)h );
		jobW        = w;
		jobH        = h;
		jobSettings = settings;
		hasJob      = true;
		lock.unlock();
		cond.notify_one();
		return true;
	}

	//Mengambil hasil terbaru kalau lebih baru dari lastSeq. Wajah terbesar ada di urutan pertama.
	bool Fetch( unsigned int& lastSeq, std::vector< FaceRect >& out )
	{
		std::lock_guard< std::mutex > lock( mutex );
		if( resultSeq == lastSeq )
			return false;
		out     = result;
		lastSeq = resultSeq;
		return true;
	}

private:
	bool LoadCascade( cv::CascadeClassifier& cascade )
	{
		PathStr folders[ 2 ] = { dir + ModelSubDir(), dir };
		std::string xml;
		PathStr foundIn;
		for( int i = 0; i < 2; ++i )
		{
			if( ReadWholeFile( MakePath( folders[ i ], FACE_CASCADE_FILE ), xml ) )
			{
				foundIn = folders[ i ];
				break;
			}
		}
		if( xml.empty() )
		{
			std::string msg = std::string( "ERROR: file model wajah tidak ditemukan: " ) + FACE_CASCADE_FILE +
							  "\nDicari di: " + AsciiOnly( folders[ 0 ] ) + "\nDan di: " + AsciiOnly( folders[ 1 ] );
			WriteFaceLog( dir, msg );
			FFGLLog::LogToHost( "TrackingBox: file model wajah tidak ditemukan" );
			return false;
		}

		//File dibaca sendiri lalu diberikan ke OpenCV lewat memori, supaya nama folder berhuruf non-ASCII tidak jadi masalah.
		try
		{
			cv::FileStorage fs( xml, cv::FileStorage::READ | cv::FileStorage::MEMORY );
			if( !fs.isOpened() || !cascade.read( fs.getFirstTopLevelNode() ) || cascade.empty() )
			{
				WriteFaceLog( dir, "ERROR: file model wajah ditemukan tapi gagal dibaca OpenCV: " + AsciiOnly( foundIn ) );
				FFGLLog::LogToHost( "TrackingBox: file model wajah gagal dibaca" );
				return false;
			}
		}
		catch( ... )
		{
			WriteFaceLog( dir, "ERROR: OpenCV melempar error saat membaca model wajah." );
			return false;
		}

		WriteFaceLog( dir, "OK: model wajah dimuat dari " + AsciiOnly( foundIn ) );
		FFGLLog::LogToHost( "TrackingBox: model wajah dimuat" );
		return true;
	}

	void Run()
	{
		cv::CascadeClassifier cascade;
		bool loaded = false;
		try
		{
			loaded = LoadCascade( cascade );
		}
		catch( ... )
		{
			loaded = false;
		}
		if( !loaded )
		{
			status = ST_FAILED;
			return;
		}
		status = ST_READY;

		std::vector< unsigned char > buffer;
		std::vector< cv::Rect > rects;
		cv::Mat flipped;

		for( ;; )
		{
			int w = 0;
			int h = 0;
			Settings settings;
			{
				std::unique_lock< std::mutex > lock( mutex );
				cond.wait( lock, [ this ] { return stopRequested || hasJob; } );
				if( stopRequested )
					return;
				buffer.swap( jobBuffer );
				w        = jobW;
				h        = jobH;
				settings = jobSettings;
				hasJob   = false;
				busy     = true;
			}

			std::vector< FaceRect > faces;
			try
			{
				cv::setNumThreads( settings.threads );
				cv::Mat gray( h, w, CV_8UC1, buffer.data() );
				cv::flip( gray, flipped, 0 );//Dari urutan OpenGL (bawah dulu) ke urutan gambar biasa (atas dulu).
				rects.clear();
				cascade.detectMultiScale( flipped, rects, settings.scaleFactor, settings.minNeighbors, cv::CASCADE_SCALE_IMAGE, cv::Size( FACE_MIN_PX, FACE_MIN_PX ) );
				for( size_t i = 0; i < rects.size(); ++i )
				{
					FaceRect f;
					f.cx   = ( (float)rects[ i ].x + 0.5f * (float)rects[ i ].width ) / (float)w;
					f.cy   = 1.0f - ( (float)rects[ i ].y + 0.5f * (float)rects[ i ].height ) / (float)h;//Balik lagi: 0 = bawah.
					f.size = (float)rects[ i ].height / (float)h;
					faces.push_back( f );
				}
				//Urutkan dari yang terbesar (insertion sort, jumlahnya kecil).
				for( size_t i = 1; i < faces.size(); ++i )
				{
					FaceRect key = faces[ i ];
					size_t j     = i;
					while( j > 0 && faces[ j - 1 ].size < key.size )
					{
						faces[ j ] = faces[ j - 1 ];
						--j;
					}
					faces[ j ] = key;
				}
			}
			catch( ... )
			{
				faces.clear();
			}

			{
				std::lock_guard< std::mutex > lock( mutex );
				result.swap( faces );
				++resultSeq;
				busy = false;
			}
		}
	}

	PathStr dir;
	std::thread worker;
	std::atomic< int > status;
	std::atomic< bool > busy;

	std::mutex mutex;
	std::condition_variable cond;
	bool stopRequested;
	bool hasJob;
	std::vector< unsigned char > jobBuffer;
	int jobW;
	int jobH;
	Settings jobSettings;
	std::vector< FaceRect > result;
	unsigned int resultSeq;
};
//FACE_DETECTOR_END
#else
//Platform tanpa OpenCV: detektor "kosong" yang selalu gagal, supaya sisa kode tidak perlu diubah.
typedef std::string PathStr;
static PathStr PluginDirectory()
{
	return PathStr();
}
class FaceDetector
{
public:
	enum Status
	{
		ST_LOADING = 0,
		ST_READY   = 1,
		ST_FAILED  = 2
	};
	struct Settings
	{
		int minNeighbors;
		double scaleFactor;
		int threads;
	};
	explicit FaceDetector( const PathStr& )
	{
		FFGLLog::LogToHost( "TrackingBox: mode Face tidak tersedia di platform ini" );
	}
	Status GetStatus() const
	{
		return ST_FAILED;
	}
	bool CanSubmit() const
	{
		return false;
	}
	bool Submit( const unsigned char*, int, int, const Settings& )
	{
		return false;
	}
	bool Fetch( unsigned int&, std::vector< FaceRect >& )
	{
		return false;
	}
};
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Plugin
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static const char* const OPT_QUALITY[]     = { "Low", "Balanced", "Full" };
static const char* const OPT_MODE[]        = { "Brightness", "Color", "Motion", "Face" };
static const char* const OPT_TRACKS[]      = { "1", "2", "3", "4", "5", "6", "7", "8" };
static const char* const OPT_SHAPE_STYLE[] = { "Square", "Circle", "Triangle", "Win2000", "Scope HUD", "L-Frame", "Grid Box", "Particle", "Prism", "Star" };
static const char* const OPT_FX_STYLE[]    = { "Normal", "Invert", "Posterize", "Monochrome", "Thermal", "Dither", "Pixelate", "Glitch", "X-Ray" };
static const char* const OPT_HUD_POS[]     = { "Right", "Left", "Bottom", "Top" };
static const char* const OPT_LINE_WAY[]    = { "Straight", "Curve" };
static const char* const OPT_LINE_STYLE[]  = { "Solid", "Dashed" };

TrackingBox::TrackingBox() :
	readback( (size_t)GRID_MAX_W * GRID_MAX_H * 4, 0 ),
	score( (size_t)GRID_MAX_W * GRID_MAX_H, 0.0f ),
	work( (size_t)GRID_MAX_W * GRID_MAX_H, 0.0f ),
	prevLum( (size_t)GRID_MAX_W * GRID_MAX_H, 0.0f ),
	havePrev( false ),
	lastGw( 0 ),
	lastGh( 0 ),
	frameCounter( 0 ),
	lastMode( -1 ),
	faceGray( (size_t)FACE_W_MAX * FACE_H_MAX, 0 ),
	faceFrames( 0 ),
	faceSeq( 0 ),
	text( "TRACK" ),
	textDirty( true ),
	textTexture( 0 ),
	hasText( false ),
	textAspect( 0.0f ),
	textUVx( 1.0f ),
	textUVy( 1.0f ),
	atlasTexture( 0 ),
	atlasAspect( 0.0f ),
	atlasCharW( 0.0f ),
	atlasUVy( 1.0f ),
	startTime( std::chrono::steady_clock::now() )
{
	for( unsigned int i = 0; i < (unsigned int)PT_COUNT; ++i )
		values[ i ] = 0.0f;

	for( int i = 0; i < MAX_TRACKS; ++i )
	{
		Track& t = tracks[ i ];
		t.x = t.tx = 0.5f;
		t.y = t.ty = 0.5f;
		t.size = t.tsize = 0.5f;
		t.alpha          = 0.0f;
		t.detected       = false;
		t.miss           = 0;
		t.seed           = (float)i * 0.37f + 0.11f;
	}

	SetMinInputs( 1 );
	SetMaxInputs( 1 );

	//Daftarkan parameter. Urutannya harus sama dengan enum ParamType.
	AddOptionParam( PT_QUALITY, "Quality", 3, OPT_QUALITY, 1 );
	AddOptionParam( PT_MODE, "Track Mode", 4, OPT_MODE, 0 );
	AddOptionParam( PT_TRACKS, "Tracks", 8, OPT_TRACKS, 3 );
	AddFloatParam( PT_THRESHOLD, "Threshold", FF_TYPE_STANDARD, 0.15f );
	AddFloatParam( PT_TARGET_R, "Target Color", FF_TYPE_RED, 1.0f );
	AddFloatParam( PT_TARGET_G, "Target Color_Green", FF_TYPE_GREEN, 0.2f );
	AddFloatParam( PT_TARGET_B, "Target Color_Blue", FF_TYPE_BLUE, 0.2f );
	AddFloatParam( PT_TOLERANCE, "Color Tolerance", FF_TYPE_STANDARD, 0.4f );
	AddFloatParam( PT_MIN_SIZE, "Min Shape Size", FF_TYPE_STANDARD, 0.2f );
	AddFloatParam( PT_MAX_SIZE, "Max Shape Size", FF_TYPE_STANDARD, 0.5f );
	AddFloatParam( PT_CHAOS, "Size Chaos", FF_TYPE_STANDARD, 0.5f );
	AddFloatParam( PT_SMOOTH, "Smoothing", FF_TYPE_STANDARD, 0.6f );
	AddBoolParam( PT_SHAPE_ON, "Shape", true );
	AddOptionParam( PT_SHAPE_STYLE, "Shape Style", 10, OPT_SHAPE_STYLE, 0 );
	AddOptionParam( PT_FX_STYLE, "FX Style", 9, OPT_FX_STYLE, 0 );
	AddFloatParam( PT_SHAPE_THICK, "Shape Thick", FF_TYPE_STANDARD, 0.3f );
	AddFloatParam( PT_TRACKER_R, "Tracker Color", FF_TYPE_RED, 1.0f );
	AddFloatParam( PT_TRACKER_G, "Tracker Color_Green", FF_TYPE_GREEN, 1.0f );
	AddFloatParam( PT_TRACKER_B, "Tracker Color_Blue", FF_TYPE_BLUE, 1.0f );
	AddBoolParam( PT_HUD_ON, "HUD Data", true );
	AddOptionParam( PT_HUD_POS, "HUD Position", 4, OPT_HUD_POS, 0 );
	AddFloatParam( PT_HUD_SCALE, "HUD Scale", FF_TYPE_STANDARD, 0.5f );
	SetParamInfo( PT_TEXT, "Text", FF_TYPE_TEXT, text.c_str() );
	AddBoolParam( PT_LINE_ON, "Line", true );
	AddOptionParam( PT_LINE_WAY, "Line Way", 2, OPT_LINE_WAY, 0 );
	AddOptionParam( PT_LINE_STYLE, "Line Style", 2, OPT_LINE_STYLE, 1 );
	AddFloatParam( PT_CURVE, "Curve Amount", FF_TYPE_STANDARD, 0.5f );
	AddFloatParam( PT_LINE_THICK, "Line Thick", FF_TYPE_STANDARD, 0.2f );
	AddFloatParam( PT_DASH, "Dash Length", FF_TYPE_STANDARD, 0.3f );
	AddFloatParam( PT_LINE_R, "Line Color", FF_TYPE_RED, 1.0f );
	AddFloatParam( PT_LINE_G, "Line Color_Green", FF_TYPE_GREEN, 1.0f );
	AddFloatParam( PT_LINE_B, "Line Color_Blue", FF_TYPE_BLUE, 1.0f );
	AddBoolParam( PT_VIDEO_BG, "Video BG", true );

	FFGLLog::LogToHost( "Created TrackingBox effect" );
}
TrackingBox::~TrackingBox()
{
}

void TrackingBox::AddFloatParam( unsigned int index, const char* name, unsigned int type, float defaultValue )
{
	values[ index ] = defaultValue;
	SetParamInfo( index, name, type, defaultValue );
}

void TrackingBox::AddBoolParam( unsigned int index, const char* name, bool defaultValue )
{
	values[ index ] = defaultValue ? 1.0f : 0.0f;
	SetParamInfo( index, name, FF_TYPE_BOOLEAN, defaultValue );
}

//Parameter pilihan (tombol). Nilai tiap elemen = nomor urutnya (0, 1, 2, ...).
void TrackingBox::AddOptionParam( unsigned int index, const char* name, int numElements, const char* const* labels, int defaultIndex )
{
	values[ index ] = (float)defaultIndex;
	SetOptionParamInfo( index, name, (unsigned int)numElements, (float)defaultIndex );
	for( int i = 0; i < numElements; ++i )
		SetParamElementInfo( index, (unsigned int)i, labels[ i ], (float)i );
}

int TrackingBox::OptionIndex( unsigned int index, int numElements ) const
{
	return ClampInt( (int)floorf( values[ index ] + 0.5f ), 0, numElements - 1 );
}

FFResult TrackingBox::InitGL( const FFGLViewportStruct* vp )
{
	std::string mainCode = std::string( _mainShaderPart1 ) + _mainShaderPart2 + _mainShaderPart3;

	if( !analysisShader.Compile( _vertexShaderCode, _analysisFragmentShaderCode ) )
	{
		DeInitGL();
		return FF_FAIL;
	}
	if( !mainShader.Compile( _vertexShaderCode, mainCode ) )
	{
		DeInitGL();
		return FF_FAIL;
	}
	if( !quad.Initialise() )
	{
		DeInitGL();
		return FF_FAIL;
	}
	if( !analysisFbo.Initialise( GRID_MAX_W, GRID_MAX_H ) )
	{
		DeInitGL();
		return FF_FAIL;
	}
	if( !faceFbo.Initialise( FACE_W_MAX, FACE_H_MAX ) )
	{
		DeInitGL();
		return FF_FAIL;
	}

	//Siapkan tekstur kosong untuk teks label dan atlas angka.
	GLuint* targets[ 2 ] = { &textTexture, &atlasTexture };
	const int widths[ 2 ]  = { TEXT_TEX_W, ATLAS_TEX_W };
	const int heights[ 2 ] = { TEXT_TEX_H, ATLAS_TEX_H };
	for( int k = 0; k < 2; ++k )
	{
		glGenTextures( 1, targets[ k ] );
		ScopedSamplerActivation activateSampler( 0 );
		Scoped2DTextureBinding textureBinding( *targets[ k ] );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
		glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, widths[ k ], heights[ k ], 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL );
		glGenerateMipmap( GL_TEXTURE_2D );
	}

	BuildAtlasTexture();

	{
		std::lock_guard< std::mutex > lock( textMutex );
		textDirty = true;
	}
	havePrev     = false;
	frameCounter = 0;
	lastMode     = -1;
	faceFrames   = 0;
	for( int i = 0; i < MAX_TRACKS; ++i )
	{
		tracks[ i ].alpha    = 0.0f;
		tracks[ i ].detected = false;
		tracks[ i ].miss     = 0;
	}

	//Use base-class init as success result so that it retains the viewport.
	return CFFGLPlugin::InitGL( vp );
}

FFResult TrackingBox::ProcessOpenGL( ProcessOpenGLStruct* pGL )
{
	if( pGL->numInputTextures < 1 )
		return FF_FAIL;

	if( pGL->inputTextures[ 0 ] == NULL )
		return FF_FAIL;

	const FFGLTextureStruct& input = *pGL->inputTextures[ 0 ];
	if( input.Width == 0 || input.Height == 0 )
		return FF_FAIL;

	//The input texture's dimension might change each frame and so might the content area.
	FFGLTexCoords maxCoords = GetMaxGLTexCoords( input );

	bool dirty = false;
	{
		std::lock_guard< std::mutex > lock( textMutex );
		dirty = textDirty;
	}
	if( dirty )
		RebuildTextTexture();

	const int qualityIndex = OptionIndex( PT_QUALITY, 3 );
	const QualityCfg& q    = QUALITY[ qualityIndex ];
	const int mode         = OptionIndex( PT_MODE, 4 );
	const int count        = OptionIndex( PT_TRACKS, 8 ) + 1;

	//Ganti mode: lepaskan semua track lama supaya memudar, lalu mulai bersih.
	if( mode != lastMode )
	{
		for( int i = 0; i < MAX_TRACKS; ++i )
		{
			tracks[ i ].detected = false;
			tracks[ i ].miss     = 0;
		}
		havePrev   = false;
		faceFrames = 0;
		lastMode   = mode;
	}

	//Kalau ukuran grid berubah (Quality diganti), data frame sebelumnya tidak berlaku lagi.
	if( q.gw != lastGw || q.gh != lastGh )
	{
		havePrev     = false;
		lastGw       = q.gw;
		lastGh       = q.gh;
		frameCounter = 0;
	}

	//------------------------------------------------------------------------------------------------------------------
	// Pass 1: kecilkan gambar jadi grid berisi skor, baca ke CPU, cari puncak. Dilakukan tiap q.interval frame.
	//------------------------------------------------------------------------------------------------------------------
	if( mode == MODE_FACE )
	{
		UpdateFaces( input, maxCoords, qualityIndex, count );
	}
	else if( frameCounter % (unsigned int)q.interval == 0 )
	{
		GLint prevViewport[ 4 ];
		glGetIntegerv( GL_VIEWPORT, prevViewport );
		{
			//RB_REVERT mengembalikan framebuffer milik Resolume setelah selesai.
			ScopedFBOBinding fboBinding( analysisFbo.GetGLID(), ScopedFBOBinding::RB_REVERT );
			glViewport( 0, 0, q.gw, q.gh );
			glClearColor( 0.0f, 0.0f, 0.0f, 0.0f );
			glClear( GL_COLOR_BUFFER_BIT );

			ScopedShaderBinding shaderBinding( analysisShader.GetGLID() );
			ScopedSamplerActivation activateSampler( 0 );
			Scoped2DTextureBinding textureBinding( input.Handle );

			analysisShader.Set( "InputTexture", 0 );
			analysisShader.Set( "MaxUV", maxCoords.s, maxCoords.t );
			analysisShader.Set( "GridSize", (float)q.gw, (float)q.gh );
			analysisShader.Set( "Mode", mode == MODE_COLOR ? 1 : 0 );
			analysisShader.Set( "Samples", q.samples );
			analysisShader.Set( "TargetColor", values[ PT_TARGET_R ], values[ PT_TARGET_G ], values[ PT_TARGET_B ] );
			analysisShader.Set( "Tol", Lerp( 0.15f, 1.0f, Clamp01( values[ PT_TOLERANCE ] ) ) );
			quad.Draw();

			glPixelStorei( GL_PACK_ALIGNMENT, 4 );
			glReadPixels( 0, 0, q.gw, q.gh, GL_RGBA, GL_UNSIGNED_BYTE, &readback[ 0 ] );
		}
		glViewport( prevViewport[ 0 ], prevViewport[ 1 ], prevViewport[ 2 ], prevViewport[ 3 ] );

		Analyze( &readback[ 0 ], q.gw, q.gh, count, mode );
	}
	++frameCounter;

	StepTracks( count );

	//------------------------------------------------------------------------------------------------------------------
	// Pass 2: gambar hasil akhir.
	//------------------------------------------------------------------------------------------------------------------
	{
		//FFGL requires us to leave the context in a default state on return, so use these scoped bindings to help us do that.
		//Unit 0 = input, unit 1 = teks label, unit 2 = atlas angka HUD.
		ScopedShaderBinding shaderBinding( mainShader.GetGLID() );
		ScopedSamplerActivation activateInputSampler( 0 );
		Scoped2DTextureBinding inputBinding( input.Handle );
		ScopedSamplerActivation activateTextSampler( 1 );
		Scoped2DTextureBinding textBinding( textTexture );
		ScopedSamplerActivation activateAtlasSampler( 2 );
		Scoped2DTextureBinding atlasBinding( atlasTexture );

		const float seconds = std::chrono::duration< float >( std::chrono::steady_clock::now() - startTime ).count();
		const float aspect  = (float)input.Width / (float)input.Height;

		mainShader.Set( "InputTexture", 0 );
		mainShader.Set( "TextTexture", 1 );
		mainShader.Set( "AtlasTexture", 2 );
		mainShader.Set( "MaxUV", maxCoords.s, maxCoords.t );
		mainShader.Set( "Aspect", aspect );
		mainShader.Set( "PixelSize", 1.0f / (float)input.Height );
		mainShader.Set( "Time", seconds );
		mainShader.Set( "TrackCount", count );

		//Ukuran tiap track: campuran luas area terang dan gerak acak (Size Chaos), di antara Min dan Max.
		float minHalf;
		float maxHalf;
		HalfRange( values, minHalf, maxHalf );
		const float chaos = Clamp01( values[ PT_CHAOS ] );

		char name[ 32 ];
		for( int i = 0; i < MAX_TRACKS; ++i )
		{
			const Track& t = tracks[ i ];
			float noise    = 0.5f + 0.5f * ( 0.6f * sinf( seconds * 0.7f + t.seed * 7.0f ) + 0.4f * sinf( seconds * 1.9f + t.seed * 3.0f ) );
			float s01      = Lerp( Clamp01( t.size * 1.3f ), noise, chaos );
			float half     = Lerp( minHalf, maxHalf, Clamp01( s01 ) ) * Lerp( 0.5f, 1.0f, Clamp01( t.alpha ) );

			snprintf( name, sizeof( name ), "TrackA[%d]", i );
			mainShader.Set( name, t.x, t.y, half, t.alpha );
			snprintf( name, sizeof( name ), "TrackB[%d]", i );
			mainShader.Set( name, floorf( Clamp01( t.x ) * 999.0f ), floorf( Clamp01( t.y ) * 999.0f ), (float)i, t.seed );
		}

		mainShader.Set( "ShapeOn", values[ PT_SHAPE_ON ] > 0.5f ? 1.0f : 0.0f );
		mainShader.Set( "ShapeStyle", OptionIndex( PT_SHAPE_STYLE, 10 ) );
		mainShader.Set( "FxStyle", OptionIndex( PT_FX_STYLE, 9 ) );
		mainShader.Set( "ShapeThick", Lerp( 0.002f, 0.015f, Clamp01( values[ PT_SHAPE_THICK ] ) ) );
		mainShader.Set( "TrackerColor", values[ PT_TRACKER_R ], values[ PT_TRACKER_G ], values[ PT_TRACKER_B ] );

		mainShader.Set( "HudOn", values[ PT_HUD_ON ] > 0.5f ? 1.0f : 0.0f );
		mainShader.Set( "HudPos", OptionIndex( PT_HUD_POS, 4 ) );
		mainShader.Set( "HudH", Lerp( 0.012f, 0.05f, Clamp01( values[ PT_HUD_SCALE ] ) ) );
		mainShader.Set( "TextAspect", hasText ? textAspect : 0.0f );
		mainShader.Set( "TextUV", textUVx, textUVy );
		mainShader.Set( "AtlasAspect", atlasAspect );
		mainShader.Set( "AtlasCharW", atlasCharW );
		mainShader.Set( "AtlasUVy", atlasUVy );

		mainShader.Set( "LineOn", values[ PT_LINE_ON ] > 0.5f ? 1.0f : 0.0f );
		mainShader.Set( "LineWay", OptionIndex( PT_LINE_WAY, 2 ) );
		mainShader.Set( "LineDashed", OptionIndex( PT_LINE_STYLE, 2 ) == 1 ? 1.0f : 0.0f );
		mainShader.Set( "CurveAmt", Clamp01( values[ PT_CURVE ] ) );
		mainShader.Set( "LineThick", Lerp( 0.001f, 0.010f, Clamp01( values[ PT_LINE_THICK ] ) ) );
		mainShader.Set( "DashLen", Lerp( 0.004f, 0.05f, Clamp01( values[ PT_DASH ] ) ) );
		mainShader.Set( "LineColor", values[ PT_LINE_R ], values[ PT_LINE_G ], values[ PT_LINE_B ] );
		mainShader.Set( "LineChain", q.chain ? 1.0f : 0.0f );

		mainShader.Set( "VideoBg", values[ PT_VIDEO_BG ] > 0.5f ? 1.0f : 0.0f );

		quad.Draw();
	}

	return FF_SUCCESS;
}

FFResult TrackingBox::DeInitGL()
{
	analysisShader.FreeGLResources();
	mainShader.FreeGLResources();
	quad.Release();
	analysisFbo.Release();
	faceFbo.Release();

	//Menghentikan thread detektor wajah (menunggu sampai selesai) sebelum plugin dibongkar.
	faceDetector.reset();
	faceSeq = 0;

	if( textTexture != 0 )
	{
		glDeleteTextures( 1, &textTexture );
		textTexture = 0;
	}
	if( atlasTexture != 0 )
	{
		glDeleteTextures( 1, &atlasTexture );
		atlasTexture = 0;
	}
	hasText     = false;
	atlasAspect = 0.0f;

	return FF_SUCCESS;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Tracking
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//pixels = grid gw x gh (RGBA8, baris pertama = bawah gambar). Kanal merah berisi skor sel.
//Mencari sampai "count" puncak yang saling berjauhan, lalu mencocokkannya dengan track yang sudah ada
//supaya tiap track tetap mengikuti objek yang sama dari frame ke frame.
void TrackingBox::Analyze( const unsigned char* pixels, int gw, int gh, int count, int mode )
{
	const int cells    = gw * gh;
	const float thr    = Lerp( 0.0f, 0.85f, Clamp01( values[ PT_THRESHOLD ] ) );
	const float floorV = thr > 0.02f ? thr : 0.02f;

	for( int i = 0; i < cells; ++i )
	{
		float l = (float)pixels[ i * 4 ] / 255.0f;
		float s = l;
		if( mode == MODE_MOTION )
		{
			//Gerak = selisih tingkat terang dengan frame analisis sebelumnya.
			s = havePrev ? fabsf( l - prevLum[ i ] ) * 4.0f : 0.0f;
			if( s > 1.0f )
				s = 1.0f;
			prevLum[ i ] = l;
		}
		score[ i ] = s;
		work[ i ]  = s;
	}
	if( mode == MODE_MOTION )
		havePrev = true;

	//Jarak minimum antar puncak: makin banyak track, makin kecil.
	float sep = (float)gh * ( 0.32f / sqrtf( (float)count ) );
	if( sep < 2.0f )
		sep = 2.0f;
	const int sepI = (int)ceilf( sep );

	Cand cands[ MAX_TRACKS ];
	int nc = 0;

	for( int k = 0; k < count; ++k )
	{
		int bx     = -1;
		int by     = -1;
		float best = 0.0f;
		for( int y = 0; y < gh; ++y )
		{
			for( int x = 0; x < gw; ++x )
			{
				float v = work[ y * gw + x ];
				if( v > best )
				{
					best = v;
					bx   = x;
					by   = y;
				}
			}
		}
		if( bx < 0 || best < floorV )
			break;

		//Posisi halus di antara sel: rata-rata berbobot dari 3x3 sel di sekitar puncak.
		float sw = 0.0f;
		float sx = 0.0f;
		float sy = 0.0f;
		for( int dy = -1; dy <= 1; ++dy )
		{
			for( int dx = -1; dx <= 1; ++dx )
			{
				int x = bx + dx;
				int y = by + dy;
				if( x < 0 || y < 0 || x >= gw || y >= gh )
					continue;
				float v = score[ y * gw + x ];
				float w = v * v;
				sw += w;
				sx += w * ( (float)x + 0.5f );
				sy += w * ( (float)y + 0.5f );
			}
		}
		float cx = ( (float)bx + 0.5f ) / (float)gw;
		float cy = ( (float)by + 0.5f ) / (float)gh;
		if( sw > 1e-6f )
		{
			cx = sx / sw / (float)gw;
			cy = sy / sw / (float)gh;
		}

		//Luas area: berapa bagian dari lingkaran di sekitar puncak yang cukup terang. Dipakai untuk ukuran shape.
		int hit   = 0;
		int total = 0;
		for( int dy = -sepI; dy <= sepI; ++dy )
		{
			for( int dx = -sepI; dx <= sepI; ++dx )
			{
				if( (float)( dx * dx + dy * dy ) > sep * sep )
					continue;
				int x = bx + dx;
				int y = by + dy;
				if( x < 0 || y < 0 || x >= gw || y >= gh )
					continue;
				++total;
				if( score[ y * gw + x ] >= 0.5f * best )
					++hit;
				work[ y * gw + x ] = 0.0f;//Sekalian tandai sebagai sudah dipakai.
			}
		}
		float area = total > 0 ? (float)hit / (float)total : 0.0f;

		cands[ nc ].x    = cx;
		cands[ nc ].y    = cy;
		cands[ nc ].area = sqrtf( area );
		++nc;
	}

	MatchCandidates( cands, nc, count, (float)gw / (float)gh, 0 );
}

//Mencocokkan calon track (hasil analisis frame ini) dengan track yang sudah ada, supaya tiap track tetap mengikuti
//objek yang sama dari frame ke frame. aspect = lebar/tinggi gambar (untuk mengukur jarak dengan benar).
//hold > 0: track yang sesaat tidak ditemukan ditahan sampai hold analisis berturut-turut sebelum mulai memudar.
void TrackingBox::MatchCandidates( const Cand* cands, int nc, int count, float aspect, int hold )
{
	//Cocokkan kandidat dengan track yang sedang terlihat: pasangan terdekat dulu.
	bool candTaken[ MAX_TRACKS ];
	bool trackTaken[ MAX_TRACKS ];
	for( int i = 0; i < MAX_TRACKS; ++i )
	{
		candTaken[ i ]  = false;
		trackTaken[ i ] = false;
	}
	const float ga = aspect;

	for( int round = 0; round < nc; ++round )
	{
		float bestD = 1e9f;
		int bi      = -1;
		int bj      = -1;
		for( int i = 0; i < count; ++i )
		{
			if( trackTaken[ i ] || tracks[ i ].alpha < 0.05f )
				continue;
			for( int j = 0; j < nc; ++j )
			{
				if( candTaken[ j ] )
					continue;
				float dx = ( cands[ j ].x - tracks[ i ].tx ) * ga;
				float dy = cands[ j ].y - tracks[ i ].ty;
				float d  = dx * dx + dy * dy;
				if( d < bestD )
				{
					bestD = d;
					bi    = i;
					bj    = j;
				}
			}
		}
		if( bi < 0 || bestD > 0.3f * 0.3f )
			break;

		tracks[ bi ].tx    = cands[ bj ].x;
		tracks[ bi ].ty    = cands[ bj ].y;
		tracks[ bi ].tsize = cands[ bj ].area;
		trackTaken[ bi ]   = true;
		candTaken[ bj ]    = true;
	}

	//Kandidat yang belum dapat pasangan: ambil slot kosong (atau yang paling pudar).
	for( int j = 0; j < nc; ++j )
	{
		if( candTaken[ j ] )
			continue;
		int slot      = -1;
		float lowest = 2.0f;
		for( int i = 0; i < count; ++i )
		{
			if( trackTaken[ i ] )
				continue;
			if( tracks[ i ].alpha < lowest )
			{
				lowest = tracks[ i ].alpha;
				slot   = i;
			}
		}
		if( slot < 0 )
			break;

		Track& t = tracks[ slot ];
		if( t.alpha < 0.05f )
		{
			//Track baru muncul langsung di tempatnya, tidak meluncur dari posisi lama.
			t.x    = cands[ j ].x;
			t.y    = cands[ j ].y;
			t.size = cands[ j ].area;
		}
		t.tx             = cands[ j ].x;
		t.ty             = cands[ j ].y;
		t.tsize          = cands[ j ].area;
		trackTaken[ slot ] = true;
		candTaken[ j ]     = true;
	}

	for( int i = 0; i < MAX_TRACKS; ++i )
	{
		Track& t = tracks[ i ];
		if( trackTaken[ i ] )
		{
			t.detected = true;
			t.miss     = 0;
		}
		else if( hold > 0 && i < count && t.detected && t.alpha >= 0.05f && ++t.miss <= hold )
		{
			//Sesaat tidak ditemukan: tetap tampil di posisi terakhir.
		}
		else
		{
			t.detected = false;
			t.miss     = 0;
		}
	}
}

//Dipanggil tiap frame saat mode Face: kirim gambar kecil ke detektor (kalau sudah waktunya dan detektor bebas),
//lalu ambil hasil terbaru kalau ada.
void TrackingBox::UpdateFaces( const FFGLTextureStruct& input, const FFGLTexCoords& maxCoords, int qualityIndex, int count )
{
	if( !faceDetector )
		faceDetector.reset( new FaceDetector( PluginDirectory() ) );

	if( faceDetector->GetStatus() == FaceDetector::ST_FAILED )
	{
		//Model wajah tidak bisa dimuat: tidak ada yang di-track. Penyebabnya ditulis di TrackingBox_face_log.txt.
		for( int i = 0; i < MAX_TRACKS; ++i )
		{
			tracks[ i ].detected = false;
			tracks[ i ].miss     = 0;
		}
		return;
	}

	const FaceCfg& fc = FACE_QUALITY[ qualityIndex ];
	++faceFrames;
	if( faceFrames >= fc.interval && faceDetector->CanSubmit() )
	{
		faceFrames = 0;

		GLint prevViewport[ 4 ];
		glGetIntegerv( GL_VIEWPORT, prevViewport );
		{
			ScopedFBOBinding fboBinding( faceFbo.GetGLID(), ScopedFBOBinding::RB_REVERT );
			glViewport( 0, 0, fc.w, fc.h );
			glClearColor( 0.0f, 0.0f, 0.0f, 0.0f );
			glClear( GL_COLOR_BUFFER_BIT );

			ScopedShaderBinding shaderBinding( analysisShader.GetGLID() );
			ScopedSamplerActivation activateSampler( 0 );
			Scoped2DTextureBinding textureBinding( input.Handle );

			analysisShader.Set( "InputTexture", 0 );
			analysisShader.Set( "MaxUV", maxCoords.s, maxCoords.t );
			analysisShader.Set( "GridSize", (float)fc.w, (float)fc.h );
			analysisShader.Set( "Mode", 2 );//Abu-abu murni.
			analysisShader.Set( "Samples", 2 );
			analysisShader.Set( "TargetColor", 1.0f, 1.0f, 1.0f );
			analysisShader.Set( "Tol", 1.0f );
			quad.Draw();

			glPixelStorei( GL_PACK_ALIGNMENT, 1 );
			glReadPixels( 0, 0, fc.w, fc.h, GL_RED, GL_UNSIGNED_BYTE, &faceGray[ 0 ] );
			glPixelStorei( GL_PACK_ALIGNMENT, 4 );
		}
		glViewport( prevViewport[ 0 ], prevViewport[ 1 ], prevViewport[ 2 ], prevViewport[ 3 ] );

		FaceDetector::Settings s;
		//Threshold di mode Face = seberapa ketat: kiri = longgar (lebih banyak wajah, kadang salah), kanan = ketat.
		s.minNeighbors = 2 + (int)floorf( Clamp01( values[ PT_THRESHOLD ] ) * 6.0f + 0.5f );
		s.scaleFactor  = fc.scale;
		s.threads      = fc.threads;
		faceDetector->Submit( &faceGray[ 0 ], fc.w, fc.h, s );
	}

	std::vector< FaceRect > faces;
	if( faceDetector->Fetch( faceSeq, faces ) )
		ApplyFaces( faces, count, (float)input.Width / (float)input.Height );
}

//Mengubah hasil deteksi wajah menjadi calon track. Ukuran shape mengikuti ukuran wajah (dalam batas Min/Max Shape Size).
void TrackingBox::ApplyFaces( const std::vector< FaceRect >& faces, int count, float aspect )
{
	float minHalf;
	float maxHalf;
	HalfRange( values, minHalf, maxHalf );
	const float range = ( maxHalf - minHalf ) > 1e-4f ? ( maxHalf - minHalf ) : 1e-4f;

	Cand cands[ MAX_TRACKS ];
	int nc = 0;
	for( size_t i = 0; i < faces.size() && nc < count; ++i )
	{
		//Shape sedikit lebih besar dari wajah. Track.size dikalikan 1,3 di ProcessOpenGL, jadi dibagi 1,3 di sini.
		const float half = 0.5f * faces[ i ].size * 1.15f;
		const float s01  = Clamp01( ( half - minHalf ) / range );
		cands[ nc ].x    = faces[ i ].cx;
		cands[ nc ].y    = faces[ i ].cy;
		cands[ nc ].area = s01 / 1.3f;
		++nc;
	}
	MatchCandidates( cands, nc, count, aspect, FACE_HOLD );
}

//Dipanggil tiap frame: haluskan gerakan, ukuran, dan kemunculan/hilangnya track.
void TrackingBox::StepTracks( int count )
{
	//Smoothing 0 = langsung loncat, 1 = sangat lambat dan mulus.
	const float k = 1.0f - 0.97f * Clamp01( values[ PT_SMOOTH ] );
	for( int i = 0; i < MAX_TRACKS; ++i )
	{
		Track& t   = tracks[ i ];
		bool want  = ( i < count ) && t.detected;
		float rate = want ? 0.35f : 0.10f;
		t.alpha += ( ( want ? 1.0f : 0.0f ) - t.alpha ) * rate;
		if( t.alpha < 0.002f )
			t.alpha = 0.0f;

		t.x += ( t.tx - t.x ) * k;
		t.y += ( t.ty - t.y ) * k;
		t.size += ( t.tsize - t.size ) * k;
	}
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Teks
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef _WIN32
//Mengubah teks UTF-8 (dari Resolume) menjadi UTF-16 untuk GDI.
static std::wstring Utf8ToWide( const std::string& s )
{
	std::wstring out;
	size_t i       = 0;
	const size_t n = s.size();
	while( i < n )
	{
		unsigned int c  = (unsigned char)s[ i ];
		unsigned int cp = 0xFFFD;
		int extra       = 0;
		if( c < 0x80 )
		{
			cp    = c;
			extra = 0;
		}
		else if( ( c & 0xE0 ) == 0xC0 )
		{
			cp    = c & 0x1F;
			extra = 1;
		}
		else if( ( c & 0xF0 ) == 0xE0 )
		{
			cp    = c & 0x0F;
			extra = 2;
		}
		else if( ( c & 0xF8 ) == 0xF0 )
		{
			cp    = c & 0x07;
			extra = 3;
		}
		else
		{
			++i;
			out.push_back( (wchar_t)0xFFFD );
			continue;
		}
		++i;

		bool ok = true;
		for( int k = 0; k < extra; ++k )
		{
			if( i >= n || ( ( (unsigned char)s[ i ] ) & 0xC0 ) != 0x80 )
			{
				ok = false;
				break;
			}
			cp = ( cp << 6 ) | ( ( (unsigned char)s[ i ] ) & 0x3F );
			++i;
		}
		if( !ok )
		{
			out.push_back( (wchar_t)0xFFFD );
			continue;
		}

		if( cp >= 0x10000 )
		{
			cp -= 0x10000;
			out.push_back( (wchar_t)( 0xD800 + ( cp >> 10 ) ) );
			out.push_back( (wchar_t)( 0xDC00 + ( cp & 0x3FF ) ) );
		}
		else
		{
			out.push_back( (wchar_t)cp );
		}
	}
	return out;
}
#endif

//MAC_TEXT_BEGIN
#ifdef __APPLE__
//Satu baris teks CoreText: font monospace tebal (Menlo Bold), warna putih. Dipakai untuk label dan atlas angka.
class MacTextLine
{
public:
	MacTextLine( const char* utf8, size_t length ) :
		line( NULL ),
		ascent( 0.0 ),
		descent( 0.0 ),
		width( 0.0 )
	{
		CFStringRef str = CFStringCreateWithBytes( kCFAllocatorDefault, (const UInt8*)utf8, (CFIndex)length, kCFStringEncodingUTF8, false );
		if( str == NULL )
			return;

		CTFontRef font          = CTFontCreateWithName( CFSTR( "Menlo-Bold" ), (CGFloat)TEXT_FONT_PX, NULL );
		CGColorSpaceRef space   = CGColorSpaceCreateDeviceRGB();
		const CGFloat white[ 4 ] = { 1.0, 1.0, 1.0, 1.0 };
		CGColorRef color        = ( space != NULL ) ? CGColorCreate( space, white ) : NULL;

		if( font != NULL && color != NULL )
		{
			const void* keys[ 2 ]   = { kCTFontAttributeName, kCTForegroundColorAttributeName };
			const void* values[ 2 ] = { font, color };
			CFDictionaryRef attrs   = CFDictionaryCreate( kCFAllocatorDefault, keys, values, 2, &kCFTypeDictionaryKeyCallBacks, &kCFTypeDictionaryValueCallBacks );
			if( attrs != NULL )
			{
				CFAttributedStringRef attributed = CFAttributedStringCreate( kCFAllocatorDefault, str, attrs );
				if( attributed != NULL )
				{
					line = CTLineCreateWithAttributedString( attributed );
					if( line != NULL )
						width = CTLineGetTypographicBounds( line, &ascent, &descent, NULL );
					CFRelease( attributed );
				}
				CFRelease( attrs );
			}
		}

		if( color != NULL )
			CGColorRelease( color );
		if( space != NULL )
			CGColorSpaceRelease( space );
		if( font != NULL )
			CFRelease( font );
		CFRelease( str );
	}
	~MacTextLine()
	{
		if( line != NULL )
			CFRelease( line );
	}
	MacTextLine( const MacTextLine& )            = delete;
	MacTextLine& operator=( const MacTextLine& ) = delete;

	//Menggambar garis ini di titik x (kiri) dengan bagian atas huruf menempel di tepi atas bitmap (tinggi bitmapH).
	void DrawTop( CGContextRef ctx, int bitmapH, double x ) const
	{
		if( line == NULL || ctx == NULL )
			return;
		CGContextSetTextPosition( ctx, (CGFloat)x, (CGFloat)bitmapH - ascent );
		CTLineDraw( line, ctx );
	}

	CTLineRef line;
	CGFloat ascent;
	CGFloat descent;
	double width;
};

//Membuat konteks gambar RGBA 8-bit premultiplied di atas pixels (baris pertama di memori = bagian atas gambar).
//Huruf putih, jadi kanal merah hasilnya = tebal huruf (0..255). Wajib di-CGContextRelease oleh pemanggil.
static CGContextRef MacCreateTextContext( std::vector< unsigned char >& pixels, int w, int h )
{
	CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
	if( space == NULL )
		return NULL;
	CGContextRef ctx = CGBitmapContextCreate( &pixels[ 0 ], (size_t)w, (size_t)h, 8, (size_t)w * 4, space, (CGBitmapInfo)kCGImageAlphaPremultipliedLast );
	CGColorSpaceRelease( space );
	if( ctx != NULL )
		CGContextSetAllowsFontSmoothing( ctx, false );//Antialias abu-abu biasa, bukan LCD.
	return ctx;
}
#endif
//MAC_TEXT_END

//Menggambar teks label ke bitmap memakai GDI Windows (font Consolas) atau CoreText (Mac), lalu mengunggahnya sebagai tekstur.
//Harus dipanggil dari thread OpenGL, makanya dipanggil dari ProcessOpenGL, bukan dari SetTextParameter.
void TrackingBox::RebuildTextTexture()
{
	std::string copy;
	{
		std::lock_guard< std::mutex > lock( textMutex );
		copy      = text;
		textDirty = false;
	}

	hasText    = false;
	textAspect = 0.0f;
	textUVx    = 1.0f;
	textUVy    = 1.0f;

#ifdef _WIN32
	if( copy.empty() )
		return;

	std::wstring wide = Utf8ToWide( copy );
	if( wide.empty() )
		return;

	HDC dc = CreateCompatibleDC( NULL );
	if( dc == NULL )
		return;

	BITMAPINFO bmi;
	memset( &bmi, 0, sizeof( bmi ) );
	bmi.bmiHeader.biSize        = sizeof( BITMAPINFOHEADER );
	bmi.bmiHeader.biWidth       = TEXT_TEX_W;
	bmi.bmiHeader.biHeight      = -TEXT_TEX_H;//Negatif = baris pertama di atas.
	bmi.bmiHeader.biPlanes      = 1;
	bmi.bmiHeader.biBitCount    = 32;
	bmi.bmiHeader.biCompression = BI_RGB;

	void* bits     = NULL;
	HBITMAP bitmap = CreateDIBSection( dc, &bmi, DIB_RGB_COLORS, &bits, NULL, 0 );
	if( bitmap == NULL || bits == NULL )
	{
		DeleteDC( dc );
		return;
	}
	HGDIOBJ oldBitmap = SelectObject( dc, bitmap );
	memset( bits, 0, (size_t)TEXT_TEX_W * TEXT_TEX_H * 4 );

	HFONT font = CreateFontW(
		-TEXT_FONT_PX, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY,
		0,//DEFAULT_PITCH | FF_DONTCARE
		L"Consolas" );
	HGDIOBJ oldFont = SelectObject( dc, font );

	SetBkMode( dc, TRANSPARENT );
	SetTextColor( dc, 0x00FFFFFF );//Putih. Yang dipakai nanti hanya kanal merah sebagai "seberapa tebal hurufnya".

	SIZE extent;
	extent.cx = 0;
	extent.cy = 0;
	GetTextExtentPoint32W( dc, wide.c_str(), (int)wide.size(), &extent );
	TextOutW( dc, 0, 0, wide.c_str(), (int)wide.size() );
	GdiFlush();

	if( extent.cx > 0 && extent.cy > 0 )
	{
		const int usedW = extent.cx > TEXT_TEX_W ? TEXT_TEX_W : (int)extent.cx;
		const int usedH = extent.cy > TEXT_TEX_H ? TEXT_TEX_H : (int)extent.cy;

		ScopedSamplerActivation activateSampler( 0 );
		Scoped2DTextureBinding textureBinding( textTexture );
		glPixelStorei( GL_UNPACK_ALIGNMENT, 4 );
		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, TEXT_TEX_W, TEXT_TEX_H, 0, GL_BGRA, GL_UNSIGNED_BYTE, bits );
		glGenerateMipmap( GL_TEXTURE_2D );

		textAspect = (float)usedW / (float)usedH;
		textUVx    = (float)usedW / (float)TEXT_TEX_W;
		textUVy    = (float)usedH / (float)TEXT_TEX_H;
		hasText    = true;
	}

	SelectObject( dc, oldFont );
	DeleteObject( font );
	SelectObject( dc, oldBitmap );
	DeleteObject( bitmap );
	DeleteDC( dc );
#elif defined( __APPLE__ )
	if( copy.empty() )
		return;

	MacTextLine textLine( copy.c_str(), copy.size() );
	if( textLine.line == NULL || textLine.width <= 0.0 )
		return;

	std::vector< unsigned char > pixels( (size_t)TEXT_TEX_W * TEXT_TEX_H * 4, 0 );
	CGContextRef ctx = MacCreateTextContext( pixels, TEXT_TEX_W, TEXT_TEX_H );
	if( ctx == NULL )
		return;
	textLine.DrawTop( ctx, TEXT_TEX_H, 0.0 );
	CGContextFlush( ctx );
	CGContextRelease( ctx );

	const int usedW = (int)std::min( (double)TEXT_TEX_W, ceil( textLine.width ) );
	const int usedH = (int)std::min( (double)TEXT_TEX_H, ceil( textLine.ascent + textLine.descent ) );
	if( usedW > 0 && usedH > 0 )
	{
		ScopedSamplerActivation activateSampler( 0 );
		Scoped2DTextureBinding textureBinding( textTexture );
		glPixelStorei( GL_UNPACK_ALIGNMENT, 4 );
		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, TEXT_TEX_W, TEXT_TEX_H, 0, GL_RGBA, GL_UNSIGNED_BYTE, &pixels[ 0 ] );
		glGenerateMipmap( GL_TEXTURE_2D );

		textAspect = (float)usedW / (float)usedH;
		textUVx    = (float)usedW / (float)TEXT_TEX_W;
		textUVy    = (float)usedH / (float)TEXT_TEX_H;
		hasText    = true;
	}
#endif
}

//Atlas angka untuk HUD: setiap karakter dari ATLAS_CHARS digambar di selnya sendiri (lebar sel sama semua),
//jadi shader bisa menyusun "XXX:YYY" cukup dengan memilih nomor sel.
void TrackingBox::BuildAtlasTexture()
{
	atlasAspect = 0.0f;
	atlasCharW  = 0.0f;
	atlasUVy    = 1.0f;

#ifdef _WIN32
	HDC dc = CreateCompatibleDC( NULL );
	if( dc == NULL )
		return;

	BITMAPINFO bmi;
	memset( &bmi, 0, sizeof( bmi ) );
	bmi.bmiHeader.biSize        = sizeof( BITMAPINFOHEADER );
	bmi.bmiHeader.biWidth       = ATLAS_TEX_W;
	bmi.bmiHeader.biHeight      = -ATLAS_TEX_H;
	bmi.bmiHeader.biPlanes      = 1;
	bmi.bmiHeader.biBitCount    = 32;
	bmi.bmiHeader.biCompression = BI_RGB;

	void* bits     = NULL;
	HBITMAP bitmap = CreateDIBSection( dc, &bmi, DIB_RGB_COLORS, &bits, NULL, 0 );
	if( bitmap == NULL || bits == NULL )
	{
		DeleteDC( dc );
		return;
	}
	HGDIOBJ oldBitmap = SelectObject( dc, bitmap );
	memset( bits, 0, (size_t)ATLAS_TEX_W * ATLAS_TEX_H * 4 );

	HFONT font = CreateFontW(
		-TEXT_FONT_PX, 0, 0, 0, FW_BOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, ANTIALIASED_QUALITY,
		0,
		L"Consolas" );
	HGDIOBJ oldFont = SelectObject( dc, font );

	SetBkMode( dc, TRANSPARENT );
	SetTextColor( dc, 0x00FFFFFF );

	const int numChars = (int)( sizeof( ATLAS_CHARS ) / sizeof( wchar_t ) ) - 1;
	SIZE cell;
	cell.cx = 0;
	cell.cy = 0;
	GetTextExtentPoint32W( dc, L"0", 1, &cell );

	if( cell.cx > 0 && cell.cy > 0 && cell.cx * numChars <= ATLAS_TEX_W && cell.cy <= ATLAS_TEX_H )
	{
		for( int i = 0; i < numChars; ++i )
		{
			SIZE one;
			one.cx = 0;
			one.cy = 0;
			GetTextExtentPoint32W( dc, &ATLAS_CHARS[ i ], 1, &one );
			TextOutW( dc, i * cell.cx + ( cell.cx - one.cx ) / 2, 0, &ATLAS_CHARS[ i ], 1 );
		}
		GdiFlush();

		ScopedSamplerActivation activateSampler( 0 );
		Scoped2DTextureBinding textureBinding( atlasTexture );
		glPixelStorei( GL_UNPACK_ALIGNMENT, 4 );
		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, ATLAS_TEX_W, ATLAS_TEX_H, 0, GL_BGRA, GL_UNSIGNED_BYTE, bits );
		glGenerateMipmap( GL_TEXTURE_2D );

		atlasAspect = (float)cell.cx / (float)cell.cy;
		atlasCharW  = (float)cell.cx / (float)ATLAS_TEX_W;
		atlasUVy    = (float)cell.cy / (float)ATLAS_TEX_H;
	}

	SelectObject( dc, oldFont );
	DeleteObject( font );
	SelectObject( dc, oldBitmap );
	DeleteObject( bitmap );
	DeleteDC( dc );
#elif defined( __APPLE__ )
	const int numChars = (int)( sizeof( ATLAS_CHARS ) / sizeof( wchar_t ) ) - 1;

	MacTextLine zero( "0", 1 );
	if( zero.line == NULL )
		return;
	const int cellW = (int)ceil( zero.width );
	const int cellH = (int)ceil( zero.ascent + zero.descent );

	if( cellW > 0 && cellH > 0 && cellW * numChars <= ATLAS_TEX_W && cellH <= ATLAS_TEX_H )
	{
		std::vector< unsigned char > pixels( (size_t)ATLAS_TEX_W * ATLAS_TEX_H * 4, 0 );
		CGContextRef ctx = MacCreateTextContext( pixels, ATLAS_TEX_W, ATLAS_TEX_H );
		if( ctx == NULL )
			return;
		for( int i = 0; i < numChars; ++i )
		{
			//ATLAS_CHARS hanya berisi karakter ASCII, jadi tiap karakter bisa disalin langsung sebagai satu byte UTF-8.
			const char ch = (char)ATLAS_CHARS[ i ];
			MacTextLine one( &ch, 1 );
			//Tiap karakter digambar di tengah selnya sendiri (lebar sel sama untuk semua).
			one.DrawTop( ctx, ATLAS_TEX_H, (double)i * cellW + ( cellW - one.width ) * 0.5 );
		}
		CGContextFlush( ctx );
		CGContextRelease( ctx );

		ScopedSamplerActivation activateSampler( 0 );
		Scoped2DTextureBinding textureBinding( atlasTexture );
		glPixelStorei( GL_UNPACK_ALIGNMENT, 4 );
		glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA8, ATLAS_TEX_W, ATLAS_TEX_H, 0, GL_RGBA, GL_UNSIGNED_BYTE, &pixels[ 0 ] );
		glGenerateMipmap( GL_TEXTURE_2D );

		atlasAspect = (float)cellW / (float)cellH;
		atlasCharW  = (float)cellW / (float)ATLAS_TEX_W;
		atlasUVy    = (float)cellH / (float)ATLAS_TEX_H;
	}
#endif
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Parameter
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

FFResult TrackingBox::SetFloatParameter( unsigned int dwIndex, float value )
{
	if( dwIndex >= PT_COUNT || dwIndex == PT_TEXT )
		return FF_FAIL;

	values[ dwIndex ] = value;
	return FF_SUCCESS;
}

FFResult TrackingBox::SetTextParameter( unsigned int index, const char* value )
{
	if( index != PT_TEXT )
		return FF_FAIL;

	std::lock_guard< std::mutex > lock( textMutex );
	text      = ( value != NULL ) ? value : "";
	textDirty = true;
	return FF_SUCCESS;
}

float TrackingBox::GetFloatParameter( unsigned int index )
{
	if( index >= PT_COUNT || index == PT_TEXT )
		return 0.0f;

	return values[ index ];
}

char* TrackingBox::GetTextParameter( unsigned int index )
{
	(void)index;
	static char textBufferForHost[ 512 ];

	std::lock_guard< std::mutex > lock( textMutex );
	size_t n = text.size() < sizeof( textBufferForHost ) - 1 ? text.size() : sizeof( textBufferForHost ) - 1;
	memcpy( textBufferForHost, text.c_str(), n );
	textBufferForHost[ n ] = '\0';
	return textBufferForHost;
}
