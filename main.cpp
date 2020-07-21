#include "stdafx.h"
#include "Random.hpp"
#include "VectorWriter.hpp"

LPCTSTR destPath = LR"(C:\Temp\2remove\simd)";

// Make them 1GB files
constexpr size_t scalarsPerTest = 1024 * 1024 * 256;

template<class TFunc, class TVec = __m256>
static inline void generate( LPCTSTR name )
{
	CPath path;
	path.m_strPath = destPath;
	const CString fileName = CString{ name } + L".bin";
	path.Append( fileName );

	CAtlFile file;
	HRESULT	 hr = file.Create( path.m_strPath, GENERIC_WRITE, 0, CREATE_ALWAYS );
	if( FAILED( hr ) )
		__debugbreak();

	VectorWriter<TVec> writer{ file };
	printf( "Generating \"%S\"\n", path.m_strPath.operator LPCWSTR() );

	constexpr size_t vectorSize = sizeof( TVec ) / 4;
	const size_t countVectors = scalarsPerTest / vectorSize;

	TFunc func;
	for( size_t i = 0; i < countVectors; i++ )
	{
		const TVec val = func.next();
		writer.write( val );
	}
}

struct GeneratorBase
{
	RandomNumbers lhs, rhs;
	GeneratorBase() : lhs( 0xC4000C40u ), rhs( 0x946E36B2u ) { }
};

struct TestSum : GeneratorBase
{
	__forceinline __m256 next()
	{
		__m256 a, b;
		lhs.next( a );
		rhs.next( b );
		return _mm256_add_ps( a, b );
	}
};

struct TestMul : GeneratorBase
{
	__forceinline __m256 next()
	{
		__m256 a, b;
		lhs.next( a );
		rhs.next( b );
		return _mm256_mul_ps( a, b );
	}
};

struct TestFma : GeneratorBase
{
	RandomNumbers acc;
	TestFma() : acc( 0xFFEE0033 ) { }

	__forceinline __m256 next()
	{
		__m256 a, b, c;
		lhs.next( a );
		rhs.next( b );
		acc.next( c );
		return _mm256_fmadd_ps( c, a, b );
	}
};

struct TestConvert : GeneratorBase
{
	__forceinline __m256 next()
	{
		__m256 a;
		lhs.next( a );
		__m256i ints = _mm256_cvtps_epi32( a );
		return _mm256_castsi256_ps( ints );
	}
};

void saveExactMathBinaries()
{
	CreateDirectory( destPath, nullptr );
	generate<TestSum>( L"vaddps" );
	generate<TestMul>( L"vmulps" );
	generate<TestFma>( L"vfmadd" );
	generate<TestConvert>( L"vcvtps2dq" );
}
#include "rcppsErrorSummary.h"

int main()
{
	// saveExactMathBinaries();
	rcppsErrorSummary();
	return 0;
}