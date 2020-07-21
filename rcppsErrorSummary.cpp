#include "stdafx.h"
#include "rcppsErrorSummary.h"
#include <omp.h>
#include <assert.h>

struct Bucket
{
	// Meaning of indices in both arrays: 0 = equal, 1 = the approximate was less than the precise, 2 = the approximate was greater than precise

	// Total count of values in the bucket
	std::array<uint32_t, 3> counts;
	// Maximum absolute difference in the bucket. Because index 0 means "they are equal", the first element is always 0.
	std::array<uint32_t, 3> maxDiff;

	__forceinline __m128i loadCount() const
	{
		return _mm_loadu_si128( ( const __m128i* )counts.data() );
	}
	__forceinline __m128i loadMaxDiff() const
	{
		return _mm_loadu_si128( ( const __m128i* )maxDiff.data() );
	}
	__forceinline void storeCount( __m128i v )
	{
		_mm_storeu_si64( counts.data(), v );
		counts[ 2 ] = _mm_extract_epi32( v, 2 );
	}
	__forceinline void storeMaxDiff( __m128i v )
	{
		_mm_storeu_si64( maxDiff.data(), v );
		maxDiff[ 2 ] = _mm_extract_epi32( v, 2 );
	}
};

struct ThreadContext
{
	// The index in the array is the high 9 bits of the precise float value, sign + exponent
	std::array<Bucket, 0x200> buckets;

	ThreadContext()
	{
		ZeroMemory( this, sizeof( ThreadContext ) );
	}

	inline void mergeFrom( const ThreadContext& other )
	{
		// This code runs very rarely, not much point in using SIMD for this, however scalar instructions can't compute max.
		const Bucket* src = other.buckets.data();
		const Bucket* const srcEnd = src + 0x200;
		Bucket* dest = buckets.data();
		for( ; src < srcEnd; src++, dest++ )
		{
			__m128i c1 = src->loadCount();
			__m128i ax1 = src->loadMaxDiff();
			__m128i c2 = dest->loadCount();
			__m128i ax2 = dest->loadMaxDiff();
			dest->storeCount( _mm_add_epi32( c1, c2 ) );
			dest->storeMaxDiff( _mm_max_epu32( ax1, ax2 ) );
		}
	}
};

// How many floats are processed by each iteration. The iterations are running in parallel, of course.
constexpr uint32_t floatsPerChunk = 1024 * 1024;
constexpr uint32_t vectorsPerChunk = floatsPerChunk / 8;

__forceinline __m128i vget_low( __m256i x )
{
	return _mm256_castsi256_si128( x );
}
__forceinline __m128i vget_high( __m256i x )
{
	return _mm256_extractf128_si256( x, 1 );
}

template<int lane16, int lane32>
__forceinline void accumulateLane( Bucket* const buckets, const int comparisons, const __m128i preciseExpVec, const __m128i absDiffVec )
{
	const uint16_t exponent = _mm_extract_epi16( preciseExpVec, lane16 );
	assert( exponent < 0x200 );

	const uint32_t absDiff = (uint32_t)_mm_extract_epi32( absDiffVec, lane32 );
	const int cmpMask = ( comparisons >> ( lane16 * 2 ) ) & 3;
	assert( 3 != cmpMask );

	Bucket& bucket = buckets[ exponent ];
	bucket.counts[ cmpMask ]++;
	bucket.maxDiff[ cmpMask ] = std::max( bucket.maxDiff[ cmpMask ], absDiff );
}

template<bool isNegative>
__forceinline void computeChunk( int index, ThreadContext& dest )
{
	// We don't want cache line sharing, that's why using local context to compute stuff, merging in the end
	ThreadContext context;

	const uint32_t firstFloat = (uint32_t)index * floatsPerChunk;
	__m256i integers = _mm256_set1_epi32( (int)firstFloat );
	integers = _mm256_add_epi32( integers, _mm256_setr_epi32( 0, 1, 2, 3, 4, 5, 6, 7 ) );
	const __m256i advanceOffsets = _mm256_set1_epi32( 8 );

	Bucket* const buckets = context.buckets.data();

	for( uint32_t i = 0; i < vectorsPerChunk; i++ )
	{
		// Compute these floats
		const __m256 floats = _mm256_castsi256_ps( integers );
		const __m256 preciseFloats = _mm256_div_ps( _mm256_set1_ps( 1 ), floats );
		const __m256 approxFloats = _mm256_rcp_ps( floats );

		// Cast to integers and extract exponent + sign portions
		const __m256i precise = _mm256_castps_si256( preciseFloats );
		const __m256i approx = _mm256_castps_si256( approxFloats );
		const __m256i expValues = _mm256_srli_epi32( precise, 32 - 9 );
		const __m128i preciseExp = _mm_packus_epi32( vget_low( expValues ), vget_high( expValues ) );

		// Subtract and compute absolute difference
		const __m256i absDiff = _mm256_abs_epi32( _mm256_sub_epi32( approx, precise ) );

		// Compare both ( approx > precise ) and ( approx < precise ), move all 16 results into scalar register
		// Negative floats have their sort order flipped relative to integers.
		// The condition is true or false for complete chunks, only testing that once per chunk outside the loop
		// Passing that value as a template argument for optimal performance, unlike normal branches `if constexpr` has no runtime overhead.
		__m256i greater, less;
		if constexpr( isNegative )
		{
			greater = _mm256_cmpgt_epi32( precise, approx );
			less = _mm256_cmpgt_epi32( approx, precise );
		}
		else
		{
			greater = _mm256_cmpgt_epi32( approx, precise );
			less = _mm256_cmpgt_epi32( precise, approx );
		}
		const __m256i blendMask = _mm256_set1_epi32( 0xFFFF );
		const __m256i comparisonsAvx = _mm256_blendv_epi8( greater, less, blendMask );	// Now we have 16-bit lanes set to either -1 or 0, based on these comparisons
		const __m128i comparisonsSse = _mm_packs_epi16( vget_low( comparisonsAvx ), vget_high( comparisonsAvx ) );	// Pack them into bytes
		const int comparisons = _mm_movemask_epi8( comparisonsSse );	// Move all 16 values into bits of the scalar register
		assert( comparisons >= 0 && comparisons < 0x10000 );

		__m128i tmp = vget_low( absDiff );
		accumulateLane<0, 0>( buckets, comparisons, preciseExp, tmp );
		accumulateLane<1, 1>( buckets, comparisons, preciseExp, tmp );
		accumulateLane<2, 2>( buckets, comparisons, preciseExp, tmp );
		accumulateLane<3, 3>( buckets, comparisons, preciseExp, tmp );

		tmp = vget_high( absDiff );
		accumulateLane<4, 0>( buckets, comparisons, preciseExp, tmp );
		accumulateLane<5, 1>( buckets, comparisons, preciseExp, tmp );
		accumulateLane<6, 2>( buckets, comparisons, preciseExp, tmp );
		accumulateLane<7, 3>( buckets, comparisons, preciseExp, tmp );

		integers = _mm256_add_epi32( integers, advanceOffsets );
	}

	dest.mergeFrom( context );
}

void print( int index, const Bucket& b )
{
	const char sign = ( index & 0x100 ) ? '-' : '+';
	printf( "%c\t", sign );
	const int exponent = index & 0xFF;
	if( 0 == exponent )
		printf( "DEN" );
	else if( 0xFF == exponent )
		printf( "NAN" );
	else
		printf( "2^%i", exponent - 127 );

	uint32_t total = b.counts[ 0 ] + b.counts[ 1 ] + b.counts[ 2 ];
	printf( "\t%i\t", total );
	for( int i = 0; i < 3; i++ )
		printf( "%i\t", b.counts[ i ] );
	printf( "%i\t%i\t%i\n", b.maxDiff[ 1 ], b.maxDiff[ 2 ], std::max( b.maxDiff[ 1 ], b.maxDiff[ 2 ] ) );
}

void rcppsErrorSummary()
{
	std::vector<ThreadContext> threads;
	threads.resize( (size_t)omp_get_max_threads() );

	constexpr int chunksCount = ( (size_t)1 << 32 ) / floatsPerChunk;

#ifdef NDEBUG
#pragma omp parallel for schedule( static, 1 )
#endif
	for( int i = 0; i < chunksCount; i++ )
	{
		ThreadContext& ctx = threads[ omp_get_thread_num() ];
		if( i < ( chunksCount / 2 ) )
			computeChunk<false>( i, ctx );
		else
			computeChunk<true>( i, ctx );
	}

	ThreadContext result;
	for( const auto& c : threads )
		result.mergeFrom( c );

	printf( "sign\texp\ttotal\texact\tless\tgreater\tmaxLess\tmaxGreater\tmaxAbs\n" );
	for( int i = 0; i < 0x100; i++ )
	{
		const int neg = i + 0x100;
		print( neg, result.buckets[ neg ] );
		print( i, result.buckets[ i ] );
	}
}