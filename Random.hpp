#pragma once

// Utility class to generate pseudo-random values, fast.
class RandomNumbers
{
	__m128i key;
	__m128i state;

public:

	RandomNumbers( uint32_t seed );

	__forceinline void next( __m128& result )
	{
		state = _mm_aesenc_si128( state, key );
		result = _mm_castsi128_ps( state );
	}

	__forceinline void next( __m256& result )
	{
		__m128 low, high;
		next( low );
		next( high );
		result = _mm256_insertf128_ps( _mm256_castps128_ps256( low ), high, 1 );
	}
};