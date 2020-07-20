#include "stdafx.h"
#include "Random.hpp"
#include <random>

RandomNumbers::RandomNumbers( uint32_t seed )
{
	std::mt19937 engine{ seed };

	alignas( 16 ) std::array<uint32_t, 4> _state, _key;
	for( int i = 0; i < 4; i++ )
		_state[ i ] = engine();
	for( int i = 0; i < 4; i++ )
		_key[ i ] = engine();

	state = _mm_load_si128( ( __m128i* )_state.data() );
	key = _mm_load_si128( ( __m128i* )_key.data() );
}