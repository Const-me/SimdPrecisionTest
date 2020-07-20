#pragma once

// Utility class to write SIMD vectors into a file, fast.
template<class TVec>
class VectorWriter
{
	CAtlFile& file;
	static constexpr size_t capacity = ( 1024 * 8 ) / sizeof( TVec );
	std::array<TVec, capacity> buffer;
	size_t count = 0;

	void flush()
	{
		const DWORD cb = (DWORD)( count * sizeof( TVec ) );
		HRESULT hr = file.Write( buffer.data(), cb );
		if( SUCCEEDED( hr ) )
		{
			count = 0;
			return;
		}
		__debugbreak();
	}

public:

	VectorWriter( CAtlFile& f ) : file( f ) { }

	~VectorWriter()
	{
		if( count > 0 )
			flush();
	}

	__forceinline void write( TVec v )
	{
		if( count < capacity )
		{
			buffer[ count++ ] = v;
			return;
		}
		flush();
		buffer[ 0 ] = v;
		count = 1;
	}
};