/*
Copyright 2015 Orlando Aguilar Vivanco

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

[[vk::binding(0, 0)]] StructuredBuffer<uint> inBuffer;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> outBuffer;

struct image_data {
    uint width;
    uint height;
};

[[vk::push_constant]]
image_data data;

//half size of kernel. The kernel is a 3x3 window, therefore half the size is floor(3/2)
static const uint shift = 1;

//total size of the data in the structured buffers
static const uint total_data = data.width * data.height;


//The kernel of the laplassian looks like this:
// -1 -1 -1
// -1  8 -1
// -1 -1 -1
//
//with 8 the coefficient of the center of the kernel.


//the shared data is used to "cache" the texels that will be used  during the convolution
//by the whole workgroup. shared_data0 will be used by the convolution on the first row
//of the kernel, shared_data1 on the second row, and so on.
groupshared float3 shared_data0[258];
groupshared float3 shared_data1[258];
groupshared float3 shared_data2[258]; 


#include "util.hlsl"
//include the definition of sample_raw

[numthreads(256, 1, 1)]
void Main(uint3 DTid : SV_DispatchThreadID, uint3 localId : SV_GroupThreadID)
{
	//index used to access the shared memory
	uint id = localId.x + shift;
	
	//indices used to access the structured buffer. Each thread will sample 3 texels to cache the data
	uint xcoord = DTid.x;
	uint ybelow = DTid.x - data.width;
	uint yabove = DTid.x + data.width;

	shared_data0[id] = sample_raw(inBuffer, total_data, ybelow);
	shared_data1[id] = sample_raw(inBuffer, total_data, xcoord);
	shared_data2[id] = sample_raw(inBuffer, total_data, yabove);
	
	//threads 0 and 255 will sample 3 extra texels for the neighbors of data 
	if (localId.x==0){
		uint xcoord = DTid.x-1;
		uint ybelow = DTid.x - data.width-1;
		uint yabove = DTid.x + data.width-1;

		shared_data0[0] = sample_raw(inBuffer, total_data, ybelow);
		shared_data1[0] = sample_raw(inBuffer, total_data, xcoord);
		shared_data2[0] = sample_raw(inBuffer, total_data, yabove);
	}
	
	if (localId.x==255){
		uint xcoord = DTid.x+1;
		uint ybelow = DTid.x - data.width+1;
		uint yabove = DTid.x + data.width+1;

		shared_data0[257] = sample_raw(inBuffer, total_data, ybelow);
		shared_data1[257] = sample_raw(inBuffer, total_data, xcoord);
		shared_data2[257] = sample_raw(inBuffer, total_data, yabove);
	}
	//make sure all the workgroup threads have filled the shared data with the texels.
	GroupMemoryBarrierWithGroupSync();

	// apply the kernel convolution. The convolution has been manually unrolled here.
	uint3 output = 8*shared_data1[id]-shared_data1[id-1]-shared_data1[id+1]
	-shared_data0[id]-shared_data0[id-1]-shared_data0[id+1]
	-shared_data2[id]-shared_data2[id-1]-shared_data2[id+1];

	//packs color back into a uint to store it in the structured buffer
	uint outcolor = output.r | (output.g<<8) | (output.b<<16) | (0xff << 24);
    outBuffer[xcoord] = outcolor;
}