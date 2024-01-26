/*
Copyright 2024 Orlando Aguilar Vivanco

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

/*
This kernel performs the filtering in the vertical axis. For an explanation of 
how the kernel works, look at the horizontal kernel GaussH.hlsl, which contains
more detailed information.
*/

[[vk::binding(0, 0)]] StructuredBuffer<uint> InBuffer;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> OutBuffer;

struct image_data {
    uint width;
    uint height;
};

[[vk::push_constant]]
image_data data;

static const uint total_data = data.width * data.height;

static const float kernel[31] =	{0.00088806, 0.00158611, 0.00272177, 0.00448744, 0.00710844, 0.01081877,
 0.01582012, 0.02222644, 0.03000255, 0.03891121, 0.04848635, 0.0580487,
 0.0667719,  0.07379436, 0.07835755, 0.07994048, 0.07835755, 0.07379436,
 0.0667719,  0.0580487 , 0.04848635, 0.03891121, 0.03000255, 0.02222644,
 0.01582012, 0.01081877, 0.00710844, 0.00448744, 0.00272177, 0.00158611,
 0.00088806};
 
 static const uint shift = 15;
  groupshared float3 shared_data[256+shift*2]; 

#include "util.hlsl"
//include the definition of sample_normalized

[numthreads(256, 1, 1)]
void Main(uint3 DTid : SV_DispatchThreadID, uint3 localId : SV_GroupThreadID)
{	
	uint id = localId.x + shift; //edges of gaussian
	
	//this kernel will sample data vertically, the formula below
	//will calculate the ycoordinate on the texture represented by
	//the strucutured buffer
	uint ycoord = (DTid.x%data.height)*data.width+DTid.x/data.height;

	shared_data[id] = sample_normalized(InBuffer, total_data, ycoord);
	if (localId.x<shift){
		uint ycoord_minus_shift = ((DTid.x-shift)%data.height)*data.width+(DTid.x-shift)/data.height;
		shared_data[id-shift] = sample_normalized(InBuffer, total_data, ycoord_minus_shift);
	}
	if (localId.x>=256-shift){
		uint ycoord_plus_shift = ((DTid.x+shift)%data.height)*data.width+(DTid.x+shift)/data.height;
		shared_data[id+shift] = sample_normalized(InBuffer, total_data, ycoord_plus_shift);
	}
	GroupMemoryBarrierWithGroupSync();

	//convolution
	float3 colorResult= float3(0.0,0.0,0.0);
	for (uint z=0;z<(shift*2+1);++z){
		colorResult+=shared_data[id-shift+z] * kernel[z];
	}
	
	
	uint red = colorResult.r*255;
	uint green = colorResult.g*255;
	uint blue = colorResult.b*255;
	uint outcolor = red | (green<<8) | (blue<<16) | (0xff << 24);
    OutBuffer[ycoord] = outcolor;
}