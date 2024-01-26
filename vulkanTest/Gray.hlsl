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

[[vk::binding(0, 0)]] StructuredBuffer<uint> InBuffer;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> OutBuffer;

struct image_data {
    uint width;
    uint height;
};

[[vk::push_constant]]
image_data data;

//total size of the data in the structured buffers
static const uint total_data = data.width * data.height;

#include "util.hlsl"
//include the definition of sample_raw

[numthreads(256, 1, 1)]
void Main(uint3 DTid : SV_DispatchThreadID)
{
	float3 color = sample_raw(InBuffer, total_data, DTid.x);
	
	//calculate grayscale, very simple just an average of the channels
	uint gray = (color.r+color.g+color.b)/3;
	
	uint outcolor = gray | (gray<<8) | (gray<<16) | (0xff << 24);
    OutBuffer[DTid.x] = outcolor;
}