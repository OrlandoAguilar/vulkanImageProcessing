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

//samples a texel an unpacks the color information into a float3
float3 sample_raw(StructuredBuffer<uint> buffer,uint size, uint index){
	//out of bounds check
	if (index>=size)	
		return float3(0.0, 0.0, 0.0);
	
	uint color = buffer[index];
	float3 result;
	result.b = (color >> 16)&0xff;
	result.g = (color >> 8)&0xff;
	result.r = color&0xff;
	return result;
}

//samples a texel an unpacks the color information into a float3
float3 sample_normalized(StructuredBuffer<uint> buffer,uint size, uint index){
	return sample_raw(buffer, size, index)/255.0f;
}