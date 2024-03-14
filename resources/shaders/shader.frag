#version 450

// A combined image sampler descriptor is represented 
// in GLSL by a sampler uniform.
layout(binding = 1) uniform sampler2D texSampler;

/* matching input from vertex shader, same name not necessary since index
*  is specified by location derivatives
*/
layout(location = 0) in vec3 fragColor;
layout(location = 1) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    // built in texture func, sampler and cordinates as args
    // sampler automatically takes care of filtering and 
    // transformations in the background
    outColor = texture(texSampler, fragTexCoord);
}