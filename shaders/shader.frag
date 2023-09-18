#version 450

/* matching input from vertex shader, same name not necessary since index
*  is specified by location derivatives
*/
layout(location = 0) in vec3 fragColor;

layout(location = 0) out vec4 outColor;

void main() {
    outColor = vec4(fragColor, 1.0);
}