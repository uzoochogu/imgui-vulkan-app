#version 450

/* Uniform Buffer object
*/
layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

/* Take input from vertex buffer into vertex attibutes using 
** the in keyword
*/
layout(location = 0) in vec2 inPosition;

/*Pass in colors just like positions*/
layout(location = 1) in vec3 inColor;

/*Pass per-vertex colors to the fragment shader, so that it can output 
 interpolated values to the framebuffer.
*/
layout(location = 0) out vec3 fragColor;

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 0.0, 1.0);
    fragColor = inColor;   //output for color
}