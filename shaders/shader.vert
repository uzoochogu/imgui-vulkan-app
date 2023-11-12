#version 450

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

vec2 positions[3] = vec2[](
    vec2(0.0, -0.5),
    vec2(0.5, 0.5),
    vec2(-0.5, 0.5)
);

void main() {
    gl_Position = vec4(positions[gl_VertexIndex], 0.0, 1.0);
    fragColor = colors[gl_VertexIndex];   //output for color
}