// The lamp's light at each occupied voxel of the streamlines' grid, for the ray
// tracer to read at its hits: a cone from the voxel toward the lamp, through the
// occupancy pyramid. Run before each frame, as the lamp turns with the camera.
// After Kraaijeveld B, Jalba AC, Vilanova A, Chamberland M. Real-Time Rendering
// of Dynamic Line Sets using Voxel Ray Tracing. arXiv:2510.09081 (2025).

{{ bindings_code }}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let size = i32(u_material.size);
    let v = gid.x + gid.y * groups.x * 64u;
    if (v >= u_material.size * u_material.size * u_material.size) {
        return;
    }
    if (s_occupancy[v] <= 0.0) {
        // Not read at the hits, but cleared: the buffer is reused for other lines.
        s_shadow[v] = 0.0;
        return;
    }
    let c = vec3<i32>(i32(v) % size, (i32(v) / size) % size, i32(v) / (size * size));
    s_shadow[v] = cone_light(vec3<f32>(c) + 0.5, lamp_direction());
}
