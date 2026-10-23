// Blended (direct) volume rendering: emission-absorption ray casting.
//
// The geometry and ray set-up follow pygfx's volume_ray.wgsl: the back faces
// of the volume's box are drawn, and each fragment marches the ray from the
// front face (or the near plane) to the back face, in voxel coordinates.
//
// The ray is cut into slabs of a fixed length, one instance each, drawn from
// the farthest slab to the nearest. Each slab composites its own samples front
// to back and writes the depth of its first visible sample, so the depth test
// drops a slab behind an opaque object already drawn, and the blend puts the
// slabs in front over it. The error is at most one slab of tissue behind the
// object, composited as if it were in front.

{$ include 'pygfx.std.wgsl' $}
$$ if colormap_dim
    {$ include 'pygfx.colormap.wgsl' $}
$$ endif
{$ include 'pygfx.volume_common.wgsl' $}


struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
    @builtin(instance_index) instance_index : u32,
};


@vertex
fn vs_main(in: VertexInput) -> Varyings {
    var geo = get_vol_geometry();
    // A mirroring world or camera transform reverses the triangles' winding, and
    // the culling would keep the front faces: swap two corners of each triangle.
    var index = i32(in.vertex_index);
    let mirrored = sign(determinant(u_wobject.world_transform)) * sign(determinant(u_stdinfo.cam_transform)) < 0.0;
    if (mirrored && index % 3 != 1) {
        index = select(index - 2, index + 2, index % 3 == 0);
    }
    let i0 = geo.indices[index];

    let data_pos = vec4<f32>(nonlinear_transform(geo.positions[i0]), 1.0);
    let world_pos = u_wobject.world_transform * data_pos;
    let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;
    let ndc_to_data = u_wobject.world_transform_inv * u_stdinfo.cam_transform_inv * u_stdinfo.projection_transform_inv;

    var varyings: Varyings;
    varyings.position = vec4<f32>(ndc_pos);
    varyings.world_pos = vec3<f32>(world_pos.xyz);
    // A back face, because the front faces are culled.
    varyings.data_back_pos = vec4<f32>(data_pos);

    // The ray through this fragment, from the near to the far plane, in voxels.
    let cam_sign = sign(determinant(u_stdinfo.cam_transform));
    let ndc_pos1 = vec4<f32>(ndc_pos.xy, -1.0 * cam_sign * ndc_pos.w, ndc_pos.w);
    let ndc_pos2 = vec4<f32>(ndc_pos.xy, cam_sign * ndc_pos.w, ndc_pos.w);
    varyings.data_near_pos = vec4<f32>(ndc_to_data * ndc_pos1);
    varyings.data_far_pos = vec4<f32>(ndc_to_data * ndc_pos2);

    // Instance 0 is the farthest slab, so the slabs are blended back to front.
    varyings.slab = f32({{ n_slabs }}u - 1u - in.instance_index);
    return varyings;
}


// A value in [0, 1) that changes from pixel to pixel, to jitter the first sample.
fn pixel_noise(p: vec2<f32>) -> f32 {
    return fract(52.9829189 * fract(dot(p, vec2<f32>(0.06711056, 0.00583715))));
}


fn to_ndc_depth(coord: vec3<f32>, sizef: vec3<f32>) -> f32 {
    let data_pos = coord * sizef - vec3<f32>(0.5);
    let world_pos = u_wobject.world_transform * vec4<f32>(data_pos, 1.0);
    let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;
    return ndc_pos.z / ndc_pos.w;
}


$$ if shading
// Samples below this share of clim are not lit.
const UNLIT_BELOW = 0.04;

// Headlight shading from the gradient, which is the normal of the local surface.
fn shade(physical: vec3<f32>, coord: vec3<f32>, view_world: vec3<f32>) -> vec3<f32> {
    // Central differences per voxel, interpolated: the same as differences of the
    // interpolated values a voxel either side, in one read.
    let grad_data = textureSampleLevel(t_grad, s_grad, coord, 0.0).xyz * u_material.gradient_scale;
    // Normals take the inverse transpose of the world transform.
    let inv = u_wobject.world_transform_inv;
    let normal = transpose(mat3x3<f32>(inv[0].xyz, inv[1].xyz, inv[2].xyz)) * grad_data;
    let len = length(normal);
    // Flat regions have no surface: there the gradient is noise, so leave them unlit.
    let strength = smoothstep(0.02, 0.1, length(grad_data) / max(abs(u_material.maprange[1] - u_material.maprange[0]), 1e-6));
    if (len < 1e-8) {
        return physical;
    }
    // The light comes from the eye, so it only matters how much the surface faces the eye.
    let facing = abs(dot(normal / len, view_world));
    let lit = physical * (0.35 + 0.65 * facing) + vec3<f32>(0.2 * pow(facing, 24.0));
    return mix(physical, lit, strength);
}
$$ endif


@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {
    {$ include 'pygfx.clipping_planes.wgsl' $}

    let sizef = vec3<f32>(textureDimensions(t_img));

    // The ray, in voxel coordinates, as in pygfx's volume_ray.wgsl.
    let back_pos = varyings.data_back_pos.xyz / varyings.data_back_pos.w;
    let far_pos = varyings.data_far_pos.xyz / varyings.data_far_pos.w;
    let near_pos = varyings.data_near_pos.xyz / varyings.data_near_pos.w;
    let view_ray = normalize(far_pos - near_pos);
    var dist = dot(near_pos - back_pos, view_ray);
    dist = max(dist, min((-0.5 - back_pos.x) / view_ray.x, (sizef.x - 0.5 - back_pos.x) / view_ray.x));
    dist = max(dist, min((-0.5 - back_pos.y) / view_ray.y, (sizef.y - 0.5 - back_pos.y) / view_ray.y));
    dist = max(dist, min((-0.5 - back_pos.z) / view_ray.z, (sizef.z - 0.5 - back_pos.z) / view_ray.z));
    let front_pos = back_pos + view_ray * dist;
    let ray_length = -dist;  // in voxels

    // Samples are a fixed step apart along the whole ray, the first one jittered
    // per pixel to hide wood-grain rings; this slab takes its share of them.
    let step = max(u_material.step_size, 0.05);
    let jitter = pixel_noise(varyings.position.xy);
    let n_total = i32(ceil(ray_length / step - jitter));
    var per_slab = n_total;
    if (u_material.slab_size > 0.0) {
        per_slab = max(1, i32(round(u_material.slab_size / step)));
    }
    let first = i32(varyings.slab + 0.5) * per_slab;
    let last = min(first + per_slab, n_total);
    if (first >= last) { discard; }

    let start_coord = (front_pos + vec3<f32>(0.5)) / sizef;
    let step_coord = view_ray * step / sizef;

    // Alpha is defined per voxel of path; correct it for the step length.
    let alpha_scale = clamp(u_material.density * u_material.opacity, 0.0, 1.0);
    // A range of zero width (minimum equal to maximum) would divide by zero below.
    let span = u_material.maprange[1] - u_material.maprange[0];
    let clim_span = select(span, 1e-12, abs(span) < 1e-12);

    $$ if shading
    let world_matrix = mat3x3<f32>(u_wobject.world_transform[0].xyz, u_wobject.world_transform[1].xyz, u_wobject.world_transform[2].xyz);
    let view_world = normalize(world_matrix * view_ray);
    $$ endif

    var acc_color = vec3<f32>(0.0);  // premultiplied by alpha
    var acc_alpha = 0.0;
    var first_coord = vec3<f32>(-1.0);
    for (var i = first; i < last; i = i + 1) {
        let coord = start_coord + (f32(i) + jitter) * step_coord;
        let value = sample_vol(coord, sizef);

        // Transparent at or below the low end of clim, densest at the high end.
        let t = clamp(((value.r {{ climcorrection }}) - u_material.maprange[0]) / clim_span, 0.0, 1.0);
        if (t <= 0.0) { continue; }
        let color = sampled_value_to_color(value);
        let voxel_alpha = t * alpha_scale * color.a;
        let alpha = 1.0 - pow(1.0 - voxel_alpha, step);
        if (alpha <= 0.0) { continue; }

        $$ if colorspace == 'srgb'
            var physical = srgb2physical(color.rgb);
        $$ else
            var physical = color.rgb;
        $$ endif
        $$ if shading
            // The faintest samples (many of a head's) are nearly transparent and
            // dark, and lighting them would not show: they are left unlit.
            if (t > UNLIT_BELOW) {
                physical = shade(physical, coord, view_world);
            }
        $$ endif

        if (first_coord.x < 0.0) { first_coord = coord; }
        acc_color += (1.0 - acc_alpha) * alpha * physical;
        acc_alpha += (1.0 - acc_alpha) * alpha;
        if (acc_alpha > 0.98) { break; }
    }

    if (acc_alpha <= 0.002) { discard; }
    do_alpha_test(acc_alpha);

    var out: FragmentOutput;
    // The pipeline blends with straight (not premultiplied) alpha.
    out.color = vec4<f32>(acc_color / acc_alpha, acc_alpha);
    out.depth = to_ndc_depth(first_coord, sizef);

    $$ if write_pick
    out.pick = (
        pick_pack(u32(u_wobject.global_id), 20) +
        pick_pack(u32(first_coord.x * 16383.0), 14) +
        pick_pack(u32(first_coord.y * 16383.0), 14) +
        pick_pack(u32(first_coord.z * 16383.0), 14)
    );
    $$ endif
    return out;
}
