// Streamlines ray traced as clipped capsules through their voxel grid, after
// Kraaijeveld B, Jalba AC, Vilanova A, Chamberland M. Real-Time Rendering of
// Dynamic Line Sets using Voxel Ray Tracing. arXiv:2510.09081 (2025).
//
// The grid's box is drawn with its back faces; each fragment casts a ray in voxel
// coordinates (the object's transform takes them to the world) and walks the
// grid, skipping the blocks of the touched pyramid that no segment reaches, and
// tests the capsules listed in each voxel it crosses. A hit counts only within
// the voxel being crossed, so the hits come in order along the ray. The hit is
// lit by the ambient light and the shadow of the voxels around it (see
// voxel_cones.wgsl), from the lamp at the camera's upper left. Opaque lines stop
// at the first hit; translucent ones blend every hit, front to back.

{$ include 'pygfx.std.wgsl' $}

struct VertexInput {
    @builtin(vertex_index) vertex_index : u32,
};

const AMBIENT = 0.4;  // the scene's light all round
const HEAD = 1.2;  // the lamp at the camera
const KEY = 1.2;  // the lamp at its upper left, which casts the shadows
const MAX_STEPS = 4096;  // cells crossed per ray, at most
const MAX_HITS = 8;  // hits kept per voxel at a time, translucent
const NO_HIT = 1e30;

const BOX = array<vec3<f32>, 8>(
    vec3<f32>(0.0, 0.0, 0.0), vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 1.0, 0.0),
    vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(1.0, 0.0, 1.0),
    vec3<f32>(0.0, 1.0, 1.0), vec3<f32>(1.0, 1.0, 1.0),
);
const FACES = array<u32, 36>(
    0u, 2u, 1u, 1u, 2u, 3u,  4u, 5u, 6u, 5u, 7u, 6u,
    0u, 1u, 4u, 1u, 5u, 4u,  2u, 6u, 3u, 3u, 6u, 7u,
    0u, 4u, 2u, 2u, 4u, 6u,  1u, 3u, 5u, 3u, 7u, 5u,
);

@vertex
fn vs_main(in: VertexInput) -> Varyings {
    let size = f32(u_material.size);
    let data_pos = vec4<f32>(BOX[FACES[in.vertex_index]] * size, 1.0);
    let world_pos = u_wobject.world_transform * data_pos;
    let ndc_pos = u_stdinfo.projection_transform * u_stdinfo.cam_transform * world_pos;
    let ndc_to_data = u_wobject.world_transform_inv * u_stdinfo.cam_transform_inv * u_stdinfo.projection_transform_inv;
    // The ray through this point of the box, from the near to the far plane, in
    // voxel coordinates.
    let cam_sign = sign(determinant(u_stdinfo.cam_transform));
    let ndc_near = vec4<f32>(ndc_pos.xy, -1.0 * cam_sign * ndc_pos.w, ndc_pos.w);
    let ndc_far = vec4<f32>(ndc_pos.xy, cam_sign * ndc_pos.w, ndc_pos.w);
    var varyings: Varyings;
    varyings.position = vec4<f32>(ndc_pos);
    varyings.world_pos = vec3<f32>(world_pos.xyz);
    varyings.data_back_pos = vec4<f32>(data_pos);
    varyings.data_near_pos = vec4<f32>(ndc_to_data * ndc_near);
    varyings.data_far_pos = vec4<f32>(ndc_to_data * ndc_far);
    return varyings;
}

fn colour_at(i: u32) -> vec3<f32> {
    return vec3<f32>(s_colours[3u * i], s_colours[3u * i + 1u], s_colours[3u * i + 2u]);
}

fn unit_or_zero(v: vec3<f32>) -> vec3<f32> {
    let n = length(v);
    return select(vec3<f32>(0.0), v / n, n > 0.0);
}

// Where the ray enters and leaves a ball, if it meets it.
fn ball_span(ro: vec3<f32>, rd: vec3<f32>, centre: vec3<f32>, r: f32) -> vec2<f32> {
    let oc = ro - centre;
    let b = dot(oc, rd);
    let disc = b * b - (dot(oc, oc) - r * r);
    if (disc <= 0.0) {
        return vec2<f32>(NO_HIT, -NO_HIT);
    }
    let root = sqrt(disc);
    return vec2<f32>(-b - root, -b + root);
}

struct Hit {
    t: f32,
    segment: u32,
    normal: vec3<f32>,
    along: f32,  // 0 at the segment's start, 1 at its end
};

// The ray's entry into the clipped capsule around segment s: the capsule (a
// cylinder between two balls, which is convex) cut by the planes across the
// tangents at its ends. t is NO_HIT if the ray misses it.
fn hit_capsule(ro: vec3<f32>, rd: vec3<f32>, s: u32) -> Hit {
    var hit: Hit;
    hit.t = NO_HIT;
    hit.segment = s;
    let a = s_segments[4u * s].xyz;
    let b = s_segments[4u * s + 1u].xyz;
    let r = u_material.radius;
    let ab = b - a;
    let len2 = dot(ab, ab);
    // The convex capsule: the span from the first entry to the last exit of its
    // parts.
    var span = ball_span(ro, rd, a, r);
    let span_b = ball_span(ro, rd, b, r);
    span = vec2<f32>(min(span.x, span_b.x), max(span.y, span_b.y));
    if (len2 > 1e-12) {
        let axis = ab / sqrt(len2);
        let oa = ro - a;
        let d_perp = rd - axis * dot(rd, axis);
        let o_perp = oa - axis * dot(oa, axis);
        let qa = dot(d_perp, d_perp);
        let qb = dot(o_perp, d_perp);
        let qc = dot(o_perp, o_perp) - r * r;
        var body = vec2<f32>(NO_HIT, -NO_HIT);
        if (qa > 1e-12) {
            let disc = qb * qb - qa * qc;
            if (disc > 0.0) {
                let root = sqrt(disc);
                body = vec2<f32>((-qb - root) / qa, (-qb + root) / qa);
            }
        } else if (qc <= 0.0) {
            body = vec2<f32>(-NO_HIT, NO_HIT);  // along the axis, inside the radius
        }
        // Between the two ends along the axis.
        let y0 = dot(oa, axis);
        let dy = dot(rd, axis);
        if (abs(dy) > 1e-12) {
            let e0 = -y0 / dy;
            let e1 = (sqrt(len2) - y0) / dy;
            body = vec2<f32>(max(body.x, min(e0, e1)), min(body.y, max(e0, e1)));
        } else if (y0 < 0.0 || y0 > sqrt(len2)) {
            body = vec2<f32>(NO_HIT, -NO_HIT);
        }
        if (body.x < body.y) {
            span = vec2<f32>(min(span.x, body.x), max(span.y, body.y));
        }
    }
    if (span.x >= span.y) {
        return hit;
    }
    // The clipping planes: kept where (p - a) . n0 >= 0 and (p - b) . n1 <= 0.
    let ends = array<vec3<f32>, 2>(s_segments[4u * s + 2u].xyz, s_segments[4u * s + 3u].xyz);
    var plane_normal = vec3<f32>(0.0);
    let den0 = dot(rd, ends[0]);
    let num0 = dot(ro - a, ends[0]);
    if (abs(den0) < 1e-9) {
        if (num0 < 0.0) { return hit; }
    } else if (den0 > 0.0) {
        let t = -num0 / den0;
        if (t > span.x) { span.x = t; plane_normal = -ends[0]; }
    } else {
        span.y = min(span.y, -num0 / den0);
    }
    let den1 = dot(rd, ends[1]);
    let num1 = dot(ro - b, ends[1]);
    if (abs(den1) < 1e-9) {
        if (num1 > 0.0) { return hit; }
    } else if (den1 < 0.0) {
        let t = -num1 / den1;
        if (t > span.x) { span.x = t; plane_normal = ends[1]; }
    } else {
        span.y = min(span.y, -num1 / den1);
    }
    if (span.x >= span.y) {
        return hit;
    }
    hit.t = span.x;
    let p = ro + rd * span.x;
    hit.along = clamp(dot(p - a, ab) / max(len2, 1e-12), 0.0, 1.0);
    if (any(plane_normal != vec3<f32>(0.0))) {
        hit.normal = plane_normal;
    } else {
        hit.normal = unit_or_zero(p - (a + ab * hit.along));
    }
    return hit;
}

// Where the ray leaves the cell of `width` voxels around `cell`.
fn cell_exit(ro: vec3<f32>, inv_rd: vec3<f32>, cell: vec3<f32>, width: f32) -> f32 {
    let far_side = (cell + select(vec3<f32>(0.0), vec3<f32>(1.0), inv_rd > vec3<f32>(0.0))) * width;
    let t = (far_side - ro) * inv_rd;
    return min(t.x, min(t.y, t.z));
}

// The hit's colour as linear light.
fn shade(hit: Hit, ro: vec3<f32>, rd: vec3<f32>, lamp: vec3<f32>) -> vec3<f32> {
    let i = bitcast<u32>(s_segments[4u * hit.segment].w);
    let srgb = mix(colour_at(i), colour_at(i + 1u), hit.along);
    let albedo = srgb2physical(srgb);
    let p = ro + rd * hit.t;
    let n = select(hit.normal, -hit.normal, dot(hit.normal, rd) > 0.0);
    var ambient = 1.0;
    if (u_material.ambient_occlusion > 0.5) {
        ambient = ambient_at(p);
    }
    let lamp_side = dot(n, lamp);
    let shadow = shadow_at(p);
    let head = max(dot(n, -rd), 0.0);
    let diffuse = AMBIENT * ambient + HEAD * head * mix(1.0, ambient, 0.5) + KEY * max(lamp_side, 0.0) * shadow;
    let halfway = normalize(lamp - rd);
    let specular = 0.15 * pow(max(dot(n, halfway), 0.0), 40.0) * shadow;
    return albedo * diffuse * RECIPROCAL_PI + vec3<f32>(specular);
}

@fragment
fn fs_main(varyings: Varyings) -> FragmentOutput {
    let sizef = f32(u_material.size);
    let size = i32(u_material.size);
    let back_pos = varyings.data_back_pos.xyz / varyings.data_back_pos.w;
    let far_pos = varyings.data_far_pos.xyz / varyings.data_far_pos.w;
    let near_pos = varyings.data_near_pos.xyz / varyings.data_near_pos.w;
    let rd = normalize(far_pos - near_pos);
    let inv_rd = 1.0 / select(rd, vec3<f32>(1e-12), abs(rd) < vec3<f32>(1e-12));
    // From the box's front face, or the near plane inside it, to its back face.
    var dist = dot(near_pos - back_pos, rd);
    let t_lo = (vec3<f32>(0.0) - back_pos) * inv_rd;
    let t_hi = (vec3<f32>(sizef) - back_pos) * inv_rd;
    let t_in = min(t_lo, t_hi);
    dist = max(dist, max(t_in.x, max(t_in.y, t_in.z)));
    if (dist >= 0.0) { discard; }
    let ro = back_pos + rd * dist;
    let t_end = -dist;

    let lamp = lamp_direction();

    let top = min(u_material.levels - 1u, 5u);
    var t = 0.0;
    var colour = vec3<f32>(0.0);
    var alpha = 0.0;
    var first_t = NO_HIT;
    var steps = 0;
    loop {
        if (t > t_end || steps >= MAX_STEPS || alpha > 0.995) { break; }
        steps++;
        let p = ro + rd * (t + 1e-3);
        // The coarsest block that no segment touches is skipped whole.
        var skipped = false;
        for (var level = top; level >= 1u; level--) {
            let width = f32(1u << level);
            let cell = floor(p / width);
            let n = i32(u_material.size >> level);
            let c = vec3<i32>(cell);
            if (inside(c, n) && s_touched[level_start(level) + voxel_index(c, n)] == 0u) {
                t = cell_exit(ro, inv_rd, cell, width);
                skipped = true;
                break;
            }
        }
        if (skipped) { continue; }
        let cell = floor(p);
        let c = vec3<i32>(cell);
        let t_exit = cell_exit(ro, inv_rd, cell, 1.0);
        if (!inside(c, size)) { t = t_exit; continue; }
        let v = voxel_index(c, size);
        let begin = s_offsets[v];
        let end = s_offsets[v + 1u];
        // Only hits within this voxel, from where the ray enters it to where it
        // leaves: later ones are found in their own voxels.
        var after = t;
        loop {
            // The nearest hits after `after`, up to MAX_HITS, in order.
            var hits: array<Hit, MAX_HITS>;
            var count = 0;
            for (var e = begin; e < end; e++) {
                let hit = hit_capsule(ro, rd, s_lists[e]);
                if (hit.t <= after || hit.t > t_exit) { continue; }
                $$ if transparent
                var slot = min(count, MAX_HITS - 1);
                if (count == MAX_HITS && hit.t >= hits[MAX_HITS - 1].t) { continue; }
                while (slot > 0 && hits[slot - 1].t > hit.t) {
                    hits[slot] = hits[slot - 1];
                    slot--;
                }
                hits[slot] = hit;
                count = min(count + 1, MAX_HITS);
                $$ else
                if (count == 0 || hit.t < hits[0].t) {
                    hits[0] = hit;
                    count = 1;
                }
                $$ endif
            }
            for (var k = 0; k < count; k++) {
                let hit = hits[k];
                $$ if transparent
                let a = u_material.opacity;
                $$ else
                let a = 1.0;
                $$ endif
                colour += (1.0 - alpha) * a * shade(hit, ro, rd, lamp);
                alpha += (1.0 - alpha) * a;
                first_t = min(first_t, hit.t);
                after = hit.t;
            }
            $$ if transparent
            if (count < MAX_HITS || alpha > 0.995) { break; }
            $$ else
            break;
            $$ endif
        }
        $$ if not transparent
        if (alpha > 0.0) { break; }
        $$ endif
        t = t_exit;
    }
    if (alpha <= 0.0) { discard; }

    let hit_world = u_wobject.world_transform * vec4<f32>(ro + rd * first_t, 1.0);
    let ndc = u_stdinfo.projection_transform * u_stdinfo.cam_transform * hit_world;
    var out: FragmentOutput;
    $$ if transparent
    out.color = vec4<f32>(colour / alpha, alpha);
    $$ else
    out.color = vec4<f32>(colour, 1.0);
    $$ endif
    out.depth = ndc.z / ndc.w;
    $$ if write_pick
    out.pick = pick_pack(u32(u_wobject.global_id), 20);
    $$ endif
    return out;
}
