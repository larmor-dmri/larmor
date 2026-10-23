// Cones through the occupancy pyramid of the streamlines' voxel grid, shared by
// the ray tracer and the pass that lights the voxels; after Kraaijeveld B, Jalba
// AC, Vilanova A, Chamberland M. Real-Time Rendering of Dynamic Line Sets using
// Voxel Ray Tracing. arXiv:2510.09081 (2025).

// Where the lamp sits in the camera's frame: above and left of it.
const LAMP = vec3<f32>(-0.6, 0.8, 1.0);
const SHADOW_TAN = 0.15;  // half the aperture of the shadow cone
const SHADOW_REACH = 48.0;  // in voxels

// The lamp's direction in voxel coordinates.
fn lamp_direction() -> vec3<f32> {
    let world = (u_stdinfo.cam_transform_inv * vec4<f32>(normalize(LAMP), 0.0)).xyz;
    return normalize((u_wobject.world_transform_inv * vec4<f32>(world, 0.0)).xyz);
}

fn level_start(level: u32) -> u32 {
    if (level < 4u) {
        return u_material.level_start_a[level];
    }
    return u_material.level_start_b[level - 4u];
}

fn voxel_index(c: vec3<i32>, size: i32) -> u32 {
    return u32(c.x + size * (c.y + size * c.z));
}

fn inside(c: vec3<i32>, size: i32) -> bool {
    return all(c >= vec3<i32>(0)) && all(c < vec3<i32>(size));
}

// The occupancy at voxel coordinates `at` on one level, linear between voxel
// centres and 0 outside the grid.
fn occupancy_on(at: vec3<f32>, level: u32) -> f32 {
    let size = i32(u_material.size >> level);
    let c = at / f32(1u << level) - 0.5;
    let base = vec3<i32>(floor(c));
    let f = c - floor(c);
    let start = level_start(level);
    var value = 0.0;
    for (var k = 0; k < 8; k++) {
        let o = vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        let p = base + o;
        if (!inside(p, size)) {
            continue;
        }
        let w = select(1.0 - f, f, vec3<bool>(o == vec3<i32>(1)));
        value += w.x * w.y * w.z * s_occupancy[start + voxel_index(p, size)];
    }
    return value;
}

// The share of light a cone toward `direction` lets through from p.
fn cone_light(p: vec3<f32>, direction: vec3<f32>) -> f32 {
    var opacity = 0.0;
    var reached = 1.0;
    let top = f32(u_material.levels - 1u);
    while (reached < SHADOW_REACH && opacity < 0.99) {
        let width = max(2.0 * reached * SHADOW_TAN, 1.0);
        let level = min(log2(width), top);
        let low = u32(floor(level));
        let high = min(low + 1u, u_material.levels - 1u);
        let at = p + direction * reached;
        let w = level - f32(low);
        var occ = (1.0 - w) * occupancy_on(at, low) + w * occupancy_on(at, high);
        occ = 1.0 - sqrt(1.0 - clamp(occ, 0.0, 1.0));  // steps of half the width
        opacity += (1.0 - opacity) * occ;
        reached += width / 2.0;
    }
    return 1.0 - opacity;
}

// The ambient light at p, from the occupied voxels around it.
fn ambient_at(p: vec3<f32>) -> f32 {
    return clamp(field_at(p, 0u) * u_material.light_scale, 0.0, 1.0);
}

// The lamp's light at p, from the occupied voxels around it.
fn shadow_at(p: vec3<f32>) -> f32 {
    return field_at(p, 1u);
}

// A value of the occupied voxels around p, linear between their centres: the
// ambient light (which 0) or the lamp's light (which 1).
fn field_at(p: vec3<f32>, which: u32) -> f32 {
    let size = i32(u_material.size);
    let c = p - 0.5;
    let base = vec3<i32>(floor(c));
    let f = c - floor(c);
    var lit = 0.0;
    var share = 0.0;
    for (var k = 0; k < 8; k++) {
        let o = vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        let q = base + o;
        if (!inside(q, size)) {
            continue;
        }
        let w = select(1.0 - f, f, vec3<bool>(o == vec3<i32>(1)));
        let weight = w.x * w.y * w.z;
        let v = voxel_index(q, size);
        if (s_occupancy[v] > 0.0) {  // only occupied voxels hold values
            lit += weight * select(s_light[v], s_shadow[v], which == 1u);
            share += weight;
        }
    }
    return select(1.0, lit / share, share > 1e-6);
}

