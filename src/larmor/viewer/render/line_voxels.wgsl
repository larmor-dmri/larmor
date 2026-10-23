// Streamlines voxelized as clipped capsules, after Kraaijeveld B, Jalba AC,
// Vilanova A, Chamberland M. Real-Time Rendering of Dynamic Line Sets using Voxel
// Ray Tracing. arXiv:2510.09081 (2025).
//
// Positions are in voxel coordinates: voxel i spans i to i + 1. The passes, in
// order: voxelize (the segments' ends and planes, the occupancy, and the
// segments per voxel), resolve (occupancy as
// floats, which voxels any segment touches, the counts to scan), the pyramids,
// the scan of the counts into offsets, fill (each voxel's segments), the ambient
// light per voxel, and the light at the points.

struct Params {
    size: u32,  // voxels along each side of the grid
    levels: u32,  // levels of the pyramids, the grid included
    segments: u32,
    points: u32,
    radius: f32,  // of the capsules, in voxels
    min_radius: f32,  // thinner capsules are widened to this and fade
    tan_half: f32,  // of the cones' aperture
    reach: f32,  // of the cones, in voxels
    cones: u32,
    blocks: u32,  // blocks of the scan
    pad0: u32,
    pad1: u32,
    level_start: array<vec4<u32>, 4>,  // where each level of a pyramid starts
    directions: array<vec4<f32>, 32>,  // of the cones, unit
};

struct Level {
    level: u32,  // the level a pyramid pass makes
    pad0: u32,
    pad1: u32,
    pad2: u32,
};

@group(0) @binding(0) var<uniform> u: Params;
@group(0) @binding(1) var<storage, read> points: array<f32>;  // (P, 3)
@group(0) @binding(2) var<storage, read> starts: array<u32>;  // first point of each segment
@group(0) @binding(3) var<storage, read_write> filled: array<atomic<u32>>;  // fixed point
@group(0) @binding(4) var<storage, read_write> counts: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> occupancy: array<f32>;  // pyramid
@group(0) @binding(6) var<storage, read_write> touched: array<u32>;  // pyramid, any segment
@group(0) @binding(7) var<storage, read_write> offsets: array<u32>;  // size^3 + 1
@group(0) @binding(8) var<storage, read_write> sums: array<u32>;  // of the scan's blocks
@group(0) @binding(9) var<storage, read_write> lists: array<u32>;  // segments per voxel
@group(0) @binding(10) var<storage, read_write> light: array<f32>;  // per voxel
@group(0) @binding(11) var<storage, read_write> point_light: array<f32>;
// Per segment: its start (w: the index of its first point), its end, and the
// normals of its clipping planes there.
@group(0) @binding(12) var<storage, read_write> segments: array<vec4<f32>>;
@group(1) @binding(0) var<uniform> pass_level: Level;

// The fixed-point scale of the summed occupancy: WGSL has integer atomics only.
const FIXED: f32 = 65536.0;
const GROUP: u32 = 64u;
const SCAN: u32 = 256u;

fn flat_index(gid: vec3<u32>, groups: vec3<u32>) -> u32 {
    return gid.x + gid.y * groups.x * GROUP;
}

fn point_at(i: u32) -> vec3<f32> {
    return vec3<f32>(points[3u * i], points[3u * i + 1u], points[3u * i + 2u]);
}

fn level_start(level: u32) -> u32 {
    return u.level_start[level / 4u][level % 4u];
}

fn voxel_index(c: vec3<i32>, size: i32) -> u32 {
    return u32(c.x + size * (c.y + size * c.z));
}

fn unit_or_zero(v: vec3<f32>) -> vec3<f32> {
    let n = length(v);
    return select(vec3<f32>(0.0), v / n, n > 0.0);
}

// The unit tangent at each end of segment s, by central differences along its
// streamline: the step before the segment counts where it is of the same one.
fn end_tangents(s: u32) -> array<vec3<f32>, 2> {
    let i = starts[s];
    let forward = point_at(i + 1u) - point_at(i);
    var t0 = forward;
    var t1 = forward;
    if (s > 0u && starts[s - 1u] + 1u == i) {
        t0 += step_before(i);
    }
    if (s + 1u < u.segments && starts[s + 1u] == i + 1u) {
        t1 += point_at(i + 2u) - point_at(i + 1u);
    }
    return array<vec3<f32>, 2>(unit_or_zero(t0), unit_or_zero(t1));
}

fn step_before(i: u32) -> vec3<f32> {
    return point_at(i) - point_at(i - 1u);
}

// The occupancy a clipped capsule gives the voxel centred at p.
fn capsule_occupancy(p: vec3<f32>, v0: vec3<f32>, v1: vec3<f32>, n0: vec3<f32>, n1: vec3<f32>) -> f32 {
    let widened = max(u.radius, u.min_radius);
    let d = v1 - v0;
    let pv0 = p - v0;
    let pv1 = p - v1;
    let h = clamp(dot(pv0, d) / max(dot(d, d), 1e-12), 0.0, 1.0);
    let capsule = length(pv0 - d * h) - widened;
    let signed = max(capsule, max(-dot(pv0, n0), dot(pv1, n1)));
    let fade = u.radius / widened;
    return clamp(0.5 - signed, 0.0, 1.0) * fade * fade;
}

// Visits every voxel the capsule of radius r around segment s touches: steps
// along the segment's major axis a voxel at a time and takes, at each step, the
// box the capsule spans across the two minor axes. Mode 0 adds the occupancy,
// mode 2 counts the segment, mode 1 writes it into the voxels' lists.
fn traverse(s: u32, mode: u32, r: f32) {
    let size = i32(u.size);
    let v0 = segments[4u * s].xyz;
    let v1 = segments[4u * s + 1u].xyz;
    var d = v1 - v0;
    let ad = abs(d);
    var major = 0u;
    if (ad.y > ad.x) { major = 1u; }
    if (ad.z > max(ad.x, ad.y)) { major = 2u; }
    let minor1 = (major + 1u) % 3u;
    let minor2 = (major + 2u) % 3u;
    var a = v0;
    var b = v1;
    if (d[major] < 0.0) {
        a = v1;
        b = v0;
    }
    d = b - a;
    let along = d[major];
    let norm_d = max(length(d), 1e-12);
    var per_step = d / max(along, 1e-12);
    var r1 = r;
    var r2 = r;
    if (along < 1e-9) {
        // A point: a sphere, stepped along the major axis.
        per_step = vec3<f32>(0.0);
        per_step[major] = 1.0;
    } else {
        // The half-widths of the capsule's cut across the major axis, an ellipse:
        // for the unit axis q, r * sqrt(1 - q_j^2) / |q_major| along minor axis i.
        let q = d / norm_d;
        let q_major = max(abs(q[major]), 1e-12);
        r1 = r * sqrt(clamp(1.0 - q[minor2] * q[minor2], 0.0, 1.0)) / q_major;
        r2 = r * sqrt(clamp(1.0 - q[minor1] * q[minor1], 0.0, 1.0)) / q_major;
    }
    a -= per_step * r;  // room for the round ends
    b += per_step * r;
    let t_min = a[major];
    let t_max = b[major];
    let first = floor(t_min);
    let count = max(i32(ceil(t_max) - first), 1);
    for (var k = 0; k < count; k++) {
        let cell = first + f32(k);
        let t0 = max(cell, t_min);
        let t1 = min(cell + 1.0, t_max);
        let p0 = a + per_step * (t0 - t_min);
        let p1 = a + per_step * (t1 - t_min);
        let j_lo = i32(floor(min(p0[minor1], p1[minor1]) - r1));
        let j_hi = i32(floor(max(p0[minor1], p1[minor1]) + r1));
        let k_lo = i32(floor(min(p0[minor2], p1[minor2]) - r2));
        let k_hi = i32(floor(max(p0[minor2], p1[minor2]) + r2));
        for (var j = j_lo; j <= j_hi; j++) {
            for (var m = k_lo; m <= k_hi; m++) {
                var c = vec3<i32>(0);
                c[major] = i32(cell);
                c[minor1] = j;
                c[minor2] = m;
                if (any(c < vec3<i32>(0)) || any(c >= vec3<i32>(size))) {
                    continue;
                }
                let v = voxel_index(c, size);
                if (mode == 0u) {
                    let n0 = segments[4u * s + 2u].xyz;
                    let n1 = segments[4u * s + 3u].xyz;
                    let o = capsule_occupancy(vec3<f32>(c) + 0.5, v0, v1, n0, n1);
                    if (o > 0.0) {
                        atomicAdd(&filled[v], u32(round(o * FIXED)));
                    }
                } else if (mode == 2u) {
                    atomicAdd(&counts[v], 1u);
                } else {
                    let slot = offsets[v] + atomicAdd(&counts[v], 1u);
                    lists[slot] = s;
                }
            }
        }
    }
}

@compute @workgroup_size(64)
fn voxelize(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let s = flat_index(gid, groups);
    if (s < u.segments) {
        let i = starts[s];
        let ends = end_tangents(s);
        segments[4u * s] = vec4<f32>(point_at(i), bitcast<f32>(i));
        segments[4u * s + 1u] = vec4<f32>(point_at(i + 1u), 0.0);
        segments[4u * s + 2u] = vec4<f32>(ends[0], 0.0);
        segments[4u * s + 3u] = vec4<f32>(ends[1], 0.0);
        // The occupancy of capsules widened against aliasing; the lists, of the
        // capsules themselves.
        traverse(s, 0u, max(u.radius, u.min_radius));
        traverse(s, 2u, u.radius);
    }
}

@compute @workgroup_size(64)
fn fill(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let s = flat_index(gid, groups);
    if (s < u.segments) {
        traverse(s, 1u, u.radius);
    }
}

// The occupancy as floats, at most 1; which voxels a segment touches; the counts
// to scan, and the counts cleared to serve as fill's cursors.
@compute @workgroup_size(64)
fn resolve(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let v = flat_index(gid, groups);
    if (v >= u.size * u.size * u.size) {
        return;
    }
    occupancy[v] = min(f32(atomicLoad(&filled[v])) / FIXED, 1.0);
    let n = atomicLoad(&counts[v]);
    touched[v] = select(0u, 1u, n > 0u);
    offsets[v] = n;
    atomicStore(&counts[v], 0u);
}

// One level of both pyramids from the one below: the mean occupancy of 2x2x2
// voxels, and whether any of them is touched.
@compute @workgroup_size(64)
fn pyramid_level(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let level = pass_level.level;
    let size = i32(u.size >> level);
    let v = flat_index(gid, groups);
    if (v >= u32(size * size * size)) {
        return;
    }
    let c = vec3<i32>(i32(v) % size, (i32(v) / size) % size, i32(v) / (size * size));
    let below = level_start(level - 1u);
    let here = level_start(level);
    var total = 0.0;
    var any_touched = 0u;
    for (var k = 0; k < 8; k++) {
        let child = 2 * c + vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        let w = below + voxel_index(child, 2 * size);
        total += occupancy[w];
        any_touched = max(any_touched, touched[w]);
    }
    occupancy[here + v] = total / 8.0;
    touched[here + v] = any_touched;
}

var<workgroup> shared_scan: array<u32, 256>;

// An inclusive scan of the workgroup's values in shared_scan.
fn scan_shared(t: u32) {
    for (var stride = 1u; stride < SCAN; stride *= 2u) {
        workgroupBarrier();
        var add = 0u;
        if (t >= stride) {
            add = shared_scan[t - stride];
        }
        workgroupBarrier();
        shared_scan[t] += add;
    }
    workgroupBarrier();
}

// The scan of the counts, first within blocks of SCAN voxels: offsets then hold
// each voxel's start within its block, and sums each block's total.
@compute @workgroup_size(256)
fn scan_blocks(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let block = wid.x + wid.y * groups.x;
    let v = block * SCAN + lid.x;
    let total_voxels = u.size * u.size * u.size;
    var n = 0u;
    if (v < total_voxels) {
        n = offsets[v];
    }
    shared_scan[lid.x] = n;
    scan_shared(lid.x);
    if (v < total_voxels) {
        offsets[v] = shared_scan[lid.x] - n;
    }
    if (lid.x == SCAN - 1u && block < u.blocks) {
        sums[block] = shared_scan[lid.x];
    }
}

// The scan of the blocks' totals, by one workgroup: each thread a run of blocks.
// The total of all ends up past the last block.
@compute @workgroup_size(256)
fn scan_sums(@builtin(local_invocation_id) lid: vec3<u32>) {
    let per = (u.blocks + SCAN - 1u) / SCAN;
    let begin = lid.x * per;
    var run = 0u;
    for (var b = begin; b < min(begin + per, u.blocks); b++) {
        let n = sums[b];
        sums[b] = run;
        run += n;
    }
    shared_scan[lid.x] = run;
    scan_shared(lid.x);
    let before = shared_scan[lid.x] - run;
    for (var b = begin; b < min(begin + per, u.blocks); b++) {
        sums[b] += before;
    }
    if (lid.x == SCAN - 1u) {
        sums[u.blocks] = shared_scan[lid.x];
    }
}

// The blocks' starts added: offsets[v] is where voxel v's list starts, and
// offsets[size^3] the end of the last.
@compute @workgroup_size(64)
fn add_sums(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let v = flat_index(gid, groups);
    let total_voxels = u.size * u.size * u.size;
    if (v < total_voxels) {
        offsets[v] += sums[v / SCAN];
    } else if (v == total_voxels) {
        offsets[v] = sums[u.blocks];
    }
}

// The occupancy at base-voxel coordinates `at` on one level, linear between
// voxel centres and 0 outside the grid.
fn sample_occupancy(at: vec3<f32>, level: u32) -> f32 {
    let size = i32(u.size >> level);
    let c = at / f32(1u << level) - 0.5;
    let base = vec3<i32>(floor(c));
    let f = c - floor(c);
    let start = level_start(level);
    var value = 0.0;
    for (var k = 0; k < 8; k++) {
        let o = vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        let p = base + o;
        if (any(p < vec3<i32>(0)) || any(p >= vec3<i32>(size))) {
            continue;
        }
        let w = select(1.0 - f, f, vec3<bool>(o == vec3<i32>(1)));
        value += w.x * w.y * w.z * occupancy[start + voxel_index(p, size)];
    }
    return value;
}

// Between the two levels around a fractional one.
fn sample_between(at: vec3<f32>, level: f32) -> f32 {
    let low = u32(floor(level));
    let high = min(low + 1u, u.levels - 1u);
    let weight = level - f32(low);
    return (1.0 - weight) * sample_occupancy(at, low) + weight * sample_occupancy(at, high);
}

// The share of the cones from each occupied voxel that nothing stops.
@compute @workgroup_size(64)
fn ambient(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let size = i32(u.size);
    let v = flat_index(gid, groups);
    if (v >= u.size * u.size * u.size) {
        return;
    }
    if (occupancy[v] <= 0.0) {
        light[v] = 0.0;
        return;
    }
    let c = vec3<i32>(i32(v) % size, (i32(v) / size) % size, i32(v) / (size * size));
    let centre = vec3<f32>(c) + 0.5;
    var sum = 0.0;
    for (var k = 0u; k < u.cones; k++) {
        let direction = u.directions[k].xyz;
        var opacity = 0.0;
        var reached = 1.0;  // from the voxel's edge, past its own lines
        while (reached < u.reach) {
            let width = max(2.0 * reached * u.tan_half, 1.0);
            let level = min(log2(width), f32(u.levels - 1u));
            var occ = clamp(sample_between(centre + direction * reached, level), 0.0, 1.0);
            // Steps of half the width: the occupancy corrected for that length.
            occ = 1.0 - sqrt(1.0 - occ);
            opacity += (1.0 - opacity) * occ;
            reached += width / 2.0;
        }
        sum += 1.0 - opacity;
    }
    light[v] = sum / f32(u.cones);
}

// The light at each point, averaged over the occupied voxels around it.
@compute @workgroup_size(64)
fn light_at_points(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let i = flat_index(gid, groups);
    if (i >= u.points) {
        return;
    }
    let size = i32(u.size);
    let c = point_at(i) - 0.5;
    let base = vec3<i32>(floor(c));
    let f = c - floor(c);
    var lit = 0.0;
    var share = 0.0;
    for (var k = 0; k < 8; k++) {
        let o = vec3<i32>(k & 1, (k >> 1) & 1, (k >> 2) & 1);
        let p = base + o;
        if (any(p < vec3<i32>(0)) || any(p >= vec3<i32>(size))) {
            continue;
        }
        let w = select(1.0 - f, f, vec3<bool>(o == vec3<i32>(1)));
        let weight = w.x * w.y * w.z;
        let v = voxel_index(p, size);
        lit += weight * light[v];
        share += select(0.0, weight, occupancy[v] > 0.0);
    }
    point_light[i] = select(1.0, lit / share, share > 1e-6);
}
