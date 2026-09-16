import { useEffect, useId, useRef } from 'react'

// The scene beside Upload and Pipeline: a schematic I-2K bus in the foreground,
// Earth's limb entering from the right, sunrise on the limb, space behind.
// Pure SVG, drawn flat: stepped ocean tones, outlined landmasses, a terminator
// shadow. Two independent motions share one requestAnimationFrame loop: the
// land and cloud deck turn on their own clock (the clouds slightly faster, for
// parallax), the satellite eases toward the pointer (or sways when nobody is
// pointing).
//
// Two stacked SVGs, not one: the sphere, its blurred rim glow and the sun are
// static and sit in a layer of their own (composited via will-change), so the
// gaussian filters are rasterised once rather than re-run under every frame
// of the animated layer above.

const VIEW = { width: 420, height: 600 }

// Earth: a circle whose centre lies past the right edge, so the viewBox crops
// it to a limb. The arc enters at about (420, 127), reaches (320, 420) and
// leaves through the bottom edge near (355, 600).
const EARTH = { cx: 800, cy: 420, r: 480 }
const RIM = { inner: 3, glow: 14, halo: 30 } // stroke widths of the rim layers
const SUN = { x: 400, y: 155, r: 70 } // on the limb, upper right
const SPIKES = [
  { angle: 0, length: 110 },
  { angle: 90, length: 80 },
  { angle: 38, length: 46 },
  { angle: -52, length: 46 },
]
const EARTH_TURN_SECONDS = 240
// Land turns a little slower than the cloud deck, for parallax between them.
const LAND_RATE = 0.8

// One clock for every instance. Upload and Pipeline each mount their own
// scene and only animate while on screen; angles derived from a shared epoch
// (not from each loop's own elapsed time) keep both Earths in the same phase,
// so switching views does not change what is in frame.
const EPOCH = performance.now()
const sceneSeconds = (now) => (now - EPOCH) / 1000

// Only the outer ~100 units of the disc are in frame (bearings 158-218 deg at
// rest), so everything on the surface sits at dist 380-480 and is spread round
// the whole circumference: a 240 s turn always has something in view.

// Cloud bands, in Earth's frame: distance from the centre, bearing (deg), and
// size. Each is laid tangent to its bearing so it reads as a band, not a blob.
const CLOUDS = [
  { bearing: 168, dist: 400, rx: 150, ry: 16, opacity: 0.2 },
  { bearing: 196, dist: 455, rx: 120, ry: 12, opacity: 0.16 },
  { bearing: 150, dist: 430, rx: 110, ry: 14, opacity: 0.22 },
  { bearing: 215, dist: 395, rx: 130, ry: 11, opacity: 0.15 },
  { bearing: 128, dist: 470, rx: 140, ry: 15, opacity: 0.18 },
  { bearing: 250, dist: 420, rx: 120, ry: 12, opacity: 0.14 },
  { bearing: 290, dist: 450, rx: 150, ry: 16, opacity: 0.2 },
  { bearing: 340, dist: 400, rx: 100, ry: 12, opacity: 0.16 },
  { bearing: 20, dist: 460, rx: 140, ry: 14, opacity: 0.21 },
  { bearing: 70, dist: 410, rx: 120, ry: 13, opacity: 0.15 },
  { bearing: 100, dist: 445, rx: 90, ry: 11, opacity: 0.18 },
]

// Landmasses, in the same frame. Each path is drawn in a local frame whose +x
// is "north" (the tangent direction; screen-up when the mass sits on the left
// limb) and whose +y points inward, away from the limb. `tone` picks the fill
// class. The first is the South Asian subcontinent: a broad northern mass
// tapering to a southern point, with an island off the tip.
const LANDS = [
  {
    bearing: 186, dist: 424, tone: 'green',
    // Local +x north (screen-up on the left limb), +y inward. Kept under 80
    // units across so the whole coast sits inside the visible sliver.
    d: 'M62,-36 C74,-32 84,-22 88,-10 C93,4 92,20 84,32 C78,40 66,40 54,36 C42,32 30,30 16,24 C2,18 -12,16 -30,10 C-48,4 -66,-2 -82,-10 C-92,-16 -90,-24 -78,-26 C-62,-28 -46,-30 -30,-34 C-12,-38 8,-42 30,-42 C42,-42 52,-40 62,-36 Z',
  },
  { bearing: 186, dist: 424, tone: 'green', d: 'M-98,-8 C-93,-15 -84,-12 -83,-3 C-82,4 -88,10 -95,7 C-102,4 -103,-2 -98,-8 Z' },
  {
    bearing: 210, dist: 445, tone: 'dry',
    d: 'M40,-30 C58,-26 70,-10 64,8 C58,24 40,30 22,26 C6,22 -8,30 -24,22 C-40,14 -46,-2 -36,-16 C-26,-30 -6,-30 10,-36 C20,-40 30,-34 40,-30 Z',
  },
  {
    bearing: 160, dist: 460, tone: 'arid',
    d: 'M-70,-14 C-56,-30 -30,-34 -8,-28 C10,-24 26,-30 44,-22 C60,-16 66,2 54,14 C42,26 20,22 2,28 C-18,34 -40,30 -58,18 C-72,10 -78,-2 -70,-14 Z',
  },
  {
    bearing: 262, dist: 425, tone: 'green',
    d: 'M-50,-40 C-20,-52 24,-50 56,-36 C80,-26 86,0 70,20 C56,38 30,40 6,34 C-14,30 -34,38 -52,26 C-70,14 -74,-10 -60,-26 C-56,-32 -54,-38 -50,-40 Z',
  },
  {
    bearing: 320, dist: 455, tone: 'dry',
    d: 'M-30,-20 C-10,-34 20,-30 40,-18 C54,-10 50,10 34,18 C18,26 0,20 -16,24 C-32,28 -46,14 -42,0 C-40,-10 -36,-16 -30,-20 Z',
  },
  {
    bearing: 30, dist: 435, tone: 'arid',
    d: 'M-60,-10 C-40,-36 4,-40 36,-30 C62,-22 74,-2 62,16 C50,32 24,30 0,34 C-24,38 -50,30 -62,12 C-66,4 -66,-4 -60,-10 Z',
  },
  {
    bearing: 90, dist: 465, tone: 'green',
    d: 'M-36,-16 C-16,-30 16,-28 34,-14 C46,-4 42,14 26,20 C10,26 -8,18 -26,20 C-42,22 -50,6 -44,-6 C-42,-10 -40,-13 -36,-16 Z',
  },
]

// ---------------------------------------------------------------------------
// Satellite: schematic wireframe of the I-2K bus. Model units, +X along the
// solar arrays, +Y up the boom, -Z nadir (the navigation antenna face).
// ---------------------------------------------------------------------------
// Where the model's origin sits in the viewBox. With REST and SCALE below, the
// wings stay at least 14 units inside the left edge and 16 clear of the limb
// through the whole idle sway; the pointer's full range can carry them further.
const SAT = { x: 198, y: 240 }
const BODY = { x: 0.5, y: 0.6, z: 0.5 } // half-extents: 1.0 x 1.2 x 1.0
const WING = { root: 0.75, length: 2.4, halfWidth: 0.45, cells: 3 }
const DISH = { z: -0.62, radius: 0.34, points: 12 }
const BOOM = { base: BODY.y, tip: 1.15, cross: 0.12 }

// Edge kinds map to classes in styles.css: structure, cell divisions, antenna.
// Faces are the two solar wings, filled so they read as surfaces.
function buildModel() {
  const points = []
  const edges = []
  const faces = []
  const point = (x, y, z) => points.push([x, y, z]) - 1
  const edge = (a, b, kind = 'structure') => edges.push([a, b, kind])

  // Body: a box.
  const corner = {}
  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      for (const sz of [-1, 1]) corner[`${sx}${sy}${sz}`] = point(sx * BODY.x, sy * BODY.y, sz * BODY.z)
    }
  }
  for (const [a, b] of [
    ['-1-1-1', '1-1-1'], ['-11-1', '11-1'], ['-1-11', '1-11'], ['-111', '111'],
    ['-1-1-1', '-11-1'], ['1-1-1', '11-1'], ['-1-11', '-111'], ['1-11', '111'],
    ['-1-1-1', '-1-11'], ['1-1-1', '1-11'], ['-11-1', '-111'], ['11-1', '111'],
  ]) {
    edge(corner[a], corner[b])
  }

  // Solar array wings, one per side: a yoke from the body, then a flat panel
  // split into cells.
  for (const side of [-1, 1]) {
    const yokeStart = point(side * BODY.x, 0, 0)
    const yokeEnd = point(side * WING.root, 0, 0)
    edge(yokeStart, yokeEnd)

    const cellLength = WING.length / WING.cells
    const top = []
    const bottom = []
    for (let i = 0; i <= WING.cells; i++) {
      const x = side * (WING.root + i * cellLength)
      top.push(point(x, WING.halfWidth, 0))
      bottom.push(point(x, -WING.halfWidth, 0))
    }
    faces.push([top[0], top[WING.cells], bottom[WING.cells], bottom[0]])
    for (let i = 0; i < WING.cells; i++) {
      edge(top[i], top[i + 1])
      edge(bottom[i], bottom[i + 1])
    }
    edge(top[0], bottom[0])
    edge(top[WING.cells], bottom[WING.cells])
    for (let i = 1; i < WING.cells; i++) edge(top[i], bottom[i], 'cell')
  }

  // Navigation antenna on the nadir face: a ring with spokes, on a short mount.
  const mount = point(0, 0, -BODY.z)
  const hub = point(0, 0, DISH.z)
  edge(mount, hub, 'antenna')
  const ring = []
  for (let i = 0; i < DISH.points; i++) {
    const a = (i / DISH.points) * Math.PI * 2
    ring.push(point(Math.cos(a) * DISH.radius, Math.sin(a) * DISH.radius, DISH.z))
  }
  ring.forEach((p, i) => {
    edge(p, ring[(i + 1) % DISH.points], 'antenna')
    edge(p, hub, 'antenna')
  })

  // Boom on +Y with a small cross-piece at the tip.
  const boomBase = point(0, BOOM.base, 0)
  const boomTip = point(0, BOOM.tip, 0)
  edge(boomBase, boomTip)
  edge(point(-BOOM.cross, BOOM.tip, 0), point(BOOM.cross, BOOM.tip, 0))

  return { points, edges, faces }
}

const MODEL = buildModel()

const DEG = Math.PI / 180
const REST = { x: 14 * DEG, y: -48 * DEG }
const RANGE = { x: 25 * DEG, y: 40 * DEG }
const CAMERA_DISTANCE = 10
const FOCAL = CAMERA_DISTANCE // model units at z = 0 project 1:1 before scaling
const SCALE = 62 // model units -> viewBox units
const IDLE_SWAY = 5 * DEG // idle: a slow sway about rest, this far either way
const IDLE_PERIOD = 40 // seconds per sway cycle
const EASE_POINTER = 6 // per second
const EASE_IDLE = 1.2 // per second
const OPACITY = { near: 1, far: 0.35 } // narrow, so far edges stay visible

function rotateX([x, y, z], a) {
  const c = Math.cos(a)
  const s = Math.sin(a)
  return [x, y * c - z * s, y * s + z * c]
}

function rotateY([x, y, z], a) {
  const c = Math.cos(a)
  const s = Math.sin(a)
  return [x * c + z * s, y, -x * s + z * c]
}

function project([x, y, z]) {
  const k = FOCAL / (z + CAMERA_DISTANCE)
  // SVG y grows downward; model +Y is up.
  return [x * k * SCALE, -y * k * SCALE]
}

// Writes one frame straight to the SVG elements; React is not involved.
function drawSatellite(lines, polygons, angleX, angleY) {
  const camera = MODEL.points.map((p) => rotateX(rotateY(p, angleY), angleX))
  const screen = camera.map(project)
  // Depth is normalised to this frame's own z range, so the nearest edge is
  // always fully opaque and the farthest always at OPACITY.far.
  let zNear = Infinity
  let zFar = -Infinity
  for (const [, , z] of camera) {
    if (z < zNear) zNear = z
    if (z > zFar) zFar = z
  }
  const zSpan = zFar - zNear || 1
  const depthOpacity = (z) => OPACITY.near + (OPACITY.far - OPACITY.near) * ((z - zNear) / zSpan)
  MODEL.faces.forEach((corners, i) => {
    const polygon = polygons[i]
    if (!polygon) return
    polygon.setAttribute('points', corners.map((c) => `${screen[c][0].toFixed(2)},${screen[c][1].toFixed(2)}`).join(' '))
    const z = corners.reduce((sum, c) => sum + camera[c][2], 0) / corners.length
    polygon.setAttribute('opacity', depthOpacity(z).toFixed(3))
  })
  MODEL.edges.forEach(([a, b], i) => {
    const line = lines[i]
    if (!line) return
    line.setAttribute('x1', screen[a][0].toFixed(2))
    line.setAttribute('y1', screen[a][1].toFixed(2))
    line.setAttribute('x2', screen[b][0].toFixed(2))
    line.setAttribute('y2', screen[b][1].toFixed(2))
    line.setAttribute('stroke-opacity', depthOpacity((camera[a][2] + camera[b][2]) / 2).toFixed(3))
  })
}

// Position on the sphere for a (bearing, dist) pair; `rotate` turns a local
// frame so +x runs along the tangent (north) and +y points inward.
function placement({ bearing, dist }) {
  const a = bearing * DEG
  return {
    cx: EARTH.cx + Math.cos(a) * dist,
    cy: EARTH.cy + Math.sin(a) * dist,
    rotate: bearing + 90,
  }
}

export default function OrbitScene() {
  // Upload and Pipeline each mount a scene, and url(#id) resolves document-wide
  // to the first match, so every gradient, filter and clip path is namespaced
  // per instance; without this one view paints with the other's (hidden) defs.
  const uid = useId().replace(/[^a-zA-Z0-9_-]/g, '')
  const containerRef = useRef(null)
  const linesRef = useRef([])
  const polygonsRef = useRef([])
  const cloudsRef = useRef(null)
  const landRef = useRef(null)

  useEffect(() => {
    const container = containerRef.current
    const lines = linesRef.current
    const polygons = polygonsRef.current
    const clouds = cloudsRef.current
    const land = landRef.current
    const motion = window.matchMedia('(prefers-reduced-motion: reduce)')

    const state = {
      current: { ...REST },
      target: { ...REST },
      pointing: false,
      frame: 0,
      last: 0,
      visible: false,
    }

    // Cloud angle from the shared clock; land follows at LAND_RATE.
    const drawEarth = (seconds) => {
      const angle = (360 / EARTH_TURN_SECONDS) * seconds
      clouds.setAttribute('transform', `rotate(${angle.toFixed(3)} ${EARTH.cx} ${EARTH.cy})`)
      land.setAttribute('transform', `rotate(${(angle * LAND_RATE).toFixed(3)} ${EARTH.cx} ${EARTH.cy})`)
    }
    drawSatellite(lines, polygons, state.current.x, state.current.y)
    drawEarth(sceneSeconds(performance.now()))

    function pointerAngles(event) {
      const rect = container.getBoundingClientRect()
      const nx = Math.min(Math.max(((event.clientX - rect.left) / rect.width) * 2 - 1, -1), 1)
      const ny = Math.min(Math.max(((event.clientY - rect.top) / rect.height) * 2 - 1, -1), 1)
      return { x: REST.x - ny * RANGE.x, y: REST.y + nx * RANGE.y }
    }

    // One loop, two clocks: Earth turns regardless of the pointer; the
    // satellite eases toward whichever target applies.
    function tick(now) {
      const dt = state.last ? Math.min((now - state.last) / 1000, 0.1) : 0
      state.last = now
      const seconds = sceneSeconds(now)
      drawEarth(seconds)

      if (!state.pointing) {
        state.target = { x: REST.x, y: REST.y + IDLE_SWAY * Math.sin((seconds / IDLE_PERIOD) * Math.PI * 2) }
      }
      const rate = state.pointing ? EASE_POINTER : EASE_IDLE
      const blend = 1 - Math.exp(-rate * dt)
      state.current.x += (state.target.x - state.current.x) * blend
      state.current.y += (state.target.y - state.current.y) * blend
      drawSatellite(lines, polygons, state.current.x, state.current.y)

      state.frame = requestAnimationFrame(tick)
    }

    function start() {
      if (state.frame || motion.matches || !state.visible) return
      state.last = 0
      state.frame = requestAnimationFrame(tick)
    }

    function stop() {
      cancelAnimationFrame(state.frame)
      state.frame = 0
    }

    function onMove(event) {
      const angles = pointerAngles(event)
      if (motion.matches) {
        // Reduced motion: no easing, no idle sway, no Earth; follow the pointer directly.
        state.current = angles
        drawSatellite(lines, polygons, angles.x, angles.y)
        return
      }
      state.pointing = true
      state.target = angles
    }

    function onLeave() {
      if (motion.matches) {
        state.current = { ...REST }
        drawSatellite(lines, polygons, REST.x, REST.y)
        return
      }
      // Ease back onto the shared sway.
      state.pointing = false
    }

    function onMotionChange() {
      stop()
      state.pointing = false
      state.current = { ...REST }
      state.target = { ...REST }
      drawSatellite(lines, polygons, REST.x, REST.y)
      start()
    }

    // Both views stay mounted while hidden, and the column is display:none on
    // narrow screens: only animate while the scene is actually on screen.
    const observer = new IntersectionObserver(([entry]) => {
      state.visible = entry.isIntersecting
      if (state.visible) start()
      else stop()
    })
    observer.observe(container)

    container.addEventListener('pointermove', onMove)
    container.addEventListener('pointerdown', onMove)
    container.addEventListener('pointerleave', onLeave)
    container.addEventListener('pointercancel', onLeave)
    motion.addEventListener('change', onMotionChange)

    return () => {
      stop()
      observer.disconnect()
      container.removeEventListener('pointermove', onMove)
      container.removeEventListener('pointerdown', onMove)
      container.removeEventListener('pointerleave', onLeave)
      container.removeEventListener('pointercancel', onLeave)
      motion.removeEventListener('change', onMotionChange)
    }
  }, [])

  const viewBox = `0 0 ${VIEW.width} ${VIEW.height}`

  return (
    <figure className="scene" ref={containerRef}>
      <div className="scene-stage">
        {/* Static layer: sphere, rim glow, sun. */}
        <svg className="scene-layer scene-static" viewBox={viewBox} aria-hidden="true">
          <defs>
            {/* Ocean: three flat tonal bands out from the lit limb, each
                boundary softened over a few percent rather than blended. */}
            <radialGradient id={`${uid}-earth-fill`} gradientUnits="userSpaceOnUse" cx="380" cy="130" r="720">
              <stop offset="0" className="stop-earth-mid" />
              <stop offset="0.3" className="stop-earth-mid" />
              <stop offset="0.35" className="stop-earth-band" />
              <stop offset="0.56" className="stop-earth-band" />
              <stop offset="0.61" className="stop-earth-deep" />
              <stop offset="1" className="stop-earth-deep" />
            </radialGradient>
            <radialGradient id={`${uid}-sun-fill`}>
              <stop offset="0" className="stop-sun-core" stopOpacity="1" />
              <stop offset="0.18" className="stop-sun-flare" stopOpacity="0.85" />
              <stop offset="0.5" className="stop-sun-flare" stopOpacity="0.22" />
              <stop offset="1" className="stop-sun-flare" stopOpacity="0" />
            </radialGradient>
            <filter id={`${uid}-rim-glow`} x="-10%" y="-10%" width="120%" height="120%">
              <feGaussianBlur stdDeviation="9" />
            </filter>
            <filter id={`${uid}-rim-halo`} x="-15%" y="-15%" width="130%" height="130%">
              <feGaussianBlur stdDeviation="26" />
            </filter>
            <filter id={`${uid}-sun-soft`} x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="10" />
            </filter>
            <filter id={`${uid}-spike-soft`} x="-20%" y="-300%" width="140%" height="700%">
              <feGaussianBlur stdDeviation="1.2" />
            </filter>
          </defs>

          <circle className="earth-halo" cx={EARTH.cx} cy={EARTH.cy} r={EARTH.r + RIM.halo / 2} strokeWidth={RIM.halo} filter={`url(#${uid}-rim-halo)`} />
          <circle className="earth-glow" cx={EARTH.cx} cy={EARTH.cy} r={EARTH.r + RIM.glow / 2} strokeWidth={RIM.glow} filter={`url(#${uid}-rim-glow)`} />
          <circle className="earth-body" cx={EARTH.cx} cy={EARTH.cy} r={EARTH.r} fill={`url(#${uid}-earth-fill)`} />

          <g transform={`translate(${SUN.x} ${SUN.y})`}>
            <circle className="sun-burst" r={SUN.r} fill={`url(#${uid}-sun-fill)`} filter={`url(#${uid}-sun-soft)`} />
            {SPIKES.map(({ angle, length }) => (
              <line
                key={angle}
                className="sun-spike"
                x1={-length}
                x2={length}
                y1="0"
                y2="0"
                transform={`rotate(${angle})`}
                filter={`url(#${uid}-spike-soft)`}
              />
            ))}
            <circle className="sun-core" r="9" filter={`url(#${uid}-spike-soft)`} />
          </g>
        </svg>

        {/* Animated layer: land, cloud bands, the terminator and inner rim over them, the satellite. */}
        <svg className="scene-layer scene-live" viewBox={viewBox} aria-hidden="true">
          <defs>
            <clipPath id={`${uid}-earth-clip`}>
              <circle cx={EARTH.cx} cy={EARTH.cy} r={EARTH.r} />
            </clipPath>
            <filter id={`${uid}-cloud-soft`} x="-10%" y="-40%" width="120%" height="180%">
              <feGaussianBlur stdDeviation="1.2" />
            </filter>
            {/* Terminator: night creeping in from the lower left, away from the sun. */}
            <linearGradient id={`${uid}-earth-terminator`} gradientUnits="userSpaceOnUse" x1="320" y1="600" x2="420" y2="180">
              <stop offset="0" className="stop-earth-night" stopOpacity="0.85" />
              <stop offset="0.45" className="stop-earth-night" stopOpacity="0.45" />
              <stop offset="1" className="stop-earth-night" stopOpacity="0" />
            </linearGradient>
            <radialGradient id={`${uid}-earth-rim-inner`} gradientUnits="userSpaceOnUse" cx={EARTH.cx} cy={EARTH.cy} r={EARTH.r}>
              <stop offset="0.9" className="stop-earth-rim" stopOpacity="0" />
              <stop offset="0.975" className="stop-earth-rim" stopOpacity="0.28" />
              <stop offset="1" className="stop-earth-rim" stopOpacity="0.55" />
            </radialGradient>
          </defs>

          <g clipPath={`url(#${uid}-earth-clip)`}>
            <g ref={landRef} className="earth-land">
              {LANDS.map((mass, i) => {
                const { cx, cy, rotate } = placement(mass)
                return (
                  <path
                    key={i}
                    className={`land land-${mass.tone}`}
                    d={mass.d}
                    transform={`translate(${cx} ${cy}) rotate(${rotate})`}
                  />
                )
              })}
            </g>
            <g ref={cloudsRef} className="earth-clouds" filter={`url(#${uid}-cloud-soft)`}>
              {CLOUDS.map((cloud, i) => {
                const { cx, cy, rotate } = placement(cloud)
                return (
                  <ellipse
                    key={i}
                    className="earth-cloud"
                    cx={cx}
                    cy={cy}
                    rx={cloud.rx}
                    ry={cloud.ry}
                    opacity={cloud.opacity}
                    transform={`rotate(${rotate} ${cx} ${cy})`}
                  />
                )
              })}
            </g>
            <circle cx={EARTH.cx} cy={EARTH.cy} r={EARTH.r} fill={`url(#${uid}-earth-terminator)`} />
            <circle cx={EARTH.cx} cy={EARTH.cy} r={EARTH.r} fill={`url(#${uid}-earth-rim-inner)`} />
          </g>

          <g transform={`translate(${SAT.x} ${SAT.y})`}>
            {MODEL.faces.map((_, i) => (
              <polygon
                key={i}
                ref={(node) => {
                  polygonsRef.current[i] = node
                }}
                className="sat-face"
              />
            ))}
            {MODEL.edges.map(([, , kind], i) => (
              <line
                key={i}
                ref={(node) => {
                  linesRef.current[i] = node
                }}
                className={`sat-edge sat-${kind}`}
              />
            ))}
          </g>
        </svg>
      </div>
      <figcaption className="scene-caption">I-2K bus, schematic</figcaption>
    </figure>
  )
}
