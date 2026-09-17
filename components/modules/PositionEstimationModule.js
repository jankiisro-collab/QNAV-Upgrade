"use client";

import {
  useEffect,
  useMemo,
  useState,
} from "react";

import {
  LineChart,
  Line,
  BarChart,
  Bar,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";

import {
  BrainCircuit,
  Target,
  TrendingDown,
  Activity,
  Play,
  Pause,
  RotateCcw,
  MapPin,
  Database,
  Settings2,
  Cpu,
  Navigation,
  Sparkles,
  FlaskConical,
  ShieldCheck,
} from "lucide-react";

import PositionEstimation3D from "../PositionEstimation3D";
import { VEHICLE_PRESETS } from "../../lib/vehiclePresets";
import { mulberry32, makeGaussian } from "../../lib/rng";
import dashboardData from "../../public/data/qnav_dashboard.json";

const PANEL = "qnav-panel p-5";

// Shorter display labels for the 5-way model comparison, keyed by the
// full model name used in stage9_model_comparison.csv / the exported JSON.
const MODEL_LABELS = {
  "BiLSTM": "BiLSTM",
  "Temporal Transformer": "Transformer",
  "Physics-Informed Transformer": "Physics-\nInformed",
  "Physics-Informed Transformer + Uncertainty": "+ Uncertainty",
  "Physics-Informed Transformer + Uncertainty + EKF": "+ EKF (final)",
};

// Metrics hidden from the "Final model metrics" grid. CEP50_m / CEP95_m are
// excluded on request; the rest are kept out because they duplicate what's
// already shown in the stat cards or ablation chart above.
const HIDDEN_METRIC_KEYS = new Set([
  "model",
  "p95_position_error_m",
  "max_position_error_m",
  "NLL",
  "coverage_95",
  "calibration_error",
  "CEP50_m",
  "CEP95_m",
]);

const BAR_COLORS = ["#94a3b8", "#64748b", "#0e7490", "#0369a1", "#0ea5e9"];

export default function PositionEstimationModule({ simulation }) {
  const {
    rows = [],
    params = {},
    geometry = [],
    noiseBudget = {},
    trajectoryMeta = {},
    waypoints = [],
  } = simulation || {};

  const [featureData, setFeatureData] = useState([]);

  useEffect(() => {
    let cancelled = false;

    fetch("/data/ml/feature_importance.json")
      .then((res) => (res.ok ? res.json() : []))
      .then((json) => {
        if (!cancelled) setFeatureData(json);
      })
      .catch(() => {
        if (!cancelled) setFeatureData([]);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const featureChartData = useMemo(() => {
    return (featureData || []).slice(0, 12).map((item) => ({
      name: formatFeatureName(item.feature),
      importance: Number(item.importance),
    }));
  }, [featureData]);

  // ---------------------------------------------------------------------
  // Section A state: interactive simulation sandbox
  // ---------------------------------------------------------------------

  const [currentTime, setCurrentTime] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [coordinate, setCoordinate] = useState("latitude");

  const trajectoryData = useMemo(() => {
    if (!rows.length) return [];

    const integration = params.integrationTimeS || 10;
    const diamonds = params.nDiamonds || 4;

    const noiseScale =
      (noiseBudget?.shotNoiseStd || 1) *
      (8 / diamonds) *
      (10 / integration);

    // Deterministic (seeded) Gaussian noise -- reproducible for a given
    // parameter set, and lightly smoothed so it reads as a plausible
    // position-error residual for this sandbox. This is a parameter-driven
    // simulation, not a live model or EKF computation -- see the "Validated
    // Model Results" section below for the real, offline-evaluated numbers.
    const rng = mulberry32(20260910);
    const gaussian = makeGaussian(rng);

    const rawLat = rows.map(() => gaussian(0, 1));
    const rawLon = rows.map(() => gaussian(0, 1));
    const rawAlt = rows.map(() => gaussian(0, 1));

    const smooth = (arr, window = 4) =>
      arr.map((_, i) => {
        const start = Math.max(0, i - window);
        const slice = arr.slice(start, i + 1);
        return slice.reduce((a, b) => a + b, 0) / slice.length;
      });

    const smoothLat = smooth(rawLat);
    const smoothLon = smooth(rawLon);
    const smoothAlt = smooth(rawAlt);

    return rows.map((row, index) => {
      const actualLat = row.lat_deg ?? row.lat ?? row.latitude ?? row.actual_lat ?? 0;
      const actualLon = row.lon_deg ?? row.lon ?? row.longitude ?? row.actual_lon ?? 0;
      const actualAlt = row.alt_m ?? row.alt ?? row.altitude ?? row.actual_alt ?? 0;

      const latNoise = smoothLat[index] * noiseScale * 0.35;
      const lonNoise = smoothLon[index] * noiseScale * 0.35;
      const altNoise = smoothAlt[index] * noiseScale * 1.8;

      return {
        time_s: row.t ?? row.time ?? row.time_s ?? index,
        actual_lat: actualLat,
        actual_lon: actualLon,
        actual_alt: actualAlt,
        predicted_lat: actualLat + latNoise / 111320,
        predicted_lon:
          actualLon +
          lonNoise / (111320 * Math.max(Math.cos((actualLat * Math.PI) / 180), 0.2)),
        predicted_alt: actualAlt + altNoise,
      };
    });
  }, [rows, params, noiseBudget]);

  const errorData = useMemo(() => {
    return trajectoryData.map((item) => {
      const latError = (item.actual_lat - item.predicted_lat) * 111320;
      const lonError =
        (item.actual_lon - item.predicted_lon) *
        111320 *
        Math.cos((item.actual_lat * Math.PI) / 180);
      const altError = item.actual_alt - item.predicted_alt;
      const error3D = Math.sqrt(latError ** 2 + lonError ** 2 + altError ** 2);

      return {
        time_s: item.time_s,
        latitude_error_m: latError,
        longitude_error_m: lonError,
        altitude_error_m: altError,
        position_error_3d_m: error3D,
      };
    });
  }, [trajectoryData]);

  useEffect(() => {
    if (!playing || !trajectoryData.length) return;

    const timer = setInterval(() => {
      setCurrentTime((previous) => {
        if (previous >= trajectoryData.length - 1) {
          setPlaying(false);
          return 0;
        }
        return previous + 1;
      });
    }, 100);

    return () => clearInterval(timer);
  }, [playing, trajectoryData.length]);

  useEffect(() => {
    setCurrentTime(0);
    setPlaying(false);
  }, [trajectoryData.length]);

  const selectedPoint = useMemo(() => {
    if (!trajectoryData.length) return null;
    return trajectoryData[currentTime] || trajectoryData[0];
  }, [trajectoryData, currentTime]);

  const selectedError = useMemo(() => {
    if (!errorData.length) return null;
    return errorData[currentTime] || errorData[0];
  }, [errorData, currentTime]);

  const comparisonData = useMemo(() => {
    return trajectoryData.map((item) => {
      const actualKey = `actual_${
        coordinate === "altitude" ? "alt" : coordinate === "latitude" ? "lat" : "lon"
      }`;
      const predictedKey = `predicted_${
        coordinate === "altitude" ? "alt" : coordinate === "latitude" ? "lat" : "lon"
      }`;

      return {
        time_s: Number(item.time_s),
        actual: Number(item[actualKey]),
        predicted: Number(item[predictedKey]),
      };
    });
  }, [trajectoryData, coordinate]);

  const configSummary = useMemo(() => {
    const preset = VEHICLE_PRESETS[params.vehicleType];
    const usingRoute = trajectoryMeta?.useWaypoints ?? waypoints.length >= 2;

    return {
      vehicleLabel: preset?.label || params.vehicleType || "—",
      vehicleIcon: preset?.icon || "",
      speed: params.speed,
      altitude: params.startAlt,
      route: usingRoute ? `Custom route — ${waypoints.length} waypoints` : "Synthetic demo pattern",
      diamonds: params.nDiamonds,
      integrationTimeS: params.integrationTimeS,
      noiseStdNT: noiseBudget?.shotNoiseStdNT,
      samples: rows.length,
      duration: trajectoryMeta?.duration ?? params.duration,
    };
  }, [params, noiseBudget, trajectoryMeta, waypoints, rows.length]);

  // ---------------------------------------------------------------------
  // Section B state: validated offline model results (real EKF fusion)
  // ---------------------------------------------------------------------

  const comparison = dashboardData.model_comparison || [];
  const trajectory = dashboardData.trajectory || [];
  const finalMetrics = dashboardData.model_performance || {};

  const baseline = comparison[0];
  const finalModel = comparison[comparison.length - 1];
  const improvementPct =
    baseline && finalModel
      ? (((baseline.position_MAE_m - finalModel.position_MAE_m) / baseline.position_MAE_m) * 100).toFixed(0)
      : null;

  const barData = comparison.map((row, i) => ({
    name: MODEL_LABELS[row.model] || row.model,
    fullName: row.model,
    MAE: row.position_MAE_m,
    fill: BAR_COLORS[i % BAR_COLORS.length],
  }));

  // Position error over time (EKF): 3D error only, horizontal error removed.
  const trajChartData = trajectory.map((p) => ({
    t: p.time_s,
    error_3d: p.position_error_3d_m,
  }));

  const positionChartData = trajectory.map((p) => ({
    t: p.time_s,
    actual_E: p.actual_E_m,
    predicted_E: p.predicted_E_m,
    actual_N: p.actual_N_m,
    predicted_N: p.predicted_N_m,
  }));

  return (
    <div className="space-y-6 qnav-fade-in">
      <div>
        <h1 className="text-xl md:text-2xl font-bold tracking-tight qnav-glow-text">
          Position Estimation &amp; AI Model
        </h1>
        <p className="text-gray-500 text-sm mt-1">
          Explore how simulation parameters shape the receiver&apos;s position error, then compare
          against the validated deep-learning + EKF pipeline evaluated offline on held-out test data.
        </p>
      </div>

      {/* ================= SECTION A: INTERACTIVE SIMULATION ================= */}

      <SectionBanner
        icon={FlaskConical}
        badge="Interactive Simulation"
        title="Live Parameter Sandbox"
        description="Every number below is generated in your browser from the current simulation parameters (vehicle, route, sensor ensemble, noise budget). It is a what-if sandbox, not a trained model or Kalman filter — use it to build intuition for how configuration choices affect position error."
        tone="sim"
      />

      <div className={PANEL}>
        <div className="flex items-center gap-2 mb-3">
          <Settings2 size={15} className="text-cyan" />
          <h2 className="font-semibold text-sm">Showing Results For</h2>
        </div>

        <div className="flex flex-wrap gap-2">
          <ConfigChip label="Vehicle" value={`${configSummary.vehicleIcon} ${configSummary.vehicleLabel}`.trim()} />
          <ConfigChip label="Speed" value={`${configSummary.speed ?? "—"} m/s`} />
          <ConfigChip label="Altitude" value={`${configSummary.altitude ?? "—"} m`} />
          <ConfigChip label="Route" value={configSummary.route} />
          <ConfigChip label="Ensemble" value={`${configSummary.diamonds ?? "—"} diamonds`} />
          <ConfigChip label="Integration" value={`${configSummary.integrationTimeS ?? "—"} s`} />
          <ConfigChip
            label="Sensor Noise"
            value={typeof configSummary.noiseStdNT === "number" ? `±${configSummary.noiseStdNT.toFixed(2)} nT` : "—"}
          />
          <ConfigChip label="Samples" value={configSummary.samples.toLocaleString()} />
        </div>

        <p className="text-[11px] text-gray-500 mt-3">
          This updates automatically whenever you change Vehicle &amp; Trajectory, Location &amp; Map or
          Sensor Ensemble and press <span className="text-cyan font-medium">Set Parameters</span>.
        </p>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        <StatCard label="Time" value={`${selectedPoint?.time_s ?? 0} s`} icon={Activity} />
        <StatCard
          label="Simulated 3D Error"
          value={`${Number(selectedError?.position_error_3d_m ?? 0).toFixed(2)} m`}
          icon={Target}
        />
        <StatCard label="Samples" value={trajectoryData.length} icon={TrendingDown} />
        <StatCard label="Pipeline" value="Sim → Features → Model → EKF" icon={BrainCircuit} small />
      </div>

      <div className={PANEL}>
        <div className="flex items-center gap-2 mb-5">
          <BrainCircuit size={17} className="text-cyan" />
          <div>
            <h2 className="font-semibold text-sm">Position Estimation Pipeline</h2>
            <p className="text-[11px] text-gray-500 mt-1">
              How a receiver position estimate is produced end to end, from raw simulation through to a
              fused position.
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          <PipelineStep icon={Database} title="Simulation" text="Vehicle motion, magnetic field and NV sensor measurements" />
          <PipelineArrow />
          <PipelineStep icon={Settings2} title="Feature Engineering" text="Normalization, feature extraction and magnetic signatures" />
          <PipelineArrow />
          <PipelineStep icon={Cpu} title="Physics-Informed Transformer" text="Learned position prediction from sensor features" />
          <PipelineArrow />
          <PipelineStep icon={Navigation} title="EKF Fusion" text="Kalman filtering of the model output into a final position" />
        </div>
      </div>

      <div className={PANEL}>
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="font-semibold text-sm">3D Receiver Trajectory (Sandbox)</h2>
            <p className="text-[11px] text-gray-500 mt-1">
              Ground-truth vs. simulated-estimate trajectory for the current parameters.
            </p>
          </div>
          <div className="text-[10px] font-mono text-gray-400">t = {selectedPoint?.time_s ?? 0} s</div>
        </div>

        <div className="h-[450px] rounded-lg overflow-hidden border border-border bg-slate-50">
          <PositionEstimation3D trajectory={trajectoryData} currentIndex={currentTime} />
        </div>

        <SimulationControls
          currentTime={currentTime}
          setCurrentTime={setCurrentTime}
          playing={playing}
          setPlaying={setPlaying}
          length={trajectoryData.length}
        />
      </div>

      {selectedError && (
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
          <SimulationCard label="Latitude Error" value={`${Number(selectedError.latitude_error_m).toFixed(2)} m`} />
          <SimulationCard label="Longitude Error" value={`${Number(selectedError.longitude_error_m).toFixed(2)} m`} />
          <SimulationCard label="Altitude Error" value={`${Number(selectedError.altitude_error_m).toFixed(2)} m`} />
          <SimulationCard
            label="3D Position Error"
            value={`${Number(selectedError.position_error_3d_m).toFixed(2)} m`}
            highlight
          />
        </div>
      )}

      <div className={PANEL}>
        <div className="flex flex-wrap items-center justify-between gap-3 mb-4">
          <div>
            <h2 className="font-semibold text-sm">Actual vs Simulated-Estimate Position</h2>
            <p className="text-[11px] text-gray-500 mt-1">
              Ground-truth trajectory vs. this sandbox&apos;s simulated position estimate.
            </p>
          </div>

          <div className="flex gap-1 bg-slate-100 rounded-lg p-1">
            <CoordinateButton active={coordinate === "latitude"} onClick={() => setCoordinate("latitude")}>
              Latitude
            </CoordinateButton>
            <CoordinateButton active={coordinate === "longitude"} onClick={() => setCoordinate("longitude")}>
              Longitude
            </CoordinateButton>
            <CoordinateButton active={coordinate === "altitude"} onClick={() => setCoordinate("altitude")}>
              Altitude
            </CoordinateButton>
          </div>
        </div>

        <div className="h-[360px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={comparisonData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#dbe3ec" />
              <XAxis dataKey="time_s" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} domain={["auto", "auto"]} />
              <Tooltip
                formatter={(value, name) => [
                  Number(value).toFixed(coordinate === "altitude" ? 2 : 6),
                  name === "actual" ? "Ground Truth" : "Simulated Estimate",
                ]}
                labelFormatter={(value) => `Time: ${value} s`}
              />
              <ReferenceLine x={selectedPoint?.time_s} stroke="#ef4444" strokeDasharray="4 4" />
              <Line type="monotone" dataKey="actual" name="actual" stroke="#0e7490" strokeWidth={2} dot={false} isAnimationActive={false} />
              <Line type="monotone" dataKey="predicted" name="predicted" stroke="#f59e0b" strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="flex justify-center gap-6 mt-3 text-[10px]">
          <div className="flex items-center gap-2">
            <span className="w-3 h-0.5 bg-cyan" />
            Ground Truth
          </div>
          <div className="flex items-center gap-2">
            <span className="w-3 h-0.5 bg-amber-500" />
            Simulated Estimate
          </div>
        </div>
      </div>

      <div className={PANEL}>
        <div className="flex items-center gap-2 mb-4">
          <Activity size={16} className="text-cyan" />
          <div>
            <h2 className="font-semibold text-sm">Simulated 3D Position Error vs Time</h2>
            <p className="text-[11px] text-gray-500 mt-1">
              Euclidean distance between actual and simulated-estimate positions in this sandbox.
            </p>
          </div>
        </div>

        <div className="h-[350px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={errorData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#dbe3ec" />
              <XAxis dataKey="time_s" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10 }} />
              <Tooltip formatter={(value) => [`${Number(value).toFixed(2)} m`, "3D Error"]} />
              <ReferenceLine x={selectedError?.time_s} stroke="#ef4444" strokeDasharray="4 4" />
              <Line type="monotone" dataKey="position_error_3d_m" stroke="#0e7490" strokeWidth={1.8} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ================= SECTION B: VALIDATED MODEL RESULTS ================= */}

      <SectionBanner
        icon={ShieldCheck}
        badge="Validated Model Results"
        title="Offline Evaluation — Physics-Informed Transformer + EKF"
        description="These charts are loaded from real offline evaluation output (the model ablation study and held-out test-set predictions), not generated live in the browser. They show the true, measured accuracy of the trained pipeline."
        tone="model"
      />

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <MiniStat label="Final model MAE" value={finalModel ? `${finalModel.position_MAE_m.toFixed(1)} m` : "—"} color="#0369a1" />
        <MiniStat label="Improvement vs BiLSTM" value={improvementPct !== null ? `${improvementPct}%` : "—"} color="#15803d" />
      </div>

      <div className={PANEL}>
        <div className="mb-3">
          <h2 className="font-semibold text-sm">Model comparison — mean position error</h2>
          <p className="text-[11px] text-gray-500 mt-1">
            BiLSTM and the Temporal Transformer are deep-learning baselines. Each step to the right adds a
            proposed component: the physics-informed loss, the uncertainty head, then EKF fusion.
          </p>
        </div>

        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={barData} margin={{ top: 10, right: 16, left: 0, bottom: 24 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
            <XAxis dataKey="name" tick={{ fontSize: 11, fill: "#64748b" }} interval={0} height={50} />
            <YAxis
              tick={{ fontSize: 11, fill: "#64748b" }}
              label={{ value: "Position MAE (m)", angle: -90, position: "insideLeft", fontSize: 11, fill: "#64748b" }}
            />
            <Tooltip
              formatter={(value) => [`${Number(value).toFixed(1)} m`, "MAE"]}
              labelFormatter={(_, payload) => payload?.[0]?.payload?.fullName || ""}
              contentStyle={{ fontSize: 12, borderRadius: 10 }}
            />
            <Bar dataKey="MAE" radius={[6, 6, 0, 0]}>
              {barData.map((entry, i) => (
                <Cell key={i} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className={PANEL}>
        <div className="mb-3">
          <h2 className="font-semibold text-sm">Actual vs predicted trajectory (East / North)</h2>
          <p className="text-[11px] text-gray-500 mt-1">Physics-Informed Transformer + EKF, one representative test mission.</p>
        </div>

        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={positionChartData} margin={{ top: 10, right: 16, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="t" tick={{ fontSize: 11, fill: "#64748b" }} label={{ value: "time (s)", position: "insideBottom", offset: -3, fontSize: 11, fill: "#64748b" }} />
            <YAxis tick={{ fontSize: 11, fill: "#64748b" }} label={{ value: "metres", angle: -90, position: "insideLeft", fontSize: 11, fill: "#64748b" }} />
            <Tooltip contentStyle={{ fontSize: 12, borderRadius: 10 }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Line type="monotone" dataKey="actual_E" stroke="#0f172a" dot={false} strokeWidth={2} name="Actual E" />
            <Line type="monotone" dataKey="predicted_E" stroke="#0ea5e9" dot={false} strokeWidth={2} strokeDasharray="4 3" name="Predicted E" />
            <Line type="monotone" dataKey="actual_N" stroke="#475569" dot={false} strokeWidth={2} name="Actual N" />
            <Line type="monotone" dataKey="predicted_N" stroke="#f59e0b" dot={false} strokeWidth={2} strokeDasharray="4 3" name="Predicted N" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className={PANEL}>
        <div className="mb-3">
          <h2 className="font-semibold text-sm">Position error over time (EKF)</h2>
          <p className="text-[11px] text-gray-500 mt-1">3D position error, Physics-Informed Transformer + EKF.</p>
        </div>

        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={trajChartData} margin={{ top: 10, right: 16, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="t" tick={{ fontSize: 11, fill: "#64748b" }} label={{ value: "time (s)", position: "insideBottom", offset: -3, fontSize: 11, fill: "#64748b" }} />
            <YAxis tick={{ fontSize: 11, fill: "#64748b" }} label={{ value: "error (m)", angle: -90, position: "insideLeft", fontSize: 11, fill: "#64748b" }} />
            <Tooltip contentStyle={{ fontSize: 12, borderRadius: 10 }} />
            <Line type="monotone" dataKey="error_3d" stroke="#dc2626" dot={false} strokeWidth={2} name="3D error" />
          </LineChart>
        </ResponsiveContainer>
      </div>

      <div className={PANEL}>
        <div className="flex items-center gap-2 mb-5">
          <BrainCircuit size={17} className="text-cyan" />
          <div>
            <h2 className="font-semibold text-sm">Feature Importance</h2>
            <p className="text-[11px] text-gray-500 mt-1">
              Features contributing most to the Physics-Informed Transformer position prediction.
            </p>
          </div>
        </div>

        <div className="space-y-3">
          {featureChartData.map((item, index) => (
            <div key={item.name} className="flex items-center gap-3">
              <div className="w-5 text-[10px] font-mono text-gray-400 text-right">{index + 1}</div>
              <div className="w-32 text-[11px] text-gray-600 truncate">{item.name}</div>
              <div className="flex-1 h-2 bg-slate-100 rounded-full overflow-hidden">
                <div
                  className="h-full bg-cyan rounded-full"
                  style={{ width: `${Math.min(item.importance * 100, 100)}%` }}
                />
              </div>
              <div className="w-16 text-right text-[10px] font-mono text-gray-500">
                {item.importance.toFixed(4)}
              </div>
            </div>
          ))}
        </div>
      </div>

      <div className={PANEL}>
        <h2 className="font-semibold text-sm">Final model metrics</h2>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 text-xs mt-3">
          {Object.entries(finalMetrics)
            .filter(([k]) => !HIDDEN_METRIC_KEYS.has(k))
            .map(([key, value]) => (
              <div key={key} className="rounded-lg bg-gray-50 px-3 py-2">
                <div className="text-gray-500 uppercase tracking-wide text-[10px]">{key}</div>
                <div className="font-mono font-semibold text-gray-800 mt-0.5">
                  {typeof value === "number" ? value.toFixed(3) : String(value)}
                </div>
              </div>
            ))}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Shared small components
// ---------------------------------------------------------------------------

function SectionBanner({ icon: Icon, badge, title, description, tone }) {
  const toneClasses =
    tone === "model"
      ? "border-emerald-300/60 bg-emerald-50/60"
      : "border-amber-300/60 bg-amber-50/60";

  const badgeClasses =
    tone === "model" ? "bg-emerald-600 text-white" : "bg-amber-500 text-white";

  return (
    <div className={`qnav-panel p-5 border ${toneClasses}`}>
      <span className={`inline-flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wide px-2.5 py-1 rounded-full ${badgeClasses}`}>
        <Icon size={12} />
        {badge}
      </span>
      <h2 className="font-semibold text-base mt-2.5">{title}</h2>
      <p className="text-[12px] text-gray-600 mt-1 leading-relaxed">{description}</p>
    </div>
  );
}

function ConfigChip({ label, value }) {
  return (
    <div className="flex items-center gap-1.5 bg-slate-50 border border-border rounded-full px-3 py-1.5">
      <span className="text-[10px] uppercase tracking-wide text-gray-400">{label}</span>
      <span className="text-[11px] font-mono font-medium text-gray-700">{value}</span>
    </div>
  );
}

function CoordinateButton({ active, onClick, children }) {
  return (
    <button
      onClick={onClick}
      className={`px-3 py-1.5 rounded-md text-[10px] font-medium transition ${
        active ? "bg-white text-cyan shadow-sm" : "text-gray-500 hover:text-gray-800"
      }`}
    >
      {children}
    </button>
  );
}

function PipelineStep({ icon: Icon, title, text }) {
  return (
    <div className="border border-border rounded-lg p-4 bg-slate-50">
      <div className="flex items-center gap-2 mb-2">
        <div className="w-7 h-7 rounded-md bg-cyan/10 flex items-center justify-center">
          <Icon size={15} className="text-cyan" />
        </div>
        <div className="font-semibold text-xs">{title}</div>
      </div>
      <div className="text-[10px] text-gray-500 leading-relaxed">{text}</div>
    </div>
  );
}

function PipelineArrow() {
  return <div className="hidden md:flex items-center justify-center text-gray-300">→</div>;
}

function SimulationControls({ currentTime, setCurrentTime, playing, setPlaying, length }) {
  const max = Math.max(length - 1, 0);

  return (
    <div className="mt-5">
      <div className="flex justify-between mb-2">
        <span className="text-[10px] uppercase tracking-wide text-gray-500">Simulation Time</span>
        <span className="font-mono text-sm font-semibold">{currentTime} s</span>
      </div>

      <input
        type="range"
        min="0"
        max={max}
        value={currentTime}
        onChange={(event) => {
          setPlaying(false);
          setCurrentTime(Number(event.target.value));
        }}
        className="w-full accent-cyan cursor-pointer"
      />

      <div className="flex justify-between text-[10px] text-gray-400 mt-1">
        <span>0 s</span>
        <span>{max} s</span>
      </div>

      <div className="flex justify-center gap-2 mt-4">
        <button
          onClick={() => setPlaying((value) => !value)}
          className="flex items-center gap-2 bg-slate-900 text-white px-4 py-2 rounded-lg text-xs font-medium"
        >
          {playing ? <Pause size={14} /> : <Play size={14} />}
          {playing ? "Pause" : "Play Simulation"}
        </button>

        <button
          onClick={() => {
            setPlaying(false);
            setCurrentTime(0);
          }}
          className="flex items-center gap-2 border border-border px-4 py-2 rounded-lg text-xs font-medium"
        >
          <RotateCcw size={14} />
          Reset
        </button>
      </div>
    </div>
  );
}

function StatCard({ label, value, icon: Icon, small }) {
  return (
    <div className="qnav-panel p-4">
      <div className="flex justify-between">
        <span className="text-[10px] uppercase tracking-wide text-gray-500">{label}</span>
        <Icon size={15} className="text-cyan" />
      </div>
      <div className={small ? "text-sm font-semibold font-mono mt-1.5 leading-snug" : "text-xl font-bold font-mono mt-1.5"}>
        {value}
      </div>
    </div>
  );
}

function SimulationCard({ label, value, highlight }) {
  return (
    <div className={`qnav-panel p-4 ${highlight ? "border-cyan/40 bg-cyan/5" : ""}`}>
      <div className="text-[10px] uppercase tracking-wide text-gray-500">{label}</div>
      <div className={`text-lg font-bold font-mono mt-1 ${highlight ? "text-cyan" : ""}`}>{value}</div>
    </div>
  );
}

function MiniStat({ label, value, color }) {
  return (
    <div className="qnav-panel p-3.5">
      <div className="text-[10px] uppercase tracking-wide text-gray-500">{label}</div>
      <div className="text-lg font-bold font-mono mt-0.5" style={{ color }}>
        {value}
      </div>
    </div>
  );
}

function formatFeatureName(name) {
  return name.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}
