package telemetry

import (
	"fmt"
	"sort"
	"sync"
	"time"
)

// Metrics tracks per-request latency and emits p50/p95/p99 summaries.
// Thread-safe: safe to Record from multiple goroutines.
type Metrics struct {
	mu      sync.Mutex
	samples []float64 // latency in ms, rolling window of last maxSamples
	total   int64
	errors  int64
}

const maxSamples = 1000

func NewMetrics() *Metrics {
	return &Metrics{}
}

// Record adds a latency sample (milliseconds).
func (m *Metrics) Record(latencyMs float64) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.samples = append(m.samples, latencyMs)
	if len(m.samples) > maxSamples {
		m.samples = m.samples[len(m.samples)-maxSamples:]
	}
	m.total++
}

// RecordError increments the error counter.
func (m *Metrics) RecordError() {
	m.mu.Lock()
	m.errors++
	m.mu.Unlock()
}

// P95 returns the 95th-percentile latency in milliseconds.
// Returns 0 if no samples recorded.
func (m *Metrics) P95() float64 { return m.percentile(0.95) }
func (m *Metrics) P50() float64 { return m.percentile(0.50) }
func (m *Metrics) P99() float64 { return m.percentile(0.99) }

func (m *Metrics) percentile(p float64) float64 {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(m.samples) == 0 {
		return 0
	}
	cp := make([]float64, len(m.samples))
	copy(cp, m.samples)
	sort.Float64s(cp)
	idx := int(float64(len(cp)) * p)
	if idx >= len(cp) {
		idx = len(cp) - 1
	}
	return cp[idx]
}

// Summary returns a human-readable latency summary string.
func (m *Metrics) Summary() string {
	return fmt.Sprintf(
		"requests=%d errors=%d p50=%.1fms p95=%.1fms p99=%.1fms",
		m.total, m.errors, m.P50(), m.P95(), m.P99(),
	)
}

// EventTelemetry represents a single fight event log entry.
type EventTelemetry struct {
	FightID    string
	Attacker   int
	PunchType  string
	Target     string
	Outcome    string
	ReceivedAt time.Time
}

// LogEvent is fire-and-forget, records a fight event to stdout.
// In production this would write to a time-series store (e.g. InfluxDB, Datadog).
func LogEvent(e EventTelemetry) {
	fmt.Printf(
		"[TELEMETRY] fight=%s at=%s attacker=%d %s->%s (%s)\n",
		e.FightID,
		e.ReceivedAt.Format(time.RFC3339Nano),
		e.Attacker,
		e.PunchType,
		e.Target,
		e.Outcome,
	)
}
