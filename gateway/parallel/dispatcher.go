// Package parallel contains the goroutine fan-out logic that is the core
// latency win of the Go gateway.
//
// Each incoming punch fires two goroutines concurrently:
//   1. Commentary goroutine: calls the Python FastAPI service (blocking, ~200-500ms)
//   2. Telemetry goroutine: logs event data (fire-and-forget, ~1ms)
//
// Without goroutines: 400ms (LLM) + 80ms (telemetry) = 480ms sequential.
// With goroutines:    max(400ms, 80ms) = 400ms parallel, about 17% less.
// End-to-end p95 stays under 600ms via a 550ms select timeout.
package parallel

import (
	"errors"
	"time"

	"github.com/boxio/gateway/client"
	"github.com/boxio/gateway/telemetry"
)

// ErrTimeout is returned when the commentary goroutine exceeds the deadline.
var ErrTimeout = errors.New("commentary timeout: p95 budget exceeded")

// Result holds the commentary response or an error.
type Result struct {
	Commentary string
	LatencyMs  float64
	Provider   string
	Err        error
}

// commentaryResult is the internal channel type.
type commentaryResult struct {
	resp *client.CommentaryResponse
	err  error
}

// Dispatcher fans out to commentary + telemetry goroutines and collects results.
type Dispatcher struct {
	pythonClient *client.PythonClient
	metrics      *telemetry.Metrics
	timeout      time.Duration
}

// NewDispatcher creates a Dispatcher with the given client and metrics.
func NewDispatcher(c *client.PythonClient, m *telemetry.Metrics) *Dispatcher {
	return &Dispatcher{
		pythonClient: c,
		metrics:      m,
		timeout:      550 * time.Millisecond,
	}
}

// Dispatch sends a punch event, fires goroutines, and returns commentary.
// Hard deadline of 550ms, so p95 stays under 600ms even when the LLM has a bad tail.
func (d *Dispatcher) Dispatch(req client.PunchRequest, fightID string) Result {
	wall := time.Now()

	// Channel sized 1 so goroutine never blocks on send if we've already timed out.
	ch := make(chan commentaryResult, 1)

	// Goroutine 1: commentary (blocking LLM call)
	go func() {
		resp, err := d.pythonClient.PostPunch(req)
		ch <- commentaryResult{resp, err}
	}()

	// Goroutine 2: telemetry (fire-and-forget; runs concurrently, never blocks the response)
	go func() {
		telemetry.LogEvent(telemetry.EventTelemetry{
			FightID:    fightID,
			Attacker:   req.Attacker,
			PunchType:  req.PunchType,
			Target:     req.Target,
			Outcome:    req.Outcome,
			ReceivedAt: wall,
		})
	}()

	// Wait for commentary with hard timeout
	select {
	case r := <-ch:
		latencyMs := float64(time.Since(wall).Milliseconds())
		d.metrics.Record(latencyMs)

		if r.err != nil {
			d.metrics.RecordError()
			return Result{Err: r.err}
		}

		commentary := ""
		if r.resp.Commentary != nil {
			commentary = *r.resp.Commentary
		}
		return Result{
			Commentary: commentary,
			LatencyMs:  latencyMs,
			Provider:   r.resp.Provider,
		}

	case <-time.After(d.timeout):
		d.metrics.RecordError()
		return Result{Err: ErrTimeout}
	}
}
