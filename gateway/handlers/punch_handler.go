package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/boxio/gateway/client"
	"github.com/boxio/gateway/parallel"
	"github.com/boxio/gateway/telemetry"
)

// PunchPayload is the JSON body accepted at POST /api/v1/punch.
type PunchPayload struct {
	Attacker     int    `json:"attacker"`
	PunchType    string `json:"punch_type"`
	Target       string `json:"target"`
	Outcome      string `json:"outcome"`
	Damage       int    `json:"damage"`
	Fighter1Name string `json:"fighter1_name"`
	Fighter2Name string `json:"fighter2_name"`
	RoundNum     int    `json:"round_num"`
	FightID      string `json:"fight_id"`
}

// PunchResponse is the JSON body returned from POST /api/v1/punch.
type PunchResponse struct {
	Commentary string  `json:"commentary"`
	LatencyMs  float64 `json:"latency_ms"`
	Provider   string  `json:"provider"`
	P95Ms      float64 `json:"p95_latency_ms"`
}

// PunchHandler handles incoming fight events.
type PunchHandler struct {
	dispatcher *parallel.Dispatcher
	metrics    *telemetry.Metrics
}

// NewPunchHandler wires up dependencies.
func NewPunchHandler(d *parallel.Dispatcher, m *telemetry.Metrics) *PunchHandler {
	return &PunchHandler{dispatcher: d, metrics: m}
}

func (h *PunchHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}

	var payload PunchPayload
	if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
		http.Error(w, fmt.Sprintf("invalid JSON: %v", err), http.StatusBadRequest)
		return
	}

	// Defaults
	if payload.Fighter1Name == "" {
		payload.Fighter1Name = "Fighter 1"
	}
	if payload.Fighter2Name == "" {
		payload.Fighter2Name = "Fighter 2"
	}
	if payload.RoundNum == 0 {
		payload.RoundNum = 1
	}
	if payload.FightID == "" {
		payload.FightID = "unknown"
	}

	req := client.PunchRequest{
		Attacker:     payload.Attacker,
		PunchType:    payload.PunchType,
		Target:       payload.Target,
		Outcome:      payload.Outcome,
		Damage:       payload.Damage,
		Fighter1Name: payload.Fighter1Name,
		Fighter2Name: payload.Fighter2Name,
		RoundNum:     payload.RoundNum,
	}

	result := h.dispatcher.Dispatch(req, payload.FightID)

	if result.Err != nil {
		if result.Err == parallel.ErrTimeout {
			http.Error(w, "commentary timeout", http.StatusGatewayTimeout)
		} else {
			http.Error(w, result.Err.Error(), http.StatusBadGateway)
		}
		return
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(PunchResponse{
		Commentary: result.Commentary,
		LatencyMs:  result.LatencyMs,
		Provider:   result.Provider,
		P95Ms:      h.metrics.P95(),
	})
}
