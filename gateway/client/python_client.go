package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// PunchRequest mirrors the Python FastAPI PunchRequest schema.
type PunchRequest struct {
	Attacker      int    `json:"attacker"`
	PunchType     string `json:"punch_type"`
	Target        string `json:"target"`
	Outcome       string `json:"outcome"`
	Damage        int    `json:"damage"`
	Knockdown     bool   `json:"knockdown"`
	FightID       string `json:"fight_id"`
	Fighter1Name  string `json:"fighter1_name"`
	Fighter2Name  string `json:"fighter2_name"`
	RoundNum      int    `json:"round_num"`
}

// CommentaryResponse mirrors the Python FastAPI CommentaryResponse schema.
type CommentaryResponse struct {
	Commentary    *string `json:"commentary"`
	LatencyMs     float64 `json:"latency_ms"`
	P95LatencyMs  float64 `json:"p95_latency_ms"`
	Provider      string  `json:"provider"`
}

// PythonClient sends punch events to the Python FastAPI service.
type PythonClient struct {
	baseURL    string
	httpClient *http.Client
}

// NewPythonClient creates a client pointing at the given Python service URL.
func NewPythonClient(baseURL string) *PythonClient {
	return &PythonClient{
		baseURL: baseURL,
		httpClient: &http.Client{
			Timeout: 560 * time.Millisecond, // hard ceiling; gateway adds ~10ms overhead
		},
	}
}

// PostPunch sends a punch event and returns the commentary response.
// Returns an error on HTTP failure or timeout.
func (c *PythonClient) PostPunch(req PunchRequest) (*CommentaryResponse, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshal: %w", err)
	}

	resp, err := c.httpClient.Post(
		c.baseURL+"/internal/punch",
		"application/json",
		bytes.NewReader(body),
	)
	if err != nil {
		return nil, fmt.Errorf("http post: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("python service returned %d", resp.StatusCode)
	}

	var result CommentaryResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	return &result, nil
}

// Health checks whether the Python service is reachable.
func (c *PythonClient) Health() bool {
	resp, err := c.httpClient.Get(c.baseURL + "/health")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == http.StatusOK
}
