// Box.IO API Gateway
//
// High-concurrency HTTP frontend for the boxing commentary system.
// Each POST /api/v1/punch fans out two goroutines:
//   - commentary: calls Python FastAPI service (LLM inference + RAG)
//   - telemetry:  logs event data (fire-and-forget)
//
// Hard 550ms timeout per request guarantees p95 < 600ms.
package main

import (
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/boxio/gateway/client"
	"github.com/boxio/gateway/handlers"
	"github.com/boxio/gateway/parallel"
	"github.com/boxio/gateway/telemetry"
)

func pythonServiceURL() string {
	if u := os.Getenv("PYTHON_SERVICE_URL"); u != "" {
		return u
	}
	return "http://localhost:8000"
}

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	pythonURL := pythonServiceURL()
	log.Printf("Go gateway starting on :%s — Python service: %s", port, pythonURL)

	// Wire up dependencies
	metrics := telemetry.NewMetrics()
	pyClient := client.NewPythonClient(pythonURL)
	dispatcher := parallel.NewDispatcher(pyClient, metrics)
	punchHandler := handlers.NewPunchHandler(dispatcher, metrics)

	// Wait for Python service to be ready (useful in Docker compose startup)
	waitForPython(pyClient)

	mux := http.NewServeMux()

	// Fight event endpoint
	mux.Handle("/api/v1/punch", punchHandler)

	// Metrics endpoint — shows p50/p95/p99
	mux.HandleFunc("/metrics", func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprintln(w, metrics.Summary())
	})

	// Health check
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		pythonOK := pyClient.Health()
		if pythonOK {
			fmt.Fprintln(w, `{"status":"ok","python_service":"up"}`)
		} else {
			w.WriteHeader(http.StatusServiceUnavailable)
			fmt.Fprintln(w, `{"status":"degraded","python_service":"down"}`)
		}
	})

	srv := &http.Server{
		Addr:         ":" + port,
		Handler:      mux,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 10 * time.Second,
		IdleTimeout:  120 * time.Second,
	}

	log.Printf("Listening on :%s", port)
	if err := srv.ListenAndServe(); err != nil {
		log.Fatalf("server error: %v", err)
	}
}

// waitForPython polls the Python service health endpoint until it responds.
// Gives Docker compose time for the Python service to finish startup.
func waitForPython(c *client.PythonClient) {
	for i := 0; i < 30; i++ {
		if c.Health() {
			log.Println("Python service is ready")
			return
		}
		log.Printf("Waiting for Python service... (%d/30)", i+1)
		time.Sleep(2 * time.Second)
	}
	log.Println("WARNING: Python service did not respond — proceeding anyway")
}
