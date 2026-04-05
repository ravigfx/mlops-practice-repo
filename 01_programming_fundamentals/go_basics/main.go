// 01 - Go Basics for MLOps
// Practice: HTTP server, JSON handling, health checks, metrics
// Run: go run main.go

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"net/http"
	"sync"
	"sync/atomic"
	"time"
)

// ── Types ─────────────────────────────────────────────────────

type PredictRequest struct {
	SepalLength float64 `json:"sepal_length"`
	SepalWidth  float64 `json:"sepal_width"`
	PetalLength float64 `json:"petal_length"`
	PetalWidth  float64 `json:"petal_width"`
}

type PredictResponse struct {
	Prediction   int                `json:"prediction"`
	ClassName    string             `json:"class_name"`
	Confidence   float64            `json:"confidence"`
	Probabilities map[string]float64 `json:"probabilities"`
	LatencyMs    float64            `json:"latency_ms"`
	ModelVersion string             `json:"model_version"`
}

type HealthResponse struct {
	Status       string    `json:"status"`
	ModelLoaded  bool      `json:"model_loaded"`
	ModelVersion string    `json:"model_version"`
	UptimeSeconds float64  `json:"uptime_seconds"`
	Timestamp    time.Time `json:"timestamp"`
}

type MetricsResponse struct {
	TotalRequests     int64   `json:"total_requests"`
	SuccessRequests   int64   `json:"success_requests"`
	ErrorRequests     int64   `json:"error_requests"`
	AvgLatencyMs      float64 `json:"avg_latency_ms"`
	ActiveConnections int64   `json:"active_connections"`
}

// ── Server ────────────────────────────────────────────────────

type MLServer struct {
	startTime         time.Time
	modelVersion      string
	classNames        []string

	totalRequests     int64 // atomic
	successRequests   int64 // atomic
	errorRequests     int64 // atomic
	activeConnections int64 // atomic

	mu             sync.RWMutex
	totalLatencyMs float64
}

func NewMLServer() *MLServer {
	return &MLServer{
		startTime:    time.Now(),
		modelVersion: "1.0.0",
		classNames:   []string{"setosa", "versicolor", "virginica"},
	}
}

// ── Simulated Model (replace with actual model loader) ────────

func (s *MLServer) predict(req PredictRequest) (int, []float64) {
	// Simple rule-based mock — replace with actual Go ONNX model
	// or call out to Python gRPC service
	score := req.PetalLength*0.5 + req.PetalWidth*0.8

	var probs [3]float64
	switch {
	case score < 2.5:
		probs = [3]float64{0.90, 0.08, 0.02}
	case score < 5.0:
		probs = [3]float64{0.05, 0.85, 0.10}
	default:
		probs = [3]float64{0.02, 0.13, 0.85}
	}

	// Add small random noise
	for i := range probs {
		probs[i] += rand.Float64() * 0.02
	}

	// Normalize
	sum := probs[0] + probs[1] + probs[2]
	for i := range probs {
		probs[i] /= sum
	}

	pred := 0
	for i, p := range probs {
		if p > probs[pred] {
			pred = i
		}
	}
	return pred, probs[:]
}

// ── Handlers ──────────────────────────────────────────────────

func (s *MLServer) handlePredict(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}

	atomic.AddInt64(&s.totalRequests, 1)
	atomic.AddInt64(&s.activeConnections, 1)
	defer atomic.AddInt64(&s.activeConnections, -1)

	start := time.Now()

	var req PredictRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		atomic.AddInt64(&s.errorRequests, 1)
		http.Error(w, `{"error":"invalid request body"}`, http.StatusBadRequest)
		return
	}

	// Validate
	for _, v := range []float64{req.SepalLength, req.SepalWidth, req.PetalLength, req.PetalWidth} {
		if v <= 0 || math.IsNaN(v) {
			atomic.AddInt64(&s.errorRequests, 1)
			http.Error(w, `{"error":"features must be positive numbers"}`, http.StatusUnprocessableEntity)
			return
		}
	}

	pred, probs := s.predict(req)
	latency := float64(time.Since(start).Microseconds()) / 1000.0

	// Track latency
	s.mu.Lock()
	s.totalLatencyMs += latency
	s.mu.Unlock()
	atomic.AddInt64(&s.successRequests, 1)

	probMap := map[string]float64{}
	for i, name := range s.classNames {
		probMap[name] = math.Round(probs[i]*10000) / 10000
	}

	resp := PredictResponse{
		Prediction:    pred,
		ClassName:     s.classNames[pred],
		Confidence:    math.Round(probs[pred]*10000) / 10000,
		Probabilities: probMap,
		LatencyMs:     latency,
		ModelVersion:  s.modelVersion,
	}

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *MLServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	uptime := time.Since(s.startTime).Seconds()
	resp := HealthResponse{
		Status:        "healthy",
		ModelLoaded:   true,
		ModelVersion:  s.modelVersion,
		UptimeSeconds: math.Round(uptime*100) / 100,
		Timestamp:     time.Now().UTC(),
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

func (s *MLServer) handleMetrics(w http.ResponseWriter, r *http.Request) {
	total := atomic.LoadInt64(&s.totalRequests)
	s.mu.RLock()
	avgLatency := 0.0
	if total > 0 {
		avgLatency = s.totalLatencyMs / float64(total)
	}
	s.mu.RUnlock()

	resp := MetricsResponse{
		TotalRequests:     total,
		SuccessRequests:   atomic.LoadInt64(&s.successRequests),
		ErrorRequests:     atomic.LoadInt64(&s.errorRequests),
		AvgLatencyMs:      math.Round(avgLatency*100) / 100,
		ActiveConnections: atomic.LoadInt64(&s.activeConnections),
	}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// ── Middleware ────────────────────────────────────────────────

func loggingMiddleware(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next(w, r)
		log.Printf("%s %s %s", r.Method, r.URL.Path, time.Since(start))
	}
}

// ── Main ──────────────────────────────────────────────────────

func main() {
	server := NewMLServer()
	mux := http.NewServeMux()

	mux.HandleFunc("/predict", loggingMiddleware(server.handlePredict))
	mux.HandleFunc("/health",  loggingMiddleware(server.handleHealth))
	mux.HandleFunc("/metrics", loggingMiddleware(server.handleMetrics))

	addr := ":8080"
	fmt.Printf("🚀 ML Server starting on %s\n", addr)
	fmt.Printf("   POST /predict  — make predictions\n")
	fmt.Printf("   GET  /health   — health check\n")
	fmt.Printf("   GET  /metrics  — server metrics\n")

	// Example test:
	// curl -X POST http://localhost:8080/predict \
	//   -H "Content-Type: application/json" \
	//   -d '{"sepal_length":5.1,"sepal_width":3.5,"petal_length":1.4,"petal_width":0.2}'

	log.Fatal(http.ListenAndServe(addr, mux))
}
