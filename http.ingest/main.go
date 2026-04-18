// http.ingest – Receive mirrored HTTP traffic, dump the raw request,
// and forward it as JSON to the BlueAgent WAF /api/analyze endpoint.
//
// This service does one thing: capture raw HTTP and forward it.
// All analysis, queuing, and logging are handled by BlueAgent.
//
// Environment variables:
//
//	PORT   – Listen port (default "9002").
//	AI_URL – BlueAgent analyze endpoint, e.g. "http://blueagent:5000/api/analyze".
package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"os"
	"strings"
	"time"
)

// analyzeRequest is the JSON body sent to POST /api/analyze.
type analyzeRequest struct {
	RawHTTP string `json:"raw_http"`
}

// shared HTTP client for forwarding (connection pooling).
var aiClient = &http.Client{Timeout: 30 * time.Second}

func main() {
	port := getenv("PORT", "9002")
	aiURL := getenv("AI_URL", "")

	mux := http.NewServeMux()

	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok\n"))
	})

	mux.HandleFunc("/favicon.ico", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusNoContent)
	})

	// Catch-all: capture any HTTP request and forward to BlueAgent.
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		defer func() { _ = r.Body.Close() }()

		// Skip requests whose Host header is an IP (not a domain name).
		if !isDomain(r.Host) {
			w.WriteHeader(http.StatusOK)
			_, _ = w.Write([]byte("ok\n"))
			return
		}

		bodyBytes, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "failed to read body", http.StatusBadRequest)
			return
		}

		rawDump := dumpRawRequest(r, bodyBytes)

		if aiURL != "" {
			go forward(aiURL, rawDump)
		}

		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok\n"))
	})

	srv := &http.Server{
		Addr:              ":" + port,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
		ReadTimeout:       10 * time.Second,
		WriteTimeout:      5 * time.Second,
		IdleTimeout:       60 * time.Second,
	}

	log.Printf("http.ingest listening on :%s -> %s\n", port, aiURL)
	log.Fatal(srv.ListenAndServe())
}

// dumpRawRequest produces a full raw HTTP dump (request line + headers + body).
func dumpRawRequest(r *http.Request, body []byte) string {
	rr := new(http.Request)
	*rr = *r
	rr.Body = io.NopCloser(bytes.NewReader(body))
	b, err := httputil.DumpRequest(rr, true)
	if err != nil {
		target := r.URL.Path
		if r.URL.RawQuery != "" {
			target += "?" + r.URL.RawQuery
		}
		return fmt.Sprintf("%s %s HTTP/1.1\r\n\r\n", r.Method, target)
	}
	return string(b)
}

// isDomain returns true if the host string looks like a domain name
// rather than a raw IP address (with or without a port).
func isDomain(host string) bool {
	// Strip port if present (e.g. "example.com:8080" → "example.com")
	h := host
	if hostname, _, err := net.SplitHostPort(host); err == nil {
		h = hostname
	}
	if h == "" {
		return false
	}
	// If it parses as an IP, it's not a domain.
	if net.ParseIP(h) != nil {
		return false
	}
	return true
}

// forward POSTs the raw HTTP text to BlueAgent's /api/analyze endpoint.
func forward(aiURL string, rawDump string) {
	payload, err := json.Marshal(analyzeRequest{RawHTTP: rawDump})
	if err != nil {
		return
	}

	req, err := http.NewRequest("POST", aiURL, bytes.NewReader(payload))
	if err != nil {
		return
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := aiClient.Do(req)
	if err != nil {
		return
	}
	defer func() { _ = resp.Body.Close() }()
	_, _ = io.Copy(io.Discard, resp.Body)
}

func getenv(key, def string) string {
	v := strings.TrimSpace(os.Getenv(key))
	if v == "" {
		return def
	}
	return v
}
