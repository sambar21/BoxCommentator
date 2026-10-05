# Deploy to kind (optional)

Runs the Python commentary service (2 replicas) and the Go gateway in a local kind cluster. The gateway is
published at `http://localhost:8080`.

Pgvector is not deployed here: without a database the service logs a warning and runs without retrieval
(prompts still get the fighter profiles and live stats). Use `docker compose up` for the full RAG stack.

```bash
docker build -t boxio-python:dev .
docker build -t boxio-gateway:dev gateway

kind create cluster --name boxio --config k8s/kind-config.yaml
kind load docker-image boxio-python:dev boxio-gateway:dev --name boxio

kubectl create secret generic boxio-secrets --from-env-file=.env    # GROQ_API_KEY, NEBIUS_API_KEY, ...
kubectl apply -f k8s/python-service.yaml -f k8s/gateway.yaml
kubectl rollout status deploy/python-service deploy/go-gateway

curl -X POST localhost:8080/api/v1/punch -H 'Content-Type: application/json' \
  -d '{"attacker":1,"punch_type":"hook","target":"head","outcome":"landed","damage":55,"knockdown":true,
       "fight_id":"demo","fighter1_name":"Alvarez","fighter2_name":"Garcia"}'
```

Use another backend by changing `COMMENTARY_BACKEND` in `k8s/python-service.yaml` (a name from
`config/backends.yaml`, e.g. `nebius`) and re-applying. Tear down with `kind delete cluster --name boxio`.

Not yet verified: this has been written but not run (no cluster was started while building it).
