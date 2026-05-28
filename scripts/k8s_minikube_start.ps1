Set-Location (Split-Path -Parent $PSScriptRoot)

kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secret.template.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/postgres-deployment.yaml
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/mlflow-deployment.yaml
kubectl apply -f k8s/streamlit-deployment.yaml
kubectl apply -f k8s/airflow-deployment.yaml
kubectl apply -f k8s/ingress.yaml

Write-Host "RetailPulse Kubernetes manifests applied."
